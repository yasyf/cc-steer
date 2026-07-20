"""Stage 2 over a hosted vLLM endpoint: exact sentinel scoring by ``prompt_logprobs``, fail-open on outage.

The HTTP sibling of :class:`~cc_steer.watcher.drafter_mlx.MlxDrafter`: it drafts against the hosted
scale-to-zero vLLM endpoint (the athome ``[serve.modal-vllm]`` recipe) and satisfies the cascade's
``Drafter`` protocol unchanged. It scores the abstain decision the same way MlxDrafter does —
first-token P(NO_STEER) at the answer position — but reads that probability from vLLM's
``prompt_logprobs`` extension: the client renders the ``[system, user, assistant=NO_STEER]`` turn to
token ids through the base tokenizer, teacher-forces ``prefix_ids + [sentinel_id]`` as a
``max_tokens=1`` completion, and reads the exact ``log P(sentinel | prefix)`` back off the last
``prompt_logprobs`` position. That probability is exact and rank-independent — every row is scorable,
unlike a top-``k`` chat-completions logprobs read where a below-top-``k`` sentinel has no reportable
value.

The answer position is found by the divergence method every scoring path shares
(:func:`~cc_steer.retrain.sentinel.prefix_and_sentinel_via`), so the ``<think></think>`` scaffold the
Qwen3 template injects is captured template-exactly rather than assumed with a hardcoded prefill
string. ``prompt_logprobs`` is a vLLM extension the pure OpenAI API does not implement, so this
drafter requires a vLLM-compatible endpoint; the steer text on a fired moment is one ordinary
``/v1/chat/completions`` call.

Two things set it apart from the local drafter. First, the endpoint lives across a network the
daemon's single-threaded scoring loop cannot afford to hang on, so every call carries an explicit
timeout from config. Second, an unreachable endpoint, a timeout, or a malformed response FAILS OPEN
to ``NO_STEER`` with a visible line on stderr — the one sanctioned fail-open in this fail-fast
codebase, so the daemon keeps observing turns while the hosted backend is down; the boundary is a
narrow ``(httpx.HTTPError, json.JSONDecodeError, DrafterResponseError)`` catch around the network
calls alone — transport faults, a non-JSON body, and an unreadable shape — never a broad swallow.
Nothing selects this backend unless an operator asks for it (``--drafter http``); mlx stays the
default.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import click
import httpx
from athome.train.spec import BASE_MODELS

from cc_steer.rendering import DRAFT_CHAR_CAP, NO_STEER, strip_think, tail_messages
from cc_steer.retrain.sentinel import prefix_and_sentinel_via
from cc_steer.watcher.cascade import DRAFT_SYSTEM, flattened
from cc_steer.watcher.types import Draft

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from cc_steer.rendering import Message

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
COMPLETIONS_PATH = "/v1/completions"
MAX_DRAFT_TOKENS = 256
DEFAULT_THRESHOLD = 0.5
WATCHER_BASE = BASE_MODELS["qwen3-8b"]


class DrafterResponseError(Exception):
    """A completions or chat-completions response the drafter could not read as a sentinel score or a steer."""


def sentinel_prob(payload: object, sentinel: int) -> float:
    """Exact first-token ``P(NO_STEER)`` from a vLLM ``prompt_logprobs`` completions payload.

    The teacher-forced prompt ends at ``sentinel``, so the last ``prompt_logprobs`` position carries
    ``log P(sentinel | prefix)`` under the ``str(sentinel)`` key; ``P(NO_STEER)`` is its ``exp``.

    Args:
        payload: The decoded ``/v1/completions`` JSON body.
        sentinel: The forced answer-position token id whose logprob is read.

    Raises:
        DrafterResponseError: When the payload carries no ``prompt_logprobs`` list, or its last
            position holds no logprob for ``sentinel``.
    """
    match payload:
        case {"choices": [{"prompt_logprobs": [*_, dict(final)]}, *_]}:
            scored = final.get(str(sentinel))
        case _:
            raise DrafterResponseError(
                f"endpoint returned no prompt_logprobs and is likely not vLLM-compatible: {str(payload)[:200]}"
            )
    match scored:
        case {"logprob": int() | float() as logprob}:
            return math.exp(float(logprob))
        case _:
            raise DrafterResponseError(
                f"sentinel {sentinel} carries no logprob at the answer position: {str(scored)[:120]}"
            )


def draft_text(payload: object) -> str:
    """The assistant message content from a chat-completions payload.

    Raises:
        DrafterResponseError: When the payload carries no message content.
    """
    match payload:
        case {"choices": [{"message": {"content": str(content)}}, *_]}:
            return content
        case _:
            raise DrafterResponseError(f"no message content in chat-completions payload: {str(payload)[:200]}")


def base_tokenizer() -> PreTrainedTokenizerBase:
    """The base-model tokenizer at its pinned revision; requires the ``http`` extra for ``transformers``."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(WATCHER_BASE.hf, revision=WATCHER_BASE.hf_revision)


class HttpDrafter:
    """Stage 2 over a hosted vLLM endpoint, scored by exact ``prompt_logprobs``, fail-open on outage.

    Implements the cascade's ``Drafter`` protocol. Scoring renders the answer position to token ids
    with the base tokenizer and reads ``log P(NO_STEER | prefix)`` from one ``max_tokens=1``
    ``/v1/completions`` call carrying vLLM's ``prompt_logprobs`` extension; the drafter abstains
    (``NO_STEER``) when that P(NO_STEER) meets ``threshold`` and otherwise issues a
    ``/v1/chat/completions`` call for the steer text. Every call carries ``timeout``, and a transport
    error, timeout, or malformed response fails open to ``Draft(NO_STEER, None)`` with a visible
    stderr line rather than stalling the daemon loop. Because ``prompt_logprobs`` is a vLLM extension
    rather than baseline OpenAI, the endpoint must be vLLM-compatible.

    Args:
        endpoint: The endpoint base URL; ``/v1/completions`` and ``/v1/chat/completions`` are appended.
        model: The served model/adapter name — the OpenAI ``model`` field.
        threshold: Abstain when first-token P(NO_STEER) meets this.
        timeout: Per-request timeout in seconds, applied to every call.
        api_key: A bearer token for the endpoint, or None for an unauthenticated one.
        transport: An httpx transport override, for tests; None uses the network.
        tokenizer: A base-model tokenizer override, for tests; None loads
            ``Qwen/Qwen3-8B`` at its pinned revision.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        threshold: float,
        timeout: float,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.threshold = threshold
        self.timeout = timeout
        # fail fast at daemon start on a broken http env (missing extra or failed HF download), not hours into the loop
        self._tokenizer = tokenizer if tokenizer is not None else base_tokenizer()
        self.client = httpx.AsyncClient(
            base_url=self.endpoint,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            transport=transport,
            # Modal's scale-to-zero wake handshake answers the first cold request with a 303
            follow_redirects=True,
        )

    async def draft(self, prompt: list[Message]) -> Draft:
        """One score-based decision over the rendered window, failing open to ``NO_STEER`` on any endpoint fault.

        Feeds the endpoint the training contract — ``NO_STEER`` when P(NO_STEER) meets the threshold,
        otherwise a freshly generated steer. A network error, timeout, or unreadable response returns
        ``Draft(NO_STEER, None)`` after a visible stderr line, so the daemon keeps observing.
        """
        tail = flattened(tail_messages(prompt, DRAFT_CHAR_CAP))
        try:
            prob = await self.nosteer_prob(tail)
            if prob >= self.threshold:
                return Draft(NO_STEER, prob)
            return Draft(await self.generate(tail), prob)
        except (httpx.HTTPError, json.JSONDecodeError, DrafterResponseError) as error:
            click.echo(f"http drafter failed open to NO_STEER ({type(error).__name__}: {error})", err=True)
            return Draft(NO_STEER, None)

    async def nosteer_prob(self, tail: str) -> float:
        """Exact first-token P(NO_STEER) at the answer position, via vLLM ``prompt_logprobs``.

        Renders the divergence-found ``prefix_ids + [sentinel_id]`` for the ``[system, user,
        assistant=NO_STEER]`` turn and teacher-forces it as a ``max_tokens=1`` completion, reading
        ``exp(log P(sentinel | prefix))`` off the last ``prompt_logprobs`` position — exact and
        rank-independent, so every row is scorable.
        """
        prefix, sentinel = self._prefix_and_sentinel(tail)
        response = await self.client.post(
            COMPLETIONS_PATH,
            json={
                "model": self.model,
                "prompt": [*prefix, sentinel],
                "max_tokens": 1,
                "temperature": 0.0,
                "prompt_logprobs": 0,
            },
        )
        response.raise_for_status()
        return sentinel_prob(response.json(), sentinel)

    async def generate(self, tail: str) -> str:
        """The steer text for a fired moment: one greedy chat completion, think-scaffold stripped.

        Unlike MlxDrafter this cannot ban the sentinel token — the OpenAI ``logit_bias`` takes
        token ids, which the pure ``/v1`` surface never exposes without the model's tokenizer.
        Generation runs only once the score already fell below the abstain threshold, so the
        greedy continuation is a steer in the ordinary case; a rare argmax collapse back to
        ``NO_STEER`` reads as an abstention downstream
        (:func:`~cc_steer.watcher.cascade.steer_or_none`).
        """
        response = await self.client.post(
            CHAT_COMPLETIONS_PATH,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": DRAFT_SYSTEM},
                    {"role": "user", "content": tail},
                ],
                "max_tokens": MAX_DRAFT_TOKENS,
                "temperature": 0.0,
            },
        )
        response.raise_for_status()
        return strip_think(draft_text(response.json()))

    def _prefix_and_sentinel(self, tail: str) -> tuple[list[int], int]:
        tokenizer = self._tokenizer

        def render(answer: str) -> list[int]:
            return tokenizer.encode(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": DRAFT_SYSTEM},
                        {"role": "user", "content": tail},
                        {"role": "assistant", "content": answer},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                ),
                add_special_tokens=False,
            )

        return prefix_and_sentinel_via(render)

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self.client.aclose()
