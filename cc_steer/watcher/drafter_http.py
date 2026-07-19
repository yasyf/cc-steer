"""Stage 2 over a hosted OpenAI-compatible endpoint: sentinel scoring by logprobs, fail-open on outage.

The HTTP sibling of :class:`~cc_steer.watcher.drafter_mlx.MlxDrafter`: it drafts against any
OpenAI-compatible ``/v1/chat/completions`` surface — the hosted scale-to-zero vLLM endpoint is
served with ``--max-logprobs 40`` — and satisfies the cascade's ``Drafter`` protocol unchanged.
It scores the abstain decision the same way MlxDrafter does — first-token P(NO_STEER) at the
answer position — but reads that probability from the endpoint's top-``k`` logprobs rather than an
exact full-vocabulary softmax, so its numbers differ from the local substrate's; calibrating a
threshold for this backend is a separate, later concern. Reaching the answer position needs the
same boundary MlxDrafter walks locally: this LoRA emits a ``<think>\n\n</think>\n\n`` scaffold
before its answer (see :mod:`cc_steer.rendering`), so scoring teacher-forces that scaffold as an
assistant prefill (``THINK_PREFILL``) with vLLM's ``continue_final_message`` /
``add_generation_prompt`` chat extensions, putting the first scored token at the answer position
rather than at the ``<think>`` opener the model generates first. ``continue_final_message`` is a
vLLM (and compatible) extension the pure OpenAI API rejects — the hosted vLLM endpoint (the athome
``[serve.modal-vllm]`` recipe) is the deployment target, and a 400 from an incompatible server
flows through the sanctioned fail-open below with a visible stderr line.

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

from cc_steer.rendering import DRAFT_CHAR_CAP, NO_STEER, strip_think, tail_messages
from cc_steer.watcher.cascade import DRAFT_SYSTEM, flattened
from cc_steer.watcher.types import Draft

if TYPE_CHECKING:
    from cc_steer.rendering import Message

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
THINK_PREFILL = "<think>\n\n</think>\n\n"
TOP_LOGPROBS = 20
MAX_DRAFT_TOKENS = 256
DEFAULT_THRESHOLD = 0.5


class DrafterResponseError(Exception):
    """A chat-completions response the drafter could not read as a first-token score or a steer."""


def sentinel_candidate(entry: object) -> tuple[str, float] | None:
    """A ``(sentinel-prefix token, logprob)`` pair for one ``top_logprobs`` entry, or None when it is not one.

    Raises:
        DrafterResponseError: When the entry is not a ``token``/``logprob`` record.
    """
    match entry:
        case {"token": token, "logprob": int() | float() as logprob} if (
            stripped := str(token).strip()
        ) and NO_STEER.startswith(stripped):
            return stripped, float(logprob)
        case {"token": _, "logprob": int() | float()}:
            return None
        case _:
            raise DrafterResponseError(f"unexpected top_logprobs entry: {str(entry)[:120]}")


def sentinel_prob(payload: object) -> float | None:
    """First-token P(NO_STEER) from a chat-completions logprobs payload, or None below the endpoint's top-k.

    Approximates :meth:`MlxDrafter.nosteer_prob` against the OpenAI surface. Without the endpoint's
    tokenizer the canonical first token of ``NO_STEER`` is unknowable, so among the top-``k``
    alternatives at the answer position that ``NO_STEER`` begins with, this takes the
    highest-probability one and returns ``exp`` of its logprob. For a model whose abstain mass
    concentrates on that canonical first token, max-prob selects it in the realistic case and
    degrades gracefully otherwise — unlike a longest-prefix pick, which BPE's merge-rank ordering
    can send to a longer non-canonical merge (e.g. ``NO_STE``) that co-occurs with the canonical
    ``NO``. A sentinel that fell outside the endpoint's top-``k`` has no reportable probability and
    yields None — the score-less state :class:`~cc_steer.watcher.types.Draft` already carries for
    the spawn path.

    Args:
        payload: The decoded ``/v1/chat/completions`` JSON body.

    Raises:
        DrafterResponseError: When the payload carries no first-token logprobs.
    """
    match payload:
        case {"choices": [{"logprobs": {"content": [{"top_logprobs": list(entries)}, *_]}}, *_]}:
            candidates = [pair for entry in entries if (pair := sentinel_candidate(entry)) is not None]
        case _:
            raise DrafterResponseError(f"no first-token logprobs in chat-completions payload: {str(payload)[:200]}")
    if not candidates:
        return None
    return math.exp(max(candidates, key=lambda pair: pair[1])[1])


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


class HttpDrafter:
    """Stage 2 over a hosted OpenAI-compatible endpoint, scored by top-k logprobs, fail-open on outage.

    Implements the cascade's ``Drafter`` protocol. Scoring is one ``max_tokens=1`` call whose
    top-``k`` logprobs yield P(NO_STEER); the drafter abstains (``NO_STEER``) when that meets
    ``threshold`` and otherwise issues a second call for the steer text. Every call carries
    ``timeout``, and a transport error, timeout, or malformed response fails open to
    ``Draft(NO_STEER, None)`` with a visible stderr line rather than stalling the daemon loop.

    Args:
        endpoint: The endpoint base URL; ``/v1/chat/completions`` is appended.
        model: The model name the endpoint serves — the OpenAI ``model`` field.
        threshold: Abstain when first-token P(NO_STEER) meets this; substrate-specific, so
            calibrating it for a hosted model is a later, separate concern.
        timeout: Per-request timeout in seconds, applied to every call.
        api_key: A bearer token for the endpoint, or None for an unauthenticated one.
        transport: An httpx transport override, for tests; None uses the network.
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
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.threshold = threshold
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=self.endpoint,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            transport=transport,
        )

    async def draft(self, prompt: list[Message]) -> Draft:
        """One score-based decision over the rendered window, failing open to ``NO_STEER`` on any endpoint fault.

        Feeds the endpoint the training contract — ``NO_STEER`` when P(NO_STEER) meets the
        threshold (or a sentinel below the endpoint's top-k leaves it unscored), otherwise a
        freshly generated steer. A network error, timeout, or unreadable response returns
        ``Draft(NO_STEER, None)`` after a visible stderr line, so the daemon keeps observing.
        """
        messages = [
            {"role": "system", "content": DRAFT_SYSTEM},
            {"role": "user", "content": flattened(tail_messages(prompt, DRAFT_CHAR_CAP))},
        ]
        try:
            prob = await self.nosteer_prob(messages)
            if prob is not None and prob >= self.threshold:
                return Draft(NO_STEER, prob)
            return Draft(await self.generate(messages), prob)
        except (httpx.HTTPError, json.JSONDecodeError, DrafterResponseError) as error:
            click.echo(f"http drafter failed open to NO_STEER ({type(error).__name__}: {error})", err=True)
            return Draft(NO_STEER, None)

    async def nosteer_prob(self, messages: list[dict[str, str]]) -> float | None:
        """First-token P(NO_STEER) from the endpoint's top-k logprobs, or None when the sentinel is below top-k.

        Teacher-forces the ``<think>\\n\\n</think>\\n\\n`` scaffold as an assistant prefill so the
        single scored token lands at the answer position — the boundary
        :meth:`MlxDrafter._prefix_and_sentinel` finds locally — rather than at the ``<think>``
        opener the LoRA generates first. ``continue_final_message`` / ``add_generation_prompt`` are
        vLLM chat extensions the pure OpenAI API rejects; a 400 from an incompatible server fails
        open through :meth:`draft`.
        """
        response = await self.client.post(
            CHAT_COMPLETIONS_PATH,
            json={
                "model": self.model,
                "messages": [*messages, {"role": "assistant", "content": THINK_PREFILL}],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": TOP_LOGPROBS,
                "add_generation_prompt": False,
                "continue_final_message": True,
            },
        )
        response.raise_for_status()
        return sentinel_prob(response.json())

    async def generate(self, messages: list[dict[str, str]]) -> str:
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
                "messages": messages,
                "max_tokens": MAX_DRAFT_TOKENS,
                "temperature": 0.0,
            },
        )
        response.raise_for_status()
        return strip_think(draft_text(response.json()))

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self.client.aclose()
