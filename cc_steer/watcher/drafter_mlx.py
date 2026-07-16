"""Stage 2 in production: the lab-trained LoRA watcher, served locally over MLX.

E2's winner — a QLoRA adapter on the 4-bit Qwen3-4B-Instruct-2507 base with
score-based sentinel abstention — is trained by the lab and registered as the
``watcher`` component; this module only loads and serves. A registered version
directory holds the mlx-lm adapter pair (``adapters.safetensors`` +
``adapter_config.json``) and a ``metadata.json`` whose keys this module reads:

* ``base_model`` — the mlx model id the adapter was trained over.
* ``thresholds`` — abstain thresholds on first-token P(NO_STEER), keyed by
  operating point (``budget``: precision-first, the shadow default; ``f1``).
* ``render_version`` — the prompt-rendering contract the adapter trained on.

The decision is score-based, never greedy string-match (the argmax trap: the
model parks on the NO_STEER sentinel even when the fire signal is sub-argmax).
``decide`` abstains iff P(NO_STEER) >= threshold and otherwise generates with
the sentinel token banned, so a fired moment always yields steer text. The
input contract is the lab's training rendering verbatim:
``flattened(tail_messages(prompt, DRAFT_CHAR_CAP))`` under ``DRAFT_SYSTEM``.

mlx-lm lives behind the ``mlx`` extra and is imported lazily, mirroring
:mod:`cc_steer.watcher.gate`'s handling of the ``gate`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cc_steer import registry
from cc_steer.rendering import DRAFT_CHAR_CAP, NO_STEER, strip_think, tail_messages
from cc_steer.watcher.cascade import DRAFT_SYSTEM, flattened
from cc_steer.watcher.types import Draft

if TYPE_CHECKING:
    from pathlib import Path

    from cc_steer.rendering import Message

COMPONENT = "watcher"
ADAPTER_NAME = "adapters.safetensors"
ADAPTER_CONFIG_NAME = "adapter_config.json"
THRESHOLD_KEYS = ("budget", "f1")
MAX_DRAFT_TOKENS = 256


class MlxDrafter:
    """The trained stage-2 drafter: base 4-bit + adapter, greedy decoding, score-based abstain.

    Implements the cascade's ``Drafter`` protocol. Loads the promoted registry
    version by default; pass ``version`` to pin one.

    Args:
        version: The registry version to load; defaults to ``current("watcher")``.
        root: The registry root override, for tests.
        operating_point: Which metadata threshold to abstain at (``budget``/``f1``).
        threshold: An explicit abstain threshold overriding the metadata.

    Raises:
        RuntimeError: When no watcher version is promoted, the artifact files
            are missing, the metadata lacks the needed keys, or the ``mlx``
            extra is not installed.
    """

    def __init__(
        self,
        version: registry.VersionInfo | None = None,
        *,
        root: Path | None = None,
        operating_point: str = "budget",
        threshold: float | None = None,
    ) -> None:
        resolved = version or registry.current(COMPONENT, root=root)
        if resolved is None:
            raise RuntimeError(
                "no promoted watcher model: train and promote one with `cc-steer retrain --component watcher`"
            )
        for name in (ADAPTER_NAME, ADAPTER_CONFIG_NAME):
            if not (resolved.path / name).exists():
                raise RuntimeError(f"watcher version {resolved.version} has no {name} at {resolved.path / name}")
        self.version = resolved
        self.base_model = str(self._metadata("base_model"))
        self.threshold = threshold if threshold is not None else self._metadata_threshold(operating_point)
        self.operating_point = "explicit" if threshold is not None else operating_point
        self.render_version = int(str(self._metadata("render_version")))
        loaded = _mlx_lm().load(self.base_model, adapter_path=str(resolved.path))
        self.model, self.tokenizer = loaded[0], loaded[1]

    async def draft(self, prompt: list[Message]) -> Draft:
        """One score-based decision over the rendered window.

        The tail-capped flattening is the training contract — never plain
        ``flattened(prompt)``. Inference runs synchronously on the caller's
        thread: MLX streams are bound to the thread that first touched the
        model, so a worker-thread hop dies with "no Stream(cpu, 0) in current
        thread". Blocking the daemon loop for the ~seconds a decision takes is
        fine — sessions are evaluated one at a time and ingest buffers in the
        transcript files.
        """
        context_tail = flattened(tail_messages(prompt, DRAFT_CHAR_CAP))
        return self.decide(context_tail)

    def decide(self, context_tail: str) -> Draft:
        """Abstain (``NO_STEER``) iff P(NO_STEER) >= threshold, else a sentinel-suppressed steer."""
        p = self.nosteer_prob(context_tail)
        if p >= self.threshold:
            return Draft(NO_STEER, p)
        return Draft(self.draft_suppressed(context_tail), p)

    def nosteer_prob(self, context_tail: str) -> float:
        """First-token P(NO_STEER sentinel) at the answer position — the abstain score.

        Sub-argmax by design: greedy may pick NO_STEER while the fire signal
        lives in this probability (the E2 diagnostic). The softmax normalizes in
        float32: bf16 ``logsumexp`` rounding shifts mid-confidence scores by up to
        ~0.03, enough to move a served decision off the value the score was fit at.
        """
        import mlx.core as mx

        prefix, sentinel = self._prefix_and_sentinel(context_tail)
        logits = self.model(mx.array(prefix)[None])[0, -1].astype(mx.float32)
        logp = logits - mx.logsumexp(logits)
        return float(mx.exp(logp[sentinel]))

    def draft_suppressed(self, context_tail: str, *, max_tokens: int = MAX_DRAFT_TOKENS) -> str:
        """Greedy draft with the NO_STEER sentinel token banned: the steer a fired moment gets."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        _, sentinel = self._prefix_and_sentinel(context_tail)
        raw = generate(
            self.model,
            self.tokenizer,
            prompt=self._chat_prompt(context_tail),
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
            logits_processors=make_logits_processors(logit_bias={int(sentinel): -1e9}),
        )
        return strip_think(raw)

    def _chat_prompt(self, context_tail: str) -> list[int]:
        messages = [{"role": "system", "content": DRAFT_SYSTEM}, {"role": "user", "content": context_tail}]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def _prefix_and_sentinel(self, context_tail: str) -> tuple[list[int], int]:
        """Templated ids up to the answer-content position and the sentinel's first token id there.

        The content start is found by diverging a NO_STEER assistant turn from
        a dummy one, so it is exact regardless of how the template tokenizes
        the injected ``<think></think>`` scaffold.
        """
        base = [{"role": "system", "content": DRAFT_SYSTEM}, {"role": "user", "content": context_tail}]
        tmpl = self.tokenizer.apply_chat_template
        ids_ns = tmpl(base + [{"role": "assistant", "content": NO_STEER}], add_generation_prompt=False)
        ids_dm = tmpl(base + [{"role": "assistant", "content": "zzz other"}], add_generation_prompt=False)
        i = 0
        while i < len(ids_ns) and i < len(ids_dm) and ids_ns[i] == ids_dm[i]:
            i += 1
        return ids_ns[:i], ids_ns[i]

    def _metadata(self, key: str) -> object:
        if key not in self.version.metadata:
            raise RuntimeError(f"watcher version {self.version.version} metadata carries no {key!r}")
        return self.version.metadata[key]

    def _metadata_threshold(self, operating_point: str) -> float:
        thresholds = self.version.metadata.get("thresholds")
        if not isinstance(thresholds, dict) or operating_point not in thresholds:
            raise RuntimeError(
                f"watcher version {self.version.version} metadata carries no thresholds[{operating_point!r}] "
                f"(expected keys {THRESHOLD_KEYS})"
            )
        return float(thresholds[operating_point])


def _mlx_lm() -> Any:
    """The mlx_lm module; requires the ``mlx`` extra."""
    try:
        import mlx_lm
    except ImportError as error:
        raise RuntimeError("the local watcher requires the 'mlx' extra: pip install 'cc-steer[mlx]'") from error
    return mlx_lm
