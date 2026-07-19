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

The base+adapter weights load lazily on first use and unload once idle: the
registry/metadata resolution and the ``mlx`` import guard stay eager (a broken
env fails fast at daemon start), but the multi-GB weight load is deferred and
an :class:`~athome.idle.IdleResource` reaps it after ``idle_ttl_s`` of quiet,
so a watcher that scores a burst then sits silent for hours releases its
memory. Every synchronous scoring entry point self-wakes through
:meth:`load`, so the daemon and the retrain sweep share one lazy path.

mlx-lm lives behind the ``mlx`` extra and is imported lazily, mirroring
:mod:`cc_steer.watcher.gate`'s handling of the ``gate`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from athome.idle import IdleResource

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
IDLE_TTL_S = 900.0


class MlxDrafter:
    """The trained stage-2 drafter: base 4-bit + adapter, greedy decoding, score-based abstain.

    Implements the cascade's ``Drafter`` protocol. Loads the promoted registry
    version by default; pass ``version`` to pin one. The base+adapter weights
    load lazily on the first scoring call and unload after ``idle_ttl_s`` idle
    seconds via :attr:`resource`; drive the reaper with ``resource.run``.

    Args:
        version: The registry version to load; defaults to ``current("watcher")``.
        root: The registry root override, for tests.
        operating_point: Which metadata threshold to abstain at (``budget``/``f1``).
        threshold: An explicit abstain threshold overriding the metadata.
        idle_ttl_s: Idle seconds the loaded weights sit unused before the reaper unloads them.

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
        idle_ttl_s: float = IDLE_TTL_S,
    ) -> None:
        resolved = version or registry.current(COMPONENT, root=root)
        if resolved is None:
            raise RuntimeError(
                "no promoted watcher model: train and promote one with `cc-steer retrain --component watcher`"
            )
        for name in (ADAPTER_NAME, ADAPTER_CONFIG_NAME):
            if not (resolved.path / name).exists():
                raise RuntimeError(f"watcher version {resolved.version} has no {name} at {resolved.path / name}")
        _mlx_lm()  # fail fast at daemon start on a broken mlx env; only the weight load below is deferred
        self.version = resolved
        self.base_model = str(self._metadata("base_model"))
        self.threshold = threshold if threshold is not None else self._metadata_threshold(operating_point)
        self.operating_point = "explicit" if threshold is not None else operating_point
        self.render_version = int(str(self._metadata("render_version")))
        self._loaded: tuple[Any, Any] | None = None
        self.resource: IdleResource[tuple[Any, Any]] = IdleResource(
            self._load_async, self._unload_async, ttl_s=idle_ttl_s
        )

    def load(self) -> tuple[Any, Any]:
        """The cached ``(model, tokenizer)``, loading the base+adapter weights on first call."""
        if self._loaded is None:
            loaded = _mlx_lm().load(self.base_model, adapter_path=str(self.version.path))
            self._loaded = (loaded[0], loaded[1])
        return self._loaded

    def unload(self) -> None:
        """Drop the loaded weights, then release MLX's buffer cache — the reference falls first."""
        import mlx.core as mx

        self._loaded = None
        mx.clear_cache()

    async def _load_async(self) -> tuple[Any, Any]:
        return self.load()

    async def _unload_async(self) -> None:
        self.unload()

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
        async with self.resource.use():
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

        model, _ = self.load()
        prefix, sentinel = self._prefix_and_sentinel(context_tail)
        logits = model(mx.array(prefix)[None])[0, -1].astype(mx.float32)
        logp = logits - mx.logsumexp(logits)
        return float(mx.exp(logp[sentinel]))

    def clear_cache(self) -> None:
        """Release MLX's buffer cache, bounding peak memory across a full-frame scoring sweep.

        MLX pools freed buffers by size, so scoring hundreds of variable-length rows
        back-to-back grows resident memory unboundedly (the E12 Jetsam kill). Calling
        this per row caps the working set at a single forward pass.
        """
        import mlx.core as mx

        mx.clear_cache()

    def draft_suppressed(self, context_tail: str, *, max_tokens: int = MAX_DRAFT_TOKENS) -> str:
        """Greedy draft with the NO_STEER sentinel token banned: the steer a fired moment gets."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        model, tokenizer = self.load()
        _, sentinel = self._prefix_and_sentinel(context_tail)
        raw = generate(
            model,
            tokenizer,
            prompt=self._chat_prompt(context_tail),
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
            logits_processors=make_logits_processors(logit_bias={int(sentinel): -1e9}),
        )
        return strip_think(raw)

    def _chat_prompt(self, context_tail: str) -> list[int]:
        _, tokenizer = self.load()
        messages = [{"role": "system", "content": DRAFT_SYSTEM}, {"role": "user", "content": context_tail}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def _prefix_and_sentinel(self, context_tail: str) -> tuple[list[int], int]:
        """Templated ids up to the answer-content position and the sentinel's first token id there.

        The content start is found by diverging a NO_STEER assistant turn from
        a dummy one, so it is exact regardless of how the template tokenizes
        the injected ``<think></think>`` scaffold.
        """
        _, tokenizer = self.load()
        base = [{"role": "system", "content": DRAFT_SYSTEM}, {"role": "user", "content": context_tail}]
        tmpl = tokenizer.apply_chat_template
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
