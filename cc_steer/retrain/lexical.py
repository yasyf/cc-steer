"""The gate-component retrain lane: refresh the promoted lexical gate when the data moved.

The E1 bake-off winner — TF-IDF word 1-2 grams + char 3-5 grams into a balanced
logistic regression, temperature-scaled — is trained here on the exported gate train
view and evaluated on the frozen gate eval. A candidate is promoted through the model
registry only when it beats the incumbent's frozen-eval PR-AUC without regressing
recall at the 2 fires/100 alert budget (:func:`~cc_steer.retrain.promotion.gate_promotable`).
Every pass — skip, reject, promote — journals one line through
:func:`~cc_steer.retrain.promotion.journal`.

The artifact spec (file name, joblib payload keys, threshold key) is owned by
:mod:`cc_steer.watcher.gate`; :class:`LexicalGateTrainer` only fills it, so the trained
model loads straight back through the inference :class:`~cc_steer.watcher.gate.LexicalGate`.
scikit-learn and scipy live behind the ``gate`` extra and are imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from cc_steer import registry
from cc_steer.retrain import data, evalset, promotion
from cc_steer.retrain.promotion import PR_AUC_KEY, RECALL_KEY
from cc_steer.watcher.gate import ARTIFACT_NAME, COMPONENT, THRESHOLD_KEY

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pyarrow as pa
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion

KEEP_VERSIONS = 3
BUDGET_FIRES_PER_100 = 2.0
THRESHOLD_METRIC = "threshold_at_2per100_viewratio_proxy"

# The E1 bake-off winner (cell "lexical", grid pick word_min_df=5, C=4.0).
RECIPE: dict[str, Any] = {
    "family": "lexical",
    "model": "tfidf(word12+char35)+logreg-balanced",
    "word_min_df": 5,
    "char_min_df": 5,
    "char_max_features": 300_000,
    "C": 4.0,
    "seed": 1729,
    "val_frac": 0.1,
}


@dataclass(frozen=True, slots=True)
class GateFrame:
    """Column-oriented slice of the gate view: the text plus the ranking labels and strata."""

    ids: tuple[str, ...]
    texts: tuple[str, ...]
    labels: np.ndarray
    kinds: tuple[str, ...]
    offset_turns: np.ndarray
    source_kinds: tuple[str, ...]

    @classmethod
    def from_table(cls, table: pa.Table) -> GateFrame:
        return cls(
            ids=tuple(table.column("id").to_pylist()),
            texts=tuple(table.column("text").to_pylist()),
            labels=np.asarray(table.column("label").to_pylist(), dtype=bool),
            kinds=tuple(table.column("kind").to_pylist()),
            offset_turns=np.asarray(table.column("offset_turns").to_pylist(), dtype=np.int64),
            source_kinds=tuple(table.column("source_kind").to_pylist()),
        )

    @classmethod
    def load_train(cls, *, dataset_dir: Path | None = None) -> GateFrame:
        import pyarrow.parquet as pq

        path = (dataset_dir or data.DATASET_DIR) / "gate" / "train.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no gate train parquet at {path}")
        return cls.from_table(pq.read_table(path))

    @classmethod
    def load_eval(cls, *, root: Path | None = None) -> GateFrame:
        return cls.from_table(evalset.load_frozen("gate", root=root))

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, indices: Sequence[int] | np.ndarray) -> GateFrame:
        idx = np.asarray(indices, dtype=np.intp)
        return GateFrame(
            ids=tuple(self.ids[i] for i in idx),
            texts=tuple(self.texts[i] for i in idx),
            labels=self.labels[idx],
            kinds=tuple(self.kinds[i] for i in idx),
            offset_turns=self.offset_turns[idx],
            source_kinds=tuple(self.source_kinds[i] for i in idx),
        )

    def strata(self) -> list[str]:
        return [f"{bool(lab)}|{kind}" for lab, kind in zip(self.labels, self.kinds, strict=True)]


class LexicalGateTrainer:
    """Fits the TF-IDF FeatureUnion + balanced logistic regression that the inference gate serves.

    The fitted ``features`` (a word + char TF-IDF union) and ``clf`` serialize into the
    one artifact spec :mod:`cc_steer.watcher.gate` loads.
    """

    def __init__(self, features: FeatureUnion, clf: LogisticRegression) -> None:
        self.features = features
        self.clf = clf

    @classmethod
    def fit(
        cls,
        texts: Sequence[str],
        labels: np.ndarray,
        *,
        word_min_df: int,
        char_min_df: int,
        char_max_features: int,
        C: float,
        seed: int,
    ) -> LexicalGateTrainer:
        from sklearn.linear_model import LogisticRegression

        features = build_features(word_min_df, char_min_df, char_max_features)
        clf = LogisticRegression(class_weight="balanced", C=C, max_iter=2000, random_state=seed)
        clf.fit(features.fit_transform(texts), labels)
        return cls(features, clf)

    def logits(self, texts: Sequence[str]) -> np.ndarray:
        return self.clf.decision_function(self.features.transform(texts))


@dataclass(frozen=True, slots=True)
class Candidate:
    """One freshly trained gate: the fitted pieces plus its frozen-eval metrics."""

    model: LexicalGateTrainer
    temperature: float
    metrics: dict[str, float]


def build_features(word_min_df: int, char_min_df: int, char_max_features: int) -> FeatureUnion:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=word_min_df, sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=char_min_df,
                    max_features=char_max_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def carve_val(frame: GateFrame, *, seed: int, frac: float = 0.1) -> tuple[GateFrame, GateFrame]:
    """Deterministic stratified (label|kind) carve of a temperature-scaling val slice.

    Returns ``(train_rest, val)``; the val slice is held out of training so the fitted
    temperature is honest.
    """
    rng = np.random.default_rng(seed)
    strata = np.asarray(frame.strata())
    val_idx: list[int] = []
    rest_idx: list[int] = []
    for stratum in sorted(set(strata)):
        idx = np.flatnonzero(strata == stratum)
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * frac))
        val_idx.extend(idx[:n_val].tolist())
        rest_idx.extend(idx[n_val:].tolist())
    return frame.take(sorted(rest_idx)), frame.take(sorted(val_idx))


def fit_temperature(labels: np.ndarray, logits: np.ndarray) -> float:
    """Scalar temperature minimizing BCE-with-logits NLL of ``sigmoid(logits / T)``."""
    from scipy.optimize import minimize_scalar

    y = np.asarray(labels, dtype=np.float64).ravel()
    z = np.asarray(logits, dtype=np.float64).ravel()

    def nll(t: float) -> float:
        zt = z / t
        return float(np.mean(np.logaddexp(0.0, zt) - y * zt))

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def probs_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    from scipy.special import expit

    return expit(np.asarray(logits, dtype=np.float64) / temperature)


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve)."""
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(np.asarray(labels).ravel(), np.asarray(scores, dtype=np.float64).ravel()))


def ece(labels: np.ndarray, probs: np.ndarray, *, bins: int = 15) -> float:
    """Expected calibration error: equal-width bins over [0, 1], weighted by bin mass."""
    counts, mean_prob, frac_pos = _bin_stats(labels, probs, bins=bins)
    return float(np.sum(counts / counts.sum() * np.abs(mean_prob - frac_pos)))


def gate_metrics(frame: GateFrame, probs: np.ndarray, *, temperature: float) -> dict[str, float]:
    """The E1 metric suite over calibrated eval probabilities, keyed for the promotion bar.

    Emits :data:`~cc_steer.retrain.promotion.PR_AUC_KEY` and
    :data:`~cc_steer.retrain.promotion.RECALL_KEY` verbatim — the two metrics
    :func:`~cc_steer.retrain.promotion.gate_promotable` reads — plus the budget threshold the
    registry serves. All numbers are at the view's ~4:1 positive ratio; the alert-budget pair
    is an explicit proxy (keys carry ``viewratio_proxy``).
    """
    labels = frame.labels
    probs = np.asarray(probs, dtype=np.float64).ravel()
    out: dict[str, float] = {
        "n_eval": float(len(frame)),
        PR_AUC_KEY: pr_auc(labels, probs),
        "ece_post_t": ece(labels, probs),
        "temperature": temperature,
        "reliability_max_gap": max(abs(b[0] - b[1]) for b in _reliability_bins(labels, probs)),
    }
    kinds = np.asarray(frame.kinds)
    for neg_kind, key in (("hard_negative", "pr_auc_pos_vs_hard"), ("random_negative", "pr_auc_pos_vs_random")):
        mask = labels | (kinds == neg_kind)
        if labels[mask].any() and (~labels[mask]).any():
            out[key] = pr_auc(labels[mask], probs[mask])
    sources = np.asarray(frame.source_kinds)
    for source_kind in ("question_answer", "transcript_message"):
        mask = ~labels | (sources == source_kind)
        if labels[mask].any() and (~labels[mask]).any():
            out[f"pr_auc_src_{source_kind}"] = pr_auc(labels[mask], probs[mask])
    threshold = promotion.threshold_for_budget(probs, fires_per_100=BUDGET_FIRES_PER_100, total_turns=len(frame))
    fired = probs >= threshold
    out[THRESHOLD_METRIC] = threshold
    out[RECALL_KEY] = float(fired[labels].mean()) if labels.any() else 0.0
    return out


def train_gate(*, dataset_dir: Path | None = None, eval_root: Path | None = None) -> Candidate:
    """The E1-winning recipe on the full gate train split, temperature-scaled and frozen-evaluated."""
    seed = int(RECIPE["seed"])
    rest, val = carve_val(GateFrame.load_train(dataset_dir=dataset_dir), seed=seed, frac=float(RECIPE["val_frac"]))
    model = LexicalGateTrainer.fit(
        rest.texts,
        rest.labels,
        word_min_df=int(RECIPE["word_min_df"]),
        char_min_df=int(RECIPE["char_min_df"]),
        char_max_features=int(RECIPE["char_max_features"]),
        C=float(RECIPE["C"]),
        seed=seed,
    )
    temperature = fit_temperature(val.labels, model.logits(val.texts))
    eval_frame = GateFrame.load_eval(root=eval_root)
    probs = probs_from_logits(model.logits(eval_frame.texts), temperature)
    metrics = gate_metrics(eval_frame, probs, temperature=temperature)
    return Candidate(model=model, temperature=temperature, metrics=metrics)


def register_candidate(candidate: Candidate, *, digest: str, root: Path | None = None) -> registry.VersionInfo:
    """Serialize per the shared artifact spec, then register, promote, and prune to ``KEEP_VERSIONS``."""
    import io

    import joblib

    vectorizers = dict(candidate.model.features.transformer_list)
    buffer = io.BytesIO()
    joblib.dump(
        {
            "word_vec": vectorizers["word"],
            "char_vec": vectorizers["char"],
            "clf": candidate.model.clf,
            "temperature": candidate.temperature,
        },
        buffer,
    )
    info = registry.register(
        COMPONENT,
        {ARTIFACT_NAME: buffer.getvalue()},
        {
            "dataset_digest": digest,
            "config": dict(RECIPE),
            "metrics": dict(candidate.metrics),
            "thresholds": {THRESHOLD_KEY: candidate.metrics[THRESHOLD_METRIC]},
        },
        root=root,
    )
    registry.promote(COMPONENT, info.version, root=root)
    registry.prune(COMPONENT, keep=KEEP_VERSIONS, root=root)
    return info


def gate_train_digest(*, dataset_dir: Path | None = None) -> str:
    """The gate train view's content digest — the retrain trigger."""
    import pyarrow.parquet as pq

    path = (dataset_dir or data.DATASET_DIR) / "gate" / "train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no gate train parquet at {path}")
    return data.dataset_digest(pq.read_table(path).to_pylist())


def retrain_gate(
    *,
    force: bool = False,
    dataset_dir: Path | None = None,
    eval_root: Path | None = None,
    registry_root: Path | None = None,
    state_dir: Path | None = None,
) -> str:
    """One gate retrain pass; returns the journaled one-line verdict."""
    incumbent = registry.current(COMPONENT, root=registry_root)
    digest = gate_train_digest(dataset_dir=dataset_dir)
    if not promotion.should_retrain(incumbent, digest, force=force):
        return promotion.journal(
            COMPONENT, f"skipped (no new data at digest {digest})", dataset_digest=digest, state_dir=state_dir
        )
    candidate = train_gate(dataset_dir=dataset_dir, eval_root=eval_root)
    if incumbent is None:
        incumbent_metrics = None
    else:
        metrics = incumbent.metadata["metrics"]
        if not isinstance(metrics, dict):
            raise TypeError(f"incumbent {incumbent.version} carries non-dict metrics {metrics!r}")
        incumbent_metrics = metrics
    verdict = promotion.gate_promotable(candidate.metrics, incumbent_metrics)
    summary = (
        f"pr_auc={candidate.metrics[PR_AUC_KEY]:.4f} ece={candidate.metrics['ece_post_t']:.4f} "
        f"recall@2/100={candidate.metrics[RECALL_KEY]:.4f} threshold={candidate.metrics[THRESHOLD_METRIC]:.4f}"
    )
    version: str | None = None
    if not verdict.promote:
        incumbent_version = incumbent.version if incumbent is not None else "?"
        line = f"rejected ({verdict.reason}); incumbent {incumbent_version} stays — {summary}"
    else:
        version = (info := register_candidate(candidate, digest=digest, root=registry_root)).version
        line = f"promoted {info.version} ({verdict.reason}) — {summary}"
    return promotion.journal(
        COMPONENT, line, dataset_digest=digest, metrics=candidate.metrics, version=version, state_dir=state_dir
    )


def _reliability_bins(labels: np.ndarray, probs: np.ndarray, *, bins: int = 15) -> list[tuple[float, float]]:
    counts, mean_prob, frac_pos = _bin_stats(labels, probs, bins=bins)
    return [(float(m), float(f)) for m, f in zip(mean_prob, frac_pos, strict=True)]


def _bin_stats(labels: np.ndarray, probs: np.ndarray, *, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels).ravel()
    probs = np.asarray(probs, dtype=np.float64).ravel()
    idx = np.minimum((probs * bins).astype(np.intp), bins - 1)
    counts = np.bincount(idx, minlength=bins)
    sum_probs = np.bincount(idx, weights=probs, minlength=bins)
    sum_pos = np.bincount(idx, weights=labels.astype(np.float64), minlength=bins)
    nonzero = counts > 0
    counts = counts[nonzero]
    return counts, sum_probs[nonzero] / counts, sum_pos[nonzero] / counts
