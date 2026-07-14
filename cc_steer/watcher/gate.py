"""Stage 1 in production: the lexical gate, loaded from the registry.

E1's bake-off winner — TF-IDF word 1-2 grams + char 3-5 grams into a balanced
logistic regression, temperature-scaled — is trained and serialized by
:mod:`cc_steer.retrain.lexical`; this module only loads and scores. The two
sides share ONE artifact spec, defined here and imported by the trainer:

* ``ARTIFACT_NAME`` (``model.joblib``) — a joblib dict with keys ``word_vec``
  and ``char_vec`` (the fitted TF-IDF vectorizers, hstacked in that order),
  ``clf`` (the LogisticRegression), and ``temperature`` (the scalar the raw
  logit is divided by before the sigmoid).
* ``metadata.json`` — the registry's stamp plus ``thresholds`` carrying
  ``THRESHOLD_KEY``, the fire threshold fitted at the 2 fires/100 turns budget.

sklearn lives behind the ``gate`` extra and is imported lazily, mirroring
:mod:`cc_steer.exemplars`'s handling of the ``embed`` extra, so the base
install never pays for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cc_steer import registry

if TYPE_CHECKING:
    from pathlib import Path

COMPONENT = "gate"
ARTIFACT_NAME = "model.joblib"
THRESHOLD_KEY = "budget_2_per_100"
ARTIFACT_KEYS = ("word_vec", "char_vec", "clf", "temperature")


class LexicalGate:
    """The trained stage-1 gate: calibrated P(steer) over the flattened window text.

    Implements the cascade's ``Gate`` protocol. Loads the promoted registry
    version by default; pass ``version`` to pin one.

    Args:
        version: The registry version to load; defaults to ``current("gate")``.
        root: The registry root override, for tests.

    Raises:
        RuntimeError: When no gate version is promoted, the artifact file is
            missing or malformed, or the ``gate`` extra is not installed.
    """

    def __init__(self, version: registry.VersionInfo | None = None, *, root: Path | None = None) -> None:
        resolved = version or registry.current(COMPONENT, root=root)
        if resolved is None:
            raise RuntimeError("no promoted gate model: train and promote one with `cc-steer retrain --component gate`")
        artifact = resolved.path / ARTIFACT_NAME
        if not artifact.exists():
            raise RuntimeError(f"gate version {resolved.version} has no {ARTIFACT_NAME} at {artifact}")
        payload = _joblib().load(artifact)
        if not isinstance(payload, dict) or any(key not in payload for key in ARTIFACT_KEYS):
            raise RuntimeError(f"{artifact} is not a gate artifact: expected a dict with keys {ARTIFACT_KEYS}")
        self.version = resolved
        self.word_vec = payload["word_vec"]
        self.char_vec = payload["char_vec"]
        self.clf = payload["clf"]
        self.temperature = float(payload["temperature"])

    @property
    def threshold(self) -> float:
        """The trained fire threshold at the 2 fires/100 turns budget."""
        thresholds = self.version.metadata.get("thresholds")
        if not isinstance(thresholds, dict) or THRESHOLD_KEY not in thresholds:
            raise RuntimeError(f"gate version {self.version.version} metadata carries no thresholds[{THRESHOLD_KEY!r}]")
        return float(thresholds[THRESHOLD_KEY])

    def score(self, text: str) -> float:
        """The calibrated steer probability for one flattened window text."""
        from scipy.sparse import hstack
        from scipy.special import expit

        features = hstack([self.word_vec.transform([text]), self.char_vec.transform([text])])
        logit = float(self.clf.decision_function(features)[0])
        return float(expit(logit / self.temperature))


def _joblib() -> Any:
    """The joblib module; requires the ``gate`` extra."""
    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("the trained gate requires the 'gate' extra: pip install 'cc-steer[gate]'") from error
    return joblib
