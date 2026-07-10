from __future__ import annotations

import io
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pytest
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from cc_steer import registry
from cc_steer.watcher.gate import ARTIFACT_NAME, COMPONENT, THRESHOLD_KEY, LexicalGate

if TYPE_CHECKING:
    from pathlib import Path

STEER = "no stop revert that change the refactor broke tests wrong approach"
NOISE = "deploy status is green please add a brand new export feature"
TEXTS = [f"{base} sample {index}" for index in range(8) for base in (STEER, NOISE)]
LABELS = np.array([True, False] * 8)


def artifact_bytes(temperature: float = 1.0) -> bytes:
    """A tiny model serialized per the shared artifact spec."""
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(TEXTS)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True).fit(TEXTS)
    features = hstack([word_vec.transform(TEXTS), char_vec.transform(TEXTS)])
    clf = LogisticRegression(class_weight="balanced", C=4.0, max_iter=2000, random_state=0).fit(features, LABELS)
    payload = {"word_vec": word_vec, "char_vec": char_vec, "clf": clf, "temperature": temperature}
    buffer = io.BytesIO()
    joblib.dump(payload, buffer)
    return buffer.getvalue()


def promoted_gate(root: Path, *, temperature: float = 1.0, threshold: float = 0.8) -> registry.VersionInfo:
    metadata = {"dataset_digest": "digest-1", "metrics": {"pr_auc": 0.95}, "thresholds": {THRESHOLD_KEY: threshold}}
    info = registry.register(COMPONENT, {ARTIFACT_NAME: artifact_bytes(temperature)}, metadata, root=root)
    registry.promote(COMPONENT, info.version, root=root)
    return info


class TestLexicalGate:
    def test_loads_current_and_scores_calibrated_probabilities(self, tmp_path: Path) -> None:
        info = promoted_gate(tmp_path)
        gate = LexicalGate(root=tmp_path)
        assert gate.version.version == info.version
        steer_score, noise_score = gate.score(STEER), gate.score(NOISE)
        assert 0.0 <= noise_score < steer_score <= 1.0
        assert gate.threshold == 0.8

    def test_loads_pinned_version_without_promotion(self, tmp_path: Path) -> None:
        metadata = {"thresholds": {THRESHOLD_KEY: 0.5}}
        info = registry.register(COMPONENT, {ARTIFACT_NAME: artifact_bytes()}, metadata, root=tmp_path)
        assert LexicalGate(version=info, root=tmp_path).score(STEER) > 0.5

    def test_temperature_flattens_scores(self, tmp_path: Path) -> None:
        promoted_gate(tmp_path / "sharp", temperature=1.0)
        promoted_gate(tmp_path / "flat", temperature=100.0)
        sharp = LexicalGate(root=tmp_path / "sharp")
        flat = LexicalGate(root=tmp_path / "flat")
        assert abs(flat.score(STEER) - 0.5) < abs(sharp.score(STEER) - 0.5)

    def test_no_promoted_version_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="no promoted gate model"):
            LexicalGate(root=tmp_path)

    def test_missing_artifact_file_raises(self, tmp_path: Path) -> None:
        info = registry.register(COMPONENT, {"other.bin": b"nope"}, {}, root=tmp_path)
        registry.promote(COMPONENT, info.version, root=tmp_path)
        with pytest.raises(RuntimeError, match=ARTIFACT_NAME):
            LexicalGate(root=tmp_path)

    def test_malformed_payload_raises(self, tmp_path: Path) -> None:
        buffer = io.BytesIO()
        joblib.dump({"word_vec": None}, buffer)
        info = registry.register(COMPONENT, {ARTIFACT_NAME: buffer.getvalue()}, {}, root=tmp_path)
        registry.promote(COMPONENT, info.version, root=tmp_path)
        with pytest.raises(RuntimeError, match="not a gate artifact"):
            LexicalGate(root=tmp_path)

    def test_missing_threshold_metadata_raises(self, tmp_path: Path) -> None:
        info = registry.register(COMPONENT, {ARTIFACT_NAME: artifact_bytes()}, {}, root=tmp_path)
        registry.promote(COMPONENT, info.version, root=tmp_path)
        with pytest.raises(RuntimeError, match="thresholds"):
            _ = LexicalGate(root=tmp_path).threshold
