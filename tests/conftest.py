from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cc_pushback.store import FeedbackStore

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def store(tmp_path: Path) -> Iterator[FeedbackStore]:
    with FeedbackStore.open(tmp_path / "feedback.db") as opened:
        yield opened
