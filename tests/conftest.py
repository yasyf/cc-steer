from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cc_pushback.store import FeedbackStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[FeedbackStore]:
    async with await FeedbackStore.open(tmp_path / "feedback.db") as opened:
        yield opened
