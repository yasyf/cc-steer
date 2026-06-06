from __future__ import annotations

from typing import TYPE_CHECKING

from cc_pushback.sources.github import GitHubReviews

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from cc_pushback.repo import Repository

CLAUDE_PR = {"number": 7, "title": "fix bug", "body": "Generated with [Claude Code]"}
HUMAN_PR = {"number": 8, "title": "manual fix", "body": "just me"}


def comment(comment_id: int, login: str, body: str, updated_at: str) -> dict[str, Any]:
    return {
        "id": comment_id,
        "user": {"login": login},
        "body": body,
        "updated_at": updated_at,
        "path": "src/a.py",
        "line": 12,
        "diff_hunk": "@@ -1 +1 @@",
        "html_url": "https://github.com/owner/repo/pull/7#c",
    }


def test_keeps_only_own_comments_on_claude_prs(
    feedback_db: Repository, fake_gh: Callable[[dict[str, Any]], None], tmp_path: Path
) -> None:
    fake_gh(
        {
            "pulls": [CLAUDE_PR, HUMAN_PR],
            "comments": {
                7: [
                    comment(1, "yasyf", "do not add a fallback", "2026-06-01T00:00:00+00:00"),
                    comment(2, "someone", "looks good", "2026-06-01T01:00:00+00:00"),
                ],
                8: [comment(3, "yasyf", "on a human PR", "2026-06-01T02:00:00+00:00")],
            },
        }
    )

    _, cursor, candidates = GitHubReviews().candidates(feedback_db, cwd=tmp_path)

    assert [c.text for c in candidates] == ["do not add a fallback"]
    assert candidates[0].pr_ref == "owner/repo#7"
    assert cursor == "2026-06-01T00:00:00+00:00"


def test_non_github_remote_yields_no_candidates(
    feedback_db: Repository, fake_gh: Callable[[dict[str, Any]], None], tmp_path: Path
) -> None:
    fake_gh({"remote": None})

    source_key, cursor, candidates = GitHubReviews().candidates(feedback_db, cwd=tmp_path)

    assert (source_key, cursor, candidates) == ("", "", [])


def test_cursor_advance_makes_second_scan_a_noop(
    feedback_db: Repository, fake_gh: Callable[[dict[str, Any]], None], tmp_path: Path
) -> None:
    fake_gh(
        {
            "pulls": [CLAUDE_PR],
            "comments": {7: [comment(1, "yasyf", "fix this", "2026-06-01T00:00:00+00:00")]},
        }
    )

    source_key, cursor, candidates = GitHubReviews().candidates(feedback_db, cwd=tmp_path)
    assert feedback_db.advance_github_cursor(source_key, cursor, candidates) == 1

    _, cursor2, candidates2 = GitHubReviews().candidates(feedback_db, cwd=tmp_path)
    assert feedback_db.advance_github_cursor(source_key, cursor2 or cursor, candidates2) == 0
