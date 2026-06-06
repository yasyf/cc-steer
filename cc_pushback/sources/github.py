from __future__ import annotations

import json
import re
from datetime import datetime
from subprocess import CalledProcessError
from typing import TYPE_CHECKING, Any

from cc_pushback.models import ContextSnapshot, ContextTurn, FeedbackCandidate, PrRef
from cc_pushback.shell import call_cli
from cc_pushback.sources.base import dedup_key

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from cc_pushback.repo import Repository

__all__ = ["GitHubReviews"]

SOURCE_KIND = "github_review"
REMOTE_RE = re.compile(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$")
CLAUDE_AUTHORSHIP_RE = re.compile(r"Co-Authored-By: Claude|Generated with \[Claude Code\]")


def parse_remote(url: str) -> tuple[str, str] | None:
    return (match.group("owner"), match.group("repo")) if (match := REMOTE_RE.search(url.strip())) else None


def comment_trigger(comment: dict[str, Any]) -> ContextSnapshot:
    return ContextSnapshot(
        before=(),
        trigger=ContextTurn(
            role="assistant",
            text=f"{comment['path']}:{comment.get('line') or comment.get('original_line')}\n{comment['diff_hunk']}",
        ),
        after=(),
    )


class GitHubReviews:
    """Extracts pull-request review comments authored on Claude-generated PRs.

    Resolves the GitHub remote and the authenticated login via ``git`` and
    ``gh``, keeps pull requests whose title or body bears a Claude authorship
    marker, and emits the user's own review comments as feedback. Comments are
    paginated with a per-source ``updated_at`` cursor so each scan only fetches
    what changed.
    """

    def candidates(self, repo: Repository, *, cwd: Path) -> tuple[str, str, list[FeedbackCandidate]]:
        """Returns ``(source_key, max_cursor, candidates)`` for ``cwd``'s remote.

        Args:
            repo: The repository holding the GitHub source cursor.
            cwd: The working directory whose ``origin`` remote is inspected.

        Returns:
            The source key, the newest comment cursor seen (empty when none),
            and the discovered candidates. An empty list when the remote is
            absent or not a GitHub remote.
        """
        if (slug := self.resolve_slug(cwd)) is None:
            return "", "", []
        owner, name = slug
        source_key = f"github:{owner}/{name}"
        login = call_cli(["gh", "api", "user", "--jq", ".login"], env={}).strip()
        since = repo.cursor_for(source_key)
        candidates = list(self.collect(owner, name, login, cwd=cwd, since=since))
        return source_key, self.max_cursor(candidates), candidates

    def resolve_slug(self, cwd: Path) -> tuple[str, str] | None:
        try:
            url = call_cli(["git", "-C", str(cwd), "remote", "get-url", "origin"], env={})
        except CalledProcessError:
            return None
        return parse_remote(url)

    def collect(
        self, owner: str, name: str, login: str, *, cwd: Path, since: str | None
    ) -> Iterator[FeedbackCandidate]:
        return (
            FeedbackCandidate(
                dedup_key=dedup_key(pr_ref, str(comment["id"]), SOURCE_KIND),
                source_kind=SOURCE_KIND,
                occurred_at=datetime.fromisoformat(comment["updated_at"]),
                text=comment["body"],
                context=comment_trigger(comment),
                pr_ref=PrRef(pr_ref),
                payload={"comment_id": comment["id"], "html_url": comment["html_url"]},
            )
            for pr in self.claude_prs(owner, name)
            for pr_ref in (f"{owner}/{name}#{pr['number']}",)
            for comment in self.pr_comments(owner, name, pr["number"], since=since)
            if comment["user"]["login"] == login
        )

    def claude_prs(self, owner: str, name: str) -> Iterator[dict[str, Any]]:
        return (
            pr
            for pr in json.loads(
                call_cli(
                    ["gh", "api", f"repos/{owner}/{name}/pulls?state=all&per_page=100"],
                    env={},
                )
            )
            if CLAUDE_AUTHORSHIP_RE.search(f"{pr.get('title') or ''}\n{pr.get('body') or ''}")
        )

    def pr_comments(self, owner: str, name: str, number: int, *, since: str | None) -> list[dict[str, Any]]:
        path = f"repos/{owner}/{name}/pulls/{number}/comments?sort=updated&direction=asc&per_page=100"
        return json.loads(call_cli(["gh", "api", path + (f"&since={since}" if since else "")], env={}))

    @staticmethod
    def max_cursor(candidates: Sequence[FeedbackCandidate]) -> str:
        return max((c.occurred_at.isoformat() for c in candidates), default="")
