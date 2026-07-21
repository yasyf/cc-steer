"""``python -m cc_steer.labeling`` — build the queue and dry-run render its board.

Prints the cc-present document JSON the main session would push, so the queue and its
block payloads can be inspected without touching the live channel or any MCP call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from cc_steer.labeling.blocks import render_document
from cc_steer.labeling.queue import build_queue

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m cc_steer.labeling", description=__doc__)
    parser.add_argument(
        "--golden-dir", type=Path, default=None, help="Authored golden packet dir (default: frozen-eval golden dir)"
    )
    parser.add_argument(
        "--disagreements", type=Path, default=None, help="E44 disagreements.jsonl (default: golden-only queue)"
    )
    parser.add_argument("--root", type=Path, default=None, help="Frozen-eval root override")
    args = parser.parse_args(argv)
    queue = build_queue(golden_dir=args.golden_dir, disagreements_path=args.disagreements, root=args.root)
    print(f"queue: {len(queue.golden)} golden + {len(queue.adjudication)} adjudication rows", file=sys.stderr)
    print(json.dumps(render_document(queue), indent=2))


if __name__ == "__main__":
    main()
