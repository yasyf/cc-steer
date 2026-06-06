from __future__ import annotations

import json
from pathlib import Path

import click
from loguru import logger


@click.group()
@click.version_option(package_name="cc-pushback")
def main() -> None:
    """Learn your pushback style from past Claude Code feedback and code reviews, and replicate it with a language model."""


def _user_messages(transcript: Path) -> list[str]:
    messages: list[str] = []
    for line in transcript.read_text().splitlines():
        match json.loads(line):
            case {"type": "user", "message": {"content": str(text)}} if not text.startswith("<"):
                messages.append(text)
            case _:
                pass
    return messages


@main.command()
@click.option(
    "--transcripts",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.home() / ".claude" / "projects",
    show_default=True,
    help="Directory of Claude Code transcript JSONL files.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("messages.jsonl"),
    show_default=True,
    help="Where to write the extracted dataset.",
)
def extract(transcripts: Path, output: Path) -> None:
    """Extract human-typed messages from Claude Code transcripts into a JSONL dataset.

    Walks every ``*.jsonl`` transcript under the transcripts directory, keeps only
    the messages the user actually typed (skipping tool results and injected
    context), and writes one ``{"session", "text"}`` record per line. This raw
    dataset is the input to pushback labeling and classifier training.
    """
    records = [
        {"session": path.stem, "text": text}
        for path in sorted(transcripts.rglob("*.jsonl"))
        for text in _user_messages(path)
    ]
    output.write_text("".join(json.dumps(record) + "\n" for record in records))
    logger.debug("extracted {} messages from {}", len(records), transcripts)
    click.echo(f"Extracted {len(records)} messages to {output}")
