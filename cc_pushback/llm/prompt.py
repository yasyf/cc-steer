from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

__all__ = ["PromptMessage"]

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def dedent_text(text: str) -> str:
    return textwrap.dedent(text).strip()


@dataclass(frozen=True, kw_only=True)
class PromptMessage:
    """Fluent builder for structured LLM prompts with system text, XML context sections, and a question.

    Chain ``.system()``, ``.context(tag, content)``, and ``.ask()`` to build prompts.
    ``str()`` renders the full prompt with XML-wrapped context blocks.

    Example:
        >>> str(PromptMessage().system("be terse").context("event", "...").ask("classify"))
    """

    system_text: str = ""
    contexts: tuple[tuple[str, str], ...] = ()
    ask_text: str = ""

    def system(self, text: str) -> PromptMessage:
        return PromptMessage(system_text=dedent_text(text), contexts=self.contexts, ask_text=self.ask_text)

    def context(self, tag: str, content: str | None) -> PromptMessage:
        if content is None or not (normalized := dedent_text(content)):
            return self
        return PromptMessage(
            system_text=self.system_text,
            contexts=(*self.contexts, (tag, normalized)),
            ask_text=self.ask_text,
        )

    def ask(self, text: str) -> PromptMessage:
        return PromptMessage(system_text=self.system_text, contexts=self.contexts, ask_text=dedent_text(text))

    @classmethod
    def from_template(cls, text: str, **vars: object) -> PromptMessage:
        """Render ``text`` via :meth:`str.format_map` and wrap it as system text.

        Args:
            text: Template string with ``{name}`` placeholders.
            **vars: Substitution values for the placeholders.

        Returns:
            A :class:`PromptMessage` whose system text is the rendered template.

        Raises:
            KeyError: If the template references a placeholder not supplied in ``**vars``.
        """
        try:
            return cls(system_text=dedent_text(text).format_map(vars))
        except KeyError as exc:
            raise KeyError(f"template variable {exc.args[0]!r} not supplied") from exc

    @classmethod
    def load(cls, name: str, *, base: str | Path | None = None, **vars: object) -> PromptMessage:
        """Load a prompt from a ``.md`` file and render it via :meth:`from_template`.

        The file path is ``<dir>/<name>.md`` where ``dir`` is ``base`` if given,
        otherwise the package's bundled ``prompts/`` directory. ``name`` may contain
        ``/`` to nest.

        Args:
            name: Prompt name without the ``.md`` suffix; may include ``/`` for nesting.
            base: Optional directory to search instead of the bundled ``prompts/``.
            **vars: Template variables substituted into the file via ``str.format_map``.

        Returns:
            A :class:`PromptMessage` whose system text is the rendered file contents.

        Raises:
            FileNotFoundError: If no matching file exists in the searched directory.
            KeyError: If the file references a placeholder not supplied in ``**vars``.
        """
        path = (Path(base) if base else PROMPTS_DIR) / f"{name}.md"
        if not path.is_file():
            raise FileNotFoundError(f"prompt {name!r} not found; searched: {path}")
        return cls.from_template(path.read_text(), **vars)

    def __str__(self) -> str:
        return "\n\n".join(
            [
                *([self.system_text] if self.system_text else []),
                *(f"<{tag}>\n{content}\n</{tag}>" for tag, content in self.contexts),
                *([self.ask_text] if self.ask_text else []),
            ]
        )
