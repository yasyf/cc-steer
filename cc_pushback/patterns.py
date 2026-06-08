from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cc_pushback.matchers import LlmMatcher, Matcher, PatternName, RegexMatcher, StructuralMatcher, rx

if TYPE_CHECKING:
    from cc_pushback.classify import FeedbackEvent

TAXONOMY_VERSION = "v1"
DENIED_EDIT_TOOLS = frozenset({"Edit", "Write"})


def denied_edit(event: FeedbackEvent) -> bool:
    return bool(event.payload) and event.payload.get("tool") in DENIED_EDIT_TOOLS


@dataclass(frozen=True, slots=True)
class Pattern:
    """One named pushback pattern in the taxonomy.

    Attributes:
        name: The stable kebab-case pattern name.
        description: A one-line gloss of what the pattern captures.
        triggers: Verbatim phrasings or regex fragments that signal the pattern.
        matcher: How the cheap pass detects the pattern, when it can.
        rule: The corrective rule, phrased imperatively for the prompt.
    """

    name: PatternName
    description: str
    triggers: tuple[str, ...]
    matcher: Matcher
    rule: str


PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        name=PatternName("no-defensive-coding"),
        description="Pushback against fallbacks, shims, and guards on impossible states.",
        triggers=(
            r"fall ?back",
            r"no defensive",
            r"crash instead",
            r"don'?t .{0,20}guard",
            r"remove the (try|guard|shim)",
            r"backwards?[- ]compat",
        ),
        matcher=RegexMatcher(
            rx(
                r"fall ?backs?\b",
                r"no defensive",
                r"crash instead",
                r"don'?t\s+(?:\w+\s+){0,3}guard",
                r"remove the (?:try|guard|shim)",
                r"backwards?[- ]compat",
                r"\bshim\b",
            )
        ),
        rule="No fallbacks, shims, or guards against impossible states; crash on the unexpected.",
    ),
    Pattern(
        name=PatternName("ask-before-assuming"),
        description="Pushback for guessing instead of asking when the request was ambiguous.",
        triggers=(
            r"ask( me)? first",
            r"don'?t assume",
            r"why didn'?t you ask",
            r"you should have asked",
            r"stop guessing",
        ),
        matcher=RegexMatcher(
            rx(
                r"ask( me)? (?:first|before)",
                r"don'?t assume",
                r"why didn'?t you ask",
                r"(?:you )?should(?:'ve| have) asked",
                r"stop guessing",
                r"clarify(?:ing)? question",
            )
        ),
        rule="When the request is ambiguous, stop and ask rather than guessing.",
    ),
    Pattern(
        name=PatternName("minimal-scope"),
        description="Pushback for going beyond what was asked.",
        triggers=(
            r"out of scope",
            r"only .{0,20}(asked|requested)",
            r"don'?t .{0,15}(refactor|rewrite|touch)",
            r"stay in scope",
            r"too (much|far)",
        ),
        matcher=RegexMatcher(
            rx(
                r"out of scope",
                r"(?:stay|keep it) in scope",
                r"only (?:do|change|touch|what)",
                r"don'?t\s+(?:\w+\s+){0,3}(?:refactor|rewrite|touch|change)",
                r"(?:that's|thats|went) too far",
                r"why did you (?:also|even)",
                r"i didn'?t ask (?:you )?(?:for|to)",
            )
        ),
        rule="Stay within scope; make the change asked for and stop.",
    ),
    Pattern(
        name=PatternName("match-surrounding-code"),
        description="Pushback for ignoring the conventions of the surrounding code.",
        triggers=(
            r"match the (surrounding|existing)",
            r"follow the (convention|pattern|style)",
            r"like the (other|rest)",
            r"consistent with",
        ),
        matcher=RegexMatcher(
            rx(
                r"match the (?:surrounding|existing|rest)",
                r"follow the (?:\w+ )?(?:convention|pattern|style)",
                r"(?:like|same as) the (?:other|rest|existing) (?:code|file|function|module|method|class|test)s?",
                r"be consistent with",
                r"this isn'?t how (?:we|the)",
            )
        ),
        rule="Follow the conventions of the file and module you are editing.",
    ),
    Pattern(
        name=PatternName("delegate-dont-bulk-read"),
        description="Pushback for reading files directly instead of delegating to a subagent.",
        triggers=(
            r"use a subagent",
            r"delegate",
            r"don'?t .{0,15}read .{0,15}(all|every|bulk)",
            r"spawn an? (explore|agent)",
        ),
        matcher=RegexMatcher(
            rx(
                r"use a sub-?agent",
                r"delegate",
                r"don'?t\s+(?:\w+\s+){0,3}bulk[- ]?read",
                r"don'?t read (?:all|every|the whole)",
                r"spawn an? (?:explore|agent|subagent)",
            )
        ),
        rule="Delegate context-gathering to a subagent instead of bulk-reading files yourself.",
    ),
    Pattern(
        name=PatternName("verbatim-feedback"),
        description="Pushback for paraphrasing review comments instead of quoting them verbatim.",
        triggers=(
            r"verbatim",
            r"don'?t paraphrase",
            r"quote .{0,15}(me|exactly|comment)",
            r"my exact words",
        ),
        matcher=RegexMatcher(
            rx(
                r"(?:quote|reproduce|copy|keep|paste|include)[^.]{0,25}verbatim",
                r"verbatim,? (?:please|don|do not|not)",
                r"don'?t paraphrase",
                r"quote (?:me|it|exactly|my|the comment)",
                r"my exact words",
                r"word[- ]for[- ]word",
            )
        ),
        rule="Reproduce review comments verbatim; never paraphrase them.",
    ),
    Pattern(
        name=PatternName("parallelize-work"),
        description="Pushback for running independent work serially instead of in parallel.",
        triggers=(
            r"in parallel",
            r"parallelize",
            r"at the same time",
            r"concurrent",
            r"one message",
        ),
        matcher=RegexMatcher(
            rx(
                r"in parallel",
                r"paralleli[sz]e",
                r"at (?:the )?same time",
                r"concurrent(?:ly)?",
                r"(?:single|one) message",
                r"all at once",
            )
        ),
        rule="Dispatch independent work concurrently rather than one step at a time.",
    ),
    Pattern(
        name=PatternName("observe-dont-infer"),
        description="Pushback for reasoning from assumption instead of inspecting real data.",
        triggers=(
            r"don'?t (assume|guess|infer)",
            r"(read|look at|check) the (file|code|data|fixture)",
            r"actually (run|read|look)",
            r"don'?t make .{0,10}up",
        ),
        matcher=RegexMatcher(
            rx(
                r"don'?t (?:assume|infer)",
                r"(?:read|look at|check|inspect) the (?:actual|file|code|data|fixture|output)",
                r"actually (?:run|read|look|check)",
                r"stop (?:assuming|inferring|making .{0,10}up)",
                r"you (?:made|are making) (?:that|it) up",
            )
        ),
        rule="Inspect the actual data before reasoning; observe, don't infer.",
    ),
    Pattern(
        name=PatternName("right-search-tool"),
        description="Pushback for using the wrong search tool (grep over semantic/LSP).",
        triggers=(
            r"use (semble|grep|the lsp)",
            r"don'?t (grep|use grep)",
            r"(semantic|lsp) search",
            r"find references",
        ),
        matcher=RegexMatcher(
            rx(
                r"use (?:semble|grep|the lsp|find\w*references)",
                r"don'?t (?:just )?grep",
                r"(?:semantic|lsp) search",
                r"find ?references",
                r"that's what (?:semble|the lsp) is for",
            )
        ),
        rule="Reach for the right search tool: semantic for intent, LSP for structure, grep only for literals.",
    ),
    Pattern(
        name=PatternName("denied-edit"),
        description="A denied Edit or Write permission request.",
        triggers=(),
        matcher=StructuralMatcher(denied_edit),
        rule="Don't make this file edit; the developer rejected it.",
    ),
    Pattern(
        name=PatternName("novel"),
        description="Catch-all for pushback no taxonomy pattern covers; resolved only by the model.",
        triggers=(),
        matcher=LlmMatcher("Use only when no other pattern fits, and prefer naming a concrete novel pattern."),
        rule="A pushback the seed taxonomy does not yet name.",
    ),
)


def by_name(name: str) -> Pattern:
    """Returns the taxonomy pattern named ``name``.

    Args:
        name: The pattern name to look up.

    Returns:
        The matching :class:`Pattern`.

    Raises:
        StopIteration: If no pattern has that name.
    """
    return next(pattern for pattern in PATTERNS if pattern.name == name)


def render_taxonomy() -> str:
    """Renders the taxonomy as ``name — rule`` lines for the classification prompt."""
    return "\n".join(f"{pattern.name} — {pattern.rule}" for pattern in PATTERNS)
