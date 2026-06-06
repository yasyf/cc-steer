You are mining one piece of developer pushback against an AI coding agent (Claude Code). The developer corrected, interrupted, or rejected something Claude did. Name the rule the developer is enforcing, using the fixed taxonomy where possible.

You are given:
- <taxonomy>: the known patterns (name — rule), one per line.
- <event>: the feedback event — what Claude was doing, then the developer's words.

Return JSON matching the schema:
- pattern_names: every taxonomy name that fits (often one; may be empty).
- novel_pattern: short kebab-case name ONLY if nothing in the taxonomy fits; else null. Never propose novel for an imperfect-but-real match.
- severity: nit | minor | major | blocking — how strongly the developer pushes back.
- what_claude_did: one sentence describing the behavior that drew the pushback.
- rule: the corrective rule in one imperative sentence, as the developer would phrase it.

Be literal. Observe what the developer actually wrote; do not infer motives they didn't state.
