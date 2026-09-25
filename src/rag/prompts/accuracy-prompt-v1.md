# Accuracy prompt v1

Loaded by `rag.judge`. To swap versions, add `accuracy-prompt-v2.md` and set `_ACCURACY_PROMPT` in `src/rag/judge.py`.

Section bodies are the prompt text. Do not add a `## ` heading inside a section.

## System

Return a JSON object and nothing else.

## Instructions

Judge whether the answer matches the reference.
The only evidence is the reference. Do not use any source text.
Pass when the answer contains the same facts as the reference. Extra wording passes.
Fail when a required fact is missing, contradicted, or replaced by a wrong entity or number.
If the reference is "The corpus does not mention this.", pass only when the answer is exactly that sentence.
Return JSON with the keys pass and reason. reason is an empty string when pass is true. When pass is false, reason names the missing or contradictory fact.

## Examples

<example>
<question>
By when must the register be updated?
</question>
<reference>
By June 2028.
</reference>
<answer>
The deadline is June 2028.
</answer>
<verdict>
{"pass": true, "reason": ""}
</verdict>
</example>
<example>
<question>
By when must the register be updated?
</question>
<reference>
By June 2028.
</reference>
<answer>
By June 2027.
</answer>
<verdict>
{"pass": false, "reason": "The answer says June 2027. The reference says June 2028."}
</verdict>
</example>
<example>
<question>
What is served in the office cafeteria?
</question>
<reference>
The corpus does not mention this.
</reference>
<answer>
The corpus does not mention this.
</answer>
<verdict>
{"pass": true, "reason": ""}
</verdict>
</example>
