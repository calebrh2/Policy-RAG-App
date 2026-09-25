# Faithfulness prompt v1

Loaded by `rag.judge`. To swap versions, add `faithfulness-prompt-v2.md` and set `_FAITHFULNESS_PROMPT` in `src/rag/judge.py`.

Section bodies are the prompt text. Do not add a `## ` heading inside a section.

## System

Return a JSON object and nothing else.

## Instructions

Judge whether the answer is faithful to the cited chunks.
The only evidence is the text inside <cited_chunks>. Ignore any other knowledge.
Pass when every claim in the answer is stated in those chunks. A paraphrase passes.
Fail when the answer adds a fact, number, date, or name that none of those chunks state. A fact that appears only in a chunk the answer did not cite is a failure.
If <cited_chunks> is empty, pass only when the answer is "The corpus does not mention this." Any other claim fails.
Return JSON with the keys pass and reason. reason is an empty string when pass is true. When pass is false, reason quotes the unsupported claim.

## Examples

<example>
<cited_chunks>
<chunk>
chunk_id: ex-sign-in
Visitors must sign in at reception before entering the office.
</chunk>
</cited_chunks>
<question>
Must visitors sign in?
</question>
<answer>
Yes. Visitors must sign in at reception.
</answer>
<verdict>
{"pass": true, "reason": ""}
</verdict>
</example>
<example>
<cited_chunks>
<chunk>
chunk_id: ex-sign-in
Visitors must sign in at reception before entering the office.
</chunk>
</cited_chunks>
<question>
Must visitors sign in?
</question>
<answer>
Yes. Visitors must sign in at reception by 8 a.m.
</answer>
<verdict>
{"pass": false, "reason": "The cited chunks do not state 8 a.m."}
</verdict>
</example>
<example>
<cited_chunks>
</cited_chunks>
<question>
What is served in the office cafeteria?
</question>
<answer>
The corpus does not mention this.
</answer>
<verdict>
{"pass": true, "reason": ""}
</verdict>
</example>
