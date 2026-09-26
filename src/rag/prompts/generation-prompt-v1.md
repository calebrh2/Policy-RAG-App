# Generation prompt v1

Loaded by `rag.generate`. To swap versions, add `generation-prompt-v2.md` and set `_PROMPT_FILE` in `src/rag/generate.py`.

Section bodies are the prompt text. Do not add a `## ` heading inside a section.

## System

Return a JSON object and nothing else.

## Instructions

Answer the question using only the chunks in <chunks>.
If none of the chunks state the answer, set supported to false, set text to "The corpus does not mention this.", and return an empty citations list.
If a chunk states the answer, set supported to true, write the answer in text, and cite that chunk.
Each citation has chunk_id copied from that chunk.
Return JSON with the keys supported, text, and citations.

## Examples

<example>
<chunks>
<chunk>
chunk_id: ex-sign-in
document_title: Visitor Policy
section_path: Reception
Visitors must sign in at reception before entering the office.
</chunk>
</chunks>
<question>
Must visitors sign in?
</question>
<answer>
{"supported": true, "text": "Yes. Visitors must sign in at reception before entering the office.", "citations": [{"chunk_id": "ex-sign-in"}]}
</answer>
</example>
<example>
<chunks>
<chunk>
chunk_id: ex-badges
document_title: Visitor Policy
section_path: Badges
The following badge colors are issued at reception:
- Blue badges for employees
- Yellow badges for visitors
</chunk>
</chunks>
<question>
What color badge do visitors receive?
</question>
<answer>
{"supported": true, "text": "Visitors receive yellow badges.", "citations": [{"chunk_id": "ex-badges"}]}
</answer>
</example>
<example>
<chunks>
<chunk>
chunk_id: ex-sign-in
document_title: Visitor Policy
section_path: Reception
Visitors must sign in at reception before entering the office.
</chunk>
</chunks>
<question>
What is served in the office cafeteria?
</question>
<answer>
{"supported": false, "text": "The corpus does not mention this.", "citations": []}
</answer>
</example>
