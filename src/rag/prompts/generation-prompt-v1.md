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
chunk_id: ex-cups
document_title: Single-use Plastic-free Policy
section_path: Prohibited items
Single-use plastic cups are banned in every office.
</chunk>
</chunks>
<question>
Are single-use plastic cups banned?
</question>
<answer>
{"supported": true, "text": "Yes. Single-use plastic cups are banned in every office.", "citations": [{"chunk_id": "ex-cups"}]}
</answer>
</example>
<example>
<chunks>
<chunk>
chunk_id: ex-list
document_title: Single-use Plastic-free Policy
section_path: Elimination & Substitution
The purchase of the following items is prohibited:
- Plastic cutlery (fork/spoon/knife)
- Plastic plates, cups, and glasses
</chunk>
</chunks>
<question>
Are plastic cups prohibited?
</question>
<answer>
{"supported": true, "text": "Yes. Plastic plates, cups, and glasses are prohibited.", "citations": [{"chunk_id": "ex-list"}]}
</answer>
</example>
<example>
<chunks>
<chunk>
chunk_id: ex-water
document_title: Water Management Policy
section_path: Monitoring
Sites record monthly water use.
</chunk>
</chunks>
<question>
What is the deadline for travel expense reports?
</question>
<answer>
{"supported": false, "text": "The corpus does not mention this.", "citations": []}
</answer>
</example>
