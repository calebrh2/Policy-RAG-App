"""Extract the Coforge PDF corpus with page provenance, tables, and coverage checks.

Run: uv run python scripts/preprocessing.py
Outputs are derived artifacts; original PDFs are never modified.
The defaults read ``data/source/previous`` and write ``data/extracted/previous``
so a plain run does not replace the Meridian markdown.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/source/previous"
OUTPUT = ROOT / "data/extracted/previous"


def tokens(text: str) -> Counter[str]:
    return Counter(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def clean(text: str) -> str:
    """Normalize layout whitespace, not policy facts or source wording."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("signifci ant", "significant")  # verified PDF ligature artifact
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^th (Review Date – 10) ", r"\1th ", text)
    return text


def table_rows(table: Any) -> tuple[list[str], list[list[str]]]:
    rows = [[clean(c or "") for c in row] for row in table.extract()]
    preamble = []
    while rows and rows[0][0] and all(not c for c in rows[0][1:]):
        # Single-row continuation tables must remain rows, not preambles.
        if rows[0][0] in {"Scope 3", "Total Emissions"}:
            break
        preamble.append(rows.pop(0)[0])
    # Known merged header: strict line detection still sees an extra divider.
    if rows and rows[0] == ["Emissions", "Emiss", "ions (tCO2e)", ""]:
        preamble.append("Emissions (tCO2e)")
        rows = [["Emissions", "UK", "India"]] + [[row[0], row[1], row[3]] for row in rows[2:]]
    return preamble, rows


def render_table(rows: list[list[str]]) -> str:
    def row_line(row: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "&#124;") for c in row) + " |"

    return "\n".join(
        [row_line(rows[0]), row_line(["---"] * len(rows[0]))] + [row_line(row) for row in rows[1:]]
    )


# Explicit headings for this corpus avoid treating bold numerical claims as headings.
HEADINGS = {
    "Commitment to achieving Net Zero",
    "Baseline Emissions Footprint",
    "Emissions reduction targets",
    "Carbon Reduction Initiatives",
    "Declaration and Sign Off",
    "Signed on behalf of the Supplier:",
    "Scope",
    "Purpose",
    "Definitions",
    "Roles & Responsibilities",
    "Goal",
    "Exemptions/Waiver",
    "Elimination & Substitution (Mandatory Prohibition)",
    "Design for Reuse",
    "Regulatory Compliance (India)",
    "Supplier Engagement & Contract Controls (including EPR where applicable)",
    "Waste Minimization & Segregation",
    "Transparency & Continual Improvement",
    "Targets & KPIs (India – All Sites)",
    "Targets",
    "Monitoring & Reporting",
    "Policy Commitment",
    "Legal and Regulatory Compliance",
    "Responsible Water Use and Conservation",
    "Water Recycling, Harvesting and Reuse",
    "Water Quality and Pollution Prevention",
    "Value Chain and Supplier Engagement",
    "Community Water Stewardship",
    "Awareness and Capacity Building",
    "Objectives and Targets",
    "Water Risk Assessment",
    "Governance and Roles & Responsibilities",
    "Monitoring, Reporting and Disclosure",
    "Grievance Mechanism",
    "Alignment with Frameworks and Standards",
    "Compliance Obligations",
    "Approved by:",
    "About Coforge",
}


def is_heading(text: str) -> bool:
    return text in HEADINGS or text.startswith(
        ("Baseline Year:", "Current Year Emission:", "Annexure A:", "A1.", "A2.", "A3.")
    )


def extract(path: Path, output: Path) -> dict[str, Any]:
    doc = pymupdf.open(path)
    events: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "source": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pages": [],
        "image_text_ocr_verified": False,
        "scope": "Corpus-specific normalization; original PDFs retained; only Markdown files are written.",
    }
    expected = []
    for index, page in enumerate(doc):
        number = index + 1
        raw = page.get_text(sort=True)
        finder = page.find_tables(strategy="lines_strict")
        tables = finder.tables
        page_events = []
        excluded = []
        # Contents pages duplicate section labels and page numbers, not policy rules.
        contents_page = any(line.strip() == "Contents" for line in raw.splitlines())
        for table in tables:
            preamble, rows = table_rows(table)
            for j, text in enumerate(preamble):
                page_events.append(
                    (
                        table.bbox[1] - 0.1 + j * 0.001,
                        table.bbox[0],
                        {
                            "kind": "heading" if is_heading(text) else "text",
                            "text": text,
                            "page": number,
                        },
                    )
                )
            if rows:
                page_events.append(
                    (
                        table.bbox[1],
                        table.bbox[0],
                        {"kind": "table", "rows": rows, "page": number, "end_page": number},
                    )
                )
        for block in page.get_text("dict", sort=True)["blocks"]:
            if block["type"] != 0:
                continue
            # Group spans at the same baseline before ordering: keeps Name: + value together.
            baseline_rows: dict[float, list[Any]] = {}
            for line in block["lines"]:
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    if any(
                        pymupdf.Rect(t.bbox).contains(pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2))
                        for t in tables
                    ):
                        continue
                    baseline_rows.setdefault(round(span["origin"][1], 1), []).append(span)
            lines = []
            for _, spans in sorted(baseline_rows.items()):
                spans.sort(key=lambda s: s["bbox"][0])
                text = clean(" ".join(s["text"] for s in spans))
                if not text:
                    continue
                if contents_page or text.startswith("©"):
                    excluded.append(text)
                    continue
                lines.append((text, min(s["bbox"][1] for s in spans)))
            pending = []
            start_y = block["bbox"][1]

            def flush(
                pending: list[str], page_events: list[Any], start_y: float, x: float, number: int
            ) -> None:
                if pending:
                    page_events.append(
                        (
                            start_y,
                            x,
                            {"kind": "text", "text": clean(" ".join(pending)), "page": number},
                        )
                    )
                    pending.clear()

            for text, y in lines:
                if is_heading(text):
                    flush(pending, page_events, start_y, block["bbox"][0], number)
                    page_events.append(
                        (y, block["bbox"][0], {"kind": "heading", "text": text, "page": number})
                    )
                # "2030." is a wrapped year. Numbered items in this corpus have text after the marker.
                elif re.match(r"^(?:[•\uf077](?:$|\s)|\d+\.\s)", text):
                    flush(pending, page_events, start_y, block["bbox"][0], number)
                    start_y = y
                    pending.append(re.sub(r"^[•\uf077]", "-", text))
                else:
                    if not pending:
                        start_y = y
                    pending.append(text)
            flush(pending, page_events, start_y, block["bbox"][0], number)
        ordered = [e[2] for e in sorted(page_events, key=lambda x: (x[0], x[1]))]
        # Annexure title wraps onto the following line in the plastic PDF.
        for i in range(len(ordered) - 1):
            if ordered[i].get("text", "").endswith("(India –") and ordered[i + 1].get(
                "text", ""
            ).startswith("Coforge Premises)"):
                ordered[i]["text"] += " Coforge Premises)"
                ordered[i + 1]["text"] = ordered[i + 1]["text"][len("Coforge Premises)") :].strip()
        # Use unsorted text for coverage: sorted extraction can merge overlapping footer glyphs.
        raw_tokens = tokens(page.get_text())
        excluded_tokens = tokens(" ".join(excluded))
        expected.append(raw_tokens - excluded_tokens)
        events.extend(e for e in ordered if e.get("kind") == "table" or e.get("text"))
        report["pages"].append(
            {
                "page": number,
                "text_characters": len(raw),
                "table_fragments": len(tables),
                "excluded_layout_text": excluded,
                "contents_page_excluded": contents_page,
                "ocr_required": not raw.strip(),
                "visual_review_required": True,
            }
        )
    # The carbon PDF stores sign-off labels and values in separate blocks.
    # Reattach adjacent labels without changing the values.
    fixed = []
    for event in events:
        if event.get("text") in {"Name:", "Title:", "Date:"} and fixed:
            prior = fixed[-1]
            if prior["kind"] == "text" and prior["page"] == event["page"]:
                prior["text"] = event["text"] + " " + prior["text"]
                continue
        fixed.append(event)
    events = fixed
    # Merge immediately adjacent continuation fragments before inserting page comments.
    merged = []
    for event in events:
        if event["kind"] == "table" and event["rows"][0][0] in {"Scope 3", "Total Emissions"}:
            if not merged or merged[-1]["kind"] != "table":
                raise ValueError("Unresolved table continuation")
            merged[-1]["rows"].extend(event["rows"])
            merged[-1]["end_page"] = event["page"]
        else:
            merged.append(event)
    parts = [f"# {path.stem}", f"<!-- source_file: {path.name} -->"]
    last_marker = None
    for event in merged:
        end = event.get("end_page", event["page"])
        marker = f"<!-- source_pages: {event['page']}-{end} -->"
        if event["kind"] == "table":
            text = render_table(event["rows"])
        else:
            text = event["text"]
            if event["kind"] == "heading":
                level = (
                    "###"
                    if text.startswith(("A1.", "A2.", "A3.", "Baseline Year:"))
                    or text in {"Targets", "Monitoring & Reporting"}
                    else "##"
                )
                text = f"{level} {text}"
        if marker != last_marker:
            parts.append(marker)
            last_marker = marker
        parts.append(text)
    body = "\n\n".join(parts) + "\n"
    wanted = sum(expected, Counter())
    actual = tokens(html.unescape(re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)))
    report["missing_content_tokens"] = dict(wanted - actual)
    report["coverage_note"] = (
        "Word-count coverage excludes documented contents/footer text; it does not certify reading order or image contents."
    )
    (output / f"{path.stem}.md").write_text(body, encoding="utf-8")
    doc.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    paths = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() == ".pdf")
    if not paths:
        parser.error(f"No PDFs found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [extract(p, args.output_dir) for p in paths]
    for report in reports:
        print(
            Path(report["source"]).name,
            len(report["pages"]),
            "pages;",
            sum(p["table_fragments"] for p in report["pages"]),
            "table fragments;",
            len(report["missing_content_tokens"]),
            "distinct missing content tokens",
        )
    if any(r["missing_content_tokens"] for r in reports):
        raise RuntimeError(
            f"Text coverage check failed: {[(r['source'], r['missing_content_tokens']) for r in reports if r['missing_content_tokens']]}"
        )
    print(f"Output: {args.output_dir}")
    print("Review tables against the original PDFs. Text coverage does not prove visual accuracy.")
    if any(p["ocr_required"] for r in reports for p in r["pages"]):
        print("WARNING: Pages without selectable text need OCR.")


if __name__ == "__main__":
    main()
