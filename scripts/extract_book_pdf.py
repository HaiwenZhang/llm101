#!/usr/bin/env python3
"""Extract page text and outline metadata from a local textbook PDF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


def flatten_outline(reader: PdfReader) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []

    def walk(items: list[object], depth: int = 0) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
                continue
            try:
                page = reader.get_destination_page_number(item) + 1
            except Exception:
                page = None
            entries.append(
                {
                    "title": getattr(item, "title", str(item)),
                    "page": page,
                    "depth": depth,
                }
            )

    walk(reader.outline)
    return entries


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: extract_book_pdf.py INPUT.pdf OUTPUT_DIR")

    pdf_path = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    outline = flatten_outline(reader)

    pages: list[dict[str, object]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            pages.append({"page": index, "text": text.strip()})

    manifest = {
        "source": pdf_path.name,
        "pages": len(pages),
        "outline_entries": len(outline),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "outline.json").write_text(
        json.dumps(outline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "pages.json").write_text(
        json.dumps(pages, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "full-text.txt").write_text(
        "\n\n".join(f"<!-- PDF page {p['page']} -->\n{p['text']}" for p in pages),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
