#!/usr/bin/env python3
"""Extract a textbook PDF with the project-local OpenDataLoader PDF runtime."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARSER = ROOT / ".venv" / "bin" / "opendataloader-pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_pdf")
    parser.add_argument("output_dir")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    source = Path(args.input_pdf).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not PARSER.is_file():
        raise FileNotFoundError(PARSER)

    started = time.monotonic()
    command = [
        str(PARSER),
        str(source),
        "-o",
        str(output),
        "-f",
        "markdown,json",
        "--reading-order",
        "xycut",
        "--table-method",
        "cluster",
        "--markdown-page-separator",
        "<!-- page:%page-number% -->",
        "--image-output",
        "off",
        "--threads",
        str(args.threads),
        "-q",
    ]
    subprocess.run(command, cwd=ROOT, check=True, timeout=1800)

    markdown = output / f"{source.stem}.md"
    structured = output / f"{source.stem}.json"
    if not markdown.is_file() or not structured.is_file():
        raise RuntimeError("OpenDataLoader did not produce the expected Markdown and JSON")
    text = markdown.read_text(encoding="utf-8", errors="replace")
    headings = [
        line.lstrip("# ").strip()
        for line in text.splitlines()
        if re.match(r"^#{1,4}\s+\S", line)
    ]
    manifest = {
        "source": source.name,
        "parser": "OpenDataLoader PDF",
        "parser_package": "opendataloader-pdf 2.5.0",
        "markdown": markdown.name,
        "json": structured.name,
        "pages": text.count("<!-- page:"),
        "characters": len(text),
        "headings": headings,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    (output / "opendataloader-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("parser", "pages", "characters", "elapsed_seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
