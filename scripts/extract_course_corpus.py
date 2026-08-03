#!/usr/bin/env python3
"""Extract the local course PDF corpus with OpenDataLoader PDF.

The downloaded files intentionally keep their original course/lecture layout,
which means many of them share names such as ``01-slides.pdf``.  OpenDataLoader
writes outputs by stem, so this script creates a temporary directory of uniquely
named symlinks, parses the whole batch in one JVM invocation, and records a
source-to-output mapping for later curriculum synthesis.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "output" / "course-materials" / "combined-manifest.json"
OUTPUT_DIR = ROOT / "output" / "course-corpus" / "opendataloader"
STAGING_DIR = ROOT / "tmp" / "opendataloader-course-input"
PARSER = ROOT / ".venv" / "bin" / "opendataloader-pdf"


def slug(value: str, limit: int = 96) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (text or "document")[:limit].rstrip("-")


def load_documents(scope: str, courses: set[str]) -> list[dict[str, Any]]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    documents: list[dict[str, Any]] = []
    for course in data["courses"]:
        if courses and course["id"] not in courses:
            continue
        for session in course["sessions"]:
            groups = [("material", session.get("materials", []))]
            if scope == "all":
                groups.append(("reading", session.get("readings", [])))
            for group, items in groups:
                for index, item in enumerate(items, start=1):
                    if item.get("status") != "available":
                        continue
                    source = ROOT / "site" / "public" / item["public_url"].lstrip("/")
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    stem = "__".join(
                        [
                            slug(course["id"], 42),
                            slug(session["id"], 16),
                            group,
                            f"{index:03d}",
                            slug(source.stem, 64),
                        ]
                    )
                    documents.append(
                        {
                            "course_id": course["id"],
                            "course_title": course["title"],
                            "session_id": session["id"],
                            "session_title": session["title"],
                            "session_title_zh": session["title_zh"],
                            "session_summary": session["summary"],
                            "group": group,
                            "kind": item.get("kind"),
                            "label": item["label"],
                            "source_url": item["url"],
                            "source_pdf": str(source.relative_to(ROOT)),
                            "source_sha256": item.get("sha256"),
                            "source_pages": item.get("pages"),
                            "stem": stem,
                            "markdown": str((OUTPUT_DIR / f"{stem}.md").relative_to(ROOT)),
                            "json": str((OUTPUT_DIR / f"{stem}.json").relative_to(ROOT)),
                        }
                    )
    return documents


def output_is_valid(document: dict[str, Any]) -> bool:
    markdown = ROOT / document["markdown"]
    structured = ROOT / document["json"]
    return (
        markdown.is_file()
        and structured.is_file()
        and markdown.stat().st_size >= 100
        and structured.stat().st_size >= 100
    )


def prepare_staging(documents: list[dict[str, Any]]) -> None:
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)
    for document in documents:
        source = ROOT / document["source_pdf"]
        target = STAGING_DIR / f"{document['stem']}.pdf"
        target.symlink_to(source)


def run_parser(batch: list[dict[str, Any]], threads: int, timeout: int) -> bool:
    if not batch:
        return True
    prepare_staging(batch)
    if not any(STAGING_DIR.iterdir()):
        return
    if not PARSER.is_file():
        raise RuntimeError(
            "OpenDataLoader PDF is missing. Install it in .venv with: "
            ".venv/bin/pip install -U opendataloader-pdf"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        str(PARSER),
        str(STAGING_DIR),
        "-o",
        str(OUTPUT_DIR),
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
        str(threads),
        "-q",
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        names = ", ".join(item["source_pdf"] for item in batch)
        print(f"OpenDataLoader timeout after {timeout}s: {names}")
        return False
    if result.returncode != 0:
        print(f"OpenDataLoader returned {result.returncode} for a batch of {len(batch)}")
        return False
    return True


def markdown_stats(document: dict[str, Any]) -> dict[str, Any]:
    markdown = ROOT / document["markdown"]
    if not markdown.exists():
        return {"status": "missing"}
    text = markdown.read_text(encoding="utf-8", errors="replace")
    headings = [
        line.lstrip("# ").strip()
        for line in text.splitlines()
        if re.match(r"^#{1,4}\s+\S", line)
    ]
    compact = re.sub(r"\s+", " ", re.sub(r"!\[[^]]*\]\([^)]+\)", "", text)).strip()
    return {
        "status": "parsed",
        "characters": len(text),
        "lines": text.count("\n") + 1,
        "page_markers": text.count("<!-- page:"),
        "heading_count": len(headings),
        "headings": headings[:80],
        "excerpt": compact[:1200],
    }


def write_index(documents: list[dict[str, Any]], scope: str, elapsed: float) -> int:
    records = []
    for document in documents:
        record = dict(document)
        record["extraction"] = markdown_stats(document)
        records.append(record)
    failures = [record for record in records if record["extraction"]["status"] != "parsed"]
    payload = {
        "schema_version": 1,
        "parser": "OpenDataLoader PDF",
        "parser_package": "opendataloader-pdf 2.5.0",
        "generated_at": "2026-08-03",
        "scope": scope,
        "document_count": len(records),
        "parsed_count": len(records) - len(failures),
        "failure_count": len(failures),
        "source_pages": sum(int(record.get("source_pages") or 0) for record in records),
        "elapsed_seconds": round(elapsed, 2),
        "documents": records,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return len(failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=["materials", "all"],
        default="materials",
        help="Parse lecture materials only (default) or include all downloaded readings.",
    )
    parser.add_argument("--course", action="append", default=[], help="Only parse this course id.")
    parser.add_argument("--limit", type=int, default=0, help="Limit documents for a smoke test.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for each batch.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads < 1 or args.batch_size < 1 or args.timeout < 1:
        raise ValueError("--threads, --batch-size, and --timeout must be at least 1")
    documents = load_documents(args.scope, set(args.course))
    if args.limit:
        documents = documents[: args.limit]
    started = time.monotonic()
    pending = [document for document in documents if args.force or not output_is_valid(document)]
    print(f"OpenDataLoader corpus: {len(documents)} documents, {len(pending)} pending")
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        print(f"Batch {offset // args.batch_size + 1}: {len(batch)} document(s)", flush=True)
        run_parser(batch, args.threads, args.timeout)
    failures = write_index(documents, args.scope, time.monotonic() - started)
    parsed = len(documents) - failures
    print(f"Parsed {parsed}/{len(documents)} documents into {OUTPUT_DIR.relative_to(ROOT)}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
