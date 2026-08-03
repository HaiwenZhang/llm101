#!/usr/bin/env python3
"""Download, validate, and parse the Kimi K3 core reading set.

The pipeline is intentionally deterministic and resumable:

- paper metadata lives in papers/manifest.json;
- PDFs are downloaded atomically into papers/;
- extracted Markdown/JSON lives under output/papers/<paper-slug>/;
- site/.vitepress/theme/data/papers.json records checksums, page counts, and extraction stats;
- existing valid artifacts are reused unless a force flag is supplied.

OpenDataLoader PDF is used because it preserves headings, tables, page boundaries,
and optionally extracted figures better than a plain text-only extractor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "papers" / "manifest.json"
PAPERS_DIR = ROOT / "papers"
OUTPUT_DIR = ROOT / "output" / "papers"
INDEX_PATH = ROOT / "site" / ".vitepress" / "theme" / "data" / "papers.json"
TMP_DIR = ROOT / "tmp" / "pdfs"


@dataclass(frozen=True)
class Paper:
    order: int
    slug: str
    title: str
    arxiv_id: str
    filename: str
    local_seed: str | None
    textbook_chapters: tuple[int, ...]
    focus: tuple[str, ...]

    @property
    def url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}"

    @property
    def pdf_path(self) -> Path:
        return PAPERS_DIR / self.filename

    @property
    def output_path(self) -> Path:
        return OUTPUT_DIR / f"{self.order:02d}_{self.slug}"


def run(
    command: list[str],
    *,
    timeout: int | None = None,
    cwd: Path = ROOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def find_executable(name: str, repo_candidate: Path | None = None) -> str:
    if repo_candidate and repo_candidate.exists():
        return str(repo_candidate)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"Required executable not found: {name}")
    return found


def load_manifest() -> list[Paper]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    rows = payload.get("papers", [])
    papers = [
        Paper(
            order=int(row["order"]),
            slug=str(row["slug"]),
            title=str(row["title"]),
            arxiv_id=str(row["arxiv_id"]),
            filename=str(row["filename"]),
            local_seed=row.get("local_seed"),
            textbook_chapters=tuple(int(x) for x in row.get("textbook_chapters", [])),
            focus=tuple(str(x) for x in row.get("focus", [])),
        )
        for row in rows
    ]
    orders = [paper.order for paper in papers]
    slugs = [paper.slug for paper in papers]
    filenames = [paper.filename for paper in papers]
    if len(set(orders)) != len(orders) or len(set(slugs)) != len(slugs) or len(set(filenames)) != len(filenames):
        raise ValueError("Manifest contains duplicate order, slug, or filename")
    return sorted(papers, key=lambda paper: paper.order)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_info(path: Path, pdfinfo: str) -> dict[str, str]:
    result = run([pdfinfo, str(path)], timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"pdfinfo failed for {path.name}: {result.stderr.strip()}")
    info: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def validate_pdf(path: Path, pdfinfo: str) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size < 10_000:
        raise RuntimeError(f"Missing or implausibly small PDF: {path}")
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise RuntimeError(f"File does not start with a PDF header: {path}")
    info = pdf_info(path, pdfinfo)
    pages = int(info.get("Pages", "0"))
    if pages < 2:
        raise RuntimeError(f"PDF has an implausible page count ({pages}): {path}")
    return info


def download_paper(paper: Paper, *, curl: str, pdfinfo: str, force: bool) -> str:
    target = paper.pdf_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        validate_pdf(target, pdfinfo)
        return "reused"

    if paper.local_seed:
        seed = ROOT / paper.local_seed
        if seed.exists() and not force:
            validate_pdf(seed, pdfinfo)
            temporary = target.with_suffix(target.suffix + ".part")
            shutil.copy2(seed, temporary)
            os.replace(temporary, target)
            validate_pdf(target, pdfinfo)
            return "seeded"

    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    command = [
        curl,
        "--http1.1",
        "-L",
        "--fail",
        "--retry",
        "4",
        "--retry-all-errors",
        "--connect-timeout",
        "20",
        "--max-time",
        "900",
        "--user-agent",
        "kimi-k3-study-pipeline/1.0",
        "-o",
        str(temporary),
        paper.url,
    ]
    result = run(command, timeout=960)
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"download failed ({paper.url}): {result.stderr.strip()}")
    validate_pdf(temporary, pdfinfo)
    os.replace(temporary, target)
    return "downloaded"


def extracted_paths(paper: Paper) -> tuple[Path, Path]:
    stem = paper.pdf_path.stem
    return paper.output_path / f"{stem}.md", paper.output_path / f"{stem}.json"


def extraction_is_fresh(paper: Paper) -> bool:
    markdown, json_path = extracted_paths(paper)
    if not markdown.exists() or not json_path.exists():
        return False
    if markdown.stat().st_size < 1_000 or json_path.stat().st_size < 1_000:
        return False
    source_mtime = paper.pdf_path.stat().st_mtime
    if markdown.stat().st_mtime < source_mtime or json_path.stat().st_mtime < source_mtime:
        return False
    return extracted_images_are_complete(paper, markdown)


def markdown_image_references(markdown: Path) -> set[str]:
    text = markdown.read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r"images/([^)><\\s]+)", text))


def extracted_images_are_complete(paper: Paper, markdown: Path) -> bool:
    references = markdown_image_references(markdown)
    if not references:
        return True
    image_dir = paper.output_path / "images"
    return all((image_dir / reference).is_file() for reference in references)


def parse_paper(
    paper: Paper,
    *,
    parser: str,
    force: bool,
    include_images: bool,
    threads: int,
) -> str:
    paper.output_path.mkdir(parents=True, exist_ok=True)
    if extraction_is_fresh(paper) and not force:
        return "reused"

    markdown, json_path = extracted_paths(paper)
    markdown.unlink(missing_ok=True)
    json_path.unlink(missing_ok=True)
    image_dir = paper.output_path / "images"
    if force and image_dir.exists():
        shutil.rmtree(image_dir)
    image_mode = "external" if include_images else "off"
    command = [
        parser,
        str(paper.pdf_path),
        "-o",
        str(paper.output_path),
        "-f",
        "markdown,json",
        "--reading-order",
        "xycut",
        "--table-method",
        "cluster",
        "--markdown-page-separator",
        "<!-- page:%page-number% -->",
        "--image-output",
        image_mode,
        "--threads",
        str(threads),
        "-q",
    ]
    if include_images:
        image_dir.mkdir(parents=True, exist_ok=True)
        command.extend(["--image-format", "png", "--image-dir", "images"])
    result = run(command, timeout=3600, cwd=paper.output_path)
    if result.returncode != 0:
        raise RuntimeError(f"parser failed for {paper.pdf_path.name}: {result.stderr.strip()}")
    if not extraction_is_fresh(paper):
        raise RuntimeError(f"parser did not create valid Markdown and JSON for {paper.pdf_path.name}")
    return "parsed"


def markdown_stats(path: Path) -> dict[str, int | bool]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "characters": len(text),
        "lines": text.count("\n") + 1,
        "headings": sum(1 for line in text.splitlines() if line.startswith("#")),
        "page_markers": text.count("<!-- page:"),
        "image_references": len(markdown_image_references(path)),
        "has_abstract": "abstract" in text.lower(),
        "has_references": "references" in text.lower(),
    }


def record_for(paper: Paper, pdfinfo: str, download_status: str, parse_status: str) -> dict[str, Any]:
    info = validate_pdf(paper.pdf_path, pdfinfo)
    markdown, json_path = extracted_paths(paper)
    return {
        "order": paper.order,
        "slug": paper.slug,
        "title": paper.title,
        "arxiv_id": paper.arxiv_id,
        "source_url": paper.url,
        "pdf": str(paper.pdf_path.relative_to(ROOT)),
        "markdown": str(markdown.relative_to(ROOT)) if markdown.exists() else None,
        "json": str(json_path.relative_to(ROOT)) if json_path.exists() else None,
        "sha256": sha256(paper.pdf_path),
        "bytes": paper.pdf_path.stat().st_size,
        "pages": int(info.get("Pages", "0")),
        "pdf_title": info.get("Title", ""),
        "download_status": download_status,
        "parse_status": parse_status,
        "textbook_chapters": list(paper.textbook_chapters),
        "focus": list(paper.focus),
        "extraction": markdown_stats(markdown) if markdown.exists() else None,
    }


def write_index(records: list[dict[str, Any]], failures: list[dict[str, str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at_unix": int(time.time()),
        "paper_count": len(records),
        "failure_count": len(failures),
        "papers": records,
        "failures": failures,
    }
    INDEX_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Parsed core papers",
        "",
        "Generated by `scripts/paper_pipeline.py`. Do not edit generated metadata by hand.",
        "",
        "| # | Paper | Pages | PDF | Parsed Markdown | Textbook chapters |",
        "|---:|---|---:|---|---|---|",
    ]
    for row in records:
        chapters = ", ".join(str(value) for value in row["textbook_chapters"])
        markdown_link = (
            f"[{row['markdown']}](../../{row['markdown']})"
            if row["markdown"]
            else "not parsed"
        )
        lines.append(
            f"| {row['order']} | {row['title']} | {row['pages']} | "
            f"[{row['pdf']}](../../{row['pdf']}) | "
            f"{markdown_link} | {chapters} |"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure['slug']}`: {failure['error']}")
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_previous_index(
    records: list[dict[str, Any]],
    failures: list[dict[str, str]],
    *,
    all_papers: list[Paper],
    selected_papers: list[Paper],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Keep a complete index after a targeted --only run.

    A failed selected paper must not retain a stale successful record. If no prior
    index exists, a targeted run still writes a truthful partial index, which the
    textbook generator will reject until the full pipeline has run once.
    """

    if len(selected_papers) == len(all_papers) or not INDEX_PATH.exists():
        return records, failures
    try:
        previous = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return records, failures

    selected_slugs = {paper.slug for paper in selected_papers}
    record_by_slug = {
        str(row["slug"]): row
        for row in previous.get("papers", [])
        if str(row.get("slug", "")) not in selected_slugs
    }
    record_by_slug.update({str(row["slug"]): row for row in records})

    failure_by_slug = {
        str(row["slug"]): row
        for row in previous.get("failures", [])
        if str(row.get("slug", "")) not in selected_slugs
    }
    failure_by_slug.update({str(row["slug"]): row for row in failures})
    failed_slugs = set(failure_by_slug)
    for slug in failed_slugs:
        record_by_slug.pop(slug, None)

    ordered_records = [record_by_slug[paper.slug] for paper in all_papers if paper.slug in record_by_slug]
    ordered_failures = [failure_by_slug[paper.slug] for paper in all_papers if paper.slug in failure_by_slug]
    return ordered_records, ordered_failures


def select_papers(papers: Iterable[Paper], selectors: list[str]) -> list[Paper]:
    if not selectors:
        return list(papers)
    wanted = {item.strip() for raw in selectors for item in raw.split(",") if item.strip()}
    selected = [
        paper
        for paper in papers
        if paper.slug in wanted or paper.arxiv_id in wanted or str(paper.order) in wanted
    ]
    matched = {paper.slug for paper in selected} | {paper.arxiv_id for paper in selected} | {
        str(paper.order) for paper in selected
    }
    missing = wanted - matched
    if missing:
        raise ValueError(f"Unknown --only selector(s): {', '.join(sorted(missing))}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download-only", action="store_true", help="Download and validate PDFs, but do not parse")
    mode.add_argument("--parse-only", action="store_true", help="Parse existing PDFs without downloading")
    parser.add_argument("--only", action="append", default=[], help="Comma-separated order, slug, or arXiv id")
    parser.add_argument("--force-download", action="store_true", help="Redownload selected PDFs")
    parser.add_argument("--force-parse", action="store_true", help="Reparse selected PDFs")
    parser.add_argument("--no-images", action="store_true", help="Do not extract embedded figures")
    parser.add_argument("--threads", type=int, default=2, help="OpenDataLoader parser threads (default: 2)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failed paper")
    parser.add_argument("--list", action="store_true", help="List manifest entries and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")

    papers = load_manifest()
    selected = select_papers(papers, args.only)
    if args.list:
        for paper in selected:
            print(f"{paper.order:02d}  {paper.arxiv_id}  {paper.slug:24s}  {paper.title}")
        return 0

    curl = find_executable("curl")
    pdfinfo = find_executable("pdfinfo")
    parser = find_executable("opendataloader-pdf", ROOT / ".venv" / "bin" / "opendataloader-pdf")
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for paper in selected:
        download_status = "not-requested"
        parse_status = "not-requested"
        try:
            print(f"[{paper.order:02d}/{len(papers)-1:02d}] {paper.title}", flush=True)
            if not args.parse_only:
                download_status = download_paper(
                    paper,
                    curl=curl,
                    pdfinfo=pdfinfo,
                    force=args.force_download,
                )
                print(f"  PDF: {download_status} -> {paper.pdf_path.relative_to(ROOT)}", flush=True)
            else:
                validate_pdf(paper.pdf_path, pdfinfo)
                download_status = "preexisting"

            if not args.download_only:
                parse_status = parse_paper(
                    paper,
                    parser=parser,
                    force=args.force_parse,
                    include_images=not args.no_images,
                    threads=args.threads,
                )
                markdown, _ = extracted_paths(paper)
                print(f"  parse: {parse_status} -> {markdown.relative_to(ROOT)}", flush=True)

            records.append(record_for(paper, pdfinfo, download_status, parse_status))
        except Exception as exc:  # Keep a useful batch report instead of hiding the failed paper.
            failure = {"slug": paper.slug, "error": str(exc)}
            failures.append(failure)
            print(f"  ERROR: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                break

    processed_records = len(records)
    processed_failures = len(failures)
    records, failures = merge_previous_index(
        records,
        failures,
        all_papers=papers,
        selected_papers=selected,
    )
    write_index(records, failures)
    print(
        f"Processed: {processed_records} paper(s), {processed_failures} failure(s); "
        f"index: {len(records)} paper(s), {len(failures)} failure(s)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
