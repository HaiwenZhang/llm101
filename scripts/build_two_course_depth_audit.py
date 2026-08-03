#!/usr/bin/env python3
"""Build a page-level depth audit for CS224N and CMU Advanced NLP.

The existing curriculum coverage file answers where a lecture is routed.  This
audit deliberately answers a different question: what is present on every
slide page, and how much tutorial material currently exists at the destination.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "output" / "course-corpus" / "opendataloader" / "index.json"
COVERAGE = ROOT / "output" / "course-corpus" / "curriculum-coverage.json"
OUTPUT_JSON = ROOT / "output" / "course-corpus" / "two-course-page-audit.json"
OUTPUT_MD = ROOT / "site" / "curriculum" / "two-course-depth-audit.md"

COURSE_IDS = ("cs224n-winter-2026", "cmu-anlp-spring-2026")
PAGE_MARKER = re.compile(r"<!--\s*page:(\d+)\s*-->")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", flags=re.MULTILINE)


def clean_markdown(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", " [代码或图表] ", text, flags=re.DOTALL)
    # Strip actual HTML/Vue tags without treating mathematical comparisons
    # such as y_{<t} as a tag opener and swallowing everything up to a later >.
    text = re.sub(
        r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^>]*)?/?>",
        " ",
        text,
    )
    text = re.sub(r"[#*_`>|]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_pages(text: str, expected_pages: int) -> list[dict]:
    matches = list(PAGE_MARKER.finditer(text))
    pages: list[dict] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw = text[start:end].strip()
        heading_match = HEADING.search(raw)
        cleaned = clean_markdown(raw)
        heading = clean_markdown(heading_match.group(1)) if heading_match else ""
        if not heading:
            heading = cleaned[:90] or "图示 / 过渡页"
        pages.append(
            {
                "page": int(match.group(1)),
                "heading": heading[:160],
                "characters": len(cleaned),
                "excerpt": cleaned[:360],
                "low_text": len(cleaned) < 70,
            }
        )
    if len(pages) != expected_pages:
        raise RuntimeError(
            f"Page marker mismatch: expected {expected_pages}, found {len(pages)}"
        )
    return pages


def lesson_metrics(link: str) -> dict:
    slug = link.rsplit("/", 1)[-1]
    path = ROOT / "site" / "beginner" / f"{slug}.md"
    if not path.exists():
        return {"slug": slug, "exists": False}
    text = path.read_text(encoding="utf-8")
    body = re.sub(r"^---.*?---\s*", "", text, count=1, flags=re.DOTALL)
    plain = clean_markdown(body)
    # A contiguous Chinese paragraph used to count as one unit while every
    # English word counted separately, which systematically marked Chinese
    # lessons as shallow.  Two CJK characters are roughly one compact reading
    # unit for this coarse depth heuristic; Latin text remains word-counted.
    cjk_characters = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", plain))
    latin_words = len(re.findall(r"[A-Za-z][A-Za-z0-9_-]*", plain))
    readable_units = round(cjk_characters / 2 + latin_words)
    components = sorted(
        set(
            re.findall(
                r"<(ConceptCheck|[A-Z][A-Za-z0-9]*(?:Lab|Diagram|Architecture))\b",
                body,
            )
        )
    )
    return {
        "slug": slug,
        "exists": True,
        "readable_units": readable_units,
        "cjk_characters": cjk_characters,
        "latin_words": latin_words,
        "headings": len(re.findall(r"^#{2,4}\s", body, flags=re.MULTILINE)),
        "figures": len(re.findall(r"<(?:figure|img)\b|```mermaid", body)),
        "interactive_components": components,
        "interactive_count": len(re.findall(r"<(?:ConceptCheck|[A-Z][A-Za-z0-9]*(?:Lab|Diagram|Architecture))\b", body)),
        "reading_section": bool(re.search(r"参考论文|推荐阅读|延伸阅读|本章阅读|<ChapterReadings\b", body)),
    }


def priority(source_pages: int, metrics: dict) -> str:
    if not metrics.get("exists"):
        return "缺页"
    units = int(metrics.get("readable_units") or 0)
    figures = int(metrics.get("figures") or 0)
    if source_pages >= 40 and (units < 600 or figures < 2):
        return "高"
    if source_pages >= 20 and (units < 950 or figures < 3):
        return "中"
    return "常规"


def main() -> None:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    coverage_by_session = {
        (row["course_id"], row["session_id"]): row
        for row in coverage["sessions"]
        if row["course_id"] in COURSE_IDS
    }

    documents = []
    session_rollup: dict[tuple[str, str], dict] = {}
    target_pages: defaultdict[str, int] = defaultdict(int)
    target_meta: dict[str, dict] = {}

    for document in corpus["documents"]:
        if document["course_id"] not in COURSE_IDS or document["group"] != "material":
            continue
        markdown_path = ROOT / document["markdown"]
        pages = split_pages(
            markdown_path.read_text(encoding="utf-8"), int(document["source_pages"])
        )
        row = coverage_by_session[(document["course_id"], document["session_id"])]
        record = {
            "course_id": document["course_id"],
            "course_title": document["course_title"],
            "session_id": document["session_id"],
            "session_title": document["session_title"],
            "session_title_zh": document["session_title_zh"],
            "label": document["label"],
            "kind": document["kind"],
            "source_pdf": document["source_pdf"],
            "markdown": document["markdown"],
            "source_pages": document["source_pages"],
            "page_markers": len(pages),
            "low_text_pages": [page["page"] for page in pages if page["low_text"]],
            "targets": row["targets"],
            "pages": pages,
        }
        documents.append(record)

        key = (document["course_id"], document["session_id"])
        if key not in session_rollup:
            session_rollup[key] = {
                "course_id": document["course_id"],
                "course_title": document["course_title"],
                "session_id": document["session_id"],
                "session_title": document["session_title"],
                "session_title_zh": document["session_title_zh"],
                "documents": 0,
                "pages": 0,
                "low_text_pages": 0,
                "targets": row["targets"],
            }
        session_rollup[key]["documents"] += 1
        session_rollup[key]["pages"] += len(pages)
        session_rollup[key]["low_text_pages"] += sum(page["low_text"] for page in pages)

    # Count a lecture's slide pages once for each destination chapter.  This is
    # a workload signal, not a claim that pages are independent across targets.
    for session in session_rollup.values():
        for target in session["targets"]:
            target_pages[target["id"]] += session["pages"]
            target_meta[target["id"]] = target

    chapter_audit = []
    for target_id, pages in sorted(target_pages.items()):
        target = target_meta[target_id]
        metrics = lesson_metrics(target["link"])
        chapter_audit.append(
            {
                "target_id": target_id,
                "display_number": target["display_number"],
                "title": target["title"],
                "link": target["link"],
                "source_slide_pages": pages,
                "priority": priority(pages, metrics),
                "lesson": metrics,
            }
        )

    material_docs = len(documents)
    material_pages = sum(document["source_pages"] for document in documents)
    reading_docs = [
        document
        for document in corpus["documents"]
        if document["course_id"] in COURSE_IDS and document["group"] == "reading"
    ]
    payload = {
        "schema_version": 1,
        "generated_at": "2026-08-03",
        "scope": list(COURSE_IDS),
        "parser": corpus["parser"],
        "summary": {
            "slide_documents": material_docs,
            "slide_pages": material_pages,
            "audited_page_records": sum(len(document["pages"]) for document in documents),
            "reading_documents": len(reading_docs),
            "reading_pages": sum(int(document["source_pages"]) for document in reading_docs),
            "page_marker_mismatches": 0,
        },
        "sessions": sorted(session_rollup.values(), key=lambda row: (COURSE_IDS.index(row["course_id"]), row["session_id"])),
        "chapters": sorted(chapter_audit, key=lambda row: row["display_number"]),
        "documents": documents,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "---",
        "title: CS224N 与 CMU ANLP 逐页深度审计",
        "description: 区分课程路由、逐页解析与教程正文深度的可复核报告",
        "---",
        "",
        "# CS224N 与 CMU ANLP：逐页深度审计",
        "",
        '<div class="lesson-lead">“某讲映射到某章”只说明应该去哪里，不等于 Slides 的观点、例子和推导已经写进正文。本页把逐页来源证据与教程深度拆开统计，用来指导后续逐章扩写。</div>',
        "",
        "::: info 审计规模",
        f"两门课共有 **{material_docs} 份 Slides / 讲义，共 {material_pages:,} 页**；OpenDataLoader 已留下 **{material_pages:,} 条逐页记录**，页码不一致 **0**。另有 **{len(reading_docs)} 份论文或指定阅读，共 {payload['summary']['reading_pages']:,} 页**。",
        ":::",
        "",
        "## 怎样理解这份报告",
        "",
        "- `来源映射`：这一讲应该补到哪些教程章；",
        "- `逐页记录`：每页标题、文本摘要、字符量和低文本页已保存在 JSON；",
        "- `正文深度`：当前章的可读文本单元（约 2 个汉字或 1 个英文词）、图解和交互组件；",
        "- `扩写优先级`：来源页多但正文短、图解少的章节先处理。",
        "",
        "逐页明细：`output/course-corpus/two-course-page-audit.json`。低文本页通常是图、表、过渡页或图片型幻灯片，扩写时必须回看原 PDF，不能据此判断为空白。",
        "",
        "## 逐讲 Slides 审计",
        "",
        "| 课程 | 讲次 | 主题 | Slides | 页数 | 低文本页 | 去向 |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for session in payload["sessions"]:
        targets = "<br>".join(
            f"[{target['display_number']} · {target['title']}]({target['link']})"
            for target in session["targets"]
        )
        course = "CS224N" if session["course_id"].startswith("cs224n") else "CMU ANLP"
        lines.append(
            f"| {course} | {session['session_id']} | {session['session_title_zh']} | {session['documents']} | {session['pages']} | {session['low_text_pages']} | {targets} |"
        )

    lines += [
        "",
        "## 教程章深度缺口",
        "",
        "| 优先级 | 教程章 | 对应 Slides 页 | 正文单元 | 图解 | 交互 | 推荐阅读段 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    order = {"高": 0, "中": 1, "常规": 2, "缺页": 3}
    for row in sorted(payload["chapters"], key=lambda item: (order[item["priority"]], item["display_number"])):
        lesson = row["lesson"]
        components = "、".join(lesson.get("interactive_components", [])) or "—"
        lines.append(
            f"| {row['priority']} | [{row['display_number']} · {row['title']}]({row['link']}) | {row['source_slide_pages']} | {lesson.get('readable_units', 0)} | {lesson.get('figures', 0)} | {components} | {'有' if lesson.get('reading_section') else '缺'} |"
        )

    lines += [
        "",
        "## 后续使用规则",
        "",
        "1. 每次扩写先打开 JSON 中该讲的所有逐页记录，再回看低文本页和关键图表原页；",
        "2. 正文至少补齐问题、直觉、公式/算法、完整例子、失败模式和系统代价；",
        "3. 章末推荐阅读区分必读与选读，并给出带问题的阅读指引；",
        "4. 只有正文、图解、实验和引用都可从页面验证时，才把对应缺口视为完成。",
        "",
    ]
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
