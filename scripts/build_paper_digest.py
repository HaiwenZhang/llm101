#!/usr/bin/env python3
"""Render curated paper notes into the Kimi K3 Chinese textbook.

The chapter is generated from three sources:

1. papers/manifest.json: canonical paper order and metadata;
2. output/papers/index.json: validated local PDF/extraction artifacts and statistics;
3. study/paper_notes.json: human-curated, source-grounded Chinese study notes.

The script owns only the text between the generated markers, so the rest of the
textbook remains hand-editable. Re-running it is deterministic and idempotent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "papers" / "manifest.json"
INDEX_PATH = ROOT / "output" / "papers" / "index.json"
NOTES_PATH = ROOT / "study" / "paper_notes.json"
TEXTBOOK_PATH = ROOT / "study" / "03_kimi_k3_textbook.md"

BEGIN_MARKER = "<!-- BEGIN GENERATED PAPER DIGEST -->"
END_MARKER = "<!-- END GENERATED PAPER DIGEST -->"
INSERT_BEFORE = "# 附录 A"

LIST_FIELDS = ("mechanism", "evidence", "k3_bridge", "caution")
TEXT_FIELDS = ("position", "reading", "one_sentence", "problem", "checkpoint")


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required input is missing: {path.relative_to(ROOT)}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path.relative_to(ROOT)}: {exc}") from exc


def relative_link(target: str) -> str:
    target_path = ROOT / target
    if not target_path.is_file():
        raise RuntimeError(f"Indexed artifact is missing: {target}")
    return os.path.relpath(target_path, start=TEXTBOOK_PATH.parent).replace(os.sep, "/")


def validate_and_join() -> list[dict[str, Any]]:
    manifest = load_json(MANIFEST_PATH)
    index = load_json(INDEX_PATH)
    notes = load_json(NOTES_PATH)

    if int(index.get("failure_count", -1)) != 0:
        raise RuntimeError("Paper index contains failures; rerun scripts/paper_pipeline.py first")

    manifest_rows = {str(row["slug"]): row for row in manifest.get("papers", [])}
    index_rows = {str(row["slug"]): row for row in index.get("papers", [])}
    note_rows = {str(row["slug"]): row for row in notes.get("papers", [])}
    key_sets = {
        "manifest": set(manifest_rows),
        "index": set(index_rows),
        "notes": set(note_rows),
    }
    expected = key_sets["manifest"]
    for name, keys in key_sets.items():
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise RuntimeError(f"{name} slug mismatch; missing={missing}, extra={extra}")

    if len(note_rows) != len(notes.get("papers", [])):
        raise RuntimeError("Duplicate slug in study/paper_notes.json")

    joined: list[dict[str, Any]] = []
    for slug, manifest_row in manifest_rows.items():
        row = {**manifest_row, **index_rows[slug], **note_rows[slug]}
        if int(row["order"]) != int(manifest_row["order"]):
            raise RuntimeError(f"Order mismatch for {slug}")
        for field in TEXT_FIELDS:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise RuntimeError(f"Missing text field {field!r} for {slug}")
        for field in LIST_FIELDS:
            values = row.get(field)
            if not isinstance(values, list) or not values or not all(isinstance(x, str) and x.strip() for x in values):
                raise RuntimeError(f"Missing list field {field!r} for {slug}")
        if not row.get("markdown") or not row.get("pdf"):
            raise RuntimeError(f"Index lacks PDF or Markdown path for {slug}")
        row["pdf_link"] = relative_link(str(row["pdf"]))
        row["markdown_link"] = relative_link(str(row["markdown"]))
        markdown = ROOT / str(row["markdown"])
        extracted = markdown.read_text(encoding="utf-8", errors="replace")
        page_markers = extracted.count("<!-- page:")
        if page_markers != int(row["pages"]):
            raise RuntimeError(
                f"Page marker mismatch for {slug}: index={row['pages']}, markdown={page_markers}"
            )
        if str(row["arxiv_id"]) not in extracted[:5_000]:
            raise RuntimeError(f"Parsed Markdown does not identify arXiv:{row['arxiv_id']} for {slug}")
        joined.append(row)
    return sorted(joined, key=lambda item: int(item["order"]))


def bullet_list(values: list[str]) -> list[str]:
    lines: list[str] = []
    for value in values:
        lines.extend([f"- {value}", ""])
    return lines


def render_digest(rows: list[dict[str, Any]]) -> str:
    total_pages = sum(int(row["pages"]) for row in rows)
    total_chars = sum(int(row.get("extraction", {}).get("characters", 0)) for row in rows)
    lines = [
        BEGIN_MARKER,
        "",
        "# 第八部分：把 13 篇论文压缩成一条技术演化链",
        "",
        "## 第 16 章　必读论文浓缩精读卡",
        "",
        f"本章由脚本从 **{len(rows)} 篇、{total_pages} 页、约 {total_chars / 10_000:.1f} 万字符**的本地解析语料生成。",
        "它不是摘要拼接，而是按“问题 → 机制 → 证据 → 与 K3 的关系 → 适用边界”重新组织。",
        "所有结论均为对本地 PDF 的转述；公式、表格和精确数字若要引用到工作文档中，必须回到 PDF 核对。",
        "",
        "> 生成说明：论文元数据来自 `papers/manifest.json`，解析统计来自 `output/papers/index.json`，编者笔记来自 `study/paper_notes.json`。运行 `python scripts/build_paper_digest.py` 可幂等重建本章。",
        "",
        "### 16.1　先按因果关系读，不按发表时间读",
        "",
        "1. **预算观**：Chinchilla——先理解固定计算下参数与数据怎样分配。",
        "2. **稀疏宽度**：DeepSeekMoE → DeepSeek-V2/V3 → LatentMoE——从专家专门化走到真实硬件瓶颈。",
        "3. **长序列**：Gated DeltaNet → Kimi Linear——从 delta rule 走到逐通道 KDA 和 3:1 hybrid。",
        "4. **网络深度**：Attention Residuals——把内容相关检索从 token 轴扩展到 layer 轴。",
        "5. **后训练与 Agent**：Kimi k1.5 → DeepSeek-R1 → Kimi K2 → Kimi K2.5——从可验证推理扩到工具、视觉和并行协作。",
        "6. **总装复盘**：最后重读 Kimi K3，检查这些模块怎样被系统共设计连接起来。",
        "",
        "### 16.2　语料索引",
        "",
        "| # | 论文 | 页数 | 在学习链中的位置 | 本地材料 |",
        "|---:|---|---:|---|---|",
    ]
    for row in rows:
        links = f"[PDF]({row['pdf_link']}) · [解析全文]({row['markdown_link']})"
        title = str(row["title"]).replace("|", "\\|")
        position = str(row["position"]).replace("|", "\\|")
        lines.append(f"| {row['order']} | {title} | {row['pages']} | {position} | {links} |")

    lines.extend(["", "### 16.3　逐篇精读卡", ""])
    for card_number, row in enumerate(rows, start=1):
        title = str(row["title"])
        lines.extend(
            [
                f"#### 16.3.{card_number}　{title}",
                "",
                f"**定位**：{row['position']}  ",
                f"**定向阅读**：{row['reading']}  ",
                f"**本地来源**：[PDF]({row['pdf_link']}) · [带页码解析全文]({row['markdown_link']}) · [arXiv:{row['arxiv_id']}]({row['source_url']})",
                "",
                f"**一句话抓住它**：{row['one_sentence']}",
                "",
                f"**它在解决什么**：{row['problem']}",
                "",
                "**核心机制**",
                "",
            ]
        )
        lines.extend(bullet_list(row["mechanism"]))
        lines.extend(["**论文给了什么证据**", ""])
        lines.extend(bullet_list(row["evidence"]))
        lines.extend(["**怎样接到 K3**", ""])
        lines.extend(bullet_list(row["k3_bridge"]))
        lines.extend(["**不要过度外推**", ""])
        lines.extend(bullet_list(row["caution"]))
        lines.extend([f"**闭卷检查**：{row['checkpoint']}", "", "---", ""])

    lines.extend(
        [
            "### 16.4　读完后的统一心智模型",
            "",
            "把现代 LLM 看成六本账，而不是一张参数表：",
            "",
            "- **训练计算账**：参数、token、optimizer 与精度格式决定一次大 run 能否完成。",
            "",
            "- **激活计算账**：MoE 让总容量和每 token FLOPs 解耦，但会增加权重读取、路由和 all-to-all。",
            "",
            "- **状态内存账**：MLA 压缩历史 K/V，KDA 用固定递归状态；二者的表达能力与成本结构不同。",
            "",
            "- **信息流账**：KDA/MLA 管 token 轴，AttnRes 管 depth 轴，LatentMoE 管 channel/专家轴。",
            "",
            "- **学习信号账**：pre-training 学分布，SFT 建立接口和冷启动，RL 在环境反馈下探索，distillation 合并策略。",
            "",
            "- **在线系统账**：prefix cache、batch、并行拓扑、sandbox 状态与调度策略决定论文里的能力能否经济地交付。",
            "",
            "如果一个新方案只给出 loss，却不报告参数读取、通信、cache、吞吐和部署形态，你还不能判断它是否真的比 K3 的对应模块更好。",
            "",
            END_MARKER,
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def replace_generated_chapter(textbook: str, digest: str) -> str:
    begin_count = textbook.count(BEGIN_MARKER)
    end_count = textbook.count(END_MARKER)
    if begin_count != end_count or begin_count > 1:
        raise RuntimeError(
            f"Invalid generated markers: begin={begin_count}, end={end_count}"
        )
    if begin_count == 1:
        pattern = re.compile(
            re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER) + r"\n?",
            flags=re.DOTALL,
        )
        return pattern.sub(digest, textbook, count=1)
    if INSERT_BEFORE not in textbook:
        raise RuntimeError(f"Insertion anchor not found in textbook: {INSERT_BEFORE!r}")
    return textbook.replace(INSERT_BEFORE, digest + "\n" + INSERT_BEFORE, 1)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the textbook chapter is stale")
    parser.add_argument("--stdout", action="store_true", help="Print the generated chapter without editing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = validate_and_join()
    digest = render_digest(rows)
    if args.stdout:
        print(digest, end="")
        return 0

    textbook = TEXTBOOK_PATH.read_text(encoding="utf-8")
    updated = replace_generated_chapter(textbook, digest)
    if args.check:
        if updated != textbook:
            print(f"STALE: {TEXTBOOK_PATH.relative_to(ROOT)}")
            return 1
        print(f"OK: {TEXTBOOK_PATH.relative_to(ROOT)}")
        return 0

    changed = updated != textbook
    if changed:
        atomic_write(TEXTBOOK_PATH, updated)
    print(
        f"{'updated' if changed else 'unchanged'}: {TEXTBOOK_PATH.relative_to(ROOT)} "
        f"({len(rows)} papers)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
