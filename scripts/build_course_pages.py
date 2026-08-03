#!/usr/bin/env python3
"""Build VitePress course archive pages from the verified PDF manifests."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = [
    ROOT / "output" / "course-materials" / "manifest.json",
    ROOT / "output" / "course-materials" / "additional-manifest.json",
    ROOT / "output" / "course-materials" / "berkeley-deeprl-manifest.json",
    ROOT / "output" / "course-materials" / "cs336-manifest.json",
]
SITE_DIR = ROOT / "site" / "courses"
PUBLIC_MANIFEST = ROOT / "site" / "public" / "course-materials" / "manifest.json"
COMBINED_MANIFEST = ROOT / "output" / "course-materials" / "combined-manifest.json"


COURSE_META = {
    "cs224n-winter-2026": {
        "page": "cs224n-2026",
        "title_zh": "Stanford CS224N 逐讲资料",
        "intro": "系统的深度学习 NLP 主线：从词向量、RNN、Transformer，一直走到后训练、RAG、Agent、评测与推理。",
        "fit": "适合作为整个学习路线的主干课程。",
    },
    "ntu-adl-fall-2025": {
        "page": "ntu-adl-2025",
        "title_zh": "台湾大学 ADL 逐讲资料",
        "intro": "课件短、图多，并配有 Recitation；覆盖 Transformer、Prompt、LoRA、RAG、MoE、生成评价与 Agent。",
        "fit": "适合快速建立概念结构，并把知识接到项目实践。",
    },
    "cmu-anlp-spring-2026": {
        "page": "cmu-anlp-2026",
        "title_zh": "CMU Advanced NLP 逐讲资料",
        "intro": "研究导向的高级 NLP 课程，按讲次同时整理 Slides、核心论文、补充阅读与公开代码。",
        "fit": "适合在掌握 Transformer 基础后，继续深入架构、训练、推理、评测和前沿研究。",
    },
    "llm-systems-spring-2025": {
        "page": "llm-systems-2025",
        "title_zh": "LLM Systems 逐讲资料",
        "intro": "从 GPU 与分布式训练出发，系统学习并行、量化、MoE、推理优化、PagedAttention 与在线服务。",
        "fit": "适合补齐“模型为什么能高效训练和部署”的工程视角。",
    },
    "cmu-llm-applications-spring-2026": {
        "page": "cmu-llm-applications-2026",
        "title_zh": "CMU LLM Applications 逐讲资料",
        "intro": "围绕检索、Agent、教育、医疗、法律、代码与产品等应用主题组织，强调从模型能力到真实场景。",
        "fit": "适合建立应用地图，按自己的方向选择专题学习。",
    },
    "berkeley-cs285-spring-2026": {
        "page": "berkeley-deeprl-2026",
        "title_zh": "Berkeley Deep RL 逐讲资料",
        "intro": "从模仿学习、MDP、策略梯度和 Actor-Critic，一直讲到 LLM RL、模型式 RL、离线 RL、探索与开放问题。",
        "fit": "适合补齐大语言模型强化学习真正需要的算法基础，并把 PPO、验证式奖励和 Agent 训练放回完整 RL 框架。",
    },
    "cs336-spring-2026": {
        "page": "cs336-2026",
        "title_zh": "Stanford CS336 逐讲资料",
        "intro": "从零实现语言模型的工程主线：Tokenizer、Transformer、PyTorch、GPU、并行、Scaling、推理、数据与后训练。",
        "fit": "适合把公式变成可运行实现，并养成计算、显存和通信三本账的习惯。",
    },
}


KIND_NAMES = {
    "slides": "Slides",
    "notes": "随讲 Notes",
    "supplement": "补充讲义",
    "recitation": "Recitation Slides",
    "reading": "论文阅读",
}


def md_text(value: str) -> str:
    return value.replace("[", "\\[").replace("]", "\\]")


def format_size(item: dict) -> str:
    return f"{item.get('pages', '?')} 页 · {item.get('size_mb', '?')} MB"


def item_line(item: dict, reading: bool = False) -> str:
    label = md_text(item["label"])
    note = item.get("note")
    if item.get("status") == "interactive":
        line = (
            f"- **可执行 Slides** · [{label}（本地互动讲义）]({item['public_url']})"
            f" · [官方课程]({item['url']})"
        )
    elif item.get("status") == "available":
        kind = KIND_NAMES.get(item.get("kind"), item.get("kind", "PDF"))
        official_pdf = item.get("pdf_url") or item["url"]
        line = (
            f"- **{kind}** · [{label}（官方 PDF）]({official_pdf})"
            f" · {format_size(item)}"
        )
        if official_pdf != item["url"]:
            line += f" · [官方来源页]({item['url']})"
    else:
        prefix = "论文阅读" if reading else KIND_NAMES.get(item.get("kind"), "资料")
        state = "官网链接失效" if note and "HTTP" in note else "仅在线"
        reason = "课程主页保留链接，但当前无法下载" if state == "官网链接失效" else "课程页没有公开直链 PDF"
        line = f"- **{prefix} · {state}** · [{label}]({item['url']}) — {reason}"
    if note:
        line += f"；{note}"
    return line


def course_stats(course: dict) -> dict:
    materials = [item for session in course["sessions"] for item in session.get("materials", [])]
    readings = [item for session in course["sessions"] for item in session.get("readings", [])]
    available = [
        item for item in materials + readings
        if item.get("status") in {"available", "interactive"}
    ]
    pdf_items = [item for item in materials + readings if item.get("status") == "available"]
    return {
        "sessions": len(course["sessions"]),
        "materials": len(materials),
        "readings": len(readings),
        "papers": sum(1 for item in readings if item.get("status") == "available"),
        "online": sum(1 for item in readings if item.get("status") == "online-only"),
        "extras": sum(len(session.get("extras", [])) for session in course["sessions"]),
        "files": len(available),
        "interactive": sum(1 for item in materials if item.get("status") == "interactive"),
        "pages": sum(item.get("pages", 0) for item in pdf_items),
        "size_mb": round(sum(item.get("bytes", 0) for item in pdf_items) / 1024 / 1024, 1),
    }


def render_course(course: dict) -> str:
    stats = course_stats(course)
    is_cs = course["id"].startswith("cs224n")
    is_ntu = course["id"].startswith("ntu-adl")
    is_berkeley = course["id"].startswith("berkeley-cs285")
    is_cs336 = course["id"].startswith("cs336")
    meta = COURSE_META[course["id"]]
    title_zh = meta["title_zh"]
    lines = [
        "---",
        f"title: {title_zh}",
        f"description: {course['description']} 的逐讲 Slides、讲义与论文阅读官方索引",
        "---",
        "",
        f"# {title_zh}",
        "",
        f"> **课程**：{course['title']}  ",
        f"> **学校**：{course['school']}  ",
        f"> **官方主页**：[{course['official']}]({course['official']})  ",
        "> **抓取与校验日期**：2026-08-03",
        "",
        "::: tip 这是来源与深挖页，不是主学习顺序",
        (
            f"本页共索引 **{stats['interactive']} 份本站可执行 Slides** 与 "
            f"**{stats['files'] - stats['interactive']} 份官方 PDF**，PDF 合计 **{stats['pages']:,} 页 / {stats['size_mb']} MB**。"
            if is_cs336 else
            f"本页共索引 **{stats['files']} 份官方 Slides / PDF**，合计 **{stats['pages']:,} 页 / {stats['size_mb']} MB**。PDF 统一链接到课程官网、论文官网或 arXiv。"
        ),
        ":::",
        "",
        "初学请先沿[大模型系统课](/beginner/)学习；需要核对某一知识点来自哪些课程时，使用[名校课程知识覆盖表](/curriculum/sources)。",
        "",
    ]
    if is_cs336:
        lines += [
            "CS336 的 9 个可执行讲次保留代码、逐步执行状态和原始图像，8 个传统 Slides 讲次保留 PDF。本站把这些材料用于补足 PyTorch 实现、资源核算和系统工程，不要求初学者另学一遍完整课程。",
            "",
        ]
    elif is_cs:
        lines += [
            "Stanford 课程表中的讲次 Slides、随讲 Notes、课程服务器上的补充讲义，以及逐讲 Suggested / Additional Readings 已分开整理。",
            f"论文阅读共列出 **{stats['readings']} 项**：其中 **{stats['papers']} 份有公开 PDF**，另有 **{stats['online']} 项**只有网页、博客、项目页或需订阅入口，因此保留在线链接并明确标注。",
            "",
            "::: warning 两个需要注意的官网状态",
            "第 15 讲“Interpretability”的 Slides 链接实际指向 2024 年旧课件；第 17、18 讲课程表目前没有公开 Slides。这里忠实保留官网状态，不把缺失文件伪装成已下载。",
            ":::",
            "",
        ]
    elif is_ntu:
        lines += [
            "台大页面里的主讲 Slides 与 Recitation Slides 均按上课日期归档。课程主页上 5 个已失效的 Recitation / 旧版补充链接，改用台大同一课程往年官方存档，并在具体条目旁注明来源年份。",
            "",
        ]
    elif is_berkeley:
        lines += [
            "Berkeley 25 讲主线从模仿学习与 MDP 开始，逐步进入策略梯度、Actor-Critic、PPO、LLM RL、模型式与离线 RL。讨论课、作业和两个默认项目已挂到最相关的讲次下。",
            f"共索引 **{stats['files']} 份 PDF / {stats['pages']:,} 页**。课程主页列出的 `Course Project Assignment` 当前返回 404，来源页会保留原链接和失败状态，不用其他文件冒充。",
            "",
            "::: tip 面向大语言模型的学习顺序",
            "不必先学完整机器人控制课程。初学者可按本站 [25–33 强化学习专题](/beginner/40-rl-language-model)学习；需要推导或原始例子时再回到本页对应 Slides。",
            ":::",
            "",
        ]
    else:
        lines += [
            meta["intro"],
            f"课程表共整理 **{stats['materials']} 份讲义条目**与 **{stats['readings']} 项论文 / 延伸阅读**；其中 **{stats['papers']} 份阅读有公开 PDF**，另有 **{stats['online']} 项**只有网页、博客、视频或受限入口，因此保留官方在线链接。",
            "",
        ]
    lines += [
        "## 建议怎么学",
        "",
        "1. 先读每一讲的“本讲抓什么”，确认自己要回答的问题。",
        "2. 第一遍快速翻 Slides，只看标题、图和结论；不在公式处停太久。",
        "3. 第二遍结合 Notes 或 Recitation，把不懂的概念补齐。",
        "4. 最后从论文阅读里挑 1–2 篇精读，并回到本站对应的零基础章节复习。",
        "",
        "[查看课程资料机器可读清单（JSON）](/course-materials/manifest.json)",
        "",
        "---",
        "",
    ]

    for session in course["sessions"]:
        lines += [
            f"## {session['id']} · {session['title_zh']}",
            "",
            f"**日期**：{session['date']}  ",
            f"**英文主题**：{session['title']}",
            "",
            f"**本讲抓什么**：{session['summary']}",
            "",
        ]
        if session.get("warning"):
            lines += [f"::: warning {session['warning']}", ":::", ""]

        materials = session.get("materials", [])
        lines += ["### Slides 与讲义", ""]
        if materials:
            lines.extend(item_line(item) for item in materials)
        else:
            lines.append("- 官网课程表目前没有公开 Slides / Notes 文件。")
        lines.append("")

        readings = session.get("readings", [])
        if readings:
            lines += ["### 论文与延伸阅读", ""]
            lines.extend(item_line(item, reading=True) for item in readings)
            lines.append("")

        extras = session.get("extras", [])
        if extras:
            lines += ["### 代码与其他资源", ""]
            lines.extend(f"- [{md_text(item['label'])}]({item['url']})" for item in extras)
            lines.append("")
        lines += ["---", ""]

    lines += [
        "## 版权与更新说明",
        "",
        "本站不随 GitHub Pages 重新分发课程 PDF；条目优先链接课程官网、出版方或 arXiv。版权归原作者、课程团队及出版方所有；引用时请使用 PDF 内的正式作者与出版信息。机器清单保留官方 URL、页数、大小与 SHA-256，方便后续复核。",
        "",
    ]
    return "\n".join(lines)


def render_index(courses: list[dict]) -> str:
    stats = {course["id"]: course_stats(course) for course in courses}
    total_files = sum(item["files"] for item in stats.values())
    total_pages = sum(item["pages"] for item in stats.values())
    total_mb = round(sum(item["size_mb"] for item in stats.values()), 1)
    lines = [f"""---
title: 名校课程原始资料与溯源
description: Stanford、CMU、Berkeley 与台湾大学七门课程的官方 PDF、可执行讲义、出处和逐讲索引
---

# 名校课程原始资料与溯源

::: warning 这里不是主教程入口
这些文件用于核对原始讲义、论文和课程出处。请先学习[大模型系统课](/beginner/)；想了解资料怎样被拆解并合入主线，请看[逐讲知识覆盖表](/curriculum/sources)。
:::

资料按“**课程 → 讲次 → Slides / Notes / 论文阅读**”索引。当前共收录 **{total_files} 份资料**（包括官方 PDF 与本站可执行 Slides），其中 PDF 合计 **{total_pages:,} 页、约 {total_mb} MB**。

## 七门来源课程
"""]
    for course in courses:
        item = stats[course["id"]]
        meta = COURSE_META[course["id"]]
        lines += [
            f"### [{course['title']}](/courses/{meta['page']})",
            "",
            (
                f"{meta['intro']} 本页含 **{item['interactive']} 份本站可执行 Slides**与 "
                f"**{item['files'] - item['interactive']} 份 PDF / {item['pages']:,} 页**。"
                if item['interactive'] else
                f"{meta['intro']} 本页含 **{item['files']} 份官方 PDF / {item['pages']:,} 页**；论文阅读中有 **{item['papers']} 份公开 PDF**、**{item['online']} 项仅在线链接。"
            ),
            "",
            meta["fit"],
            "",
        ]
    lines.append("""## 推荐组合方法

1. 先按本站[零基础系统课](/beginner/)建立中文直觉。
2. 同主题看台大 ADL Slides，快速确认概念结构。
3. 再沿 CS224N 主线学习，用 CMU Advanced NLP 把研究脉络补深。
4. 想做训练与部署时转到 LLM Systems；想做产品与场景时转到 LLM Applications。
5. 每一讲先读 Slides，再挑 1–2 篇论文精读；不必从第一天就把全部论文读完。

::: info PDF 与部署说明
课程 PDF 只保存在开发者本地的 `resources/` 资料库中，并由 Git 忽略。公开网站直接链接课程官网、出版方或 arXiv，仓库只保存教程、图片、交互实验和资料清单。
:::

[打开完整 JSON 清单](/course-materials/manifest.json)
""")
    return "\n".join(lines)


def main() -> None:
    courses = []
    for manifest_path in MANIFESTS:
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            courses.extend(data["courses"])

    combined = {
        "generated_at": "2026-08-03",
        "scope": "Official lecture slides, executable notes, course supplements, and schedule readings for seven public NLP, LLM, and deep reinforcement learning courses.",
        "courses": courses,
    }
    combined_text = json.dumps(combined, ensure_ascii=False, indent=2) + "\n"
    COMBINED_MANIFEST.write_text(combined_text, encoding="utf-8")
    PUBLIC_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    public_combined = json.loads(combined_text)
    for course in public_combined["courses"]:
        for session in course["sessions"]:
            for item in session.get("materials", []) + session.get("readings", []):
                if item.get("status") == "available":
                    item["public_url"] = item.get("pdf_url") or item["url"]
                item.pop("local_path", None)
    PUBLIC_MANIFEST.write_text(
        json.dumps(public_combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "index.md").write_text(render_index(courses), encoding="utf-8")
    for course in courses:
        page = COURSE_META[course["id"]]["page"]
        (SITE_DIR / f"{page}.md").write_text(render_course(course), encoding="utf-8")
    print(f"Wrote {len(courses)} course pages to {SITE_DIR}")


if __name__ == "__main__":
    main()
