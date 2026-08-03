#!/usr/bin/env python3
"""Download and verify the public Berkeley CS 185/285 Spring 2026 PDFs.

The course page is the source of truth.  This script downloads every public
lecture slide, section, homework, and project PDF linked by the page, verifies
that it is a readable PDF, and emits the same manifest shape used by the other
course archives in this repository.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COURSE_URL = "https://rail.eecs.berkeley.edu/deeprlcourse/"
PUBLIC_ROOT = ROOT / "site" / "public" / "course-materials" / "berkeley-cs285-spring-2026"
MANIFEST = ROOT / "output" / "course-materials" / "berkeley-deeprl-manifest.json"


LECTURES = [
    ("Introduction", "导论：强化学习解决什么问题", "先建立智能体、环境、观测、动作和奖励的总地图，知道 RL 与监督学习的边界。"),
    ("Behavioral Cloning", "行为克隆（一）：从专家示范学习", "把模仿学习看成状态到动作的监督学习，并识别分布偏移为什么会让小错误滚成大错误。"),
    ("Behavioral Cloning Part 2", "行为克隆（二）：数据聚合与模仿学习", "用 DAgger 等方法修复训练分布与执行分布不一致，为离线数据和在线交互建立直觉。"),
    ("RL Basics", "强化学习基础：MDP、回报与价值", "理解 MDP、轨迹概率、折扣回报、价值函数与 Q 函数；这是后续 LLM RL 的共同语言。"),
    ("Policy Gradients", "策略梯度：直接提高好动作的概率", "推导 REINFORCE，理解 reward-to-go、baseline 与高方差问题。"),
    ("Actor Critic", "Actor-Critic：用价值网络指导策略", "让 Actor 负责行动、Critic 负责估值，并用 TD 误差降低策略梯度方差。"),
    ("Value-Based RL", "价值型强化学习", "理解贝尔曼方程、动态规划与 Q-learning，学会区分价值型和策略型方法。"),
    ("Q-learning in Practice", "Q-learning 实践", "理解经验回放、目标网络、过估计与离线数据分布偏移。"),
    ("Advanced Policy Gradients Part 1", "高级策略梯度（一）：重要性采样", "用概率比率复用旧策略数据，连接 on-policy、off-policy 与 PPO。"),
    ("Advanced Policy Gradients Part 2", "高级策略梯度（二）：约束更新幅度", "理解自然梯度、TRPO、PPO 与 KL 约束为什么能让训练更稳定。"),
    ("Variational Inference", "变分推断基础", "建立概率推断、ELBO 与潜变量直觉，为控制即推断和最大熵 RL 铺路。"),
    ("VI in RL", "强化学习中的变分推断", "把策略优化写成概率推断问题，理解熵奖励和随机策略的意义。"),
    ("Control as Inference", "控制即推断", "把高奖励轨迹视为更可能的轨迹，连接最大熵 RL、偏好建模和语言模型采样。"),
    ("LLM RL", "大语言模型强化学习", "把 prompt、token、回答和奖励正式放进 MDP，学习 LLM 策略梯度、PPO 与验证式奖励。"),
    ("Model-Based RL Part 1", "模型式强化学习（一）", "学习环境动力学模型、规划和模型误差，理解世界模型怎样帮助智能体。"),
    ("Model-Based RL Part 2", "模型式强化学习（二）", "进一步理解模型预测控制、数据收集与规划—学习闭环。"),
    ("Offline RL Part 1", "离线强化学习（一）", "只用固定数据集学习策略，理解分布外动作和保守估值。"),
    ("Offline RL Part 2", "离线强化学习（二）", "比较行为约束、保守 Q 学习和离线到在线微调。"),
    ("Exploration", "探索：怎样发现更好的行为", "理解随机探索、内在奖励与信息增益，以及语言智能体如何避免只重复已知解法。"),
    ("RL Theory", "强化学习理论", "用样本复杂度、性能差距与误差传播理解算法能保证什么、不能保证什么。"),
    ("Midterm Review Part 1", "期中复习（一）", "把 MDP、价值函数、策略梯度和 Actor-Critic 串成一张知识图。"),
    ("Midterm Review Part 2", "期中复习（二）", "复盘变分推断、PPO、模型式与离线 RL 的联系。"),
    ("Advanced Exploration", "高级探索", "学习乐观估计、后验采样与基于模型的探索。"),
    ("Multi-task RL", "多任务强化学习", "理解条件策略、迁移、元学习和多任务数据怎样改善泛化。"),
    ("Challenges and Open Problems", "挑战与开放问题", "识别奖励设计、稳定性、泛化、长期信用分配和真实世界安全等未解难题。"),
]


ATTACHMENTS = {
    2: [
        ("/deeprlcourse/static/homeworks/hw1.pdf", "Homework 1: Imitation Learning", "supplement"),
        ("/deeprlcourse/static/sections/section-1.pdf", "Section 1: Imitation Learning", "recitation"),
    ],
    4: [
        ("/deeprlcourse/static/sections/section-2-1.pdf", "Section 2.1: RL Basics", "recitation"),
        ("/deeprlcourse/static/sections/section-2-2.pdf", "Section 2.2: PyTorch / RL Practice", "recitation"),
    ],
    5: [
        ("/deeprlcourse/static/homeworks/hw2.pdf", "Homework 2: Policy Gradients", "supplement"),
        ("/deeprlcourse/static/sections/section-3.pdf", "Section 3: Policy Gradients and Actor Critic", "recitation"),
    ],
    7: [
        ("/deeprlcourse/static/homeworks/hw3.pdf", "Homework 3: Q-Learning and Actor Critic", "supplement"),
        ("/deeprlcourse/static/sections/section-4.pdf", "Section 4: Value-Based RL", "recitation"),
    ],
    9: [("/deeprlcourse/static/sections/section-5.pdf", "Section 5: Advanced Policy Gradients", "recitation")],
    11: [("/deeprlcourse/static/sections/section-6.pdf", "Section 6: Variational Inference", "recitation")],
    14: [
        ("/deeprlcourse/static/homeworks/hw4.pdf", "Homework 4: LLM RL", "supplement"),
        ("/deeprlcourse/static/sections/section-7.pdf", "Section 7: IRL and LLM RL", "recitation"),
        ("/deeprlcourse/static/misc/llm_rl_default_final_project.pdf", "Default Final Project: LLM RL", "supplement"),
    ],
    15: [("/deeprlcourse/static/sections/section-8.pdf", "Section 8: Model-Based RL", "recitation")],
    17: [
        ("/deeprlcourse/static/homeworks/hw5.pdf", "Homework 5: Offline RL", "supplement"),
        ("/deeprlcourse/static/sections/section-9.pdf", "Section 9: Offline RL", "recitation"),
        ("/deeprlcourse/static/misc/offline_to_online_rl_default_final_project.pdf", "Default Final Project: Offline-to-Online RL", "supplement"),
    ],
    25: [
        ("/deeprlcourse/static/homeworks/project_assignment.pdf", "Course Project Assignment", "supplement"),
        ("/deeprlcourse/static/misc/final_project_outline.pdf", "Final Project Outline", "supplement"),
    ],
}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "kimi3-paper-course-archiver/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def download_item(path: str, label: str, kind: str, folder: str) -> dict:
    url = urllib.parse.urljoin(COURSE_URL, path)
    filename = Path(urllib.parse.urlparse(path).path).name
    destination = PUBLIC_ROOT / folder / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size < 1000:
        try:
            payload = fetch(url)
        except urllib.error.HTTPError as error:
            return {
                "kind": kind,
                "label": label,
                "url": url,
                "status": "online-only",
                "note": f"课程主页保留此链接，但抓取时返回 HTTP {error.code}",
            }
        if not payload.startswith(b"%PDF"):
            raise ValueError(f"Not a PDF: {url}")
        destination.write_bytes(payload)
        time.sleep(0.08)
    payload = destination.read_bytes()
    if not payload.startswith(b"%PDF"):
        raise ValueError(f"Invalid PDF: {destination}")
    info = subprocess.run(
        ["pdfinfo", str(destination)], capture_output=True, text=True, check=True
    ).stdout
    match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not read page count: {destination}")
    pages = int(match.group(1))
    return {
        "kind": kind,
        "label": label,
        "url": url,
        "pdf_url": url,
        "status": "available",
        "public_url": "/" + str(destination.relative_to(ROOT / "site" / "public")),
        "pages": pages,
        "bytes": len(payload),
        "size_mb": round(len(payload) / 1024 / 1024, 2),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    html = fetch(COURSE_URL).decode("utf-8", errors="replace")
    linked_pdfs = set(re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, flags=re.I))
    sessions = []
    archived_urls = set()
    for number, (title, title_zh, summary) in enumerate(LECTURES, start=1):
        slide_path = f"/deeprlcourse/static/slides/lec-{number}.pdf"
        if slide_path not in linked_pdfs:
            raise RuntimeError(f"Official course page no longer links {slide_path}")
        materials = [download_item(slide_path, f"Lecture {number}: {title}", "slides", "slides")]
        archived_urls.add(slide_path)
        for path, label, kind in ATTACHMENTS.get(number, []):
            if path not in linked_pdfs:
                raise RuntimeError(f"Official course page no longer links {path}")
            folder = "sections" if "/sections/" in path else "homeworks" if "/homeworks/" in path else "projects"
            materials.append(download_item(path, label, kind, folder))
            archived_urls.add(path)
        sessions.append(
            {
                "id": f"L{number:02d}",
                "date": "Spring 2026",
                "title": title,
                "title_zh": title_zh,
                "summary": summary,
                "materials": materials,
                "readings": [],
                "extras": [],
            }
        )

    official_pdfs = {
        path for path in linked_pdfs
        if path.startswith("/deeprlcourse/static/") and any(
            marker in path for marker in ("/slides/", "/sections/", "/homeworks/", "/misc/")
        )
    }
    skipped = sorted(official_pdfs - archived_urls)
    if skipped:
        raise RuntimeError(f"Unclassified official PDF link(s): {skipped}")

    course = {
        "id": "berkeley-cs285-spring-2026",
        "page": "berkeley-deeprl-2026",
        "title": "UC Berkeley CS 185/285: Deep Reinforcement Learning (Spring 2026)",
        "school": "University of California, Berkeley",
        "official": COURSE_URL,
        "description": "从模仿学习、策略梯度和 Actor-Critic，到 LLM RL、模型式 RL、离线 RL 与探索的系统强化学习课程",
        "sessions": sessions,
    }
    files = [
        item for session in sessions for item in session["materials"]
        if item.get("status") == "available"
    ]
    payload = {
        "generated_at": "2026-08-03",
        "source": COURSE_URL,
        "verified_with": "Poppler pdfinfo",
        "course_count": 1,
        "file_count": len(files),
        "page_count": sum(item["pages"] for item in files),
        "courses": [course],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("course_count", "file_count", "page_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
