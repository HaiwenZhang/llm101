#!/usr/bin/env python3
"""Generate chapter-level lecture and paper reading paths.

The component data is derived from the local course manifest and curriculum
mapping, so every link remains local and every recommendation is traceable to a
downloaded PDF.  The script also attaches one ChapterReadings component to the
end of every beginner lesson, without duplicating existing tags.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSES = ROOT / "output" / "course-materials" / "combined-manifest.json"
COVERAGE = ROOT / "output" / "course-corpus" / "curriculum-coverage.json"
PAPERS = ROOT / "output" / "papers" / "index.json"
BEGINNER = ROOT / "site" / "beginner"
OUTPUT = ROOT / "site" / ".vitepress" / "theme" / "data" / "chapter-readings.json"

COURSE_SHORT = {
    "cs224n-winter-2026": "Stanford CS224N",
    "cmu-anlp-spring-2026": "CMU Advanced NLP",
    "cs336-spring-2026": "Stanford CS336",
    "ntu-adl-fall-2025": "NTU Applied Deep Learning",
    "llm-systems-spring-2025": "LLM Systems",
    "cmu-llm-applications-spring-2026": "CMU LLM Applications",
    "berkeley-cs285-spring-2026": "Berkeley Deep RL",
}

COURSE_PRIORITY = {
    "cs224n-winter-2026": 0,
    "cmu-anlp-spring-2026": 1,
    "cs336-spring-2026": 2,
    "ntu-adl-fall-2025": 3,
    "berkeley-cs285-spring-2026": 4,
    "llm-systems-spring-2025": 5,
    "cmu-llm-applications-spring-2026": 6,
}

SUPPLEMENTAL_SESSIONS = {
    "06": [("cs224n-winter-2026", "L13"), ("cmu-anlp-spring-2026", "L09")],
    "14": [("cs224n-winter-2026", "L07"), ("cmu-anlp-spring-2026", "L06")],
    "15": [("cmu-anlp-spring-2026", "L04")],
    "21": [("cmu-anlp-spring-2026", "L10")],
    "32": [("cs224n-winter-2026", "L13"), ("cmu-anlp-spring-2026", "L09")],
    "42": [("cmu-anlp-spring-2026", "L16")],
    "43": [("cmu-anlp-spring-2026", "L16")],
    "44": [("cmu-anlp-spring-2026", "L16")],
    "47": [("cmu-anlp-spring-2026", "L18")],
    "08-post-training": [("cs224n-winter-2026", "L08"), ("cmu-anlp-spring-2026", "L17")],
    "09-k3-map": [("cs224n-winter-2026", "L07"), ("cmu-anlp-spring-2026", "L06")],
    "53-k3-capstone": [("cs224n-winter-2026", "L07"), ("cmu-anlp-spring-2026", "L17")],
}

KEYWORDS = {
    "00": "language model neural architecture resource foundation scaling",
    "01": "token byte bpe wordpiece sentencepiece tokenization",
    "02": "word embedding glove representation vector word2vec",
    "03": "backprop gradient neural optimization calculus training",
    "04": "attention self-attention bahdanau transformer",
    "05": "transformer attention normalization rmsnorm rope gqa",
    "06": "decoding speculative inference cache generation language model",
    "07": "expert moe switch routing mixture",
    "10": "language model rnn recurrent gpt llama autoregressive",
    "11": "decoding sampling beam generation meta-generation",
    "12": "evaluation benchmark mmlu helm judge",
    "13": "architecture transformer mamba recurrent state space",
    "14": "bert masked pretraining encoder contextual",
    "15": "encoder decoder translation sequence-to-sequence attention",
    "16": "gpt llama autoregressive recurrent state space mamba",
    "17": "in-context few-shot prompt gpt-3 instruction",
    "18": "chain-of-thought reasoning react prompt self-consistency",
    "19": "adapter parameter-efficient fine-tuning lora peft",
    "20": "lora low-rank qlora adapter fine-tuning",
    "21": "model editing memory knowledge parametric",
    "22": "retrieval rag knowledge memory",
    "23": "retrieval dense passage nearest embedding search",
    "24": "rag retrieval attributable scholar generation",
    "25": "scaling chinchilla compute data fineweb pretraining",
    "26": "optimization pytorch training gpu normalization",
    "27": "distributed parallel megatron zero communication training",
    "28": "instruction rlhf dpo preference fine-tuning alignment",
    "29": "distillation teacher student orca",
    "30": "quantization int8 bit qlora low precision",
    "31": "flashattention attention mamba sequence ring long context",
    "32": "speculative serving inference paged cache decoding",
    "33": "agent tool react web swe environment",
    "34": "vision image multimodal clip vit chameleon",
    "35": "application dialogue writing code personalization culture",
    "36": "evaluation benchmark judge error bar experimental",
    "37": "safety risk harm bias fairness social attack",
    "38": "serving deployment system inference cache monitoring",
    "39": "research experiment evaluation design error bar",
    "40": "reinforcement rlhf preference language model",
    "41": "reinforcement markov value policy ppo gae",
    "42": "policy gradient reinforce ppo reinforcement",
    "43": "actor critic gae ppo advantage",
    "44": "ppo gae proximal policy trpo importance",
    "45": "rlhf preference reward dpo instructgpt",
    "46": "reasoning r1 grpo verifiable ppo preference",
    "47": "agent web environment swe osworld exploration",
    "48": "rlhf preference system agent safety evaluation",
    "49": "reasoning chain-of-thought test-time verify speculative",
    "50": "multilingual cross-lingual tokenization xlm fairness",
    "51": "diffusion flow matching ddpm vae image generation",
    "52": "interpretability circuit mechanistic probe attribution",
}

CORE_FALLBACKS = {
    "04": ["kimi_linear"],
    "05": ["attention_residuals", "kimi_linear"],
    "06": ["deepseek_v2", "pagedattention_vllm"],
    "07": ["switch_transformers", "deepseek_moe"],
    "13": ["kimi_k3", "mamba2"],
    "16": ["mamba2", "kimi_linear"],
    "20": ["dpo"],
    "25": ["chinchilla", "kimi_k3"],
    "26": ["megatron_lm"],
    "27": ["megatron_lm", "zero"],
    "28": ["instructgpt", "dpo"],
    "29": ["mopd"],
    "30": ["deepseek_v3"],
    "31": ["flash_attention_2", "ring_attention", "yarn"],
    "32": ["pagedattention_vllm", "mooncake"],
    "33": ["react"],
    "34": ["kimi_vl"],
    "38": ["mooncake", "pagedattention_vllm"],
    "40": ["instructgpt"],
    "41": ["instructgpt"],
    "42": ["instructgpt"],
    "43": ["instructgpt"],
    "44": ["instructgpt", "deepseek_math"],
    "45": ["instructgpt", "dpo"],
    "46": ["deepseek_r1", "deepseek_math"],
    "47": ["react"],
    "48": ["instructgpt", "dpo"],
    "49": ["deepseek_r1"],
    "09-k3-map": ["kimi_k3"],
    "53-k3-capstone": ["kimi_k3"],
}

MUST_READ_PATTERNS = {
    "01": r"rare words with subword",
    "02": r"efficient estimation of word representations",
    "03": r"backprop|vectorization",
    "04": r"jointly learning to align|attention is all you need",
    "05": r"^attention is all you need$",
    "06": r"speculative decod|meta-generation",
    "07": r"switch transformer|mixture-of-experts",
    "10": r"neural probabilistic language model|recurrent neural network",
    "11": r"meta-generation|decod",
    "12": r"helm|mmlu",
    "14": r"^bert:|bidirectional transformers",
    "15": r"encoder-decoder|jointly learning to align",
    "16": r"mamba|state space|autoregressive",
    "17": r"few-shot learners|in-context",
    "18": r"chain-of-thought",
    "19": r"lora|adapter",
    "20": r"^lora|low-rank adaptation",
    "21": r"parametric.*non-parametric|model edit",
    "22": r"retrieval-augmented|retrieval survey",
    "23": r"dense passage|retriev",
    "24": r"self-rag|attributable|openscholar",
    "25": r"compute optimal|chinchilla",
    "27": r"megatron|zero",
    "28": r"follow instructions with human feedback|instructgpt",
    "29": r"distill",
    "30": r"llm\.int8|quantiz",
    "31": r"flashattention",
    "32": r"pagedattention|speculative decod",
    "33": r"^react:|webgpt|swe-agent",
    "34": r"^clip|vision transformer",
    "36": r"helm|error bars",
    "41": r"advantage estimation|ppo",
    "42": r"proximal policy optimization|policy gradient",
    "43": r"advantage estimation|gae",
    "44": r"proximal policy optimization",
    "45": r"follow instructions with human feedback|direct preference optimization",
    "46": r"deepseek-r1|verifiable",
    "47": r"swe-agent|webgpt|webshop|osworld",
    "48": r"follow instructions with human feedback|direct preference optimization",
    "49": r"deepseek-r1|let.s verify|test-time",
    "50": r"do all languages cost|xlm-r",
    "51": r"flow matching|denoising diffusion",
    "52": r"interpret|circuit",
}

# Keep especially relevant newly archived readings visible in the three-item
# chapter panel even when a broad course session contains many older papers.
PINNED_READING_PATTERNS = {
    "31": r"^flashattention 4\b",
}

QUESTIONS = {
    "00": "模型的参数、数据、目标函数和计算资源分别决定什么，哪些结论不能由参数量单独推出？",
    "01": "分词算法优化的目标是什么，它对序列长度、稀有词和不同语言造成了什么代价？",
    "02": "表示是怎样从共现信号学出来的，内在相似度又为什么不等于下游任务有效？",
    "03": "损失怎样沿计算图传回每张矩阵，作者用什么检查梯度和优化是否正确？",
    "04": "Query、Key、Value 分别承担什么角色，注意力权重改变时信息流怎样变化？",
    "05": "论文究竟改变了 Transformer Block 的哪一步，收益来自表达能力还是训练/系统效率？",
    "06": "训练时并行预测与推理时逐 token 生成有何不同，缓存和验证把成本转移到了哪里？",
    "07": "总参数、激活参数与通信量要怎样分开报告，路由失衡会怎样破坏收益？",
    "10": "每代语言模型解决了前代的哪个瓶颈，又引入了什么新的依赖或成本？",
    "11": "解码算法改变的是模型分布还是搜索过程，它怎样影响质量、多样性与延迟？",
    "12": "基准分数对应什么任务分布，污染、提示格式和统计不确定性会怎样改变结论？",
    "13": "架构差异改变的是状态、信息混合还是计算路径，比较时哪些预算必须保持一致？",
    "14": "双向 Masked LM 学到了什么表示，它为什么适合编码任务而不能直接因果生成？",
    "15": "Encoder、跨注意力与 Decoder 各自看见什么信息，训练和推理的可见性是否一致？",
    "16": "自回归、循环状态与显式 Attention 各自保存什么历史，长序列代价怎样不同？",
    "17": "模型从示例中临时学到了什么，效果究竟来自任务说明、示例内容还是示例顺序？",
    "18": "提示方法是在增加信息、改变搜索，还是只改变输出格式，失败时如何区分？",
    "19": "可训练参数减少后，表示瓶颈落在哪里，论文是否用同数据和同预算公平比较？",
    "20": "低秩分解限制了哪些更新方向，rank、目标层和缩放系数怎样影响容量？",
    "21": "一次知识更新是否同时满足有效性、泛化、局部性和持久性？",
    "22": "检索到底补充了哪类外部记忆，检索错、证据错和生成错怎样分开定位？",
    "23": "召回器、向量表示和重排器分别优化什么指标，离线召回怎样传导到最终答案？",
    "24": "模型是否忠实使用证据，引用正确、答案正确与证据充分为什么不是同一件事？",
    "25": "在固定 FLOPs 下参数量与训练 token 怎样配平，数据质量变化会怎样移动最优点？",
    "26": "理论 FLOPs、显存占用与真实吞吐为何不同，瓶颈在计算、内存还是数据管线？",
    "27": "模型状态、激活和计算分别怎样切分，哪一步通信会成为扩展瓶颈？",
    "28": "SFT、偏好学习和在线 RL 分别提供什么监督，策略分布何时发生变化？",
    "29": "教师传递的是答案、logits 还是当前策略轨迹，学生怎样避免只模仿表面输出？",
    "30": "权重、激活和 KV Cache 分别量化到什么精度，误差怎样被校准或吸收？",
    "31": "方法是否保持精确 Attention，它节省的是算术量、显存还是 HBM 访问？",
    "32": "吞吐与首 token/逐 token 延迟如何权衡，缓存、批处理和调度各解决什么问题？",
    "33": "Agent 的动作怎样改变环境状态，工具错误、规划错误和验证错误如何区分？",
    "34": "视觉信息在哪一步变成 token 或潜变量，模态对齐损失与生成损失各教会了什么？",
    "35": "应用价值来自底座模型、外部知识还是工作流约束，失败后应改模型还是改系统？",
    "36": "实验的比较对象、预算和置信区间是否成立，平均分是否掩盖了子群退化？",
    "37": "风险来自训练数据、模型行为还是部署场景，缓解措施是否只改变表面拒答率？",
    "38": "线上质量、延迟、成本和可恢复性如何共同定义成功，监控信号能否定位根因？",
    "39": "论文的因果主张由哪个消融支撑，替代解释是否被同预算实验排除？",
    "40": "状态、动作、奖励和策略在语言生成中分别是什么，终局奖励怎样传回早期 token？",
    "41": "价值函数估计的是哪段未来回报，bootstrap 带来的偏差和方差怎样权衡？",
    "42": "log-prob 梯度为什么能提高高回报动作概率，baseline 怎样降方差而不改期望？",
    "43": "Actor 与 Critic 分别学什么，GAE 的参数怎样在偏差和方差之间移动？",
    "44": "新旧策略比率为什么会失控，clip 或 trust region 实际限制了什么？",
    "45": "偏好标签怎样变成奖励或直接变成策略梯度，参考模型承担什么约束？",
    "46": "可验证奖励覆盖了哪些能力，组内相对优势和长轨迹采样会引入什么偏差？",
    "47": "离线轨迹与当前策略分布差多远，探索、信用分配和环境可靠性怎样影响 Agent？",
    "48": "rollout、训练器、奖励/验证器和版本管理如何保持同一策略语义？",
    "49": "测试时多花的计算用在生成、验证还是聚合，收益曲线什么时候开始饱和？",
    "50": "跨语言共享参数带来迁移还是干扰，分词效率和评测覆盖是否公平？",
    "51": "加噪、去噪与概率流怎样连接，训练目标和采样算法各自近似了什么？",
    "52": "方法发现的是相关特征还是因果机制，干预实验能否复现预测？",
    "09-k3-map": "K3 的架构、训练与系统组件分别解决哪个规模瓶颈，论文证据支持到什么程度？",
    "53-k3-capstone": "能否把 K3 的每个关键设计还原成问题、机制、资源账本、证据和适用边界？",
}


def normalized_title(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value.lower()).strip()


def target_for_slug(slug: str) -> str | None:
    if slug == "08-post-training":
        return "28"
    if slug in {"09-k3-map", "53-k3-capstone"}:
        return None
    prefix = slug[:2]
    return prefix if prefix.isdigit() else None


def extract_title(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^title:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else path.stem


def score_reading(item: dict, key: str, index: int) -> int:
    title = normalized_title(item["label"])
    words = KEYWORDS.get(key, "").split()
    score = 80 if item["course_id"] in {"cs224n-winter-2026", "cmu-anlp-spring-2026"} else 20
    score += max(0, 12 - index)
    score += sum(18 for word in words if word in title)
    if "survey" in title or "chapter" in title:
        score += 3
    return score


def available_materials(session: dict) -> list[dict]:
    return [
        item
        for item in session.get("materials", [])
        if item.get("status") in {"available", "interactive"}
        and (item.get("public_url") or item.get("pdf_url") or item.get("url"))
    ]


def public_resource_url(item: dict) -> str:
    if item.get("status") == "interactive":
        return item["public_url"]
    return item.get("pdf_url") or item["url"]


def main() -> None:
    course_data = json.loads(COURSES.read_text(encoding="utf-8"))
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    paper_data = json.loads(PAPERS.read_text(encoding="utf-8"))
    session_index = {
        (course["id"], session["id"]): {**session, "course_id": course["id"]}
        for course in course_data["courses"]
        for session in course["sessions"]
    }
    sessions_by_target: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    target_titles: dict[str, str] = {}
    for row in coverage["sessions"]:
        for target in row["targets"]:
            sessions_by_target[target["id"]].append((row["course_id"], row["session_id"]))
            target_titles[target["id"]] = target["title"]

    papers_by_slug = {paper["slug"]: paper for paper in paper_data["papers"]}
    lesson_files = sorted(path for path in BEGINNER.glob("[0-9][0-9]-*.md"))
    output: dict[str, dict] = {}

    for lesson_path in lesson_files:
        slug = lesson_path.stem
        target = target_for_slug(slug)
        key = slug if slug in {"09-k3-map", "53-k3-capstone"} else (target or slug)
        session_keys = list(sessions_by_target.get(target or "", []))
        session_keys += SUPPLEMENTAL_SESSIONS.get(key, [])
        session_keys += SUPPLEMENTAL_SESSIONS.get(slug, [])
        session_keys = list(dict.fromkeys(session_keys))
        session_keys.sort(key=lambda item: (COURSE_PRIORITY.get(item[0], 99), item[1]))

        lectures = []
        for course_id, session_id in session_keys:
            session = session_index.get((course_id, session_id))
            if not session:
                continue
            materials = sorted(
                available_materials(session),
                key=lambda item: (0 if item.get("kind") == "slides" else 1, item.get("filename", "")),
            )
            for material in materials:
                lectures.append(
                    {
                        "course": COURSE_SHORT.get(course_id, course_id),
                        "session": session_id,
                        "title": session.get("title_zh") or session["title"],
                        "label": material["label"],
                        "url": public_resource_url(material),
                        "pages": material.get("pages") or 0,
                    }
                )
                break
            if len(lectures) >= 3:
                break

        reading_pool = []
        for course_id, session_id in session_keys:
            session = session_index.get((course_id, session_id))
            if not session:
                continue
            for index, reading in enumerate(session.get("readings", [])):
                if reading.get("status") != "available":
                    continue
                reading_pool.append(
                    {
                        "title": reading["label"],
                        "url": public_resource_url(reading),
                        "pages": reading.get("pages") or 0,
                        "source": f"{COURSE_SHORT.get(course_id, course_id)} {session_id}",
                        "course_id": course_id,
                        "score": score_reading({**reading, "course_id": course_id}, key, index),
                    }
                )

        deduped = {}
        for reading in sorted(reading_pool, key=lambda item: (-item["score"], item["title"])):
            deduped.setdefault(normalized_title(reading["title"]), reading)
        ranked_readings = list(deduped.values())
        selected = ranked_readings[:3]

        fallback_keys = list(CORE_FALLBACKS.get(key, []))
        fallback_keys += list(CORE_FALLBACKS.get(target or "", []))
        if slug in {"09-k3-map", "53-k3-capstone"}:
            fallback_keys.insert(0, "kimi_k3")
        for paper_slug in dict.fromkeys(fallback_keys):
            if len(selected) >= 3:
                break
            paper = papers_by_slug.get(paper_slug)
            if not paper or any(normalized_title(item["title"]) == normalized_title(paper["title"]) for item in selected):
                continue
            selected.append(
                {
                    "title": paper["title"],
                    "url": f"/{paper['pdf']}",
                    "guide_url": f"/papers/{paper['slug']}",
                    "pages": paper["pages"],
                    "source": "核心论文库",
                    "course_id": "core-papers",
                    "score": 0,
                }
            )

        # A small number of course topics (currently the social-risk lecture)
        # publish a substantial slide deck but no downloadable assigned paper.
        # Keep the recommendation honest by labeling that deck as a course
        # reading instead of inventing a paper citation or linking online-only
        # material that is absent from the local archive.
        if not selected and lectures:
            lecture = lectures[0]
            selected.append(
                {
                    "title": f"{lecture['title']}（课程专题讲义）",
                    "url": lecture["url"],
                    "pages": lecture["pages"],
                    "source": f"{lecture['course']} {lecture['session']}",
                    "course_id": "course-material",
                    "score": 0,
                }
            )

        if slug in {"09-k3-map", "53-k3-capstone"}:
            selected = []
            for paper_slug in ("kimi_k3", "kimi_linear", "latent_moe"):
                paper = papers_by_slug[paper_slug]
                selected.append(
                    {
                        "title": paper["title"],
                        "url": f"/{paper['pdf']}",
                        "guide_url": f"/papers/{paper['slug']}",
                        "pages": paper["pages"],
                        "source": "核心论文库",
                        "course_id": "core-papers",
                        "score": 10_000,
                    }
                )

        must_pattern = MUST_READ_PATTERNS.get(key)
        if must_pattern:
            candidates = ranked_readings + selected
            must = next(
                (item for item in candidates if re.search(must_pattern, normalized_title(item["title"]), flags=re.IGNORECASE)),
                None,
            )
            if must:
                selected = [must] + [
                    item
                    for item in selected
                    if normalized_title(item["title"]) != normalized_title(must["title"])
                ][:2]

        pinned_pattern = PINNED_READING_PATTERNS.get(key)
        if pinned_pattern:
            pinned = next(
                (
                    item
                    for item in ranked_readings
                    if re.search(pinned_pattern, normalized_title(item["title"]), flags=re.IGNORECASE)
                ),
                None,
            )
            if pinned and not any(
                normalized_title(item["title"]) == normalized_title(pinned["title"])
                for item in selected
            ):
                selected = selected[:2] + [pinned]

        question = QUESTIONS.get(
            key,
            f"它怎样改变“{target_titles.get(target or '', extract_title(lesson_path))}”中的关键步骤，证据与代价分别是什么？",
        )
        papers = []
        for index, reading in enumerate(selected):
            papers.append(
                {
                    "level": "必读" if index == 0 else "选读",
                    "role": "建立核心机制" if index == 0 else ("比较另一条路线" if index == 1 else "补证据与边界"),
                    "title": reading["title"],
                    "url": reading["url"],
                    "guide_url": reading.get("guide_url", ""),
                    "pages": reading["pages"],
                    "source": reading["source"],
                    "question": question,
                }
            )

        output[slug] = {
            "lesson_title": extract_title(lesson_path),
            "target_id": target,
            "reading_question": question,
            "lectures": lectures,
            "papers": papers,
        }

        text = lesson_path.read_text(encoding="utf-8").rstrip()
        if "<ChapterReadings" not in text:
            text += f"\n\n<ChapterReadings lesson=\"{slug}\" />\n"
            lesson_path.write_text(text, encoding="utf-8")

    missing = [slug for slug, chapter in output.items() if not chapter["papers"]]
    if missing:
        raise RuntimeError(f"Lessons without paper recommendations: {missing}")
    if len(output) != 54:
        raise RuntimeError(f"Expected 54 beginner lessons, generated {len(output)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "lessons": len(output),
                "with_lectures": sum(bool(item["lectures"]) for item in output.values()),
                "with_papers": sum(bool(item["papers"]) for item in output.values()),
                "must_read_papers": sum(bool(item["papers"]) for item in output.values()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
