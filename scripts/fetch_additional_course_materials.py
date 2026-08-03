#!/usr/bin/env python3
"""Archive three additional public LLM/NLP courses and their readings."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

from lxml import html

from fetch_course_materials import pdf_metadata, safe_segment, valid_pdf


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / "site" / "public" / "course-materials"
OUTPUT_ROOT = ROOT / "output" / "course-materials"
MANIFEST_PATH = OUTPUT_ROOT / "additional-manifest.json"


SOURCES = {
    "cmu-anlp-spring-2026": "https://cmu-l3.github.io/anlp-spring2026/",
    "llm-systems-spring-2025": "https://llmsystem.github.io/llmsystem2025spring/docs/Syllabus/",
    "cmu-llm-applications-spring-2026": "https://cmu-llms.org/schedule/",
}


TITLE_ZH = {
    # CMU Advanced NLP
    "Introduction & Fundamentals": "导论与高级 NLP 基础",
    "Fundamentals: Learned Representations": "学习表示：文字如何进入模型",
    "Fundamentals: Autoregressive Language Modeling": "自回归语言建模",
    "Architectures I: Recurrent Neural Networks": "架构一：循环神经网络",
    "Architectures II: Attention and Transformers": "架构二：Attention 与 Transformer",
    "Learning I: Pretraining": "学习一：预训练",
    "Scaling Laws and In-Context Learning": "Scaling Laws 与上下文学习",
    "Learning III: Fine-tuning and Distillation": "微调与知识蒸馏",
    "Inference II: Decoding Algorithms": "推理：解码算法",
    "Modeling I: Retrieval and RAG": "建模一：检索与 RAG",
    "Modeling II: Multimodal I": "建模二：多模态基础",
    "Modeling III: Multimodal II": "建模三：多模态生成",
    "Evaluation Techniques": "评测技术",
    "Research Skills and Experimental Design": "研究技能与实验设计",
    "Modeling IV: Diffusion and Flows": "建模四：扩散模型与 Flow",
    "Reinforcement Learning I: Fundamentals": "强化学习一：基础",
    "Reinforcement Learning II: Applications": "强化学习二：大模型应用",
    "Language Model-Based Agents": "基于语言模型的 Agent",
    "Quantization": "量化",
    "Parallelism and Distributed Training": "并行与分布式训练",
    "Mixture of Experts": "混合专家模型 MoE",
    "Scaling Sequence Length": "扩展序列长度",
    "Test-Time Scaling": "测试时扩展",
    # LLM Systems
    "Introduction to LLM": "大语言模型导论",
    "GPU Programming Basics 1": "GPU 编程基础一",
    "GPU Programming Basics 2": "GPU 编程基础二",
    "Learning algorithm and Auto Differentiation": "学习算法与自动微分",
    "Deep Learning Frameworks Design": "深度学习框架设计",
    "Transformer": "Transformer 系统基础",
    "Pre-trained LLMs": "预训练大语言模型",
    "Tokenization": "Tokenization",
    "LLM Decoding": "大模型解码",
    "GPU Acceleration": "GPU 加速",
    "Accelerating Transformer on GPU Part 1": "GPU 上的 Transformer 加速一",
    "Accelerating Transformer on GPU Part 2": "GPU 上的 Transformer 加速二",
    "Distributed Model Training": "分布式模型训练一",
    "Distributed Model Training II": "分布式模型训练二",
    "Distributed Model Training III": "分布式模型训练三",
    "Model Quantization": "模型量化一",
    "Model Quantization II": "模型量化二",
    "Efficient fine-tuning for Large Models": "大模型高效微调",
    "Large models with Mixture-of-Expert": "大模型与 MoE",
    "Optimizing Attention for Modern Hardware (Tri Dao)": "面向现代硬件优化 Attention",
    "Communication Efficient Distributed Training": "通信高效的分布式训练",
    "LLM Serving with PageAttention (Woosuk Kwon)": "PagedAttention 与大模型服务",
    "Better KV Cache for LLM Serving (Yuhan Liu)": "更高效的 KV Cache 服务",
    "DistServe: Disaggregated Prefill-Decoding (Hao Zhang)": "DistServe：Prefill/Decode 分离",
    "LLM serving with SGL (Ying Sheng)": "SGLang 大模型服务",
    "Efficient Reinforcement Learning System for LLMs": "高效大模型强化学习系统",
    "App Stack and Model Serving": "应用栈与模型服务",
    "GPU just-in-time compilation": "GPU 即时编译",
    "Speculative Decoding": "推测解码",
    "Retrieval-augmented Language Models": "检索增强语言模型",
    "Nearest Vector Search for Embeddings": "Embedding 近邻向量检索",
    "Multimodal LLMs": "多模态大语言模型",
    "Deepseek V3 and R1": "DeepSeek V3 与 R1",
    "Efficient Streaming Language Models with Attention Sinks": "Attention Sink 与流式语言模型",
    "Advanced Large Model Serving": "高级大模型服务",
    "Dynamo": "Dynamo 推理系统",
    # CMU LLM Applications
    "Origins of LLMs": "大语言模型的起源",
    "Natural language understanding vs generation": "自然语言理解与生成",
    "The science of prompting": "Prompting 的科学方法",
    "Deciding when to finetune and finetuning efficiently": "何时微调，以及怎样高效微调",
    "Learning representations and embeddings": "表示学习与 Embedding",
    "Retrieval 1: Storing and retrieving knowledge": "检索一：知识的存储与召回",
    "Retrieval 2: Retrieval augmented generation, deep research": "检索二：RAG 与 Deep Research",
    "Retrieval 3: Retrieval augmented generation (2) and deep research": "检索三：进阶 RAG 与 Deep Research",
    "Deep research": "Deep Research 系统",
    "Task-Oriented Dialogue": "任务型对话",
    "Tool-use, chitchat, personas, and companionship": "工具使用、闲聊、角色与陪伴",
    "Writing and ideation assistants and AI creative writing": "写作、创意助手与 AI 创作",
    "LLMs for evaluation: Synthetic data generation, simulation, automatic evaluation, AI-as-judge": "用大模型做评测：合成数据、模拟与 AI Judge",
    "Multi-agent systems": "多 Agent 系统",
    "Harms caused by LLM applications": "大模型应用造成的风险",
    "Attacking LLMs and LLM applications": "攻击大模型及其应用",
    "Code-writing assistants (guest lecture from Zora Wang)": "代码助手",
    "[tentative] Image generation and conversing about images": "图像生成与视觉对话",
    "LLMs for Non-English Languages and non-American Cultures (guest lecture from Shaily Bhatt)": "非英语语言与非美国文化中的大模型",
    "World models (guest lecture from Mingkai Deng)": "世界模型",
    "LLMs for biological understanding (guest lecture by Prof Lei Li)": "大模型与生物学理解",
    "Music generation (guest lecture by Prof Chris Donahue)": "音乐生成",
    "Numbers": "大模型与数字",
    "Robots and embodied AI (guest lecture from Leena Mathur)": "机器人与具身智能",
    "Deployment": "大模型应用部署",
    "Project presentations": "课程项目展示",
}


SUMMARY_RULES = [
    ("experimental design", "学习怎样把研究问题变成可复现的实验：设置基线、控制变量、选择指标、报告不确定性，并避免从偶然结果得出过强结论。"),
    ("learned representations", "理解离散符号如何变成连续向量，以及表示学习如何把相似性、类别与上下文信息编码进模型可计算的空间。"),
    ("autoregressive", "从概率链式法则出发理解下一个 Token 预测，连接训练目标、似然、困惑度与逐步生成。"),
    ("recurrent", "学习 RNN 如何沿时间保存状态，以及门控结构、梯度传播和序列建模的主要限制。"),
    ("attention and transformer", "拆解 Q/K/V、自注意力、多头机制、位置表示与残差归一化，并理解 Transformer 的并行计算路径。"),
    ("pretraining", "把预训练放进数据、目标函数、模型规模与计算预算的共同框架中，理解基础模型能力从何而来。"),
    ("scaling laws", "用经验幂律连接模型大小、数据量、计算量与损失，并理解上下文学习如何在不更新参数时适配任务。"),
    ("fine-tun", "比较全参数微调、LoRA、量化微调和指令适配，判断不同数据与硬件预算下该更新哪些参数。"),
    ("distillation", "理解教师模型如何把输出分布、解释轨迹或任务能力迁移到更小模型，并分析压缩和性能之间的取舍。"),
    ("decoding", "比较贪心、Beam Search、采样和推测解码，理解质量、多样性、延迟与吞吐之间的权衡。"),
    ("retrieval", "从文档切分、索引、向量召回、重排到答案生成，建立 RAG 与知识检索系统的完整数据流。"),
    ("deep research", "把多轮检索、查询改写、证据整合、来源核对和长答案生成组织成可追踪的研究工作流。"),
    ("multimodal", "理解文本、图像等模态如何被编码、对齐与统一生成，以及视觉 Token 和跨模态训练的关键设计。"),
    ("image", "学习图像 Token、离散表征与生成模型如何让语言模型处理和生成视觉内容。"),
    ("evaluation", "从数据集、指标、人工评价到 LLM-as-a-Judge，识别数据污染、偏差、方差和评测失真的风险。"),
    ("diffusion", "从逐步加噪与去噪建立扩散模型直觉，并比较 Flow Matching 等连续生成方法。"),
    ("reinforcement learning", "理解状态、动作、奖励、策略梯度与 PPO，并连接到偏好对齐、推理训练和 Agent 行为优化。"),
    ("agent", "把模型放进规划、行动、工具调用、观察与修正循环，分析记忆、环境接口和长程任务可靠性。"),
    ("quantization", "理解权重与激活从高精度映射到低比特的过程，比较误差、显存、速度和硬件支持之间的取舍。"),
    ("parallel", "拆解数据并行、张量并行、流水线并行与通信开销，学习怎样把大模型训练分布到多张 GPU。"),
    ("distributed", "理解参数、梯度、优化器状态和激活如何在设备间切分，以及同步通信为何决定训练效率。"),
    ("mixture", "学习路由器如何只激活少量专家，理解稀疏计算、负载均衡、通信与专家容量问题。"),
    ("sequence length", "分析长上下文的显存与计算瓶颈，并比较 FlashAttention、分块、环形注意力和状态空间模型。"),
    ("test-time", "研究怎样在推理阶段投入更多搜索、采样、验证和反思计算，以换取更高答案质量。"),
    ("gpu programming", "从线程、Block、内存层级和 Kernel 入手建立 GPU 并行计算直觉，为后续算子优化打基础。"),
    ("auto differentiation", "用计算图理解前向与反向模式自动微分，以及深度学习框架如何生成和执行梯度计算。"),
    ("framework", "从张量、算子、自动微分、执行图与设备调度理解深度学习框架的核心设计。"),
    ("gpu acceleration", "围绕访存、并行度、融合 Kernel 和硬件利用率分析算子为什么慢，以及如何定位优化空间。"),
    ("accelerating transformer", "把 Transformer 拆成矩阵乘、归一化、Attention 与通信步骤，理解 Kernel 融合和端到端加速。"),
    ("communication efficient", "分析 All-Reduce 等集合通信的成本，并学习通过状态切分、重叠计算和拓扑感知降低通信瓶颈。"),
    ("pagedattention", "理解 KV Cache 的分页管理与连续批处理，连接显存碎片、请求调度和服务吞吐。"),
    ("kv cache", "研究 KV Cache 的传输、复用、压缩与混合策略，降低长上下文服务的首 Token 延迟和显存占用。"),
    ("distserve", "理解 Prefill 与 Decode 的计算特征差异，以及将两阶段分离部署对延迟、吞吐和资源配置的影响。"),
    ("serving", "从请求调度、连续批处理、KV Cache、并行和可观测性建立大模型在线服务的系统视角。"),
    ("tokenization", "理解 BPE、SentencePiece 与词表训练，观察 Token 粒度如何影响成本、多语言公平和模型输入长度。"),
    ("prompt", "把 Prompt 视作实验变量，学习模板、示例、顺序和措辞如何影响模型，并用评测而不是感觉选择方案。"),
    ("embedding", "理解文本向量的训练目标、相似度与索引方式，并连接搜索、聚类、推荐与 RAG。"),
    ("task-oriented dialogue", "学习意图、槽位、状态跟踪、策略与自然语言生成如何协作完成有明确目标的多轮对话。"),
    ("tool-use", "比较工具调用、闲聊、Persona 与陪伴型助手的目标和边界，关注一致性、安全与长期交互。"),
    ("writing", "分析大模型怎样支持构思、改写与创作，同时处理原创性、作者控制和评价标准。"),
    ("multi-agent", "研究多个 Agent 的分工、通信、协商与聚合，并识别错误放大、成本和难以评测的问题。"),
    ("harm", "系统识别偏见、隐私、误导、依赖与社会影响，并把风险评估加入产品设计和部署流程。"),
    ("attack", "了解 Prompt Injection、越狱、数据与工具攻击的威胁模型，并学习分层防护和红队评测。"),
    ("code-writing", "理解代码助手的补全、编辑、测试与 Agent 工作流，并关注执行反馈、软件仓库上下文和安全。"),
    ("non-english", "观察 Tokenizer、数据覆盖和文化假设如何造成语言差异，并讨论本地化评测与文化适配。"),
    ("world model", "理解模型怎样学习环境状态与动态，并用预测、规划和模拟支持智能体决策。"),
    ("biological", "探索序列与结构模型如何用于蛋白质、分子和生物知识任务，并辨析语言类比的适用边界。"),
    ("music", "把音乐表示为可建模的序列或连续信号，理解结构、控制条件、评价与版权问题。"),
    ("number", "分析语言模型在算术、数值表示和数量推理上的能力与缺陷，以及工具增强方法。"),
    ("robot", "把语言模型与视觉、动作和物理反馈结合，理解具身智能中的感知、规划、控制与安全。"),
    ("origin", "从统计语言模型、词向量和神经语言模型回顾大模型的技术脉络，建立后续应用课程的历史坐标。"),
    ("understanding vs generation", "比较理解型任务与生成型任务的目标、训练信号和评测方法，辨清同一模型在两类任务中的角色。"),
    ("introduction", "建立课程总地图，认识模型、数据、训练、推理和系统工程之间的依赖关系。"),
]


def summarize(title: str) -> str:
    lower = title.lower()
    for key, summary in SUMMARY_RULES:
        if key in lower:
            return summary
    return "围绕本讲主题建立概念、方法、系统实现与评价标准之间的联系，并结合课件和论文理解关键设计取舍。"


def clean_text(node) -> str:
    return " ".join(node.text_content().split())


def iso_date(value: str, year: int) -> str:
    value = value.strip()
    try:
        parsed = datetime.strptime(value, "%m/%d/%Y")
        return parsed.strftime("%Y-%m-%d")
    except ValueError:
        try:
            parsed = datetime.strptime(value, "%m/%d")
            return f"{year}-{parsed.month:02d}-{parsed.day:02d}"
        except ValueError:
            return value or "补充主题"


def pdf_url(url: str) -> str | None:
    parsed = urlparse(url)
    host, path = parsed.netloc.lower(), parsed.path
    lower = path.lower()
    if "arxiv.org" in host:
        match = re.search(r"/(?:abs|pdf)/([^/?#]+)", path)
        if match:
            paper_id = match.group(1).removesuffix(".pdf")
            return f"https://arxiv.org/pdf/{paper_id}.pdf"
    if "aclanthology.org" in host:
        paper_id = path.rstrip("/").split("/")[-1].removesuffix(".pdf")
        return f"https://aclanthology.org/{paper_id}.pdf" if paper_id else None
    if "openreview.net" in host and "/forum" in path and parsed.query:
        if "qrwe7XHTmYb" in parsed.query:
            return "https://arxiv.org/pdf/2006.16668.pdf"
        return f"https://openreview.net/pdf?{parsed.query}"
    if lower.endswith(".pdf") or "/viewfile/" in lower:
        return url
    return None


def add_reading(label: str, url: str, index: int) -> dict:
    direct = pdf_url(url)
    item = {
        "label": label,
        "kind": "reading",
        "url": url,
        "pdf_url": direct,
        "status": "pending" if direct else "online-only",
    }
    if direct:
        item["filename"] = f"{index:02d}-{safe_segment(label)[:72]}.pdf"
    if "openreview.net/forum?id=qrwe7XHTmYb" in url:
        item["note"] = "OpenReview 下载端点拒绝自动访问；改用作者公开的 arXiv 版本。"
    return item


def make_course(course_id: str, title: str, school: str, official: str, description: str, page: str, index_description: str) -> dict:
    return {
        "id": course_id,
        "title": title,
        "title_zh": title,
        "school": school,
        "official": official,
        "description": description,
        "page": page,
        "index_description": index_description,
        "sessions": [],
    }


def parse_anlp(path: Path) -> dict:
    course = make_course(
        "cmu-anlp-spring-2026", "CMU Advanced NLP · Spring 2026", "Carnegie Mellon University",
        SOURCES["cmu-anlp-spring-2026"], "Advanced Natural Language Processing", "cmu-anlp-2026",
        "研究导向的高级 NLP 主线，覆盖语言建模、现代架构、多模态、RL、Agent、效率和测试时扩展。",
    )
    document = html.parse(str(path))
    for row in document.xpath('//*[@id="schedule"]//ul[contains(@class,"table")]/li[contains(@class,"table-row")]'):
        cols = row.xpath("./div")
        if len(cols) < 4 or "Lecture" not in clean_text(cols[1]):
            continue
        event = clean_text(cols[0])
        number_match = re.search(r"#\s*(\d+)", event)
        date_match = re.search(r"\d{2}/\d{2}/\d{4}", event)
        if not number_match or not date_match:
            continue
        topic_col, resources = cols[2], cols[3]
        direct_text = [" ".join(t.split()) for t in topic_col.xpath("./text()[normalize-space()]")]
        title = direct_text[0] if direct_text else clean_text(topic_col)
        title = re.sub(r"\s*\[(?:slides|code)\].*$", "", title).strip()
        materials, extras = [], []
        for anchor in topic_col.xpath(".//a[@href]"):
            label, url = clean_text(anchor), urljoin(course["official"], anchor.get("href").strip())
            if "slides" in label.lower() and pdf_url(url):
                materials.append({"label": "Slides", "kind": "slides", "url": url, "filename": "01-slides.pdf"})
            elif "code" in label.lower():
                extras.append({"label": "Code", "url": url})
        readings = []
        for index, anchor in enumerate(resources.xpath(".//a[@href]"), 1):
            readings.append(add_reading(clean_text(anchor), urljoin(course["official"], anchor.get("href").strip()), index))
        course["sessions"].append({
            "id": f"L{int(number_match.group(1)):02d}", "date": iso_date(date_match.group(), 2026),
            "title": title, "title_zh": TITLE_ZH.get(title, title), "category": clean_text(topic_col.xpath("./strong")[0]) if topic_col.xpath("./strong") else "",
            "summary": summarize(title), "materials": materials, "readings": readings, "extras": extras,
        })
    return course


def parse_llm_systems(path: Path) -> dict:
    course = make_course(
        "llm-systems-spring-2025", "Large Language Model Systems · Spring 2025", "Carnegie Mellon University",
        SOURCES["llm-systems-spring-2025"], "Large Language Model Systems", "llm-systems-2025",
        "从 GPU 编程、自动微分和 Transformer 算子一路走到分布式训练、量化、MoE、KV Cache 与在线 Serving。",
    )
    document = html.parse(str(path))
    current_date = ""
    number = 0
    skip = {"spring break", "final project presentation"}
    for row in document.xpath("//article//table//tbody/tr"):
        cells = row.xpath("./td")
        if len(cells) < 3:
            continue
        date_text = clean_text(cells[0])
        if date_text:
            current_date = iso_date(date_text, 2025)
        topic = clean_text(cells[1]).replace("[slides]", "").strip()
        if not topic or topic.lower() in skip:
            continue
        topic_links = cells[1].xpath(".//a[@href]")
        reading_links = cells[2].xpath(".//a[@href]")
        if not topic_links and not reading_links and topic.lower() == "deepseek v3 and r1":
            # Keep the advertised topic even though the page does not publish files.
            pass
        elif not topic_links and not reading_links and not clean_text(cells[2]):
            continue
        number += 1
        materials = []
        for anchor in topic_links:
            url = urljoin(course["official"], anchor.get("href").strip())
            if pdf_url(url):
                materials.append({"label": "Slides", "kind": "slides", "url": url, "filename": "01-slides.pdf"})
        readings = [
            add_reading(clean_text(anchor), urljoin(course["official"], anchor.get("href").strip()), index)
            for index, anchor in enumerate(reading_links, 1)
        ]
        course["sessions"].append({
            "id": f"L{number:02d}", "date": current_date or "补充主题", "title": topic,
            "title_zh": TITLE_ZH.get(topic, topic), "summary": summarize(topic),
            "materials": materials, "readings": readings,
        })
    return course


def parse_llm_apps(path: Path) -> dict:
    course = make_course(
        "cmu-llm-applications-spring-2026", "CMU Large Language Model Applications · Spring 2026",
        "Carnegie Mellon University", SOURCES["cmu-llm-applications-spring-2026"],
        "Large Language Model Applications", "cmu-llm-applications-2026",
        "面向真实产品与跨领域应用，覆盖 Prompt、微调、RAG、对话、创作、评测、多 Agent、安全、文化、科学、音乐和机器人。",
    )
    document = html.parse(str(path))
    number = 0
    for row in document.xpath('//li[contains(concat(" ", normalize-space(@class), " "), " table-row-lecture ")]'):
        date_nodes = row.xpath('.//div[@data-label="Date"]')
        title_nodes = row.xpath('.//div[@data-label="Description"]//b')
        if not date_nodes or not title_nodes:
            continue
        number += 1
        raw_date = clean_text(date_nodes[0]).split()[0]
        title = clean_text(title_nodes[0])
        materials = []
        for anchor in row.xpath('.//a[@title="Download slides"]'):
            url = (anchor.get("href") or "").strip()
            if url and pdf_url(url):
                materials.append({"label": "Slides", "kind": "slides", "url": url, "filename": "01-slides.pdf"})
        resource_nodes = row.xpath('.//div[@data-label="Course Material"]')
        readings = []
        if resource_nodes:
            anchors = [a for a in resource_nodes[0].xpath('.//a[@href]') if a.get("title") != "Download slides"]
            readings = [add_reading(clean_text(a), a.get("href").strip(), i) for i, a in enumerate(anchors, 1)]
        course["sessions"].append({
            "id": f"L{number:02d}", "date": iso_date(raw_date, 2026), "title": title,
            "title_zh": TITLE_ZH.get(title, title), "summary": summarize(title),
            "materials": materials, "readings": readings,
        })
    return course


def enrich_paths(courses: list[dict]) -> list[dict]:
    records = []
    for course in courses:
        for session in course["sessions"]:
            folder = f"{session['id'].lower()}-{safe_segment(session['title'])}"
            for item in session.get("materials", []):
                rel = Path(course["id"]) / folder / item["filename"]
                item.update({
                    "relative_path": rel.as_posix(), "public_url": "/course-materials/" + rel.as_posix(),
                    "local_path": str(PUBLIC_ROOT / rel),
                })
                records.append({"course": course, "session": session, "item": item})
            for item in session.get("readings", []):
                if not item.get("pdf_url"):
                    continue
                rel = Path(course["id"]) / folder / "readings" / item["filename"]
                item.update({
                    "relative_path": rel.as_posix(), "public_url": "/course-materials/" + rel.as_posix(),
                    "local_path": str(PUBLIC_ROOT / rel),
                })
                records.append({"course": course, "session": session, "item": item})
    return records


def download_one(record: dict, force: bool) -> tuple[str, str]:
    item = record["item"]
    target = Path(item["local_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if not force and valid_pdf(target):
        return "cached", str(target)
    part = target.with_suffix(target.suffix + ".part")
    if part.exists():
        part.unlink()
    url = item.get("pdf_url") or item["url"]
    try:
        subprocess.run([
            "curl", "-L", "--fail", "--silent", "--show-error", "--retry", "2",
            "--connect-timeout", "20", "--max-time", "300",
            "-A", "Mozilla/5.0 (course-materials-archiver)", "-o", str(part), url,
        ], check=True)
        if not valid_pdf(part):
            raise RuntimeError("downloaded content is not a valid PDF")
        os.replace(part, target)
        return "downloaded", str(target)
    except Exception:
        if part.exists():
            part.unlink()
        raise


def fetch_sources() -> dict[str, Path]:
    paths = {}
    source_dir = OUTPUT_ROOT / "source-pages"
    source_dir.mkdir(parents=True, exist_ok=True)
    for course_id, url in SOURCES.items():
        path = source_dir / f"{course_id}.html"
        subprocess.run([
            "curl", "-L", "--fail", "--silent", "--show-error", "--retry", "2",
            "-A", "Mozilla/5.0 (course-materials-archiver)", "-o", str(path), url,
        ], check=True)
        paths[course_id] = path
    return paths


def run(force: bool, workers: int) -> None:
    print("Refreshing three official course schedules...", flush=True)
    paths = fetch_sources()
    courses = [
        parse_anlp(paths["cmu-anlp-spring-2026"]),
        parse_llm_systems(paths["llm-systems-spring-2025"]),
        parse_llm_apps(paths["cmu-llm-applications-spring-2026"]),
    ]
    records = enrich_paths(courses)
    print("Sessions:", ", ".join(f"{c['id']}={len(c['sessions'])}" for c in courses), flush=True)
    print(f"Preparing {len(records)} downloadable PDF files...", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, record, force): record for record in records}
        for index, future in enumerate(as_completed(futures), 1):
            record = futures[future]
            label = f"{record['course']['id']}/{record['session']['id']}/{record['item']['filename']}"
            try:
                status, _ = future.result()
                print(f"[{index:03d}/{len(records)}] {status:10s} {label}", flush=True)
            except Exception as exc:
                failures.append((label, str(exc)))
                print(f"[{index:03d}/{len(records)}] FAILED    {label}: {exc}", file=sys.stderr, flush=True)
    for record in records:
        item = record["item"]
        path = Path(item["local_path"])
        if valid_pdf(path):
            item.update(pdf_metadata(path))
            item["status"] = "available"
        else:
            item["status"] = "failed"
    manifest = {
        "generated_at": "2026-08-03",
        "scope": "Official lecture slides and schedule readings for three additional public courses.",
        "courses": courses,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {MANIFEST_PATH}")
    if failures:
        print("\nDownload failures:", file=sys.stderr)
        for label, error in failures:
            print(f"- {label}: {error}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.force, max(1, min(args.workers, 8)))


if __name__ == "__main__":
    main()
