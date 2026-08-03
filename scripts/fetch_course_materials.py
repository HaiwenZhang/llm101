#!/usr/bin/env python3
"""Download and index the public lecture PDFs linked by CS224N and NTU ADL.

The course data below is intentionally explicit.  That makes the local archive
auditable: every file keeps its official URL, lecture grouping, and role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse

from lxml import html
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / "site" / "public" / "course-materials"
OUTPUT_ROOT = ROOT / "output" / "course-materials"


def material(label: str, kind: str, url: str, filename: str, note: str = "") -> dict:
    return {"label": label, "kind": kind, "url": url, "filename": filename, "note": note}


CS_BASE = "https://web.stanford.edu/class/cs224n/"
NTU_BASE = "https://www.csie.ntu.edu.tw/~miulab/f114-adl/"


CS224N = {
    "id": "cs224n-winter-2026",
    "title": "Stanford CS224N · Winter 2026",
    "school": "Stanford University",
    "official": CS_BASE,
    "description": "Natural Language Processing with Deep Learning",
    "sessions": [
        {
            "id": "L01", "date": "2026-01-06", "title": "History of NLP", "title_zh": "NLP 的历史与课程地图",
            "summary": "从规则系统、统计学习一路走到神经网络与大语言模型。先建立“语言任务—表示方法—训练范式”的总地图，再理解后面每一讲解决的是哪一类问题。",
            "materials": [
                material("课程导论 Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture01-intro.pdf", "01-course-introduction.pdf"),
                material("NLP 历史 Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture01-history.pdf", "02-history-of-nlp.pdf"),
            ],
        },
        {
            "id": "L02", "date": "2026-01-08", "title": "Word Vectors", "title_zh": "词向量：让词变成可计算的坐标",
            "summary": "理解分布式假设、Word2Vec、负采样与 GloVe。重点不是背公式，而是看懂“相似上下文中的词为何会靠近”以及词向量该如何训练和评价。",
            "materials": [
                material("Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture02-wordvecs.pdf", "01-word-vectors-slides.pdf"),
                material("Lecture Notes 1", "notes", CS_BASE + "readings/cs224n_winter2023_lecture1_notes_draft.pdf", "02-word-vectors-notes-1.pdf"),
                material("Lecture Notes 2", "notes", CS_BASE + "readings/cs224n-2019-notes02-wordvecs2.pdf", "03-word-vectors-notes-2.pdf"),
            ],
        },
        {
            "id": "S01", "date": "2026-01-09", "title": "Python Review Session", "title_zh": "Python 复习课",
            "summary": "补齐 Python、NumPy 与向量化计算的操作基础。适合在正式写神经网络之前快速检查数组形状、切片、广播和矩阵运算是否熟练。",
            "materials": [
                material("Python Review Slides", "slides", CS_BASE + "slides_w25/2024%20CS224N%20Python%20Review%20Session%20Slides.pptx.pdf", "01-python-review-slides.pdf", "官网 2026 课程表沿用 2024 版本。"),
            ],
        },
        {
            "id": "L03", "date": "2026-01-13", "title": "Backpropagation and Neural Network Basics", "title_zh": "神经网络与反向传播",
            "summary": "把前向计算、损失函数、链式法则和参数更新连成一条训练流水线；学会沿计算图反向传递梯度，并用矩阵形式高效实现。",
            "materials": [
                material("Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture03-neuralnets.pdf", "01-neural-networks-slides.pdf"),
                material("Lecture Notes", "notes", CS_BASE + "readings/cs224n-2019-notes03-neuralnets.pdf", "02-neural-networks-notes.pdf"),
                material("矩阵微积分补充讲义", "supplement", CS_BASE + "readings/gradient-notes.pdf", "03-matrix-calculus-notes.pdf"),
                material("微分复习讲义", "supplement", CS_BASE + "readings/review-differential-calculus.pdf", "04-differential-calculus-review.pdf"),
            ],
        },
        {
            "id": "L04", "date": "2026-01-15", "title": "Language Models and RNNs", "title_zh": "语言模型与循环神经网络",
            "summary": "从概率链式法则定义语言模型，再用 RNN 的隐藏状态压缩历史信息；理解时间展开、教师强制、困惑度，以及梯度消失为何限制长距离记忆。",
            "materials": [
                material("Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture04-rnnlm.pdf", "01-rnn-language-models-slides.pdf"),
                material("Lecture Notes", "notes", CS_BASE + "readings/cs224n-2019-notes05-LM_RNN.pdf", "02-rnn-language-models-notes.pdf"),
            ],
        },
        {
            "id": "L05", "date": "2026-01-20", "title": "Transformers", "title_zh": "Transformer",
            "summary": "逐层拆解 Q、K、V、自注意力、多头注意力、位置表示、残差连接与层归一化，理解 Transformer 如何并行地建立任意 Token 之间的关系。",
            "materials": [
                material("Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture05-transformers.pdf", "01-transformer-slides.pdf"),
                material("Lecture Notes", "notes", CS_BASE + "readings/cs224n-self-attention-transformers-2023_draft.pdf", "02-transformer-notes.pdf"),
            ],
        },
        {
            "id": "L06", "date": "2026-01-22", "title": "Final Projects: Custom and Default; Practical Tips", "title_zh": "研究项目设计与实践建议",
            "summary": "学习怎样把宽泛想法收敛成可验证的问题：确定任务与指标、建立基线、控制变量、做误差分析，并规划提案、里程碑和最终报告。",
            "materials": [material("Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture06-final-project.pdf", "01-final-project-practical-tips.pdf")],
        },
        {
            "id": "L07", "date": "2026-01-27", "title": "Pretraining (Scaling, Systems, Data)", "title_zh": "预训练：规模、系统与数据",
            "summary": "把大模型预训练看作数据、计算、模型和工程系统共同作用的过程；关注 Scaling Laws、训练数据治理、分布式训练与吞吐效率之间的取舍。",
            "materials": [material("Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture07-pretraining.pdf", "01-pretraining-slides.pdf")],
        },
        {
            "id": "L08", "date": "2026-01-29", "title": "Post-training (RLHF, SFT, DPO)", "title_zh": "后训练：SFT、RLHF 与 DPO",
            "summary": "理解预训练模型如何通过示范数据与偏好数据变成更会遵循指令的助手；比较监督微调、奖励模型、策略优化和直接偏好优化的目标与流程。",
            "materials": [material("Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture08-posttraining.pdf", "01-post-training-slides.pdf")],
        },
        {
            "id": "L09", "date": "2026-02-03", "title": "Efficient Adaptation (Prompting + PEFT)", "title_zh": "高效适配：Prompt 与 PEFT",
            "summary": "比较上下文学习、Prompting、Adapter 与 LoRA 等适配方式；重点理解在不完整更新模型参数时，如何用更少显存和数据完成任务迁移。",
            "materials": [material("Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture09-peft.pdf", "01-efficient-adaptation-slides.pdf")],
        },
        {
            "id": "L10", "date": "2026-02-05", "title": "Agents, Tool Use, and RAG", "title_zh": "Agent、工具调用与 RAG",
            "summary": "把模型从“只生成文字”扩展为会检索、调用工具、观察结果再行动的系统；理解 RAG 数据流、Agent 循环、ReAct 思路及其常见失败模式。",
            "materials": [material("Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture10-rag-agents.pdf", "01-rag-agents-tools-slides.pdf")],
        },
        {
            "id": "S02", "date": "2026-02-06", "title": "Hugging Face Transformers Tutorial Session", "title_zh": "Hugging Face Transformers 实作课",
            "summary": "从模型与 Tokenizer 加载开始，练习推理、批处理、微调和常用 API，把课程里的抽象结构落到可运行代码。",
            "materials": [material("Tutorial Slides", "slides", CS_BASE + "materials/hf_transformers_tutorial.pdf", "01-hugging-face-transformers-tutorial.pdf")],
        },
        {
            "id": "L11", "date": "2026-02-10", "title": "Benchmarking and Evaluation", "title_zh": "基准测试与评测",
            "summary": "学习指标、数据集、人工评价和 LLM-as-a-Judge 的使用边界；识别数据污染、捷径学习、分布偏移和单一分数掩盖能力差异等问题。",
            "materials": [material("Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture11-evaluation.pdf", "01-benchmarking-evaluation-slides.pdf")],
        },
        {
            "id": "L12", "date": "2026-02-12", "title": "Reasoning 1", "title_zh": "推理（一）：让模型展开思考",
            "summary": "从 Chain-of-Thought、自洽采样到强化学习推理，理解“多生成一些中间步骤”何时能提高正确率，以及推理轨迹为何不等于可靠解释。",
            "materials": [material("Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture12-reasoning-part1.pdf", "01-reasoning-part-1-slides.pdf")],
        },
        {
            "id": "L13", "date": "2026-02-17", "title": "Reasoning 2", "title_zh": "推理（二）：验证与测试时计算",
            "summary": "进一步讨论过程监督、验证器、搜索与测试时扩展；理解把额外计算花在候选生成、评分和修正上的方法，以及推理速度优化。",
            "materials": [material("Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture13-reasoning-part2.pdf", "01-reasoning-part-2-slides.pdf")],
        },
        {
            "id": "L14", "date": "2026-02-19", "title": "Tokenization and Multilinguality", "title_zh": "分词与多语言",
            "summary": "理解 BPE/子词切分、词表训练与 Token 成本；观察同一 Tokenizer 对不同语言的不均衡影响，以及多语言表示与数据覆盖问题。",
            "materials": [material("Guest Lecture Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture14-guest-julie-tokenization-multilinguality.pdf", "01-tokenization-multilinguality-slides.pdf")],
        },
        {
            "id": "L15", "date": "2026-02-24", "title": "Interpretability", "title_zh": "可解释性（官网链接待核对）",
            "summary": "课程表主题是可解释性，但官网当前 Slides 链接指向一份 2024 年旧文件。资料仍按官网原样保存；学习时请先看本地文件首页，避免把文件内容误当成本讲正式课件。",
            "warning": "官网课程表标题与所链接 PDF 的文件名/年份不一致。",
            "materials": [material("官网当前链接的 Slides", "slides", CS_BASE + "slides/cs224n-spr2024-lecture15-life-after-dpo-lambert.pdf", "01-official-linked-slides-title-mismatch.pdf", "课程页当前链接到 2024 旧课件，可能与本讲标题不匹配。")],
        },
        {
            "id": "L16", "date": "2026-02-26", "title": "Social and Broader Impacts of NLP (Risks)", "title_zh": "NLP 的社会影响与风险",
            "summary": "从偏见、公平、隐私、滥用、虚假信息和环境成本等角度审视 NLP 系统，学习把风险识别、评测与缓解措施放进模型开发流程。",
            "materials": [material("Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture16-impact-on-humanity.pdf", "01-social-impact-risks-slides.pdf")],
        },
        {
            "id": "L17", "date": "2026-03-03", "title": "Multimodality", "title_zh": "多模态",
            "summary": "课程表聚焦文本与图像等模态的统一表示、生成与推理；官网当前只列出推荐阅读，没有公开 Slides 文件。",
            "warning": "截至本次抓取，官网未提供 Slides。", "materials": [],
        },
        {
            "id": "L18", "date": "2026-03-05", "title": "Tinker and LoRA Without Regret", "title_zh": "Tinker 与 LoRA 实践",
            "summary": "客座课主题围绕模型微调服务与 LoRA 实践；官网当前没有公开 Slides 文件。",
            "warning": "截至本次抓取，官网未提供 Slides。", "materials": [],
        },
        {
            "id": "L19", "date": "2026-03-10", "title": "Open Questions in NLP 2026", "title_zh": "2026 年 NLP 开放问题",
            "summary": "用研究视角回看全课：哪些能力仍缺乏可靠定义，哪些评测正在失效，数据、推理、对齐、多语言和社会影响还有哪些值得继续追问的问题。",
            "materials": [material("Slides", "slides", CS_BASE + "slides_w26/cs224n-2026-lecture19-open-questions.pdf", "01-open-questions-in-nlp-2026.pdf")],
        },
    ],
}


NTU = {
    "id": "ntu-adl-fall-2025",
    "title": "NTU Applied Deep Learning · Fall 2025",
    "school": "National Taiwan University",
    "official": NTU_BASE,
    "description": "Applied Deep Learning 2025",
    "sessions": [
        {
            "id": "P00", "date": "自学 / 先修", "title": "Prerequisites", "title_zh": "自学先修：从机器学习走到反向传播",
            "summary": "用三份课件补齐课程入口：先认识深度学习与任务类型，再学习神经网络的层、激活函数与损失，最后用链式法则理解反向传播。",
            "materials": [
                material("Introduction", "slides", NTU_BASE + "doc/250901_Introduction.pdf", "01-introduction.pdf"),
                material("Neural Network Basics", "slides", NTU_BASE + "doc/250901_NNBasics.pdf", "02-neural-network-basics.pdf"),
                material("Backpropagation", "slides", NTU_BASE + "doc/250901_Backprop.pdf", "03-backpropagation.pdf"),
            ],
        },
        {
            "id": "W01", "date": "2025-09-01", "title": "Course Logistics + Sequence Modeling", "title_zh": "第 1 周：课程说明与序列建模",
            "summary": "先了解课程任务与学习方式，再从序列输入输出形式进入 RNN、LSTM 等模型，理解文本为何不能简单当作互不相关的独立词。",
            "materials": [
                material("Course Logistics", "slides", NTU_BASE + "doc/250901_Course.pdf", "01-course-logistics.pdf"),
                material("Sequence Modeling", "slides", NTU_BASE + "doc/250901_SeqModel.pdf", "02-sequence-modeling.pdf"),
            ],
        },
        {
            "id": "W02", "date": "2025-09-08", "title": "Attention, Transformer, Tokenization, BERT", "title_zh": "第 2 周：Attention 到 BERT",
            "summary": "一周串起现代 NLP 的主干：注意力负责选择信息，Transformer 负责堆叠表示，Tokenizer 负责把文字变成 Token，BERT 用遮盖预训练学习双向上下文。",
            "materials": [
                material("Attention Mechanism", "slides", NTU_BASE + "doc/250908_Attention.pdf", "01-attention-mechanism.pdf"),
                material("Transformer", "slides", NTU_BASE + "doc/250908_Transformer.pdf", "02-transformer.pdf"),
                material("Tokenization", "slides", NTU_BASE + "doc/250908_Tokenization.pdf", "03-tokenization.pdf"),
                material("BERT", "slides", NTU_BASE + "doc/250908_BERT.pdf", "04-bert.pdf"),
                material("NLP Lifecycle（Recitation）", "recitation", "https://www.csie.ntu.edu.tw/~miulab/f113-adl/doc/w2-ProjLife.pdf", "05-nlp-project-lifecycle.pdf", "2025 课程页原链接失效；改用台大同课程 2024 官方存档。"),
            ],
        },
        {
            "id": "SUP", "date": "弹性补充", "title": "Word Embeddings + BERT Variants", "title_zh": "弹性补充：词向量与 BERT 变体",
            "summary": "回补 Word2Vec、GloVe 与词向量评价，再比较 XLNet、RoBERTa、mBERT 等预训练设计，适合在 BERT 主课之后查漏补缺。",
            "materials": [
                material("Word Embeddings", "slides", "https://www.csie.ntu.edu.tw/~miulab/f111-adl/doc/220929_WordEmbeddings.pdf", "01-word-embeddings.pdf", "课程页链接到 2022 版补充课件。"),
                material("BERT Variants", "slides", "https://www.csie.ntu.edu.tw/~miulab/f113-adl/doc/240918_BERTVariants.pdf", "02-bert-variants.pdf", "2025 课程页原链接失效；改用台大同课程 2024 官方存档。"),
            ],
        },
        {
            "id": "W03", "date": "2025-09-15", "title": "Pretraining & Prompt Learning", "title_zh": "第 3 周：预训练与 Prompt Learning",
            "summary": "理解预训练—下游适配的基本范式，以及离散 Prompt、连续 Prompt 和上下文学习如何改变模型接收任务的方式。",
            "materials": [
                material("Pretraining & Prompt Learning", "slides", NTU_BASE + "doc/250915_Pretraining.pdf", "01-pretraining-prompt-learning.pdf"),
                material("Underlying Logics of Projects（Recitation）", "recitation", "https://www.csie.ntu.edu.tw/~miulab/f112-adl/doc/w3-UnderlyLogic.pdf", "02-project-underlying-logics.pdf", "2025 课程页原链接失效；改用台大同课程 2023 官方存档。"),
            ],
        },
        {
            "id": "W04", "date": "2025-09-22", "title": "Post-Training + LLM Adaptation", "title_zh": "第 4 周：后训练与大模型适配",
            "summary": "从 SFT、偏好学习等后训练方法走向参数高效微调；结合 LoRA 实作理解“冻结大模型，只训练少量低秩参数”的工程价值。",
            "materials": [
                material("Post-Training", "slides", NTU_BASE + "doc/250922_PostTraining.pdf", "01-post-training.pdf"),
                material("LLM Adaptation", "slides", NTU_BASE + "doc/250922_Adaptation.pdf", "02-llm-adaptation.pdf"),
                material("LLM LoRA Training（Recitation）", "recitation", "https://www.csie.ntu.edu.tw/~miulab/f113-adl/doc/w5-LoRA.pdf", "03-lora-training.pdf", "2025 课程页原链接失效；改用台大同课程 2024 官方存档。"),
            ],
        },
        {
            "id": "W05", "date": "2025-10-13", "title": "Retrieval-Augmented Generation", "title_zh": "第 5 周：RAG 与 MoE",
            "summary": "主讲课覆盖文档切分、向量检索、重排、上下文组织与生成；Recitation 补充大模型架构和 MoE 路由，建立系统级视角。",
            "materials": [
                material("Retrieval-Augmented Generation", "slides", NTU_BASE + "doc/251013_RAG.pdf", "01-retrieval-augmented-generation.pdf"),
                material("LLM Basics & MoE（Recitation）", "recitation", "https://www.csie.ntu.edu.tw/~miulab/f113-adl/doc/w4-LLMBasicsMOE.pdf", "02-llm-basics-and-moe.pdf", "2025 课程页原链接失效；改用台大同课程 2024 官方存档。"),
            ],
        },
        {
            "id": "W06", "date": "2025-10-27", "title": "NLG Decoding + Evaluation", "title_zh": "第 6 周：生成解码、推理与评价",
            "summary": "比较贪心、Beam Search、Top-k、Top-p 等生成策略，再学习自动指标、人工评价与 LLM 评价；Recitation 把推理效率与评测流程连接起来。",
            "materials": [
                material("NLG Decoding", "slides", NTU_BASE + "doc/251027_NLG.pdf", "01-nlg-decoding.pdf"),
                material("NLG Evaluation", "slides", NTU_BASE + "doc/251027_NLGEval.pdf", "02-nlg-evaluation.pdf"),
                material("LLM Inference & Evaluation（Recitation）", "recitation", "https://www.csie.ntu.edu.tw/~miulab/f113-adl/doc/w6-LLMInferenceEval.pdf", "03-llm-inference-evaluation.pdf", "2025 课程页原链接失效；改用台大同课程 2024 官方存档。"),
            ],
        },
        {
            "id": "W07", "date": "2025-11-03", "title": "Issues and Development in Pre-Trained Models", "title_zh": "第 7 周：预训练模型的问题与发展",
            "summary": "从预训练模型的偏见、遗忘、数据与知识局限出发，讨论模型为什么会失败，以及研究路线如何针对这些问题演进。",
            "materials": [material("Issues and Development in Pre-Trained Models", "slides", NTU_BASE + "doc/251103_Issues.pdf", "01-pretrained-model-issues.pdf")],
        },
        {
            "id": "W08", "date": "2025-11-10", "title": "Language Agents", "title_zh": "第 8 周：语言 Agent",
            "summary": "把语言模型放进“规划—行动—观察—修正”的循环，学习工具调用、记忆、环境反馈与多步任务分解，并关注 Agent 的评价和可靠性。",
            "materials": [material("Language Agents", "slides", NTU_BASE + "doc/251110_LangAgent.pdf", "01-language-agents.pdf")],
        },
        {
            "id": "W09", "date": "2025-11-17", "title": "Knowledge, Multimodality", "title_zh": "第 9 周：知识与多模态",
            "summary": "官网课程表列出本讲主题，但当前没有可直接下载的 Slides。",
            "warning": "截至本次抓取，官网未提供 Slides。", "materials": [],
        },
        {
            "id": "W10", "date": "2025-11-24", "title": "Personalization", "title_zh": "第 10 周：个性化",
            "summary": "官网课程表列出个性化主题，但当前没有可直接下载的 Slides。",
            "warning": "截至本次抓取，官网未提供 Slides。", "materials": [],
        },
        {
            "id": "W11", "date": "2025-12-01", "title": "Reasoning", "title_zh": "第 11 周：推理",
            "summary": "官网课程表列出推理主题，但当前没有可直接下载的 Slides。",
            "warning": "截至本次抓取，官网未提供 Slides。", "materials": [],
        },
    ],
}


COURSES = [CS224N, NTU]


CS_SESSION_BY_DATE = {
    "Jan 6": "L01", "Jan 8": "L02", "Jan 13": "L03", "Jan 15": "L04",
    "Jan 20": "L05", "Jan 22": "L06", "Jan 27": "L07", "Jan 29": "L08",
    "Feb 3": "L09", "Feb 5": "L10", "Feb 10": "L11", "Feb 12": "L12",
    "Feb 17": "L13", "Feb 19": "L14", "Feb 24": "L15", "Feb 26": "L16",
    "Mar 3": "L17", "Mar 5": "L18", "Mar 10": "L19",
}


READING_PDF_OVERRIDES = {
    # The course's old university mirror now returns HTML, while Nature requires
    # a subscription. Keep the official reading link but do not claim a PDF.
    "http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf": (
        None,
        "课程给出的旧 PDF 地址目前返回网页；Nature 正文需要订阅，因此仅保留在线条目。",
    ),
    # PNAS blocks automated PDF downloads; the authors also released the paper
    # as an arXiv preprint, which is a public and auditable copy.
    "https://www.pnas.org/doi/10.1073/pnas.2406675122": (
        "https://arxiv.org/pdf/2310.16410.pdf",
        "PNAS 下载端点拒绝自动访问；改用作者公开的 arXiv 版本。",
    ),
}


def safe_segment(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "session"


def official_pdf_url(url: str) -> str | None:
    """Return a direct public PDF URL when it can be derived without guessing."""
    if url in READING_PDF_OVERRIDES:
        return READING_PDF_OVERRIDES[url][0]
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path
    lower_path = path.lower()
    if "arxiv.org" in host:
        match = re.search(r"/(?:abs|pdf)/([^/?#]+)", path)
        if match:
            paper_id = match.group(1).removesuffix(".pdf")
            return f"https://arxiv.org/pdf/{paper_id}.pdf"
    if "aclweb.org" in host and "/anthology/" in lower_path:
        paper_id = path.rstrip("/").split("/")[-1].removesuffix(".pdf")
        return f"https://aclanthology.org/{paper_id}.pdf"
    if "aclanthology.org" in host:
        if lower_path.endswith(".pdf"):
            return url
        paper_id = path.rstrip("/").split("/")[-1]
        if paper_id:
            return f"https://aclanthology.org/{paper_id}.pdf"
    if "pnas.org" in host and "/doi/10." in path and "/doi/pdf/" not in path:
        return url.replace("/doi/", "/doi/pdf/", 1)
    if lower_path.endswith(".pdf") or "/viewfile/" in lower_path:
        return url
    return None


def fetch_course_page(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "curl", "-L", "--fail", "--silent", "--show-error", "--retry", "2",
        "--connect-timeout", "20", "--max-time", "120",
        "-A", "Mozilla/5.0 (course-materials-archiver)", "-o", str(path), url,
    ], check=True)


def attach_cs224n_readings(source_html: Path) -> None:
    """Attach every schedule reading link and flag which ones expose a PDF."""
    session_map = {session["id"]: session for session in CS224N["sessions"]}
    document = html.parse(str(source_html))
    for row in document.xpath('//section[@id="schedule"]//tr | //*[@id="schedule"]//tr'):
        cells = row.xpath("./td")
        if len(cells) < 3:
            continue
        date_text = " ".join(cells[0].text_content().split())
        session_id = next((value for key, value in CS_SESSION_BY_DATE.items() if key in date_text), None)
        if not session_id:
            continue
        links = cells[2].xpath(".//a[@href]")
        if not links:
            continue
        session = session_map[session_id]
        readings = session.setdefault("readings", [])
        for link_index, anchor in enumerate(links, 1):
            label = " ".join(anchor.text_content().split()) or f"Reading {link_index}"
            source_url = urljoin(CS_BASE, anchor.get("href"))
            pdf_url = official_pdf_url(source_url)
            entry = {
                "label": label,
                "kind": "reading",
                "url": source_url,
                "pdf_url": pdf_url,
                "status": "pending" if pdf_url else "online-only",
            }
            if source_url in READING_PDF_OVERRIDES:
                entry["note"] = READING_PDF_OVERRIDES[source_url][1]
            if pdf_url:
                entry["filename"] = f"{len(readings) + 1:02d}-{safe_segment(label)[:72]}.pdf"
            readings.append(entry)


def enrich_paths() -> list[dict]:
    files = []
    for course in COURSES:
        for session in course["sessions"]:
            folder = f"{session['id'].lower()}-{safe_segment(session['title'])}"
            for item in session["materials"]:
                rel = Path(course["id"]) / folder / item["filename"]
                item["relative_path"] = rel.as_posix()
                item["public_url"] = "/course-materials/" + rel.as_posix()
                item["local_path"] = str(PUBLIC_ROOT / rel)
                files.append({"course": course, "session": session, "item": item})
            for item in session.get("readings", []):
                if not item.get("pdf_url"):
                    continue
                rel = Path(course["id"]) / folder / "readings" / item["filename"]
                item["relative_path"] = rel.as_posix()
                item["public_url"] = "/course-materials/" + rel.as_posix()
                item["local_path"] = str(PUBLIC_ROOT / rel)
                files.append({"course": course, "session": session, "item": item})
    return files


def valid_pdf(path: Path) -> bool:
    try:
        return path.stat().st_size > 1024 and path.read_bytes()[:5] == b"%PDF-"
    except OSError:
        return False


def download_one(record: dict, force: bool = False) -> tuple[str, str]:
    item = record["item"]
    target = Path(item["local_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if not force and valid_pdf(target):
        return ("cached", str(target))
    part = target.with_suffix(target.suffix + ".part")
    if part.exists():
        part.unlink()
    cmd = [
        "curl", "-L", "--fail", "--silent", "--show-error", "--retry", "2",
        "--connect-timeout", "20", "--max-time", "300",
        "-A", "Mozilla/5.0 (course-materials-archiver)",
        "-o", str(part), item.get("pdf_url") or item["url"],
    ]
    try:
        subprocess.run(cmd, check=True)
        if not valid_pdf(part):
            raise RuntimeError("downloaded content is not a valid PDF")
        os.replace(part, target)
        return ("downloaded", str(target))
    except Exception:
        if part.exists():
            part.unlink()
        raise


def pdf_metadata(path: Path) -> dict:
    data = path.read_bytes()
    result = {
        "bytes": len(data),
        "size_mb": round(len(data) / 1024 / 1024, 2),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    try:
        reader = PdfReader(path)
        result["pages"] = len(reader.pages)
        snippets = []
        for page in reader.pages[:3]:
            text = (page.extract_text() or "").replace("\x00", " ")
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                snippets.append(text)
        excerpt = " ".join(snippets)[:500]
        if excerpt:
            result["text_excerpt"] = excerpt
    except Exception as exc:
        result["metadata_warning"] = str(exc)
    return result


def write_manifest() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": "2026-08-03",
        "scope": "Official lecture slides, lecture notes, and course-hosted supplements linked from the two course schedules.",
        "courses": COURSES,
    }
    path = OUTPUT_ROOT / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    public_manifest = PUBLIC_ROOT / "manifest.json"
    public_manifest.parent.mkdir(parents=True, exist_ok=True)
    public_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_download(force: bool, workers: int) -> None:
    source_page = OUTPUT_ROOT / "source-pages" / "cs224n-winter-2026.html"
    print("Refreshing the official CS224N schedule...", flush=True)
    fetch_course_page(CS224N["official"], source_page)
    attach_cs224n_readings(source_page)
    records = enrich_paths()
    print(f"Preparing {len(records)} PDF files...", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(download_one, record, force): record for record in records}
        for index, future in enumerate(as_completed(future_map), 1):
            record = future_map[future]
            item = record["item"]
            label = f"{record['course']['id']}/{record['session']['id']}/{item['filename']}"
            try:
                status, _ = future.result()
                print(f"[{index:02d}/{len(records)}] {status:10s} {label}", flush=True)
            except Exception as exc:
                failures.append((label, str(exc)))
                print(f"[{index:02d}/{len(records)}] FAILED    {label}: {exc}", file=sys.stderr, flush=True)

    for record in records:
        path = Path(record["item"]["local_path"])
        if valid_pdf(path):
            record["item"].update(pdf_metadata(path))
            record["item"]["status"] = "available"
        else:
            record["item"]["status"] = "failed"
    manifest = write_manifest()
    print(f"Manifest: {manifest}")
    if failures:
        print("\nDownload failures:", file=sys.stderr)
        for label, error in failures:
            print(f"- {label}: {error}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="download again even when a valid local PDF exists")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_download(args.force, max(1, min(args.workers, 8)))


if __name__ == "__main__":
    main()
