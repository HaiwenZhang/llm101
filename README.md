# 大模型系统课 · LLM 101

面向零基础学习者的大语言模型系统教程。内容从 Token、向量、Attention 和 Transformer 开始，延伸到预训练、Scaling、分布式训练、后训练与强化学习、高效推理、RAG、Agent、多模态、评测、安全和部署，并以 Kimi K3 作为综合案例。

## 本地运行

```bash
npm ci
npm run docs:dev
```

打开 `http://localhost:5173/`。

## 构建

```bash
npm run docs:build
npm run docs:preview
```

GitHub Pages 通过 `.github/workflows/deploy.yml` 自动发布到：

<https://haiwenzhang.github.io/llm101/>

## PDF 资料

课程 Slides、论文和书籍 PDF 统一存放在本地 `resources/`，该目录不会提交到 Git。公开教程使用课程官网、出版方或 arXiv 链接，仓库只包含教程正文、教学图片、交互实验、可执行讲义和资料索引。

`npm run content:build` 是维护者使用的内容再生成命令，需要本地解析语料；普通阅读、开发和 GitHub Actions 构建不需要这些大型文件。
