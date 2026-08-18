---
title: "Improving-Biomedical-Abstractive-Summarisation-with-Knowledg"
source: https://aclanthology.org/2023.emnlp-main.40.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:47:30"
field: "生物医学自然语言处理"
keywords: ["biomedical summarization", "citation network", "knowledge aggregation", "abstractive summarization", "pre-trained language models", "BioCiteDB"]
innovations: ["提出基于注意力机制的引文知识聚合框架，动态融合多篇引用文献摘要特征", "构建并发布大规模生物医学引用增强摘要数据集BioCiteDB（>10,000篇）", "实证表明外部引用知识可显著提升模型对领域术语与写作风格的捕获能力"]
benchmarks: ["BioCiteDB", "ROUGE", "BERTScore", "BartScore", "Flesch-Kincaid"]
---

# 论文速读：Improving-Biomedical-Abstractive-Summarisation-with-Knowledg

## 一句话总结
本文提出了一种基于注意力机制的引文知识聚合框架，通过动态融合源论文所引用文献的摘要信息，显著提升预训练语言模型在生物医学抽象文本摘要任务上的性能，并同步构建发布了包含超10,000篇论文的大规模引用增强数据集 BioCiteDB。

## 研究问题与动机
1. **领域知识鸿沟**：生物医学文本高度专业化、术语密集，通用或纯语料预训练语言模型缺乏深层领域背景，难以生成符合专家水准的摘要。
2. **现有方法局限**：既往增强工作多依赖源论文内部的显式标注（如命名实体、关系、生物本体），极少利用引文网络中外部参考文献所蕴含的共享研究背景与写作风格。
3. **数据资源匮乏**：现有生物医学摘要数据集（如 TAC 2014）仅包含313例且无引用关系，制约了引用感知型摘要模型的研究进展。
4. **知识利用低效**：简单拼接单条引文易引入噪声，缺乏对多篇引用摘要的差异化权重分配与动态聚合机制。

## 核心贡献（创新点）
1. **构建 BioCiteDB 数据集**：从 Allen Institute 开源语料中清洗出超10,000条高质量样本，每条附带平均16篇引用文献及结构化引用图，填补了公开引文增强摘要数据的空白。（与以往仅依赖单篇源文档的数据集本质不同）
2. **提出注意力引文聚合框架**：设计 `Tok^CLS` / `Tok^ABS` 特殊标记拼接策略，并通过可学习参数对多篇引用摘要的 CLS 隐藏状态进行加权聚合，实现外部知识的动态注入。（与直接拼接或随机选单条引文的朴素方法本质不同）
3. **全面性能突破与机制验证**：在 Pubmed-BART 基础上实现 ROUGE-1 recall 提升 5.7%、ROUGE-2 提升 8.1%，并在 PPL、BeS、BaS、可读性指标及人工评估四项维度均取得 SOTA；消融实验证实多引文聚合有效抑制了单引文引入的噪声。（与常规 PLM 微调基线本质不同）

## 方法详解
- **任务形式化**：输入源论文文档 $d_i$ 与其引用集合 $D^c=\{d_1^c,...,d_k^c\}$，建模条件概率 $P(Y | X_{doc}, X_{citations})$ 生成摘要序列 $Y$。
- **输入编码**：以 BART 为骨干，使用 BPE 将主文档与 $N$ 篇引用摘要按顺序拼接为序列 $Q_j$，通过特殊 Token `Tok^CLS`（全局上下文）与 `Tok^ABS`（分隔符）标记边界，得到嵌入 $E_{Q_j} = \text{concat}(E_{doc}, E_{abs_j^c})$。
- **知识聚合（Attention Module）**：所有 $Q_j$ 拼接后经 Encoder 得到隐藏状态矩阵 $Q \in \mathbb{R}^{N \times L \times M}$。提取首位置 CLS 向量 $Q^{CLS}$，经线性投影 $Attn\_logits = Q^{CLS} W^Q$ 与 Softmax 得到注意力权重 $Attn \in \mathbb{R}^{N \times 1}$。最终聚合特征为 $F = Attn^T Q \in \mathbb{R}^{L \times M}$，实现对多篇引用信息的选择性融合。
- **自回归解码**：解码器隐状态 $H_t = \text{Decoder}(y_{<t}, F)$，输出词概率 $P(y_t|y_{<t}, F) = \text{softmax}(H_t W^D)$，通过采样/贪心生成 $y_t$。
- **训练目标**：标准词级交叉熵损失 $\mathcal{L} = -\frac{1}{N}\sum_{t=1}^N \log P(y_t | y_{<t}, X)$。

## 实验与结果
- **数据集**：BioCiteDB（Train 9,144 / Val 1,143 / Test 1,143；平均每篇 6.2 个不同引用、16.6 次总引用、529 句、9,858 词）。
- **基线模型**：LEAD-3、ORACLE、ChatGPT (zero-shot)、LED、PEGASUS、BART、Pubmed-LED、Pubmed-PEGASUS、Pubmed-BART。
- **核心结果**：`Pubmed-BART + citation agg.` 在全部 ROUGE-F1 上显著领先，ROUGE-1 F1=0.3522、ROUGE-2 F1=0.0973、ROUGE-L F1=0.3233；PPL 降至 10.54（优于基线 10.80）。补充指标 BeS=83.60、BaS=-3.35、FLK=13.82、CLI=13.20 均最优。
- **关键结论**：引用知识注入使模型 recall 显著提升，表明模型能更充分吸收参考摘要中的相似表达；单引文输入（`-w one citation`）仅带来微弱提升且 PPL 上升，验证了注意力聚合机制在去噪与特征选择上的必要性；随引用数量增加，ROUGE 分数呈单调上升趋势。

## 相关工作脉络
1. **An et al. (2021) / Yasunaga et al. (2019)**：在开放域科学论文摘要中探索引用网络，本文将其垂直细化至生物医学领域并配套专用大规模数据集。
2. **领域本体/实体增强方法（Chandu et al. 2017; Schulze & Neves 2016）**：依赖静态知识库或内部标注，本文转向利用动态文献引文中的语义与风格上下文。
3. **纯语料预训练模型（BioBERT / Pubmed-BART）**：仅通过隐式概率学习领域知识，本文显式引入外部引用摘要作为结构化知识源。
4. **TAC 2014 Biomedical Summarization**：历史基准数据集仅 313 例且无引用关系，本文数据集规模扩大约 30 倍并自带完整引用图结构。
5. **长文档摘要模型（LED / PEGASUS）**：聚焦输入长度扩展与抽取策略优化，本文通过外部知识注入解决领域术语泛化与写作风格迁移问题。

## 局限性与未来方向
1. **仅覆盖抽象式摘要**：受资源与时间限制，未针对抽取式摘要开展对照实验，尽管理论上聚合框架可迁移。
2. **浅层图结构**：仅采用一阶邻居（$hop_{max}=1$）且单篇最多聚合 12 篇引用，未结合 Graph Attention Network 挖掘多层级引文依赖。
3. **尺度扩展受限**：当前实验基于 <5B 参数模型，直接迁移至 Llama-2/Baichuan 等大模型受限于显存与算力。
4. **未来方向**：开发更高效的多跳图特征融合网络；设计主文档与引用文献的联合关键信息抽取机制，进一步提升语言理解与知识压缩能力。

## 研究启发与可借鉴点
1. **外部参考文献作为强相关上下文**：将“引用关系”视为高置信度领域知识源，而非依赖通用知识库或内部实体，为学术文本生成提供了可迁移的知识注入范式。
2. **轻量级 CLS 注意力聚合**：用可学习向量对多条序列的全局表征进行加权融合，避免了拼接导致的序列超长与噪声放大，适用于多文档摘要、检索增强生成（RAG）等场景。
3. **严谨的数据清洗 Pipeline**：Algorithm 1 中的硬性过滤规则（保留 Introduction、≥3 个不同引用、完整 UID/Citation）与 Algorithm 2 的 BFS 引用图遍历，为构建学术引用数据集提供了可复用的工程模板。
4. **多维评估闭环**：同时报告 ROUGE、语义相似度（BeS/BaS）、可读性（FLK/CLI）与专家人工评估，有效规避了单一自动指标的指标钻营风险。

## 关键术语表
- **BioCiteDB**：本文构建的大规模生物医学文献引用摘要数据集，包含源论文、摘要及结构化引用关系。
- **Abstractive Summarisation**：抽象文本摘要，模型基于语义理解重新组织语言生成非原文摘录的摘要。
- **Citation Aggregation**：引文聚合模块，通过注意力机制动态权衡多篇引用摘要对当前生成任务的重要性。
- **Pubmed-BART**：在 Pubmed 海量生物医学语料上进一步预训练的 BART 变体，作为本文骨干网络。
- **ROUGE**：基于 n-gram 重叠的自动评估指标族，广泛用于摘要任务的精度/召回/F1 衡量。
- **BERTScore (BeS) / BartScore (BaS)**：分别基于 BERT 与 BART 语义空间的相似度评估指标，弥补传统 ROUGE 的语义对齐缺陷。
- **Flesch-Kincaid (FLK) / Coleman-Liau Index (CLI)**：基于词汇长度与句长的传统可读性量化指标。

## 可复现要素
- **数据集**：基于 Allen Institute 开源语料构建，论文提供完整构建算法与统计数据，但未提供独立开源仓库链接。
- **代码/权重**：基线权重均源自 Hugging Face 公开 checkpoint（LED, PEGASUS, BART, Pubmed-LED/PEGASUS/BART）；自定义聚合模块未声明独立开源。
- **关键超参**：Epoch=10，Batch Size=16，LR=$1e^{-4}$，Optimizer=Adam；$hop_{max}=1$，$neighbor_{max}=12$；输入按模型限制分块（BART 为 1024 tokens）。
- **评估实现**：ROUGE (`rouge_score`)、BeS (`bert_score`)、BaS (官方 GitHub)、FLK/CLI (`py-readability-metrics`)，均在 Python 环境复现。
