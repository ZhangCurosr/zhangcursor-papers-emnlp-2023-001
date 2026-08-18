---
title: "Learning-the-Visualness-of-Text-Using-Large-Vision-Language"
source: https://aclanthology.org/2023.emnlp-main.147.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:49:54"
---

# 论文速读：Learning-the-Visualness-of-Text-Using-Large-Vision-Language

## 一句话总结
本文提出句子级文本视觉度（Visualness/Imageability）预测任务，通过两阶段微调策略适配CLIP等大规模视觉-语言模型，使其仅凭文本输入即可准确区分视觉与非视觉句子，同时保留嵌入空间对下游文图检索任务的可用性。

## 研究问题与动机
1. **长文本图文匹配的触发盲区**：现有文生图/文图检索模型（如DALL-E、Stable Diffusion）在输入混杂视觉与非视觉句子的长文档时，无法自动判断哪些句子值得配图，导致非视觉句触发无关生成。
2. **词级视觉度不可直接聚合**：Prior work（如MRC词典、Visual Genome启发式）仅在词/短语级别量化图像度，实验证明词级分数简单平均无法有效推导句子级视觉度。
3. **标准VLM微调假设图文双输入**：CLIP/UNITER等模型的下游适配通常需同时提供图像与文本，而本任务推理阶段仅提供纯文本，需改造训练目标以适配单模态输入场景。
4. **弱监督数据噪声需目标设计抑制**：自动构建的大规模图文对齐语料存在边界模糊问题，需设计抗噪对比目标以保证分类边界清晰且表征可复用。

## 核心贡献（创新点）
1. **构建首个句子级视觉度评测基准TIMED**：通过自动规则初筛+9人Likert量表人工标注，提供3,620条高质量英语句子视觉度评分，填补了从词级到句子级视觉度评测的空白。
2. **提出TIP-CLIP修改版对比学习目标**：将非视觉句统一映射至固定NULL图像、视觉句保留原图配对，无需添加任何额外网络结构即可实现纯文本输入的二分类推断。
3. **揭示对比式目标对下游表征复用的保护作用**：对比“全视觉句统一匹配单图”的二分类替代方案，证明保留图文对比学习结构可避免嵌入空间畸变，使微调后模型在文图检索任务上仍保持高MRR（0.937 vs 0.014）。
4. **验证远端监督+人工精调的两阶段训练范式**：第一阶段在48k自动标签语料上预训练，第二阶段在3k人工标注集上校准，相较单阶段训练带来超5个绝对F1提升，兼顾数据规模与标注质量。

## 方法详解
1. **数据构建**：
   - **自动标签阶段**：从Common Crawl 45万PDF中提取段落与图像，用NLTK分句后计算每句与同页图像的CLIP相似度。设定阈值$T_{pos}=0.35$（Top 1%为视觉句）与$T_{neg}=0.18$（Bottom 5%为非视觉句），共获48,077条对齐样本（15,359视觉句+对应原图，32,718非视觉句+NULL图像）。
   - **人工标注阶段**：从另一批20万PDF中筛选宣传册/杂志类文档，按相同阈值策略抽取3,620句，9名AMT标注者按7点Likert量表评分，多数决划分为visual（n=1132）、non-visual（n=2108）与ambiguous。
2. **模型基础**：以CLIP ViT/B-32为骨干，训练阶段保留图文双编码器，推理阶段仅输入文本。
3. **训练目标（TIP-CLIP）**：修改标准InfoNCE对比损失，约束条件明确区分视觉/非视觉样本的图像嵌入选择：
   $$\mathcal{L} = -\frac{1}{2N}\sum_{j=1}^{N}\log\frac{\exp(\langle I_j^e, T_j^e\rangle/\tau)}{\sum_{k=1}^{N}\exp(\langle I_j^e, T_k^e\rangle/\tau)}, \quad I_m^e = \begin{cases} I_{null}^e, & m \in \bar{\mathcal{V}} \\ I_m^e, & m \in \mathcal{V} \end{cases}$$
   其中$\tau$为可学习温度参数，$I_{null}^e$为随机RGB NULL图像嵌入。
4. **两阶段微调流程**：Stage 1在自动标签集上训练5 epoch（lr=$5\times10^{-5}$, Adam, Tesla T4）；Stage 2切换至TIMED训练集继续2 epoch（同超参）。推理时计算文本嵌入与NULL图像嵌入的余弦相似度，视觉度得分$S = 1 - \langle I_{NULL}^e, T^e\rangle$，阈值设为0.79判定类别。
5. **对比基线**：Random、Average MRC-I、VG-Objects浓度启发式、MRC-I+w2v扩展、Fine-tuned BERT分类器、Pre-trained CLIP直接推理。

## 实验与结果
1. **TIMED测试集主实验**：TIP-CLIP取得Macro-F1 0.865、Accuracy 0.871，较次优的Fine-tuned BERT（F1 0.753）提升约11个百分点，较Pre-trained CLIP（F1 0.694）提升近17个百分点。
2. **两阶段训练消融**：仅用自动标签训练F1为0.751，仅用人标注训练F1为0.810；两阶段串联后达0.865，较单阶段人标注训练提升约5个绝对F1点，验证远端监督预训练的有效性。
3. **下游文图检索复用性**：在515条测试集视觉样本上，TIP-CLIP检索MRR为0.937（CLIP为0.989）；而“全视觉句统一
