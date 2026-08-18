---
title: "Coarse-to-Fine-Contrastive-Learning-in-Image-Text-Graph-Spac"
source: https://aclanthology.org/2023.emnlp-main.56.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:09:20"
field: "视觉语言组合推理"
keywords: ["Vision-Language Models", "Compositionality", "Contrastive Learning", "Scene Graph", "Hard Negative Mining", "Coarse-to-Fine Learning", "CLIP"]
innovations: ["场景图引导的文本分解与粗到细多粒度对比学习目标", "基于图变换（节点/边替换、子图拼接）的属性-关系硬负样本挖掘", "引入Tree-Score分析语言编码器的层次化理解能力"]
benchmarks: ["ARO", "CREPE-Systematicity", "SVO-Probes", "VL-Checklist", "ELEVATER", "ImageNet", "COCO/Flickr30K Retrieval"]
---

# 论文速读：Coarse-to-Fine Contrastive Learning in Image-Text-Graph Space for Improved Vision-Language Compositionality

## 一句话总结
本文提出 **MosaiCLIP**，通过场景图分解将文本转化为多个复杂度不同的子图句子，并以"粗到细"的对比学习目标将它们与同一图像匹配，从而显著提升视觉语言模型在属性绑定、关系理解、系统泛化和生产力方面的组合推理能力，同时保持与 CLIP 相当的通用多模态性能。

## 研究问题与动机
1. **CLIP 类模型在组合推理上严重不足**：近年研究（Yuksekgonul 等, 2022; Thrush 等, 2022; Ma 等, 2022）表明，对比训练的 VLMs 难以正确绑定属性到对应对象、理解对象间关系、系统泛化到未见概念组合，以及处理更长的复合句子。
2. **已有改进方法各有缺陷**：NegCLIP 对训练数据质量高度敏感；Teaching SVLC 依赖昂贵的 LLM 做难负样本挖掘；Syn-CLIP 使用合成数据导致与真实数据的域差距，需特殊处理。
3. **场景图可提供结构化语义表征**：场景图以对象、属性、关系构成图结构表示图像/文本语义，是精细层次化理解的有效代理；但图像场景图生成器需要大量监督标注且存在覆盖偏差。
4. **需要一种通用的组合能力提升方案**：旨在开发一种不依赖 LLM/合成数据、适用于所有对比训练 VLMs 的一般性方法，实现"粗到细"的层次化文本理解，从而改善视觉-语言组合能力。

## 核心贡献（创新点）
1. **场景图引导的文本分解与增广框架 + 粗到细对比学习目标**：将文本场景图分解为多个正性子图（不同复杂度），并将它们与同一图像匹配；与 NegCLIP 仅做单句负样本挖据的本质区别在于**利用多粒度子图匹配实现层次化理解**。
2. **面向属性绑定与关系理解的难负样本图变换策略**：提出节点交换/替换、边缘替换、子图拼接三种最小扰动变换（$f_{obj}, f_{attr}, f_{rel}$），生成针对属性与关系的硬负样本；与已有工作相比**不依赖 LLM 或合成数据，直接在图空间构造语义最小扰动负例**。
3. **Image-to-Multi-Text / Text-to-Image 对比损失**：设计一对多正样本的对比学习目标，一个图像可匹配多个子图文本；这是与传统 CLIP 一对一匹配的本质区别，支撑"粗到细"层级对齐。
4. **基于 Tree-Score 的层次化理解分析方法**：首次引入 Tree-Score（Murty 等, 2022）证明 MosaiCLIP 的语言编码器展现出更强的类树状层级计算，从而解释组合能力提升的内在机理；这一分析视角是前人工作未涉及的。
5. **两阶段课程学习 + WiSE-FT 鲁棒微调策略**：第一阶段采样至多 1 个正/负子图以缩小预训练-微调目标差距，第二阶段扩展到多子图；配合 weight-space ensembling 避免微调对预训练表征的破坏。

## 方法详解
### 3.2 图像-文本-图对齐
将标准 CLIP 图像-文本对比学习视为图像场景图 $\mathcal{G}_I$ 与其子图（文本场景图 $\mathcal{G}_T$）之间的隐式匹配：$\mathcal{G}_T \subset \mathcal{G}_I$。由此，$\forall g \in S_{\mathcal{G}_T}$，$(g, \mathcal{G}_I)$ 均为正匹配对。

### 3.3 场景图引导的文本分解
- 使用现成的文本场景图解析器（Wu 等, 2019）生成文本场景图 $G_T = (V_T, E_T)$，视作图像场景图的子图代理。
- 将其分解为 $k \le M$ 个正性子图 $P_g = \{g_1, g_2, ..., g_k\}$，每个子图代表图像的某个部分。
- 通过模板将子图转为句子，例如两节点+关系模板 `"{N1} {R} {N2}"`，形成正文本集 $P_t$。

### 3.4 难负样本子图构建
基于外部对象集 $\mathcal{N}$（1594 个）、属性集 $\mathcal{A}$（524 个）、关系集 $\mathcal{R}$（50 个）定义三种图变换：
- $f_{obj}$：对单对象子图的每个属性进行随机替换/洗牌。
- $f_{attr}$：对含关系的子图的属性节点进行替换/洗牌。
- $f_{rel}$：对对象节点洗牌/替换、关系边替换，并将正子图与从外部随机采样的子图拼接。
将 $P_g$ 经变换生成负子图集 $N_g$，再转文本得 $N_t$，作为批次内的硬负样本。

### 3.5 粗到细对比学习损失
对批处理中的图像-文本对：
- **Image-to-Multi-Text loss**：
$$\mathcal{L}_{i2t}^{MC} = -\sum_{i=1}^{|B_I|} \frac{1}{|P(i)|} \sum_{k \in P(i)} \log \frac{\exp(\tau \mathbf{u}_i^T \mathbf{v}_k)}{\sum_{j=1}^{|B_t|} \exp(\tau \mathbf{u}_i^T \mathbf{v}_j)}$$
其中 $P(i) = \{k | g_k \subseteq \mathcal{G}_i\}$ 为图像 $i$ 对应的正子图文本集合。
- **Text-to-Image loss**（仅在正文本上计算）：
$$\mathcal{L}_{t2i}^{MC} = -\sum_{j=1}^{|B_t^{pos}|} \log \frac{\exp(\tau \mathbf{u}_{p(j)}^T \mathbf{v}_j)}{\sum_{i=1}^{|B_I|} \exp(\tau \mathbf{u}_i^T \mathbf{v}_j)}$$
总损失：$\mathcal{L}_{MosaiCLIP} = (\mathcal{L}_{t2i}^{MC} + \mathcal{L}_{i2t}^{MC}) / 2$。

### 3.6 两阶段课程学习与鲁棒微调
- **阶段一**：每图像至多采样 1 个正/负子图，缩小与预训练目标的差距。
- **阶段二**：扩展到多个正/负子图。
- **WiSE-FT**：对视觉编码器进行 fine-tuning 前后的权重空间集成，防止在组合任务提升的同时损失通用多模态性能。

## 实验与结果
### 数据集与基线
- **组合推理基准**：ARO（属性/关系/词序）、CREPE-Systematicity（系统泛化）、SVO（动词理解）、VL-Checklist（对象/属性/关系）
- **通用任务基准**：ELEVATER（20 个数据集）、ImageNet、COCO/Flickr30k 检索
- **基线**：CLIP、CLIP-FT、NegCLIP、Teaching SVLC（LLM-based）、Syn-CLIP（合成数据）
- **微调数据**：COCO、CC-FT（100K）、YFCC-FT（100K）、CC3M
- **预训练数据**：CC-12M、YFCC-15M
- **骨干网络**：OpenAI CLIP-ViT-B/32、OpenCLIP RN-50、Swin-Tiny

### 主要结果（微调）
- 相对 NegCLIP：**系统泛化 +18%**、关系理解 +16.5%、属性理解 +5.3%、词序理解 +32.3%（ARO 数据集）。
- 相对 CLIP：**CREPE +11.5%、ARO +9.1%**，多项基准提升超过 20%。
- ELEVATER 分类任务平均 **+3.3%（vs NegCLIP）**、**+6.3%（vs CLIP）**，检索任务 Recall@1 **+5.4 点**。

### 主要结果（预训练）
- CREPE CU（未见过组合）最高提升达 **42.5%**（vs NegCLIP）；属性理解 +8.3%，关系理解 +12.0%。
- MosaiCLIP 在不同预训练数据集上表现更稳健，NegCLIP 在 YFCC-15M（噪声更大）上性能下降明显。
- 仅使用 **0.3x 预训练/微调数据** 即可达到或超越 NegCLIP 性能。

### 泛化到高级 VLM（BLIP）
- 在 COCO 微调后，ARO 平均成绩：**BLIP 66.0 → BLIP-FT 68.7 → BLIP+NegCLIP 72.6 → BLIP+MosaiCLIP 77.0**，关系理解提升达 **+6.3%**（vs NegCLIP）。

### 消融与分析
- **分解 ablation**：去掉 $f_{rel}$ 或 $f_{attr}$ 均导致 ARO 下降，二者均有助于关系与属性理解。
- **课程学习 ablation**：MosaiCLIP vs MosaiCLIP$_{NoCurric}$ 在 ARO 平均提升约 1-2%。
- **Tree-Score**：MosaiCLIP 语言编码器 Tree-Score 显著高于 NegCLIP，证明其层级理解能力提升。
- **编码器解耦分析**：冻结视觉/语言编码器发现，语言编码器改进幅度始终大于视觉编码器；MosaiCLIP 在不使用图像负样本挖掘的情况下仍显著改善视觉端表现。
- **鲁棒性**：在 YFCC-FT（高噪声）微调时，NegCLIP 性能下降 >10%，MosaiCLIP 保持稳定；预训练模型在 ImageNet 自然分布偏移上亦优于 NegCLIP。

## 相关工作脉络
1. **CLIP (Radford 等, 2021)**：大规模对比预训练的 VLM 基线，本文在其基础上改进组合推理，但不改变其双塔对比框架。
2. **NegCLIP (Yuksekgonul 等, 2022)**：通过图像和文本的难负样本挖掘提升组合能力；本文认为其依赖数据清洁度且在未见组合上泛化有限，MosaiCLIP 通过多子图匹配提供更强层次监督。
3. **Teaching SVLC (Doveh 等, 2023)**：使用 LLM 生成难负样本；本文避免 LLM 调用，以轻量图变换替代，降低计算与成本开销。
4. **Syn-CLIP (Cascante-Bonilla 等, 2023)**：基于 Unity3D 合成百万级图像-文本对；本文指出合成数据存在域差距且需特殊处理，而 MosaiCLIP 在自然数据上达到或超越其效果。
5. **CREPE / ARO / VL-Checklist / SVO**：均为评估 VLM 组合推理的新基准，本文在这四个基准的 11 个子数据集上进行全面评测。
6. **Tree-Score (Murty 等, 2022)**：本文借用该指标量化语言编码器的树状计算程度，首次用于解释 VLM 组合能力的内在机制。

## 局限性与未来方向
1. **每批次计算成本更高**：由于子图分解导致有效文本批次增大，需更多计算；作者建议通过减少数据量（0.3x-0.6x）换取性能-计算权衡，并计划在后续工作中设计更低开销的粗到细目标。
2. **对更先进 VLM 的验证不足**：目前主要针对 CLIP 类双塔模型；虽然验证了 BLIP 并见成效，但对 X-VLM、LXMERT 等跨模态交互模型的全面评估仍待开展。
3. **句子模板较为简单**：当前使用手工模板将子图转为句子，可能产生"人造感"文本；未来可引入 GPT-4、BLOOM 等 LLM 直接从场景图生成自然语言句子，但需权衡计算开销。

## 研究启发与可借鉴点
1. **"场景图分解 + 模板生成"的数据增广范式**：无需 LLM 或合成数据即可从现有文本中派生多粒度正/负样本，可迁移至其他视觉-语言组合能力提升任务。
2. **粗到细（Coarse-to-Fine）层次化对比学习**：将单一文本匹配扩展为多层级子图匹配的思想，可推广到多粒度特征对齐、跨模态检索等场景。
3. **图变换生成难负样本的策略**（节点/边替换、子图拼接）：可作为一般性的"最小扰动负例"构造框架，应用于其他依赖对比学习的多模态模型。
4. **Tree-Score 分析框架**：可用于解释/诊断 VLM 的语言理解能力，作为组合推理改进方法的定量分析工具。
5. **课程学习缓解预训练-微调目标偏移**：两阶段从"少子图"到"多子图"的微调策略，对任何引入额外训练信号的方法均有参考价值；配合 WiSE-FT 可实现组合能力与通用性能的兼得。

## 关键术语表
**Scene Graph（场景图）**：用图结构表示图像语义的形式，节点为对象/属性，边为对象间关系。
**Coarse-to-Fine Contrastive Learning（粗到细对比学习）**：将同一图像与不同复杂度的多个文本子图进行匹配的对齐训练策略。
**Tree-Score（树评分）**：衡量 Transformer 语言编码器在句法处理中呈现类树状层级计算程度的指标。
**Systematic Generalization（系统泛化）**：模型将已学概念重组并泛化到未见组合的能力。
**Hard Negative Mining（难负样本挖掘）**：在训练批次中引入与正样本高度相似但语义错误的样本以提升判别力。
**WiSE-FT（Weight Space Ensemble Fine-Tuning）**：通过 fine-tuning 前后视觉编码器权重的空间集成来缓解微调带来的表征遗忘。
**CREPE / ARO / VL-Checklist / SVO**：评估 VLM 组合推理能力的四个主流基准测试。

## 可复现要素
- **数据集**：预训练数据 CC-12M、YFCC-15M（公开）；微调数据 COCO、CC-FT、YFCC-FT、CC3M（公开）；基准测试 ARO、CREPE、VL-Checklist、SVO、ELEVATER、ImageNet（公开）；外部集合 $\mathcal{N}, \mathcal{A}, \mathcal{R}$ 来自 Visual Genome（公开）。
- **代码/权重**：论文未明确提供开源链接，但使用了 OpenCLIP、NegCLIP 官方代码与 OpenAI CLIP-ViT-B/32、Swin-Tiny、ResNet-50 公开权重。
- **关键超参**：
  - 微调：5 epochs，batch size=256，cosine LR，warmup=50 steps，初始 LR=1e-5，AdamW；每图像最多 3 个正子图 + 6 个负子图。
  - 预训练：32 epochs，batch size=4096，warmup=5000 steps，初始 LR=1e-3，weight decay=0.1。
  - 负样本采样概率：$p_2 = p_3$，$p_1 \in \{0, 0.08, 0.15\}$（微调阶段按 ARO val 调优）。
  - 温度参数 $\tau$：可学习。
  - 硬件：预训练 64× NVIDIA A100，微调 4× NVIDIA A100。
