---
title: "Self-Influence-Guided-Data-Reweighting-for-Language-Model-Pr"
source: https://aclanthology.org/2023.emnlp-main.125.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:30:47"
field: "语言模型预训练与数据选择"
keywords: ["language model pre-training", "data reweighting", "self-influence", "TracIn", "multilingual pre-training", "XTREME"]
innovations: ["首次将自影响力SI分数应用于无标签预训练数据重加权", "提出两阶段在线微批次重加权策略（先直接后逆加权）", "证明SI分数可有效识别预训练数据中的噪声和域不匹配样本"]
benchmarks: ["XTREME", "XQuAD", "MLQA", "TyDi QA", "XNLI", "WikiAnn NER"]
---

# 论文速读：Self-Influence-Guided-Data-Reweighting-for-Language-Model-Pre-training

## 一句话总结
本文提出了 PRESENCE（Pretraining data Re-weighting with Self-influence），首次将自影响力（Self-Influence, SI）分数应用于语言模型预训练阶段的数据重加权。该方法通过两阶段策略（先放大高SI样本、后抑制高SI样本）实现在线联合重加权，在多语言下游任务上显著提升预训练效果。

## 研究问题与动机
- **预训练数据同等对待的局限性**：现有大语言模型预训练将所有语料样本视为同等重要，但网络爬取数据的质量差异显著（含噪声、域不匹配等），忽略数据质量并非最优策略。
- **现有方法不适用于预训练**：已有的基于模型的数据选择和重加权方法均面向监督学习设定，依赖带标签的验证集、代理模型或loss/不确定性信号，无法直接迁移至无标签预训练场景。
- **预训练验证性能≠下游性能**：预训练阶段的验证loss与下游任务表现无直接相关性，使得传统离线过滤策略难以评估效果。
- **大规模预训练数据选择昂贵**：使用代理模型对海量预训练数据进行离线筛选成本高昂（Liu et al., 2022），需更高效的在线方法。

## 核心贡献（创新点）
1. **首次系统探索预训练数据重加权**：建立了自影响力（SI）分数与预训练数据质量（噪声文本、域不匹配）之间的关系，证明SI分数可有效识别低质量样本。
2. **提出 PRESENCE-Sequential 离线过滤方法**：利用SI分数对大规模网页语料进行顺序过滤，在XTREME多语言基准上优于随机采样基线。
3. **提出 PRESENCE 在线联合重加权框架**：设计了基于微批次（microbatch）的两阶段SI重加权策略（初期直接加权τ>0促进 novelty，后期逆加权τ<0抑制噪声），显著提升预训练稳定性和下游性能。
4. **验证了微批次重加权的优越性**：相比逐样本SI计算，微批次级别SI评分更平滑稳定，为大模型预训练提供了可扩展的在线重加权范式。

## 方法详解

### 1. 自影响力（SI）分数计算
采用 TracIn（Pruthi et al., 2020）计算样本对自身的影响：
$$\text{TracInSI}(f_\theta, z) = \mathbf{g}(f_\theta, z) \cdot \mathbf{g}(f_\theta, z)$$
其中 $\mathbf{g}(f_\theta, z) = \nabla l(f_\theta, z)$ 为样本梯度。为提升计算效率，仅使用编码器第一层和解码器第一层的梯度：
$$\text{TracInSI}_\mathcal{K}(f_\theta, z) = \sum_{k \in \mathcal{K}} \text{TracInSI}(f_{\theta,k}, z)$$

### 2. PRESENCE-Sequential 离线过滤
1. 随机采样训练一个 SI 评分模型 $F_\theta$（mT5-base，200K steps）。
2. 对大规模语料库 $D'$ 中所有样本计算 SI 分数。
3. 按SI升序排序，剔除最高 SI 的 $N'-N$ 个样本，保留 $N$ 个低SI样本构建过滤后预训练集 $D$。

### 3. PRESENCE 在线两阶段重加权
- **归一化与加权**：对微批次内样本SI分数 $S$ 进行标准化后，用softmax计算权重：
$$w_i = \frac{e^{\tau \cdot s_i}}{\sum_{s_j \in S} e^{\tau \cdot s_j}}$$
梯度更新：$G = \sum_{z_i \in B} w_i \cdot g(f_\theta, z_i)$

- **两阶段策略**：
$$\tau = \begin{cases} \tau_1 > 0, & i \leq I \\ \tau_2 < 0, & i > I \end{cases}$$
  - 第一阶段（direct）：τ=1，放大高SI样本梯度，推动模型学习 novelty；
  - 第二阶段（inverse）：τ=-1，抑制高SI样本（噪声/异常），稳定训练。
  - 切换步数 $I = 100{,}000$（总1M steps的一半）。

- **微批次实现**：将 minbatch 分为 $n$ 个 microbatch，计算每个 microbatch 的SI后重加权梯度（Algorithm 2）。

## 实验与结果

### 实验设置
- **数据集**：mC4（预训练），XTREME基准（评估：XQuAD、MLQA、TyDi QA、XNLI、WikiAnn NER）
- **模型**：mT5-base（batch=1024）、mT5-large（batch=512）
- **训练**：1M steps，学习率1.0，warmup 10K，inverse square root decay
- **评估**：跨语言 zero-shot 迁移 与 translate-train

### 主要结果

**Table 2 - PRESENCE-Sequential vs 基线（translate-train）**：
| 模型 | XQuAD F1 | MLQA F1 | TyDi QA F1 | XNLI Acc | WikiAnn Span-F1 |
|------|---------|---------|-----------|---------|----------------|
| mt5-base* | 78.26 | 65.45 | 52.75 | 76.76 | 80.86 |
| mT5+PRESENCE-Sequential | **78.96** | **66.04** | **57.65** | **71.22** | **44.63** |

**Table 3 - PRESENCE vs 基线（translate-train，mT5-large）**：
| 模型 | XQuAD F1 | MLQA F1 | TyDi QA F1 | XNLI Acc | WikiAnn Span-F1 |
|------|---------|---------|-----------|---------|----------------|
| mt5-large* | 78.76 | 64.33 | 59.95 | 77.56 | 79.45 |
| mT5-large+PRESENCE | **83.15** | **70.30** | **69.04** | **79.72** | **77.26** |

- **提升幅度**：mT5-large+PRESENCE 在 XQuAD 上 +4.39 F1，TyDi QA 上 +9.09 F1
- **两阶段策略有效性**（Table 5）：完整 PRESENCE 优于单一 direct/inverse 加权及反转顺序
- **在线 vs 离线**（Figure 4）：PRESENCE 与 PRESENCE-Sequential 性能相当，且计算更高效

## 相关工作脉络
1. **TracIn 与自影响力**（Pruthi et al., 2020; Yeh et al., 2018）：原有方法用于监督学习的 outlier/noisy sample 检测，本文首次扩展至无标签预训练场景。
2. **数据选择与过滤**（Raffel et al., 2020; Wenzek et al., 2020; Xie et al., 2023 DoReMi）：依赖人工规则或额外代理模型计算域权重，计算成本高；本文提出在线自适应方案。
3. **监督学习中的数据加权**（Swayamdipta et al., 2020; Mindermann et al., 2022 RHO-loss; Paul et al., 2021）：依赖标签和验证集，难以直接用于预训练。
4. **多语言预训练**（Conneau et al., 2020 XLS-R; Xue et al., 2021 mT5）：关注低资源语言数据增强，本文从样本质量角度改进预训练数据利用效率。
5. **影响力函数应用**（Koh & Liang, 2017; Guu et al., 2023 Simfluence）：主要用于模型解释性和联邦学习隐私分析，本文开拓其在预训练数据重加权中的应用。

## 局限性与未来方向
- **计算开销**：相比无重加权的基线训练时间增加约 30%（尽管远少于离线过滤方法）。
- **超参数依赖**：两阶段切换步数 $I$ 和温度 τ 的选取较 ad-hoc，缺乏自动化策略。
- **SI计算仅用浅层**：为效率仅使用 encoder/decoder 第一层梯度，可能损失部分代表性信息。
- **未来方向**：
  - 探索跨语言、跨域、跨数据源的 SI 分数关系；
  - 设计基于训练 loss、SI 分数的自动化温度调度策略；
  - 利用微批次梯度独立性进一步优化并行计算。

## 研究启发与可借鉴点
1. **两阶段重加权策略可迁移**：预训练早期的"探索"与后期的"稳定"需求可通过正负温度切换实现，此思路可应用于其他预训练优化场景。
2. **微批次级SI计算平衡效率与效果**：相比逐样本计算，微批次级别SI更平滑且计算可控，为大模型预训练中的数据加权提供了可扩展范式。
3. **SI分数作为数据质量指标的创新验证**：首次证明SI可区分噪声/域不匹配文本，这一洞察可用于其他无监督预训练的数据清洗任务。
4. **在线联合训练 vs 离线两步法的权衡**：PRESENCE 证明在线方法可在保持效果的同时大幅降低计算成本，为大规模预训练数据策略设计提供新思路。

## 关键术语表
- **Self-Influence (SI)**：样本对自身预测的影响程度，通过自身梯度与自身的内积计算，用于衡量样本重要性。
- **TracIn**：一种可扩展的一阶梯度近似方法，通过追踪训练过程中梯度的累积变化来估计样本影响力。
- **PRESENCE**：Pretraining data Re-weighting with Self-influence 的缩写，本文提出的在线数据重加权方法。
- **XTREME**：大规模多语言多任务基准测试套件，涵盖问答、句子对、结构预测等任务。
- **mC4**：Multilingual Common Crawl，包含101种语言的网页文本预训练数据集。
- **两阶段重加权**：预训练初期使用正温度放大高SI样本，后期使用负温度抑制高SI样本的策略。
- **Microbatch**：将大 mini-batch 划分为更小批次以节省显存并支持在线SI计算的训练单元。
- **Translate-train**：微调时同时使用英语训练数据和目标语言翻译数据的评估设置。

## 可复现要素
- **数据集**：mC4（公开）、XTREME基准（公开）
- **代码**：论文未明确声明开源，使用 T5X/seqio 框架
- **模型权重**：未公开
- **关键超参**：
  - 预训练步骤：1,000,000（PRESENCE）/ 200,000（PRESENCE-Sequential）
  - 学习率：1.0，warmup 10,000 steps
  - 微批次划分：mT5-base 用 n=8，mT5-large 用 n=4
  - 两阶段切换点：I = 100,000 steps
  - 温度：τ₁ = 1，τ₂ = -1
  - SI计算层：encoder第一层 + decoder第一层
  - Loss归一化因子：mT5-base 为 234,496，mT5-large 为 117,248
