---
title: "MProto-Multi-Prototype-Network-with-Denoised-Optimal-Transpo"
source: https://aclanthology.org/2023.emnlp-main.145.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:50:35"
field: "远程监督命名实体识别"
keywords: ["distantly supervised NER", "prototype network", "optimal transport", "denoising", "intra-class variance", "label noise"]
innovations: ["多原型分类器刻画实体类内方差", "去噪最优传输算法识别不完全标注噪声"]
benchmarks: ["CoNLL03", "BC5CDR"]
---

# 论文速读：MProto: Multi-Prototype Network with Denoised Optimal Transport for Distantly Supervised Named Entity Recognition

## 一句话总结
本文提出一种噪声鲁棒的原型网络 MProto，用于远程监督命名实体识别（DS-NER）任务。通过多原型表征类内方差，并结合去噪最优传输（DOT）算法缓解不完全标注噪声，在多个 DS-NER 基准上达到 SOTA 性能。

## 研究问题与动机
- **远程监督 NER 的标签噪声问题**：DS-NER 仅依赖知识库/词典和无标注语料自动生成标注，存在两类噪声——（1）未收录实体的"不完全标注"（被错误标为 O 类）；（2）同类实体因语义差异形成的类内方差。
- **现有方法不足**：AutoNER、负采样、MPU 等方法主要处理 O 类噪声，但未考虑类内方差；单一原型分类器无法刻画同一实体类型的语义子簇。
- **目标**：设计能同时建模类内方差和抵抗不完全标注噪声的 DS-NER 框架。

## 核心贡献（创新点）
1. **多原型分类器（Multi-Prototype Classifier）**：每个实体类别用 M 个原型表示，通过 token-原型相似度进行分类，刻画类内方差——与单原型方法相比，能覆盖同一实体类型的多个语义子簇。
2. **最优传输 token-原型分配**：将 token 到 ground-truth 原型的分配建模为 OT 问题，使用 Sinkhorn-Knopp 算法高效求解，避免所有 token 集中于单一原型的退化解。
3. **去噪最优传输（DOT）算法**：针对 O 类 token 设计，允许所有原型参与分配；通过分配结果区分真实负样本（分配到 O 原型）和未标注实体 token（分配到实体原型），后者不参与损失计算——与负采样/PU-learning 等直接过滤 O 类的方法不同，DOT 利用特征相似性主动识别噪声。
4. **紧凑损失（Compactness Loss）**：额外优化 token 与 ground-truth 原型之间的绝对距离，增强同类子簇内的特征紧凑性。

## 方法详解
- **Token 编码**：使用预训练语言模型（BERT-base-cased / BioBERT-base-cased）提取 token 特征 $\mathbf{H} = f_\theta(X)$。
- **多原型分类**：每类 $c$ 有 $M$ 个原型 $\mathcal{P}_c = \{\mathbf{p}_{c,1}, \dots, \mathbf{p}_{c,M}\}$，预测类别为 $\hat{c}_i = \arg\max_c \max_m \, s(\mathbf{h}_i, \mathbf{p}_{c,m})$，其中 $s$ 为余弦相似度。
- **Token-原型分配（实体 token）**：对标签为 $c$ 的 token 集合 $\mathcal{T}_c$，求解 OT 问题 $\min_\gamma \sum_{i,j} \gamma_{i,j}^c \mathrm{d}(\mathbf{h}_i, \mathbf{p}_{c,j})$，约束为每 token 分配一个原型、每原型分配均衡数量的 token（均匀分布）。用 Sinkhorn-Knopp 算法求解（正则化权重 $\lambda^r=0.001$，迭代 100 次）。
- **去噪最优传输（O 类 token）**：O 类 token 与全部 $KM$ 个原型求解 OT，其中 O 原型分配比例设为 $\beta$（BC5CDR 取 0.01，CoNLL03 取 0.05）。分配到非 O 原型的 token 视为噪声，不参与损失。
- **损失函数**：交叉熵损失 $\ell^{\mathrm{CE}}$ + 紧凑损失 $\ell^{\mathrm{c}} = \sum_i(1-s(\mathbf{h}_i,\mathbf{p}_{c_i,m}))^2$，总损失 $\ell = \ell^{\mathrm{CE}} + \lambda^{\mathrm{c}}\ell^{\mathrm{c}}$（$\lambda^{\mathrm{c}} \in [0.01, 0.1]$）。
- **原型更新**：使用指数移动平均（EMA）$\mathbf{p}_{c,m}^t = \alpha \mathbf{p}_{c,m}^{t-1} + (1-\alpha)\frac{\sum_{i\in\mathcal{T}}\mathbf{h}_i}{|\mathcal{T}|}$，$\alpha \in [0.5, 0.9]$。

## 实验与结果
- **数据集**：CoNLL03（通用域，4类）和 BC5CDR（生物医学域，Chemical/Disease），各生成 4 种远程标注版本（不同词典覆盖率/KB匹配）。
- **评估指标**：span 级 F1（精确率/召回率）。
- **主要结果**（F1 分数）：

| 方法 | BC5CDR (Big Dict) | BC5CDR (Small Dict) | CoNLL03 (KB) | CoNLL03 (Dict) |
|------|-------------------|---------------------|--------------|----------------|
| Conf-MPU（前 SOTA） | 77.22 | 71.85 | 79.16 | 81.89 |
| **MProto** | **81.47** (+4.25) | **74.30** (+2.45) | **79.58** (+0.42) | **82.49** (+0.60) |

- **提升幅度**：相比前 SOTA，在四个数据集上分别提升 +4.25%、+2.45%、+0.42%、+0.60% F1；在所有字典覆盖率（20%-100%）下均表现稳健，词典覆盖率降至 20% 时仅下降 7-9%，而负采样/早停法下降 47-56%。
- **消融结果**：移除多原型 -0.95%~-1.26%；移除紧凑损失 -2.23%~-0.29%；移除 DOT -21.89%~-34.14%（DOT 贡献最大）。

## 相关工作脉络
- **AutoNER**：重新设计 tag 方案将 token pair 分为 tie/break/unknown，unknown 对不参与损失；本文用 DOT 替代，无需修改标注体系。
- **负采样（Negative Sampling, Li et al. 2021/2022）**：随机采样部分 O 类 token 参与训练；本文 DOT 基于特征相似度主动筛选噪声，效果更优。
- **MPU / Conf-MPU（Zhou et al. 2022）**：将 O 类视为 unlabeled 并用 PU-learning 无偏估计风险；本文从原型相似度角度识别噪声，思路不同。
- **早停法（Early Stopping, Liang et al. 2020）**：训练早期停止以防过拟合噪声；本文在模型结构层面设计去噪机制，不依赖训练策略。
- **单原型分类网络**：传统 Prototypical Network 每类一个原型；本文扩展为多原型，首次将类内方差建模引入 DS-NER。

## 局限性与未来方向
- 仅验证于 DS-NER，未探索 Few-shot NER 和全监督 NER 场景。
- 原型数量 M 需通过实验选取（本文最优 M=3），缺乏自适应确定 M 的机制。
- 未处理嵌套实体（nested NER）场景。
- 未来方向：迁移到少样本 NER、全监督 NER，以及开发自动选择 M 的方法。

## 研究启发与可借鉴点
- **多原型思想**：类内方差建模可迁移至关系抽取、事件提取等序列标注任务中特征空间存在多子簇的场景。
- **OT 分配框架**：Sinkhorn-Knopp 求解的高效性使 token-原型分配可集成到端到端训练中，适合需要软分配的对比学习/度量学习任务。
- **DOT 去噪思路**：利用"未过拟合前真实样本仍与真实类原型相近"的观察来识别噪声，可推广至其他存在不完全标注的弱监督场景（如弱监督序列标注、远程关系抽取）。
- **紧凑损失设计**：$\ell^{\mathrm{c}} = \sum(1-s)^2$ 可直接迁移到任何基于原型的分类器中增强类内凝聚性。
- **低词典覆盖率下的鲁棒性验证**：通过变化词典覆盖率（20%-100%）测试方法鲁棒性的实验设计值得借鉴，可用于评估其他 DS-NER 方法的泛化能力。

## 关键术语表
- **Distantly Supervised NER（DS-NER）**：仅需知识库/词典和无标注语料即可自动生成长实体标注的命名实体识别范式。
- **Incomplete labeling noise**：因词典覆盖不全导致真实实体被错误标注为 O 类的噪声类型。
- **Intra-class variance**：同一实体类别内部因语义差异形成的多个子簇现象。
- **Optimal Transport（OT）**：研究如何将一个分布"最优"地传输到另一个分布的数学框架，此处用于 token-原型分配。
- **Sinkhorn-Knopp algorithm**：求解带熵正则化的 OT 问题的迭代算法，可通过 GPU 高效加速。
- **Denosed Optimal Transport（DOT）**：本文提出的 O 类 token 分配算法，通过允许所有原型参与分配来区分噪声与真实负样本。
- **Prototype**：每个实体类别的代表性特征向量，用于衡量 token 与其的相似度。
- **EMA（Exponential Moving Average）**：原型更新策略，对历史原型和当前批次均值做加权平均。

## 可复现要素
- **数据集**：CoNLL03 和 BC5CDR（公开），远程标注数据按论文附录方法生成。
- **代码**：已开源，地址见论文摘要中的 GitHub 链接。
- **关键超参**：M=3，$\lambda^c \in \{0.01, 0.05, 0.1\}$，$\alpha \in \{0.5, 0.9\}$，$\beta \in \{0.01, 0.05\}$，$\lambda^r=0.001$，学习率 0.0001，batch size=32，10 epochs。
- **模型 backbone**：CoNLL03 用 BERT-base-cased，BC5CDR 用 BioBERT-base-cased。
- **硬件**：单张 RTX-3090（24G）。
