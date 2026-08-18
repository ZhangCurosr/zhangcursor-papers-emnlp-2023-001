---
title: "Mitigating-Backdoor-Poisoning-Attacks-through-the-Lens-of-Sp"
source: https://aclanthology.org/2023.emnlp-main.60.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:51:04"
field: "NLP安全与鲁棒性"
keywords: ["backdoor poisoning", "spurious correlation", "z-score", "textual backdoor defense", "data poisoning mitigation"]
innovations: ["首次将后门中毒攻击建模为虚假相关现象，提出z-score统计检测框架", "无需外部模型和干净验证集的轻量级无模型防御（Z-TOKEN/Z-TREE/Z-SEQ）", "双特征（词汇+句法）联合过滤显著提升对Syntactic等复杂攻击的检测能力"]
benchmarks: ["SST-2", "OLID", "AG News", "QNLI"]
---

# 论文速读：Mitigating-Backdoor-Poisoning-Attacks-through-the-Lens-of-Spurious-Correlation

## 一句话总结
本文首次从**虚假相关（spurious correlation）**视角分析文本后门中毒攻击，提出通过计算词汇（unigram）与句法特征（祖先路径）的 **z-score** 来检测并过滤中毒样本；该方法无需外部模型或干净验证集，在 BadNet 和 InsertSent 攻击上实现近乎完美防御，整体显著优于现有基线。

## 研究问题与动机
- 大规模无监督/弱标注数据被广泛用于预训练和微调，容易受到恶意**数据中毒（data poisoning）**攻击，在训练集中植入触发词并篡改标签，导致模型在触发器出现时输出目标错误标签。
- 现有防御方法多依赖在中毒数据上训练模型（PCA、Clustering、DAN）或需要外部大语言模型（ONION），且对复杂攻击（如 Syntactic）防御效果不足；而中毒攻击的本质——将特定文本特征与目标标签强绑定——与 NLP 中经典的**虚假相关**现象高度同构。
- 现有工作（如 Manoj & Blum, 2021）虽短暂提及后门触发器与虚假相关的联系，但未提出具体防御方案，缺乏系统性的检测和过滤机制。

## 核心贡献（创新点）
1. **视角创新**：首次系统性地建立后门中毒攻击与虚假相关之间的理论联系，论证触发器本质上是与目标标签形成异常强关联的虚假特征，可被 z-score 检测。
2. **轻量级无模型防御框架（Z-defence）**：仅基于 z-score 统计量过滤中毒样本，无需训练模型、无需外部干净验证集或大型预训练语言模型，相比 PCA/Clustering/DAN/ONION 开销极低。
3. **双特征检测机制（Z-TOKEN + Z-TREE）**：提出同时利用词汇特征（unigram/bigram）和句法特征（ constituency tree 祖先路径）计算 z-score，在保留高识别率的同时降低假阳性。
4. **实证验证多数据集多攻击**：在 4 个数据集（SST-2、OLID、AG News、QNLI）上对 3 类攻击（BadNet、InsertSent、Syntactic）全面评估，证明 Z-SEQ 在插入类攻击上实现接近零 ASR 的防御，整体平均 ASR 最低。

## 方法详解
- **问题建模**：后门中毒将子集 $\mathcal{S} \subseteq \mathcal{D}$ 的实例 $(x_i, y_i)$ 经中毒函数 $f(\cdot)$ 变为 $(x'_i, y'_i)$，其中 $x'_i$ 含触发器、$y'_i$ 为攻击者目标标签。
- **虚假相关度量（z-score）**：借鉴 Gardner et al. (2021)，对于特征 $a$（如某 unigram 或祖先路径）：
  - 令 $\hat{p}(y|a) = \frac{\sum_i \mathbb{1}(a \in x_i) \cdot \mathbb{1}(y_i=y)}{\sum_i \mathbb{1}(a \in x_i)}$ 为条件频率；
  - 假设先验 $p(y|a) \approx p(y)$（从训练集估计），计算：
    $$z^* = \frac{\hat{p}(y|a) - p(y|a)}{\sqrt{p(y|a) \cdot (1-p(y|a))/n}}$$
  - $|z^*|$ 极大（实验中取 >18σ）时判定为异常触发特征。
- **特征设计**：
  - **Z-TOKEN**：对训练集中每个 unigram 计算 z-score，过滤包含高 z-score 词汇的样本。
  - **Z-TREE**：基于 constituency tree，提取从 ROOT 到 pre-terminal 节点的**祖先路径**（如 `ROOT→NP→ADJP→RB`），计算其 z-score 并过滤。
  - **Z-SEQ**：先执行 Z-TREE、再执行 Z-TOKEN 的顺序过滤；另有 **Z-SEQ (union)** 变体（任一特征命中即过滤）。
- **阈值选择**：通过 z-score 分布观察，极端离群点集中在 18σ 附近，该阈值固定且**无需在中毒数据上调参**。

## 实验与结果
- **数据集**：SST-2（2类情感）、OLID（2类冒犯语言）、AG News（4类新闻）、QNLI（2类NLI），中毒率统一设为 20%。
- **攻击方法**：BadNet（插入稀有词）、InsertSent（插入完整句子）、Syntactic（句式改写的句法模板触发）。
- **基线**：PCA、Clustering、ONION（需 GPT2-large）、DAN（需干净验证集）。
- **检测能力（FRR/FAR，表2）**：Z-TOKEN 在 BadNet/InsertSent 上 FAR 达到 0.0%（几乎零漏检），Z-TREE 对 Syntactic 检测 FAR=0.5%~1.2%；Z-SEQ 综合 FAR 最低。
- **端到端防御（ASR/CACC，表3）**：
  - Z-SEQ 在 SST-2 平均 ASR = **14.0%**（次低基线 DAN 为 36.3%，其余≈98%+）；
  - Z-SEQ 在 QNLI 平均 ASR = **10.0%**（最优）；
  - Z-SEQ 对 BadNet 和 InsertSent 在所有数据集上 ASR 均低于 4%，与"干净模型在中毒测试集上的 BASR"下界接近，实现**近乎完美防御**；
  - CACC 损失极小（平均下降 < 0.2%），OLID 上甚至略有提升（+0.1%）。
- **低中毒率测试（1%~20%，表5/6）**：Z-TOKEN 在 5% 中毒率下仍可使 QNLI ASR 降至 5.3%（FAR=0.7%），远超 Clustering/DAN（低中毒率下失效）。
- **不同模型（表7）**：Z-SEQ 为 model-free，ASR 与 CACC 结果与 BERT/RoBERTa 型号无关；在 roberta-large 上仍实现 48.3% ASR 降幅。
- **最强结果**：Z-SEQ 在 QNLI+Syntactic 上 ASR=19.1%（对比无防御 99.1%），在 SST-2+InsertSent 上 ASR=3.4%，CACC=92.6%。

## 相关工作脉络
1. **BadNet（Gu et al., 2017）**：图像后门攻击开创性工作，后被迁移至 NLP（Kurita et al., 2020）；本文将其作为词插入型攻击的基线。
2. **ONION（Qi et al., 2021a）**：使用 GPT2-large 移除中毒样本中的离群词，属于测试/训练阶段外部模型驱动的防御；本文方法无需任何预训练模型。
3. **Syntactic 攻击（Qi et al., 2021b）**：利用句法模板改写注入触发结构，比词插入更隐蔽；本文用 Z-TREE（祖先路径 z-score）专门应对此类攻击。
4. **Spurious Correlation（Gardner et al., 2021; Wu et al., 2022）**：通过 z-score 分析数据特征与标签的异常关联以去除噪声；本文首次将此思路系统引入后门防御领域。
5. **PCA/Clustering（Tran et al., 2018; Chen et al., 2018）**：基于潜表征聚类的中毒检测，需在中毒数据上训练模型；本文避免了该开销和对表征质量的依赖。
6. **DAN（Chen et al., 2022b）**：利用干净验证集的潜表征区分中毒样本；本文方法在无需任何额外数据的前提下达到可比或更优性能。

## 局限性与未来方向
- 仅针对**微调阶段**的中毒攻击有效；对预训练阶段被植入后门的模型（如 BadPre 攻击）不适用，需另行设计防御。
- 特征设计基于**已知攻击模式**（词插入/句法改写），对新兴攻击（如 Chen et al., 2022c 的写作风格触发器）需扩展特征工程，存在猫鼠游戏。
- 对 AG News 上的 Syntactic 攻击防御效果有限（ASR 仍高达 99.5%），因该类攻击可能过度破坏原始输入，难以通过现有特征充分捕捉。
- 纯**经验性观测研究**，缺乏理论分析保证方法在其他数据集/攻击上的泛化性与鲁棒性边界。

## 研究启发与可借鉴点
1. **spurious correlation 视角可推广**：将后门攻击统一建模为"虚假特征-标签强绑定"问题，为跨模态（视觉、多模态）后门防御提供统一分析框架。
2. **z-score 过滤范式可复用**：该方法仅依赖统计量，无需模型训练，可直接移植到自监督预训练数据清洗、RAG 知识库质量过滤等场景。
3. **双特征联合（lexical + syntactic）策略**：Z-SEQ 证明了正交特征联合可显著提升检测鲁棒性；可进一步引入语义特征（如词向量 co-occurrence）、篇章结构特征等扩展。
4. **无需额外数据的轻量防御设计哲学**：对于资源受限场景（如边缘设备、低资源语言），model-free 和无需干净验证集的防御具有实际部署优势，可作为后续研究的重要 baseline。
5. **低中毒率下的持续有效性**：5% 中毒率仍保持高效检测，说明 z-score 对弱信号敏感，可启发对少量高质量中毒样本的早期预警系统设计。

## 关键术语表
- **Backdoor Poisoning Attack（后门中毒攻击）**：通过在训练数据中植入触发器并篡改标签，使模型在正常输入上表现良好，但在触发器出现时输出攻击者指定标签的攻击方式。
- **Spurious Correlation（虚假相关）**：模型学习到的简单特征与标签之间的表面强关联，在分布外数据上失效，导致模型依赖捷径而非真正语义。
- **Z-score（Z 分数）**：衡量某特征条件下标签条件频率偏离先验频率的标准分数，用于量化特征与标签之间的异常关联强度。
- **Constituency Tree（成分树）**：句法分析中将句子分解为短语成分的层级树结构，用于提取句法特征（如祖先路径）。
- **CACC（Clean Accuracy）**：后门模型在原始干净测试集上的准确率，衡量防御方法对正常性能的保留程度。
- **ASR（Attack Success Rate）**：后门模型在中毒测试集上的准确率，越低表示防御效果越好。
- **FAR / FRR**：False Acceptance Rate（漏检率）与 False Rejection Rate（误杀率），衡量数据过滤方法的精确性与召回权衡。
- **Z-SEQ**：本文提出的顺序防御方法，依次应用 Z-TREE 和 Z-TOKEN 过滤可疑训练样本。

## 可复现要素
- **数据集**：SST-2、OLID、AG News、QNLI，均为公开数据集（论文未单独提供代码仓库链接，但使用了标准 HuggingFace/学术来源）。
- **代码/权重**：论文未提供独立代码仓库；使用 Transformers 库（Wolf et al., 2020）的 `bert-base-uncased`，超参：Adam 优化器，lr=$2\times10^{-5}$，batch size=32，max seq len=128，weight decay=0，训练 3 epochs，单卡 V100。
- **关键超参**：z-score 阈值 = 18σ（18 个标准差），中毒率 = 20%（低中毒率实验覆盖 1%/5%/10%/20%）。
