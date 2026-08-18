---
title: "Decoding-the-Silent-Majority-Inducing-Belief-Augmented-Socia"
source: https://aclanthology.org/2023.emnlp-main.4.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:11:53"
field: "Social NLP / 社交媒体响应预测"
keywords: ["response forecasting", "belief-augmented graph", "large language model", "social network", "sentiment prediction", "heterogeneous graph transformer", "lurker"]
innovations: ["用LLM抽取用户潜在信念人格并构建信念增强异构图，桥接跨社区同信用户", "在监督与零样本双设置下统一建模，提出Social Prompt模拟图传播", "系统评估并在lurker/unseen user场景验证强泛化能力"]
benchmarks: ["RFPN (Response Forecasting on Personas for News)", "Twitter social graph (18,634 users, 1,744,664 edges)"]
---

# 论文速读：Decoding-the-Silent-Majority-Inducing-Belief-Augmented-Socia

## 一句话总结
本文提出 SOCIALSENSE 框架，利用大语言模型从用户画像和历史帖文中提取潜在信念人格，在现有社交网络上构建以信念为中心的异构图（含用户-新闻交互边和信念节点），并通过 HGT 进行信息传播，实现了对社交媒体新闻响应的精准情感强度与极性预测，在监督与零样本设置下均显著优于现有基线，且在 lurker 和 unseen user 场景下展现更强泛化能力。

## 研究问题与动机
- **核心问题**：自动响应预测（Response Forecasting）——给定用户 persona 和一条新闻消息，预测其响应的情感极性（Positive/Negative/Neutral）与强度（0-3 级）。
- ** lurker 困境**：Twitter 上 97% 的推文由 25% 活跃用户产生，大量 lurker（历史互动极少）难以通过传统文本建模可靠预测；现有方法对这类用户利用不足。
- **隐式信念网络的结构性 gap**：初步分析发现，超过 44.6% 的用户与至少相隔两跳的邻居共享相似社会信念，但仅依赖显式社交图谱会遗漏这类跨社区的信念连接。
- **历史帖文噪声与可解释性缺失**：直接使用原始历史帖文作为特征会引入噪声，且缺乏对用户内在价值取向的显式建模与解释。

## 核心贡献（创新点）
1. **首次将 LLM 用于隐式信念人格抽取并构建信念增强图**：用 ChatGPT 从用户画像+历史帖文提取道德价值（Moral Foundations Theory）和人类价值（Schwartz Theory），形成可消费的 latent persona，与已有仅使用原始文本或简单 profile 的方法本质不同。
2. **提出信念增强的异构图结构（Belief-Augmented Social Network）**：在用户-用户关注边和用户-新闻交互边基础上，引入固定集合的信念节点（价值观维度），使信念成为跨社区用户的"捷径"，相比 InfoVGAE 等仅依赖显式交互边的图模型，能捕获隐性社区结构。
3. **监督与零样本双范式统一框架**：监督设置使用 HGT 图传播；零样本设置提出 SOCIAL PROMPT，以邻居影响力排序筛选 Top-K 邻域，再经 LLM 聚合邻居 latent persona 后预测，证明社会上下文即使在 zero-shot 下也显著提升 LLM 表现。
4. **系统性验证 lurker / unseen user 泛化能力**：实验表明在 lurker 场景（<50 条历史响应）下，SOCIALSENSE 情感极性 F1 较 DeBERTa/RoBERTa 优势从 5.99% 扩大到 11.26%，凸显信念传播对用户历史稀缺场景的鲁棒性。

## 方法详解
- **阶段一：LLM 抽取潜在人格 User_L**
  - 输入用户 profile + 历史帖文 concat，设计 prompt P_l 引导 LLM 输出结构化 latent persona，包含：Dominant Human Values（Schwartz 10 类）、Dominant Moral Values（Moral Foundations 5 对 10 类）、Ideologies、Interested Topics、Issues & Stance、Entities & Stance、Professions 等。
  - 公式：$\text{User}_L = \text{LLM}(\text{profile}, \text{history}, P_l)$
- **阶段二：构建 Belief-Augmented Social Network**
  - 三类节点：$\mathcal{V}^U$（用户）、$\mathcal{V}^M$（新闻标题）、$\mathcal{V}^B$（固定信念值节点，共 20 个：10 道德 + 10 人类价值）。
  - 三类边：$\mathcal{E}^I$（用户-新闻交互，表示用户响应过该新闻）、$\mathcal{E}^F$（用户-用户关注，有向）、$\mathcal{E}^B$（用户-信念值，无向，表示该用户具有此价值倾向）。
  - 共 18,634 用户、1,744,664 关注边。
- **阶段三：图信息传播（监督设置）**
  - 节点初始化：用户节点 u、新闻节点 m 用 DeBERTa 编码；信念节点 b 随机初始化。
  - 采用 Heterogeneous Graph Transformer (HGT) 在异构图上进行多跳消息传递，显式区分 node type 和 edge type 的元路径注意力。
  - 输出：将更新后的用户表示与新闻 embedding 拼接，过 MLP + softmax 分类（极性 3 类；强度 0-3 为回归/分类任务，使用 Spearman/Pearson 相关 + MiF1/MaF1 评估）。
  - 损失：交叉熵。
- **零样本 Social Prompt（SOCIAL PROMPT）**
  - 邻居过滤：基于权威度（follower 数）排序，取 Top-K（K=25）邻域。
  - 对每个邻居 n 抽取 $\text{User}_L^n$，通过 prompt P_s 让 LLM 汇总社区背景并生成中心用户的社交人格 $\text{User}_S$。
  - 最终预测 prompt P_p 同时输入中心用户自身的 $\text{User}_L^c$ 与 $\text{User}_S$，由 LLM 输出评论文本、极性、强度。
  - 公式：$\mathscr{R} = \text{LLM}(P_p, \text{U}_L^c, \text{LLM}(P_s, \{\text{U}_L^n\}_{n \in \mathcal{N}^K}))$

## 实验与结果
- **数据集**：RFPN（Sun et al., 2023），13.3k 响应来自 8.4k 用户、3.8k 新闻标题（Twitter）；自建社交图 18,634 用户、1,744,664 边。训练/验证/测试样本分别为 10,977 / 1,341 / 1,039。
- **评估指标**：强度用 Spearman $r_s$、Pearson $r$；极性用 Micro-F1（MiF1）与 Macro-F1（MaF1）。
- **主要结果（监督）**：
  - SOCIALSENSE 全面领先：$r_s=61.82$, $r=61.98$, MiF1=70.45%, MaF1=65.71%。
  - 相对次优 InfoVGAE 提升：强度相关 +3.21%（Spearman）、极性 MiF1 +2.99%。
  - 相对纯文本 DeBERTa：MiF1 +5.68%, MaF1 +6.41%。
  - 弱于 ChatGPT 零样本（MiF1=58.61%）——表明单一 LLM prompt 不足以捕捉图结构信息。
- **零样本结果**：SocialSense_Zero 在三项指标上均优于 ChatGPT_L 与原始 ChatGPT（MiF1 60.54% vs 59.77%/58.61%）。
- **Ablation**：去除信念节点降 MiF1 4.83%、相关 1.91%；去除用户-新闻边降相关 6.63%；去掉 profile/history 分别造成 1.93%/4.45% 损失；随机初始化节点表征 MiF1 下降 8.97%。
- **Lurker 场景**（<50 条历史响应，745 测试样本）：SOCIALSENSE MiF1=71.01%, MaF1=63.88%，较 DeBERTa/RoBERTa 优势扩至 +11.26%（MiF1）。
- **Unseen User 场景**：SOCIALSENSE MiF1=62.55%, MaF1=55.37%，显著优于所有基线，验证跨用户泛化。

## 相关工作脉络
- **传统响应预测（文本地表征）**：Lin & Chen (2008)、Artzi et al. (2012)、Li et al. (2019) 等依赖用户文本与 DNN，缺少图结构建模与 persona 抽象；SOCIALSENSE 以信念图扩展此类工作。
- **生成式评论预测**：Yang et al. (2019)、Wu et al. (2021)、Lu et al. (2022) 把响应预测建模为生成任务，缺乏定量情感度量；本文聚焦可解释的强度/极性数值预测。
- **图表示学习与情感极化**：InfoVGAE (Li et al., 2022) 在用户-新闻 bipartite 图上做无监督 belief representation；本文在其基础上扩充用户-用户边与信念节点，并使用有监督 HGT 训练，性能更高。
- **Social-NLP 与大模型 prompting**：Zhou et al. (2022)、Kojima et al. (2022) 等探索 LLM 在 NLP 上的 few-shot/zero-shot 能力；本文进一步表明，将社会上下文和 persona 注入 prompt 可显著提升 LLM 在社交预测任务上的表现。
- **Lurker / 冷启动表示学习**：McClain et al. (2021) 指出 Twitter 长尾分布；本文通过信念边实现跨社区知识迁移，缓解少历史用户的表征稀疏问题。

## 局限性与未来方向
- 依赖高质量社交网络数据：当 Twitter 图数据缺失或噪声较高时，预测性能会下降。
- 领域与文化泛化未充分验证：模型在美国 Twitter 数据上训练，对不同国家/语言/平台的可迁移性待研究。
- LLM 抽取信念的人格准确性仍有提升空间（人工评估平均 3.9/5），prompt 设计可能引入系统偏差。
- 未来方向包括：跨领域/跨文化迁移、动态演化图上的在线响应预测、以及更通用的 Social Prompt 策略。

## 研究启发与可借鉴点
1. **LLM 结构化人格抽取管线**：可将本文的 P_l prompt 设计迁移到任何需要用户画像抽象的任务（如推荐系统、stance detection），得到可消费的 latent persona 作为下游特征。
2. **信念/价值节点作为图捷径**：在异构图上加固定语义节点（道德、政治倾向、兴趣标签）能桥接远距离同信用户；这一思路可推广至 fact-checking、 misinformation diffusion 等任务。
3. **Social Prompt 作为零样本图传播代理**：对无法部署 GNN 的工业场景，可用 LLM 替代消息传递完成邻域聚合，兼具零样本能力与可解释性，值得在实际工程中试点。
4. **lurker 友好型评估协议**：建议将 "历史互动 < K 条" 的子集评测纳入标准协议，以检验模型在冷启动场景的真实可用性与鲁棒性。

## 关键术语表
- **Response Forecasting（响应预测）**：给定用户与新闻输入，定量预测其公开响应的情感极性（正/负/中）与强度（0-3）。
- **Latent Persona（潜在人格）**：由 LLM 从用户画像与历史帖文中抽取的结构化价值观、立场、兴趣等内在表征。
- **Belief-Augmented Social Network（信念增强社交图）**：在用户-用户与用户-新闻图之上附加信念值节点所构成的异构图，信念边充当跨社区连接捷径。
- **HGT（Heterogeneous Graph Transformer）**：区分节点/边类型的 Transformer 式图神经网络，支持多类型元路径 attention。
- **Social Prompt**：通过 LLM prompt 模拟图传播的零样本推理机制，包括邻域过滤与聚合生成。
- **Lurker（潜伏用户）**：历史互动极少（<50 条响应）的用户，其文本信号稀疏、难以直接用传统方法建模。
- **Moral Foundations Theory**：Graham 等提出的五对道德基础维度（Care/Harm, Fairness/Cheating, Loyalty/Betrayal, Authority/Subversion, Purity/Degradation）。
- **Schwartz Theory of Basic Values**：Schwartz 提出的人类基本价值十维模型（Conformity, Tradition, Security, Power, Achievement, Hedonism, Stimulation, Self-Direction, Universalism, Benevolence）。

## 可复现要素
- **数据集**：RFPN (Sun et al., 2023)；Twitter 社交图（18,634 用户、1,744,664 边）——论文声明将在代码发布时提供匿名化版本，未给出当前公开链接。
- **代码/权重**：论文未提供公开开源仓库，仅说明源码将随匿名数据集一起发布。
- **关键超参**：epoch=1000、patience=300、lr=5e-4、batch=1、GNN层数=3、attention heads=4、隐藏维度=128、dropout=0.2、激活 ReLU、优化器 RAdam、warmup ratio=0.06；零样本 K=25。
- **硬件**：单卡 NVIDIA RTX A6000 48GB；训练耗时 <30 分钟；参数量约 10.48M。
- **依赖库**：PyTorch、Huggingface Transformers 4.8.2、PyG 2.0.3。
