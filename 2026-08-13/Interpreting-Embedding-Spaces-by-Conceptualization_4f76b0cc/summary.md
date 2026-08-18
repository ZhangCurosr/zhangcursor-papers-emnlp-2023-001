---
title: "Interpreting-Embedding-Spaces-by-Conceptualization"
source: https://aclanthology.org/2023.emnlp-main.106.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:47:48"
field: "LLM 可解释性"
keywords: ["嵌入可解释性", "概念空间", "大语言模型", "后处理方法", "知识图谱", "语义保真"]
innovations: ["提出 CES 黑盒映射算法将隐嵌入转为概念向量", "按需动态细化算法实现自适应粒度概念空间", "新型人机联合评估协议测量语义保真度"]
benchmarks: ["AG News", "BBC News", "DBpedia 14", "Yahoo", "20 Newsgroup", "Ohsumed", "R8", "Wikipedia Triplet"]
---

# 论文速读：Interpreting Embedding Spaces by Conceptualization

## 一句话总结
本文提出 CES（Conceptualizing Embedding Spaces），一种模型无关的后处理算法，将任意 LLM 的隐式嵌入向量通过知识图谱（如 Wikipedia 分类）映射为人类可读的概念空间向量，并支持按需动态细化粒度，辅以新型人机联合评估方法验证其语义保真性。

## 研究问题与动机
- LLM 嵌入空间虽性能优异，但维度不可解释，难以用于**调试、模型对比和隐性偏差检测**。
- 既有可解释性工作主要分三类：(1) 监督/注意力探针（只解释决策，不解释空间本身）；(2) 基于嵌入矩阵的正交变换（需访问矩阵，且假设每维对应一概念）；(3) 重新训练可解释模型（牺牲原模型性能）。三者均存在**强假设或信息丢失**的问题。
- 现有评估方法仅比较分类准确率，无法区分"语义一致"与"特征重命名"——两个 80% 准确率的分类器可能只在 60% 样本上达成一致（本文引入 Cohen's kappa 解决此缺陷）。
- 缺乏一个**无需额外训练、仅用黑盒映射**即可将隐空间向量转化为概念向量并支持按需细化的统一框架。

## 核心贡献（创新点）
1. **提出 CES 算法**：将任意隐嵌入空间 $L$ 映射到人类可理解的概念空间 $\mathcal{C}$，本质区别在于**不假设任何隐维与显式概念一一对应**，仅依赖 $f:\tau(c)\to \hat{c}$ 的黑盒投影。
2. **按需动态细化算法（Selective Refinement）**：从 $C^1$ 出发迭代扩展权重最高的概念节点及其 top-$p\%$ 子节点，本质区别在于**粒度不再固定 depth，而是由上下文文本 $T'$ 驱动的自适应空间划分**，避免深层节点爆炸（$|C^3|=3467$ vs $|C^1|=37$）。
3. **新型评估协议（人机联合 + 模型评分）**：用人类标注者与 LLM 评分器在独立测试集上对 CES 向量做分类，以 Cohen's kappa 衡量语义保真，本质区别在于**直接测量"概念表示是否保留原嵌入语义"而非仅比较下游任务准确率**。
4. **三大应用场景演示**：(1) 跨模型语义对比（SBERT vs ST5 vs SRoBERTa 对同一文本的概念视图差异）；(2) LLM 层追踪（BERT/GPT2 各层权重演变可视化）；(3) 分类器解释，本质区别在于**后处理无需访问模型内部参数**，通用性强。

## 方法详解
**核心映射**：给定概念集 $C=\{c_1,\dots,c_n\}$ 与映射 $\tau:c_i\mapsto\text{text}$，预计算 $\hat{c}_i=f(\tau(c_i))\in L$。对输入向量 $l\in L$：
$$\text{CES}^{f,C,\tau}(l)=\langle \text{sim}(l,\hat{c}_1),\dots,\text{sim}(l,\hat{c}_n)\rangle^\top$$
使用 cosine similarity 且 $f$ 归一化时，等价于矩阵乘法 $M\cdot l$（$M_{ij}=u_i\cdot\hat{c}_j$，$U$ 为标准基），大幅加速。

**按需细化（Algorithm 1）**：
1. 初始 $C\leftarrow C^1$（Wikipedia 一级分类，37 个节点）。
2. 用当前 $C$ 对上下文 $T'$ 中所有文本投影到 CES，取平均得各节点权重。
3. 选择权重最大节点 $\hat{c}$，取 top-$p\%$ 子节点（按 siblings score 排序）加入 $C$。
4. 若 `removeP=True`，删除父节点 $\hat{c}$；否则保留（论文发现保留更优）。
5. 迭代至 $|C|=$ size（默认 768，与 SRoBERTa 维度对齐）。

**Siblings score**（边重要性打分）：
$$\text{score}(p,c)=\text{AVG}_{s\in\text{siblings}(c,p)}\frac{| \text{parents}(c)\cap\text{parents}(s) |}{|\text{parents}(c)|}$$
去除每个节点低分边的 $\lambda\%$（实验中 $\lambda=35$）。

**概念→文本映射 $\tau$**：两种实现——(1) 直接用概念名；(2) 拼接"概念名+子概念名"生成完整句（论文发现差异不显著，Appendix C）。

**含分类标签的细化**：若用于分类任务，将训练集标签熵线性加到节点权重中，使跨类区分度高的概念优先扩展。

## 实验与结果
**数据集**：7 个英文分类数据集——AG News、Ohsumed、R8、Yahoo、BBC News、DBpedia 14、20Newsgroup（每个 $\leq 10000$ 样本），均为话题分类（非情感）。

**评估协议**：
- **RF 分类器**（100 树、depth=5，10-fold）：CES 与 LLM 原始嵌入训练的分类器之间的 raw agreement / Cohen's kappa。
- **KNN 分类器**（n=5，cosine sim）：同上指标。
- **人类评估**：3 名研究生仅看 CES top-3 概念判断类别，与 LLM-RF/NC 分类结果比较。
- **LLM 评分器**：SBERT/ST5/SRoBERTa 以 cosine sim 评估概念与类别名的匹配度。
- **对比基线**：Dimension Meaning Assignment（DMA-words/concepts/C*）——将每维映射到最优词/概念。

**最强结果**：
- BBC News（RF）：raw agreement = $0.96\pm0.01$，kappa = $0.95\pm0.02$，CES 准确率 95.8% vs LLM 97.0%（差距仅 1.2%）。
- AG News（KNN）：raw agreement = $0.92\pm0.01$，kappa = $0.90\pm0.01$。
- 人类评估：与 NC 分类器平均 agreement 0.77，高于与 RF 分类器的 0.70（NC 更简单反而更一致）。
- 与 DMA 基线相比，CES 在所有数据集上均显著占优（Table 5，SBERT 评分器下 CES 0.80/0.90 vs DMA-words 0.55/0.50）。
- 三元组相似度任务（Wikipedia triplet，n=1000）：CES($C^3$) vs SRoBERTa raw agreement = 0.82，kappa = 0.64，高于与 true label 的一致性（0.726/0.452）。

## 相关工作脉络
- **Ribeiro et al. (2016a, 2018)**、**Lundberg & Lee (2017)**：SHAP/LIME 等监督解释方法，针对分类决策而非嵌入空间本身。
- **Dufter & Schütze (2019)**、**Park et al. (2017)**：正交变换使嵌入维度可解释，**依赖嵌入矩阵且假设每维有显式语义**。
- **Clark et al. (2019)**、**Vig et al. (2020)**：Probing 方法识别注意力头或神经元功能，**需要模型内部访问**。
- **Senel et al. (2018, 2022)**：双向对齐维度与语义概念进行重新训练，**改变原模型表征**。
- **nostalgebraist (2020, Logit Lens)**、**Dar et al. (2022)**：通过 unembedding 矩阵映射回 token 空间，**需访问模型权重**。
- **本文定位**：完全黑盒、无训练、不假设维度语义对应、无需访问模型内部，仅依赖 $f$ 与外部知识图谱。

## 局限性与未来方向
- 层追踪（Section 4.2）仅使用 token embedding 的平均池化，方式较简化。
- 实验几乎全部使用 SRoBERTa，对其它架构泛化性未充分验证。
- 未展示 CES 在真实模型调试和偏差解释上的应用（作者明确列为未来工作）。
- 本体仅使用 Wikipedia 分类图，未尝试其他知识图谱（如 DBpedia、WordNet）。
- 概念空间粒度由 size 超参控制，但自动选择最优 size 的方法未讨论。

## 研究启发与可借鉴点
1. **黑盒转换范式**：将任何"向量输出型"模型（不仅是文本 LLM，如图像 encoder、多模态 encoder）嵌入映射到概念空间的方法论可迁移至视觉、语音等领域。
2. **按需细化算法**：迭代扩展高权重节点的设计可复用于任何需要**自适应粒度语义表示**的场景（如文档主题演化追踪）。
3. **评估协议创新**：Cohen's kappa + 人类独立标注的混合评估方式，比单纯准确率更能反映"语义保真度"，可作为嵌入可解释性工作的**标准评估范式**。
4. **Siblings score 设计**：基于父节点重叠度评估概念边重要性的思路，可用于知识图谱边权重的自动学习。
5. **跨模型对比应用**：用 CES 对齐不同模型的共享概念空间后，可直接比较其语义偏差（如 SBERT 对 "FC Barcelona" 偏向地理而非体育），为模型审计提供新工具。

## 关键术语表
- **CES（Conceptualizing Embedding Spaces）**：本文核心算法，将隐嵌入向量投影到概念相似度向量的后处理框架。
- **Latent embedding space（隐嵌入空间）**：LLM 内部的高维连续向量空间，维度对人类不可解释。
- **Conceptual space（概念空间）**：由知识图谱概念节点定义的语义向量空间，维度含义清晰。
- **Selective refinement（按需细化）**：根据上下文文本动态扩展高权重概念节点及其子节点的算法。
- **Siblings score（兄弟分数）**：衡量同一父节点下子节点间语义相似度的连续评分，用于概念空间边权重排序。
- **Cohen's kappa**：衡量两个标注者一致性的统计量，消除随机一致的影响，范围 [-1, 1]。
- **DMA（Dimension Meaning Assignment）**：基线方法，为每个嵌入维分配最优词或概念名称。
- **Black-box embedding**：仅通过输入文本获取输出向量、无需访问模型内部结构的嵌入方法。

## 可复现要素
- **数据集**：AG News、Ohsumed、R8、Yahoo、BBC News、DBpedia 14、20Newsgroup、Wikipedia triplet（Ein-Dor et al., 2018）——均为公开数据集。
- **代码**：论文声明代码在线可用（标注脚注 1），但具体 URL 在正文中未给出（需查阅原文页脚）。
- **模型**：SRoBERTa（默认参数）、SBERT、ST5、BERT、GPT2，均为公开预训练模型。
- **超参**：size=768（与 SRoBERTa 维度对齐）、removeP=False（默认）、p% 未明确数值、$\lambda=35$（去边比例）、tau 使用概念名（平均 4.25 词）。
- **评估**：RF（100 树、depth=5）、KNN（n=5）、10-fold 交叉验证。
