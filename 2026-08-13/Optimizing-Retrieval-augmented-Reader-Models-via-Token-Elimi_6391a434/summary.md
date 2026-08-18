---
title: "Optimizing-Retrieval-augmented-Reader-Models-via-Token-Elimi"
source: https://aclanthology.org/2023.emnlp-main.93.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:52:03"
field: "检索增强语言模型效率优化"
keywords: ["retrieval-augmented generation", "token filtering", "decoder efficiency", "long-form question answering", "FiD", "cross-attention", "inference optimization"]
innovations: ["提出基于 cross-attention 分数的 Token Filtering 方法，在解码阶段动态剔除冗余输入 token", "首次系统性分析 FiD 在长输出场景下 decoder 延迟占比高达 95%+，并提出 token 过滤与 layer skipping 的组合优化", "在 MS MARCO 上实现 62.2% 延迟降低且性能下降不超过 2%，达到 ELI5 KILT 榜单 SOTA"]
benchmarks: ["ELI5", "MS MARCO Passage Ranking", "Natural Questions"]
---

# 论文速读：Optimizing-Retrieval-augmented-Reader-Models-via-Token-Elimi

## 一句话总结
本文针对检索增强生成模型 FiD 在长形式问答（LFQA）任务中的解码效率瓶颈，提出了一种在解码过程中动态筛选高重要性输入 token 的 Token Filtering 方法，并与 CALM 解码器层跳过技术相结合，实现了最高 62.2% 的延迟降低，性能下降不超过 2%，且在 ELI5 KILT 榜单上达到 SOTA。

## 研究问题与动机
1. **FiD 长文生成解码效率低**：FiD 需要将大量检索到的文档拼接后通过 decoder 的 cross-attention 处理，每生成一个 token 都要对整个输入序列做交叉注意力，计算开销极大，尤其对 LFQA 等长输出任务被严重放大。
2. **现有优化手段不足**：此前方法（如 FiD-Light、COLT5、FIDO 等）主要集中于减少 encoder 输入或修改架构，但 FiD 的 encoder 延迟占比相对较小， decoder 才是推理延迟的主要来源（长输出时可超过总延迟的 95%）。
3. **冗余 token 未被有效利用**：大量输入 token 对答案生成并无实质贡献，但现有的编码器端过滤方法（如 Yu et al., 2021）仅作用于 encoder 层，未触及 decoder 阶段的 token 级冗余。
4. **缺乏对 cross-attention 贡献度的系统分析**：尚未有工作系统分析 decoder 各层在生成各阶段对不同输入 token 的关注模式，以指导 token 筛选。

## 核心贡献（创新点）
1. **系统分析了 FiD encoder 与 decoder 的延迟分布及 cross-attention 模式**，揭示了在 LFQA 等长输出场景下 decoder 是延迟主因，且早期生成阶段的 cross-attention 分数能有效区分 gold passage 与噪声 passage 的 token。
2. **提出 Token Filtering 方法**：在解码阶段基于 cross-attention 分数动态筛选 top-p% 高贡献输入 token，并将无关 token 从 key-value 状态中移除，使后续所有 token 的解码计算基于更紧凑的输入序列——与 Goyal et al. (2020) 等 encoder 端过滤方法的本质区别在于本文聚焦于 decoder 阶段且作用于 token 级别而非 passage 级别。
3. **提出 Combined 方法（Token Filtering + CALM）**：将 token 级过滤与 decoder 层跳过（Schuster et al., 2022）相结合，实现"输入精简 + 层精简"双重加速，在大多数数据集上达到更优的性能-效率权衡。
4. **在三大 LFQA 基准上验证了方法有效性**：MS MARCO 延迟降低 62.2%（性能降 ≤2%）、NQ 降低 54.9%、ELI5 降低 40.9%，且在无计算限制时达到 ELI5 KILT 榜单 SOTA。

## 方法详解
**Token Filtering 核心流程：**
- 在解码步骤 t、选定层 l 处，计算每个输入 token 的平均交叉注意力分数：$S_{t,l} = \frac{1}{h} \sum_{i=1}^{h} A_{t,l}^{i}$，其中 $A_{t,l}^{i}$ 为第 i 个 attention head 在 token t、层 l 处的交叉注意力分数。
- 对 $S_{t,l}$ 做降序排序，选取 top $p\%$ 个输入 token 索引，得到 $Top_{t,l}$。
- 从 cross-attention 的 past key-value 状态（$K_{past}, V_{past}$）中仅保留 $Top_{t,l}$ 对应位置的条目，之后所有后续 token 的解码均基于此压缩后的 KV 状态进行。
- 过滤只在推理时执行一次（针对每个生成的 token），显著减少了后续所有步骤的计算维度。

**关键分析结论（指导方法设计）：**
- Decoder 的较低层（第 2-3 层）对 gold passage token 的区分能力最强，适合作为 cross-attention 分数的计算层。
- 在生成早期（约第 20 个 token 附近）cross-attention 分数对 gold passage 的区分度最高，之后逐渐下降，因此应在生成早期完成过滤。
- 过滤比例 $p$ 不宜过高（如 50%），否则会将大量低 rank passage 的噪声 token 引入，降低性能。

**Combined 方法：**
- 将 Token Filtering 与 CALM（Confident Adaptive Language Modeling，Schuster et al., 2022）结合，CALM 通过 confidence classifier 在每个 decoder 层判断是否可以提前退出，从而跳过冗余层计算。两者联合使用时，Token Filtering 减少输入维度，CALM 减少层数，产生叠加加速效果。

**超参数搜索空间：**
- 过滤比例 $p$：{10%, 30%, 50%}
- 过滤层：[1, L]（L 为 decoder 总层数）
- 过滤 token 索引：[1, 20]
- CALM 置信度阈值：[0.2, 0.9]
- 评估策略：在 dev 集上以 ROUGE-L vs 延迟为横纵坐标构建 Max Curve，选取最优超参数组合。

## 实验与结果
**数据集：**
- **ELI5**：来自 Reddit 论坛的长形式问答，训练/验证/测试集分别为 272,634 / 3,000 / 1,507（另有 KILT held-out test 600）。
- **MS MARCO**：Bing 查询的问答数据，训练/验证/测试分别为 498,000 / 3,000 / 6,980。
- **Natural Questions (NQ)**：Google Search 真实查询，训练/验证/测试分别为 55,622 / 3,000 / 6,489。

**基线模型：**
- 标准 FiD（T5-Base 和 T5-Large）、CALM（Schuster et al., 2022）、FiD-Light、RBG（Su et al., 2022）。

**主要结果：**

| 模型 | ELI5 R-L | ELI5 F1 | MS MARCO R-L | MS MARCO F1 |
|------|----------|---------|--------------|-------------|
| RBG FiD | 25.70 | 28.55 | 24.64 | 27.08 |
| RBG | 26.46 | 29.04 | 27.08 | 27.52 |
| FiD (ours) | 26.24 | 31.46 | 24.33 | 27.11 |
| FiD TF | 26.65 | 30.32 | 24.75 | 27.26 |
| **FiD Comb** | **26.97** | **31.76** | **25.11** | **27.41** |

- **延迟降低**：MS MARCO（FiD-Base）最高达 **62.2%**，NQ 达 **54.9%**，ELI5 达 **40.9%**；性能（ROUGE-L）下降 ≤ 2%。
- **BERTScore F1**：FiD Comb 在 ELI5（83.79）和 MS MARCO（85.27）上均优于原始 FiD（83.68 / 85.07）。
- **KILT 榜单**：FiD Comb 以 ROUGE-L 25.61 / F1 29.99 位列 ELI5 第一。

**结论**：Combined 方法在多数情况下优于单独使用 Token Filtering 或 CALM，实现了最佳的 性能-效率权衡。

## 相关工作脉络
1. **FiD（Izacard & Grave, 2021）**：本文的基础模型，采用 T5 作为 reader，对多路检索文档分别编码后在 decoder 端做 cross-attention，是本文方法优化的目标架构。
2. **CALM（Schuster et al., 2022）**：提出基于 confidence classifier 的 decoder 层跳过机制，本文将其与 Token Filtering 结合以实现双重加速。
3. **FiD-Light / COLT5（de Jong et al., 2022; Ainslie et al., 2023）**：分别从预训练优化和条件计算角度改进 FiD 效率，与本文侧重于推理时 token 级过滤形成互补视角。
4. **Passage Re-ranking in Reader（Yu et al., 2021）**：在 encoder 端引入 passage re-ranker 过滤无关文档，本文指出 encoder 对长输出延迟贡献有限，主张在 decoder 端进行更精细的 token 级过滤。
5. **Power-BERT（Goyal et al., 2020）**：在 BERT 推理时渐进消除低重要性 word-vector，属于 encoder 端的 token 淘汰思路，本文借鉴了其思想但应用于 decoder cross-attention 阶段。
6. **RBG（Su et al., 2022）**：在 ELI5 上提出 Read-Before-Generate 框架，本文与其进行同等规模模型的公平对比，达到更高 ROUGE-L 和 F1。

## 局限性与未来方向
1. **检索器优化缺失**：本文仅关注 reader 优化，未涉及 retriever 的改进，检索质量直接影响输入 token 的噪声水平。
2. **超参数依赖**：Token Filtering 的过滤比例 $p$、过滤层和 token 索引均为人工搜索的固定超参数，缺乏自适应动态选择机制。
3. **跨架构泛化待验证**：实验主要聚焦于 FiD 架构，对其他 encoder-decoder 模型（如 T5-Large 以上规模）的效果提升有限，未做广泛复现对比。
4. **超参数搜索空间受限**：受计算资源限制，搜索空间仅为全空间的子集，可能有更优配置未被发现。
5. **未来方向**：可探索学习型的 token 重要性预测器以替代手动搜索；将方法扩展至非 FiD 架构；研究端到端联合优化 retriever-reader 效率。

## 研究启发与可借鉴点
1. **Cross-attention 分数作为 token 重要性代理指标**：该方法可直接迁移到任何其他 decoder 需要处理大量拼接上下文的任务（如长文档摘要、多文档推理），作为轻量级输入压缩策略。
2. **"输入过滤 + 层跳过"的组合范式**：本文验证了同时优化输入维度和网络深度的协同收益，这一思路可推广至其他架构（如 Mixture of Experts、条件计算）的效率优化中。
3. **早期生成阶段的分析指导方法设计**：本文通过分析发现 cross-attention 分数在生成早期（约第 20 个 token）对 gold passage 区分度最高，从而指导过滤时机——这种"分析先行、方法后置"的研究路径值得借鉴。
4. **Max Curve 评估范式**：以 ROUGE-L vs 延迟构建 Pareto 前沿曲线，并以曲线上下关系评价方法优劣，比单一指标对比更全面，适合性能-效率权衡类研究。
5. **延迟度量优先于 FLOPs**：本文强调使用真实推理延迟而非 FLOPs（MACs）作为效率指标，更贴合实际部署需求，对工程导向研究具有参考价值。

## 关键术语表
**FiD (Fusion-in-Decoder)**：一种将检索到的多个文档分别在 encoder 中编码、再在 decoder 中通过 cross-attention 融合以生成答案的检索增强生成架构。

**Token Filtering**：本文提出的在 decoder 解码阶段，基于 cross-attention 分数动态筛选 top-p% 高贡献输入 token 并从 key-value 状态中剔除其余 token 的效率优化方法。

**Cross-attention Score**：Decoder 中 query token 对 encoder 输入 token 的注意力权重，本文用作衡量输入 token 对答案生成贡献度的重要性指标。

**CALM (Confident Adaptive Language Modeling)**：Schuster et al. (2022) 提出的 decoder 层跳过方法，通过 confidence classifier 在每个 decoder 层判断是否可提前退出，以减少计算量。

**LFQA (Long-Form Question Answering)**：长形式问答任务，要求模型生成详细、结构化的长段落回答，以 ELI5 数据集为代表。

**ODQA (Open-Domain Question Answering)**：开放域问答任务，利用外部知识库（通常为大规模文档集合）为模型提供回答问题所需的背景信息。

**Max Curve**：在性能-效率评估中，对同一方法在不同超参数设置下的 (延迟, 性能) 点集，按延迟区间取各区间内最高性能点所构成的曲线，用于公平比较不同方法。

## 可复现要素
- **数据集**：ELI5、MS MARCO Passage Ranking、Natural Questions；均基于公开数据，检索索引为 Wikipedia dump（已通过 Elasticsearch 构建）
- **代码**：论文使用了 FiD 官方实现（HuggingFace Transformers），CALM 基于 Schuster et al. (2022) 的开源代码；Token Filtering 细节见 Appendix D
- **模型权重**：模型在 KILT ELI5 榜单上排名第一，权重可能通过 KILT 平台获取
- **关键超参**：训练使用 T5-Base/Large，AdamW，LR=5e-5，batch size=64，max seq len=235，training steps=60000，warmup=1000，torch.bfloat16；Token Filtering 最优参数在 dev 集上通过网格搜索确定（过滤比例 {10%, 30%, 50%}，过滤层 [1,L]，token 索引 [1,20]）
