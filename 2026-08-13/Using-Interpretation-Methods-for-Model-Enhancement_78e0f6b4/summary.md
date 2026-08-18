---
title: "Using-Interpretation-Methods-for-Model-Enhancement"
source: https://aclanthology.org/2023.emnlp-main.28.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:34:29"
field: "可解释NLP / 低资源语言理解"
keywords: ["模型解释", "rationale增强", "低资源NLP", "Input Marginalization", "DiffMask", "对比损失", "可解释深度学习"]
innovations: ["提出通用框架UIMER统一整合多种解释方法通过gold rationale增强模型", "首次将Input Marginalization和DiffMask用于模型增强并设计对比损失与交替训练策略", "在IC/SF/NLI三任务多低资源设定下系统验证，证明新方法在低资源下显著优于梯度类方法"]
benchmarks: ["SNIPS Intent Classification", "SNIPS Slot Filling", "e-SNLI Natural Language Inference"]
---

# 论文速读：Using-Interpretation-Methods-for-Model-Enhancement

## 一句话总结
本文提出了 **UIMER**（Utilizes Interpretation Methods and gold Rationales）框架，通过将模型生成的人工解释（attribution scores）与黄金rationale对齐作为额外损失，利用各类解释方法增强模型性能；提出的两种新方法（基于划除/替换和基于提取器）在大多数低资源场景下均优于已有梯度类方法。

## 研究问题与动机
- **核心问题**：当训练数据中存在或可快速生成黄金rationale（对预测结果关键的输入token子集）时，如何有效利用这些rationale来增强下游任务模型？
- **现有方法的不足**：此前仅少数工作利用基于梯度（gradient-based）的解释方法进行模型增强，既未系统比较不同解释方法的效果，也未在多样化任务上全面评估；此外，这些方法在低资源场景下仍有较大提升空间。
- **动机**：梯度类方法容易受到分布外（OOD）问题影响；划除/替换类（erasure/replace-based）和提取器类（extractor-based）方法尚未被用于模型增强，存在探索空白。

## 核心贡献（创新点）
1. **提出通用框架 UIMER**：在任务损失之外引入对齐attribution scores与gold rationales的额外损失，并可兼容多种解释方法；与先前基于梯度的工作的本质区别在于——本文是一个统一框架，而非单一方法。
2. **提出 UIMER-IM**（基于Input Marginalization划除/替换方法的新实例）：设计了针对多token的归因分数计算方式及对比间隔损失 $L_{int}$，无需单独计算每个token的分数即可高效对齐rationale；此前类似工作（Du et al., 2019）存在严重OOD问题且效果不稳定。
3. **提出 UIMER-DM**（基于DiffMask提取器方法的新实例）：将独立的可微掩码提取器嵌入框架，设计了新的对比损失及交替训练策略；与简单将解释方法作为正则项的做法相比，本文通过冻结提取器/任务模型的异步训练保证了二者同步性。
4. **系统验证与深入分析**：在IC、SF、NLI三个任务、多组低资源设定下全面评估，并揭示了解释质量与任务性能之间的正向关联（Table 4）。

## 方法详解
- **整体框架**：总损失为 $L_\theta = L_{task}(x, y) + \alpha L_{int}(\boldsymbol{a}, \boldsymbol{g})$，其中 $\boldsymbol{a}$ 为解释方法产生的attribution scores，$\boldsymbol{g}$ 为 $0/1$ 编码的gold rationale向量，$\alpha$ 为平衡系数。
- **Warm-up Training**：先仅优化 $L_{task}$ 对模型进行预热训练（使模型具备基本任务知识），再引入 $L_{int}$ 联合训练，避免从零开始直接对齐解释与rationale。
- **UIMER-GB**（第3.1节）：将先前工作（Ghaeini et al., 2019；Huang et al., 2021）统一为框架特例，其中 $a_i = f(\partial J / \partial x)$，$L_{int}$ 度量 $\boldsymbol{a}$ 与 $\boldsymbol{g}$ 之间的距离。
- **UIMER-IM**（第3.2节）：采用Input Marginalization（Kim et al., 2020）计算归因，通过将rationale / non-rationale token替换并衡量对输出的影响，扩展至多token版本 $a_R$ 和 $a_N$；损失为对比间隔损失 $L_{int} = \max(a_N - a_R + \epsilon, 0)$，而非简单最大化 $a_R$ 最小化 $a_N$，以保留non-rationale可能提供的有用信息。
- **UIMER-DM**（第3.3节）：采用DiffMask（De Cao et al., 2020）提取器 $Ext_\phi$ 产生per-token的 $a \in [0,1]$；$L_{int}$ 为 $L_{int} = \sum_{i: g_i=1} \min\left(\frac{a_i}{\max_{j:g_j=0} a_j} - 1, 0\right)$，鼓励所有rationale的分数高于最高non-rationale分数；训练时采用**交替异步策略**：冻结任务模型优化提取器 $\phi$，再冻结提取器优化任务模型 $\theta$，以避免二者不同时训练导致解释与模型状态失配。

## 实验与结果
- **数据集与任务**：SNIPS（Intent Classification + Slot Filling，人工/正则构造rationale）、e-SNLI（NLI，自带人类标注rationale）。
- **低资源设定**：IC/SF采用1/3/10/30-shot及full；NLI采用100/500训练样本。基线：BERT-base-uncased + Softmax（IC）、+CRF（SF）、线性层（NLI）；强基线：UIMER-GB（Ghaeini et al. 2019、Huang et al. 2021）。
- **主要结果**：所有UIMER变体平均得分均优于Base和UIMER-GB。UIMER-IM在12组设定中8组取得最佳，1组次佳，例如NLI 100-sample下"+ BERT (warm.)"相比Base提升**14.86%**（62.08→68.89）；IC 1-shot下"+ Uniform (warm.)"达75.79%（Base 65.71%）。UIMER-DM在SF的1/3-shot下最优（41.32 / 53.10 F1），多轮交替训练在几乎所有设定下均有明显增益（表4显示：UIMER-DM多轮IC 1-shot从66.75→70.21）。整体趋势：低资源下性能差距更大，印证rationale在数据匮乏时的价值。

## 相关工作脉络
1. **Gradient-based增强（Ghaeini et al., 2019；Huang et al., 2021）**：本文将其统一为UIMER-GB特例，指出先前工作仅聚焦梯度类方法，缺乏跨方法对比。
2. **Rationale regularization（Du et al., 2019）**：与方法UIMER-IM相关，但原方法受OOD问题困扰且在实验中未稳定超越基线；本文通过Input Marginalization的多token扩展和warm-up策略克服了该问题。
3. **Input Marginalization（Kim et al., 2020）**：原作为解释方法提出，本文首次将其应用于模型增强，并扩展至多token归因分数及对比损失设计。
4. **DiffMask（De Cao et al., 2020）**：原作为可微掩码提取方法，本文首次将其嵌入模型增强框架并设计交替训练策略，解决了提取器与任务模型同步更新的难题。
5. **Attention作为解释**：Jain & Wallace (2019) 指出注意力权重不可靠，本文框架不直接使用attention scores作为解释，而是依赖更稳健的梯度/划除/提取器方法。
6. **Saliency learning（Ghaeini et al., 2018）**：早期利用注意力/梯度解释的工作，但仅在NLI等单一任务上验证，本文在3任务×多资源设定下进行全面对比。

## 局限性与未来方向
- 在**高资源（full）设定下新方法并非最优**：UIMER-IM和UIMER-DM均表现平平，作者解释为充足训练数据下模型已隐式学到足够信息，注入rationale的收益有限。
- Rationale获取依赖人工/规则：本文IC使用人工构建关键词字典、SF使用正则表达式，泛化性受限，难以直接迁移到无rationale标注的任务。
- 实验仅限IC、SF、NLI三类任务，未验证生成类任务（如机器翻译、文本生成）中的效果。
- 未来方向：扩展至更多类型的rationale表示形式（如实数权重）和更多解释方法。

## 研究启发与可借鉴点
1. **Warm-up训练策略可广泛复用**：先预训练任务模型再引入解释对齐损失，避免模型从零学习"关注什么"，这一策略可迁移至任何结合外部知识的训练范式。
2. **交替异步训练解决双模型耦合问题**：UIMER-DM中冻结一方优化另一方的策略，对任何"主模型+辅助解释模型"的架构均有参考价值。
3. **多token归因扩展技巧**：将逐token的IM归因扩展为rationale集合级 ($a_R$) 与non-rationale集合级 ($a_N$) 的对比，大幅降低计算复杂度，可推广至其他需要group-level解释的方法。
4. **低资源场景的系统评测范式**：在多个shot级别（1/3/10/30/full）下综合评估，清晰展示方法的有效边界，值得作为低资源NLU研究的标准评测流程。
5. **Rationale可通过低成本规则/字典构建**：IC任务仅用15分钟构建关键词字典，SF任务用正则表达式30分钟内完成rationale生成，表明在特定领域低成本获取rationale是可行的工程路径。

## 关键术语表
- **UIMER**：Utilizes Interpretation Methods and gold Rationales的缩写，本文提出的通用模型增强框架。
- **Attribution Score（归因分数）**：衡量输入token对模型最终预测贡献程度的数值指标。
- **Gold Rationale**：人工标注或通过规则/知识源生成的、对预测结果最关键的输入token子集（0/1向量表示）。
- **Input Marginalization（IM）**：通过逐token替换并度量输出概率变化来计算归因分数的解释方法（Kim et al., 2020）。
- **DiffMask（DM）**：基于可微掩码的提取器方法，用独立轻量模型学习掩码以解释任务模型的关注区域（De Cao et al., 2020）。
- **Warm-up Training**：先在纯任务损失上训练模型，再引入解释对齐损失的逐步训练策略。
- **Contrastive Margin Loss**：UIMER-IM中设计的损失函数 $\max(a_N - a_R + \epsilon, 0)$，强制rationale归因高于non-rationale并留出margin。
- **Multi-round Alternating Training**：UIMER-DM中交替冻结任务模型和提取器分别优化的多轮训练策略。

## 可复现要素
- **数据集**：SNIPS（公开）、e-SNLI（公开）；rationale为人工构建/正则提取/数据集自带。
- **代码**：已开源（https://github.com/Chord-Chen-30/UIMER）。
- **关键超参**：$\alpha \in [0.001, 20]$，$\epsilon \in [0.01, 10]$，learning rate依shot数在 $8\times10^{-5} \sim 1\times10^{-3}$ 范围调优；batch size 8-32；optimizer为AdamW + linear schedule with warmup；每种子种子4次取平均。详细超参见论文Appendix Table 6-8。
