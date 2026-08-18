---
title: "Explanation-Selection-Using-Unlabeled-Data-for-Chain-of-Thou"
source: https://aclanthology.org/2023.emnlp-main.41.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:39:54"
field: "大语言模型提示工程与推理"
keywords: ["chain-of-thought", "in-context learning", "prompt optimization", "explanation selection", "proxy metric", "silver labeling"]
innovations: ["提出基于无标签开发的解释组合两阶段搜索框架，通过留一法生成候选解释并用代理指标高效筛选", "设计单样本银标准确率 S_OSAcc 和单样本对数似然 S_OSLL 两种低成本代理指标，用于优先排序解释组合", "证明优化后的解释具有跨域泛化能力，可在未见过推理数据集上持续提升性能"]
benchmarks: ["GSM", "ECQA", "E-SNLI", "STRATEGYQA", "SVAMP", "MAWPS"]
---

# 论文速读：Explanation Selection Using Unlabeled Data for Chain-of-Thought Prompting

## 一句话总结
本文提出了一种在**黑盒模式下**利用无标签数据优化链式思维（CoT）提示中解释（explanation）组合的方法，通过留一法生成候选解释，并设计两个代理指标（单样本银标准确率 $S_{\text{OSAcc}}$ 与单样本对数似然 $S_{\text{OSLL}}$）高效筛选出能提升下游文本推理任务准确率的解释组合。

## 研究问题与动机
- 现有 CoT 提示中，即使针对同一问题，不同解释也会导致下游任务准确率出现显著波动（最大差距可达 20%），且众包标注的"种子解释"未必是最优选择。
- 手动工程化地编写或挑选解释成本高昂，即使是专业"prompt 工程师"也难以轻易找到最佳组合。
- 已有工作多依赖大量全标注数据微调模型或使用可微分提示模板，难以直接适用于**黑盒 LLM** 场景下的解释组合优化。
- 需要在仅有一组少量 few-shot 示例（含种子解释）和一组无标签开发集的前提下，以有限计算预算找到更优的解释组合。

## 核心贡献（创新点）
1. **提出了基于解释组合搜索的 ICL 优化框架**：通过将每个问题的候选解释集合限制为有限候选集（而非整个词汇空间），将指数级搜索空间降维至 $N^K$，再通过两阶段策略搜索最优组合。与以往仅优化 verbalization 或 exemplar 顺序的工作相比，本文聚焦于**解释内容的选择与组合优化**。
2. **引入伪标签（silver-labeling）机制评估解释组合**：利用无标签开发集 $V$，通过采样多个解释组合并进行多数投票生成伪标签 $\hat{a}$，以银标准确率作为替代目标函数，避免了对全标注数据的依赖。与 Zelikman 等人的 self-play/微调方法不同，本文完全不更新模型参数。
3. **设计了两个低成本的代理指标（proxy metrics）用于优先排序**：$S_{\text{OSAcc}}$（基于无标签集的 one-shot 银标准确率之和）与 $S_{\text{OSLL}}$（基于训练集的 one-shot log likelihood 之和），可在计算预算内快速筛选出高潜力的解释组合。与 Gonen 等提出的 perplexity-based 筛选策略不同，这两个指标直接建模了解释组合与下游表现的关系。

## 方法详解
**整体框架（两阶段搜索）：**

1. **候选解释生成（Leave-one-out）**：对每个 exemplar $(q_i, \tilde{e}_i, a_i)$，使用其余 $K-1$ 个含种子解释的示例作为 in-context prompt，让 LLM 以 temperature=0.7 采样生成 $N=40$ 个候选解释 $(\hat{e}, \hat{a})$，仅保留答案 $\hat{a} = a_i$ 正确的样本，每个问题最多保留 8 个候选解释，并始终包含原始种子解释。

2. **无标签开发集伪标注**：从候选组合中随机采样若干组合，对无标签开发集 $V$ 中的每个问题 $q$，通过多数投票得到银标 $\hat{a}$：
$$\hat{a} = \arg\max_a \sum_C \mathbb{1}[a = \arg\max_{\bar{a}} p(\bar{a} \mid \{(q_i, e_i, a_i)\}_{i=1:K}, q; \theta)]$$

3. **代理指标设计**：
   - **$S_{\text{OSAcc}}$**（单样本银标准确率）：计算组合中每个解释在开发集上的 one-shot 准确率之和，作为组合性能代理：
   $$S_{\text{OSAcc}}(C) = \sum_{i=1:K} \sum_{q_j \in V} \mathbb{1}[\hat{a}_j = \arg\max p(\bar{a} \mid (q_i, e_i, a_i), q_j; \theta)]$$
   - **$S_{\text{OSLL}}$**（单样本对数似然）：利用训练集 $T$，计算每对解释之间的 one-shot log probability 之和，无需开发集：
   $$S_{\text{OSLL}} = \sum_{j=1:K} \sum_{i=1:K \wedge i \neq j} \log p(e_j, a_j \mid (q_i, e_i, a_i), q_j; \theta)$$
   - **ENSEMBLE**：分别用 $S_{\text{OSAcc}}$ 和 $S_{\text{OSLL}}$ 各选出一批候选组合，取并集后用银标准确率 $\mathcal{O}(C)$ 最终排序。

4. **预算约束下的搜索**：在总计算预算 $B$（以 PASS 为单位，1 PASS = 对开发集做完整 K 示例推理）内，先用代理指标筛选 Top-K 组合，再用银标准确率精排。

## 实验与结果
- **数据集**：GSM（小学数学）、ECQA（常识问答）、E-SNLI（自然语言推断）、STRATEGYQA（Yes-No 多步推理），共 4 个任务。
- **基线模型**：code-davinci-002（主实验）、text-davinci-003（验证泛化性）。
- **主要结果（Table 3，greedy decoding）**：
  - GSM：SEED 62.6 → OPTIMIZED **66.0**（+3.4%）
  - ECQA：SEED 77.0 → OPTIMIZED **83.0**（+6.0%）
  - E-SNLI：SEED 75.2 → OPTIMIZED **82.8**（+7.6%）
  - STRATEGYQA：SEED 71.3 → OPTIMIZED **71.6**（+0.3%，不显著）
  - **平均提升约 4%**。
- **代理指标有效性（Table 2）**：在 MAX@16 设置下，$S_{\text{OSAcc}}$ 在 GSM/ECQA/E-SNLI 上分别优于随机基线 1.0%/0.9%/1.4%；$S_{\text{OSLL}}$ 在 ECQA/STRATEGYQA 上分别优于随机 2.0%/0.9%。两种指标各有优势，ENSEMBLE 综合最优。
- **Self-consistency 扩展（Table 4）**：在 5~40 次采样下，OPTIMIZED 解释始终优于 SEED，小样本时差距更明显。
- **跨模型泛化（Table 5）**：在 text-davinci-003 上同样取得显著提升（GSM +3.1%，ECQA +2.6%，E-SNLI +7.6%）。
- **跨域泛化（Table 6）**：在 GSM 上优化的解释迁移到 SVAMP、SINEQ 等未见过的数学推理数据集，均有提升。
- **低预算设置（Table 7）**：预算从 50 降至 20 时，仍可实现约 2%~6% 的提升。

## 相关工作脉络
1. **Chain-of-Thought Prompting**（Wei et al., 2022；Nye et al., 2021）：引入解释性中间推理步骤；本文与其区别在于，CoT 工作侧重于"是否加解释"，而本文聚焦于"如何选择最优解释组合"。
2. **Exemplar/Example Selection for ICL**（Ye et al., 2023；Rubin et al., 2022）：研究如何在众多示例中选择子集；本文与其互补，研究的是在给定示例集合内如何为每个示例选择更优的解释文本。
3. **Perplexity-based Prompt Optimization**（Gonen et al., 2022；Fu et al., 2022 的 BESTPPL/BESTLEN）：利用 prompt 的困惑度或长度作为筛选准则；本文的 $S_{\text{OSLL}}$ 与之相关但扩展到组合评估，且引入了基于银标的 $S_{\text{OSAcc}}$ 作为补充。
4. **Self-Play / Bootstrapping Reasoning**（Zelikman et al., 2022 的 STAR；Huang et al., 2022）：利用 LLM 生成数据并微调模型；本文的差异化在于完全**黑盒**操作，不调用梯度也不更新模型参数，仅优化 prompt 中的解释组合。
5. **Black-box Prompt Optimization**（Deng et al., 2022 的 RLPrompt；Prasad et al., 2022 的 GRIPS）：在离散模板或短字符串上进行搜索；本文将搜索空间扩展到自然语言解释的组合，规模更大、更具挑战性。

## 局限性与未来方向
- **高度依赖 LLM 能力**：方法全程使用 LLM 生成候选解释、伪标注和打分，对能力较弱的模型收益有限。
- **优化成本仍存在**：虽推理阶段开销与普通 few-shot 相同，但优化阶段需对开发集做多次推理（等价于约 500 条 self-consistency 推理量），对于高频部署场景仍需权衡。
- **代理指标并非始终有效**：在 STRATEGYQA 等二分类简单任务上，$S_{\text{OSAcc}}$ 失效；在存在严重分布偏移的任务上，$S_{\text{OSLL}}$ 可能因过拟合训练分布而失效。
- **仅优化解释，忽略其他因素**：未同时优化 verbalization、exemplar 顺序等，后续可将本方法与这些方向联合优化。
- **仅评估英文推理数据集**：对多语言场景及其他推理类型（如纯符号推理）的适用性待验证。

## 研究启发与可借鉴点
1. **代理指标 + 银标精排的两阶段搜索范式**可迁移至其他 prompt 组件优化场景（如 verbalization、ordering、selection），以低开销筛选高潜力候选。
2. **留一法（leave-one-out）生成候选解释**的思路可扩展到其他需要"变体生成"的 prompt 工程问题，如生成不同风格的 few-shot 示例。
3. **无标签开发集 + 多数投票伪标注**的评估策略，在缺少 gold label 的实际部署场景中具有重要实用价值。
4. **ENSEMBLE 策略**（融合多个互补代理指标）是一种稳健的启发式组合方案，可推广至多指标排序场景。
5. **跨域泛化验证**（Table 6）表明优化后的解释具有迁移性，提示后续可在单一领域优化后跨任务复用 prompt。

## 关键术语表
- **Chain-of-Thought (CoT) Prompting**：在 few-shot 示例中加入自然语言推理步骤（解释），引导 LLM 进行多步推理的技术。
- **In-Context Learning (ICL)**：不更新模型参数，仅在 prompt 中提供少量示例，让 LLM 直接完成目标任务的学习范式。
- **Silver-labeling（伪标注）**：利用模型自身预测（如多数投票）为无标签数据生成近似标签，用于评估或训练。
- **Proxy Metric（代理指标）**：成本低、易于计算的指标，用来近似难以直接优化的真实目标（如银标准确率）。
- **Leave-one-out Generation**：在生成某示例的候选解释时，临时排除该示例本身，使用其余示例作为 in-context prompt 进行采样。
- **Self-consistency Decoding**：对同一 prompt 多次采样（temperature > 0），取输出中出现频率最高的答案作为最终预测。
- **Compute Budget（计算预算 B）**：以"对开发集做一次完整 K 示例推理"（1 PASS）为单位衡量的评估开销上限。
- **ENSEMBLE 策略**：分别用 $S_{\text{OSAcc}}$ 和 $S_{\text{OSLL}}$ 选出候选组合，取并集后再用银标准确率排序的最终筛选方法。

## 可复现要素
- **数据集**：GSM（Cobbe et al., 2021）、ECQA（Aggarwal et al., 2021）、E-SNLI（Camburu et al., 2018）、STRATEGYQA（Geva et al., 2021），均为公开数据集；SVAMP 和 MAWPS 子集亦为公开。
- **代码/权重**：论文未提及开源代码或模型权重；实验使用 OpenAI API（code-davinci-002、text-davinci-003），非开源模型。
- **关键超参**：候选解释采样数 N=40、每问题最多保留 8 个候选、temperature=0.7、开发集大小 M=256、计算预算 B=50（主实验）/B=20（低预算实验）、self-consistency 采样数 5~40；E-SNLI 使用 9 个 exemplar（每类 3 个），其余数据集使用 8 个 exemplar。
