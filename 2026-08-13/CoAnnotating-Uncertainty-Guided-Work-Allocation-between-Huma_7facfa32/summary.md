---
title: "CoAnnotating-Uncertainty-Guided-Work-Allocation-between-Huma"
source: https://aclanthology.org/2023.emnlp-main.92.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:08:50"
field: "人机协作与数据标注"
keywords: ["LLM annotation", "human-AI collaboration", "uncertainty estimation", "data annotation", "prompt perturbation", "Pareto optimization"]
innovations: ["提出实例级不确定性引导的人机协同标注框架CoAnnotating", "发现多提示扰动显著优于同提示重复采样以增强不确定性估计", "揭示LLM自报告置信度不可靠并提出基于熵的稳定替代方案"]
benchmarks: ["AG News", "TREC", "MRPC", "TempoWiC", "Stance Detection", "Conversation Gone Awry"]
---

# 论文速读：CoAnnotating-Uncertainty-Guided-Work-Allocation-between-Huma

## 一句话总结
本文提出 **CoAnnotating** 框架，利用大语言模型（LLM）的响应不确定性（自评估置信度与熵）在实例级别估计其标注能力，从而在人类与LLM之间智能分配标注任务，以更低成本实现接近甚至达到人类水平的标注质量。

## 研究问题与动机
- 尽管 ChatGPT 等 LLM 在多项文本标注任务上展现出接近甚至超过人类 annotator 的零样本能力，但现有工作多将人类与 LLM 视为"竞争关系"，仅比较准确率，未探索两者如何**协作互补**。
- 手工标注成本高、效率低、主观性强；而 LLM 在复杂语义推理、细粒度理解任务上仍有明显短板，直接全部外包存在质量风险。
- 现有方法缺乏在**同一数据集内部**对单个样本进行细粒度分配的能力，难以在"标注质量"与"标注成本"之间取得平衡。
- 实践中需一种可操作的决策框架，帮助研究者在预算约束下决定哪些样本交给 LLM、哪些保留给人工标注。

## 核心贡献（创新点）
1. **提出 CoAnnotating 协作标注新范式**：首次将人类与 LLM 定位为互补协作者而非竞争者，从资源分配视角系统研究人机协同标注。
2. **实例级不确定性估计算法**：不同于以往仅做数据集级别的 LLM 能力评估，本文在样本级别量化 LLM 的不确定性，实现更精细的分配决策。
3. **多提示扰动增强不确定性估计**：设计 7 种不同类型的提示模板（如指令、序列交换、真/假问答、多项选择等），通过多次提示获得多样化响应，相比同提示重复采样显著提升不确定性度量的区分能力。
4. **引入 Pareto 效率进行策略选择**：将协同标注建模为多目标优化问题，通过 Pareto 前沿分析帮助实践者在不同预算下做出最优的成本-质量权衡决策。
5. **实证发现 LLM 自报告置信度的不可靠性**：实验表明 ChatGPT 的 self-evaluation 置信度分布高度偏斜（94.3% 样本置信度 > 0.8），无法有效区分标注质量，而基于熵的方法更稳定可靠。

## 方法详解
- **Prompt 构建**：为每个标注样本 $t_i$ 设计一组多样化提示 $P_i = \{p_{i1}, p_{i2}, ..., p_{ik}\}$，涵盖 7 种类型：基本指令（Instruction）、序列交换（Sequence Swapping）、语义改写（Paraphrase）、真/假题（True/False）、问答（Question Answering）、选择题（Multiple Choice）及确认偏置题（Confirmation Bias），以捕捉模型响应的敏感性。
- **不确定性计算**（两种方式）：
  - **自评估（Self-Evaluation）**：要求 LLM 输出 0-1 置信度，不确定性为 $u_i = 1 - \frac{1}{k} \sum_{j=1}^{k} P_\theta(a_{ij}|p_{ij})$，其中 $P_\theta$ 为模型直接给出的评分。
  - **熵（Entropy）**：基于 k 次不同提示的响应分布计算信息熵 $u_i = -\sum_{j=1}^{k} P_\theta(a_{ij}|p_{ij}) \ln P_\theta(a_{ij}|p_{ij})$，其中概率取各预测的频率，熵越大表示不确定程度越高。
- **工作分配策略**：
  - **随机分配（Random）**：随机选取 n 个样本交由 LLM，作为基线。
  - **自评估引导分配（Self-Evaluation Guided）**：按置信度降序排列，选取最不确定的样本分配给人工标注。
  - **熵引导分配（Entropy Guided）**：按熵值升序排列，选取最低不确定性的样本优先交由 LLM 标注，其余由人工完成；LLM 最终标签采用多数投票。
- **Pareto 策略选择**：绘制各分配比例下的测试性能（macro F1）与成本曲线，插值得到 Pareto 前沿，选取无法在不增加成本的前提下提升性能的点作为最优策略。

## 实验与结果
- **数据集**：6 个英文分类数据集，涵盖三类任务：主题分类（AG News、TREC）、语义相似度（MRPC、TempoWiC）、细腻理解（Stance Detection、Conversation Gone Awry）。
- **评估基线**：随机分配（Random）；自评估引导（Self-Evaluation）；熵引导（Entropy-Diff Prompts、Entropy-Same Prompt）。
- **主要结果**：
  - 在所有数据集上，**熵引导分配**普遍优于随机基线和自评估引导，平均性能提升最高达 **21%**（如 TempoWiC 在 40% LLM 比例下 F1 从 53.2 提升至 56.9）。
  - 在 AG News 等较简单任务中，通过 Pareto 分析发现可将约 **33% 的标注工作**外包给 ChatGPT 而保持人类水平性能。
  - 在 Stance Detection 和 Conversation Gone Awry 等细腻理解任务上，任何程度的 LLM 外包均导致质量下降，印证了任务难度依赖性。
  - 消融实验表明：**多提示扰动**的熵计算方法显著优于同提示重复采样的方式，后者因绝大多数样本熵为 0 而无法有效区分。
- **自评估不可靠**：在 AG News 上 94.3% 样本自报告置信度在 0.8-1 之间，区分力极弱；在 MRPC 上高置信度反而对应更差的标注对齐率。

## 相关工作脉络
1. **Weak Supervision 系列**（Ratner et al., 2017; Bach et al., 2019; Zhang et al., 2022）：利用启发式规则、特征标注或预训练模型生成噪声标签，本文与之不同在于使用 LLM 替代弱标签源，并通过不确定性做分配而非直接合并噪声标签。
2. **LLM 作为标注工具的比较研究**（Ding et al., 2022; Huang et al., 2023; Kuzman et al., 2023）：将 LLM 与人工标注对比准确率，视二者为竞争关系；本文转向协作视角，探索互补分配策略。
3. **Wang et al. (2021) "Want to reduce labeling cost?"**：首次利用 GPT-3 的 logit 做主动标注分配，是本文的核心灵感来源；本文扩展至多提示扰动和不一致性度量，并引入 Pareto 分析框架。
4. **Ziems et al. (2023)**：提出通过多数投票融合人类与 LLM 标注结果；本文不依赖融合，而是通过不确定性做选择性分配，减少 LLM 错误标注的污染。
5. **Kang et al. (2023) "Distill or annotate?"**：探索 LLM 蒸馏与人工标注的随机组合；本文在此基础上系统化地引入不确定性引导的分配策略。

## 局限性与未来方向
- **数据泄漏风险**：LLM 可能在训练过程中见过实验数据集，导致不确定性估计偏低，评估结果可能过于乐观。
- **未探索超人级性能**：当前目标仅为达到人类水平，未研究 LLM 在特定任务上超越人类的可能及条件。
- **仅考虑二分组的人机协作**：未纳入组内差异（如专家 vs. 众包工人、不同 LLM 版本之间的协作）。
- **任务类型受限**：仅在分类任务和英语数据集上验证，未扩展到生成类任务或多语言场景。
- **提示工程可进一步优化**：当前 7 种提示模板的设计具有经验性，尚未探索系统性优化提示以提升 LLM 标注能力的策略。

## 研究启发与可借鉴点
1. **多提示扰动增强不确定性估计**：通过多样化提示模板获取模型响应的分布特征，比单提示重复采样更能反映模型真实不确定性，该方法可迁移至其他 LLM 可靠性评估场景。
2. **Pareto 前沿用于资源分配决策**：将成本-质量权衡建模为多目标优化并可视化 Pareto 前沿，为实践者提供直观的策略选择工具，这一范式可推广至其他 AI-人工协作场景。
3. **对 LLM 自报告置信度的警示**：实验揭示自评估置信度在实际应用中可能严重失真，提示后续研究需谨慎对待 LLM 自我报告的可靠性，优先考虑基于响应分布的熵等方法。
4. **任务难度作为分配策略的设计依据**：简单任务可适当外包给 LLM，复杂细腻任务应保留给人工，这一发现可指导构建自适应的人机协作 pipeline。

## 关键术语表
**CoAnnotating**：人类与 LLM 协同标注框架，通过不确定性引导实例级别的工作分配。
**Uncertainty Estimation**：利用模型响应的不一致性（熵或自报告置信度）量化 LLM 对特定样本标注的把握程度。
**Pareto Efficiency/Frontier**：多目标优化中无法在不恶化某一目标的前提下改进另一目标的解集合，此处用于刻画成本-质量的权衡边界。
**Self-Evaluation**：要求 LLM 直接输出对预测结果的 0-1 置信度评分作为不确定性估计。
**Entropy-based Uncertainty**：基于多次提示响应的分布信息熵计算不确定性，熵越大表示预测越不一致。
**Instance-level Expertise**：在单个样本级别评估 LLM 的标注能力，而非仅做数据集级别的平均性能评估。
**Confirmation Bias Prompt**：引导性提问模板，预先表达某种立场以观察 LLM 是否受先验暗示影响，用于探测响应稳定性。

## 可复现要素
- **数据集**：AG News、TREC、MRPC、TempoWiC、SemEval-2016 Stance Detection（abortion topic）、Conversation Gone Awry，均为公开数据集。
- **代码**：已开源，见 https://github.com/SALT-NLP/CoAnnotating
- **关键超参**：
  - ChatGPT：gpt-3.5-turbo，temperature=0.7，max_tokens=800，top_p=0.95
  - RoBERTa base：Adam optimizer，learning_rate=2e-5
  - 提示数量 k=7（不同提示类型）
  - API 价格：$0.002/1k tokens
  - 人工标注成本：按数据集论文报告或假设每人$15/hour × 5 annotators
- **LLM 标注**：ambiguous 响应（如"I cannot determine..."）编码为新类别；最终标签采用多数投票。
