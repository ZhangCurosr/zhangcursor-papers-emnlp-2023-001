---
title: "PEFTDebias-Capturing-debiasing-information-using-PEFTs"
source: https://aclanthology.org/2023.emnlp-main.122.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:51:56"
field: "自然语言处理中的公平性与偏见缓解"
keywords: ["去偏见", "参数高效微调", "PEFT", "性别偏见", "种族偏见", "基础模型", "Counterfactual Data Augmentation"]
innovations: ["提出 PEFTDebias 两阶段框架，上游学习去偏 PEFT 参数、下游冻结保留去偏信息", "系统比较 Adapters/Prompt Tuning/LoRA/SFT 在偏见轴上的去偏效果，发现 Prompt Tuning 最优", "验证去偏参数的跨数据集迁移能力，证明 task-agnostic 轴去偏的有效性"]
benchmarks: ["BiasBios", "GHC", "MNLI", "LHC", "StereoSet", "CrowS-Pairs"]
---

# 论文速读：PEFTDebias-Capturing-debiasing-information-using-PEFTs

## 一句话总结
本文提出 PEFTDebias，一种利用参数高效微调（PEFT）在预训练阶段捕获特定偏见轴的去偏参数的方法，通过在下游微调时冻结这些参数，有效减轻基础模型中的性别和种族偏见，同时保持任务性能。

## 研究问题与动机
- **核心问题**：预训练基础模型（如 BERT、GPT-3）存在隐含的社会偏见（性别、种族），这些偏见会传播到下游任务中（bias transfer hypothesis）。
- **现有方法不足**：
  1. 大多数去偏方法仅在下游微调阶段应用，需要额外的标注数据（如偏见属性标注）和辅助训练目标，增加了 fine-tuning 的难度。
  2. 上游去偏后再进行下游微调，无法保证去偏效果能转移到下游任务（修改所有参数可能导致 fairness forgetting）。
  3. 现有 PEFT 去偏研究主要针对下游阶段，未充分探索 upstream PEFT 的去偏参数跨任务迁移能力。

## 核心贡献（创新点）
1. **提出 PEFTDebias 框架**：通过上游 PEFT + CDA（反事实数据增强）在特定偏见轴上学习去偏参数，下游微调时冻结这些参数以保留去偏信息。与以往方法本质区别在于"上游去偏参数捕获→下游冻结保持"的两阶段设计，避免了 fairness forgetting。
2. **系统比较多种 PEFT 方法的去偏效果**：对比 Adapters、Prompt Tuning、LoRA、Sparse Finetuning 四种 PEFT 在性别/种族偏见轴上的表现，发现 Prompt Tuning 去偏效果最优且对任务性能影响最小。
3. **验证去偏参数的跨数据集迁移能力**：证明沿特定偏见轴学到的 PEFT 去偏参数可泛化到其他同轴数据集（BiasBios→MNLI 性别轴、GHC→LHC 种族轴），且性能与全量微调相当。

## 方法详解
PEFTDebias 包含两个阶段：

**上游阶段（Upstream Phase）**：
- 使用 PEFT 方法（Adapters/Prompt Tuning/LoRA/SFT）配合 Counterfactual Data Augmentation（CDA）在偏见轴特定数据上进行训练。
- CDA 原理：交换与偏见相关的属性词（如 he↔she，black↔caucasian）生成反事实样本。
- 基础模型参数冻结，仅训练 PEFT 参数 $\phi_{PEFT}^A$，捕获任务无关的特定轴去偏信息。

**下游阶段（Downstream Phase）**：
- 将上游学到的去偏 PEFT 参数注入可训练的预训练语言模型中。
- 在下游任务微调过程中**冻结 PEFT 参数**，仅训练基础模型参数 $\theta_{FM}$。
- 假设：冻结的去偏参数能保留上游去偏效应，防止模型在任务微调过程中重新习得偏见。

**算法伪代码关键步骤**（A.3）：
```
/* Upstream stage */
φ*_PEFT^A ← FT(θ_FM, φ_PEFT, D_upstream, A)  /* A 为偏见轴 */
/* Downstream stage */
θ*_FM ← FT(θ_FM, φ*_PEFT^A, D_labeled)  /* PEFT 参数冻结 */
return θ*_FM ∪ φ*_PEFT^A
```

**关键超参**：
- 所有 PEFT 参数量约为基础 LM 的 1%
- 上游训练：学习率 $1e^{-5}$，MLM 训练 10,000 步，每 1,000 步评估
- 下游训练：batch size 32，10 epochs
- 对于样本不平衡数据（GHC、Stormfront），对少数类（仇恨言论）损失加权 10× 或 6.7×

## 实验与结果
**数据集**：
- 上游：BiasBios（性别）、GHC（种族）
- 下游（同数据集）：BiasBios（性别）、GHC（种族）
- 迁移测试：MNLI（性别轴）、LHC（种族轴）

**评估指标**：
- 内在偏见指标：StereoSet（SS LM↑、SS Score↓）、CrowS-Pairs（↓）
- 外在偏见指标：TPR-GAP（性别）、FPRD / FPRD_IPTTS（种族）

**主要结果**：

| PEFT | SS LM ↑ | SS Score ↓ | CrowS ↓ |
|------|---------|-----------|---------|
| BERT | 85.68 | 60.03 | 57.25 |
| + Adapter | 86.45 | 57.1 | 53.82 |
| **+ Prompt** | 85.54 | **56.64** | **51.91** |
| + LoRA | 86.21 | 58.85 | 54.20 |

- **上游阶段**：Prompt Tuning 在 BiasBios 上获得最优内在偏见分数（StereoSet: 56.64, CrowS: 51.91）；Adapter 在 GHC 上表现较好（StereoSet: 58.56, CrowS: 55.15）。

**下游阶段关键结果**：
- 同数据集下游：所有 PEFT 方法任务性能与 BERT 基线差距在 5% 以内，但外在偏见显著降低。Prompt Tuning 在 BiasBios 上 TPR-GAP 降至 11.98（FT 为 13.05），GHC 上 FPRD 降至 0.54（FT 为 1.01）。
- 跨数据集迁移：Prompt Tuning 在 BiasBios→MNLI 迁移中 FN 升至 0.21（FT 仅 0.02），在 GHC→LHC 迁移中 FPRD 为 0.34（与 FT 持平）。

**最强结果**：Prompt Tuning 综合表现最佳——上游内在偏见降低最多，下游外在偏见减少显著，且跨数据集迁移能力优异。

## 相关工作脉络
1. **Jin et al. (2021)**：上游去偏方法，通过 fine-tuning 预训练模型并注入偏见属性标注，再用于下游任务。本文定位差异：无需下游任务标注数据，且用 PEFT 冻结保留去偏信息避免 fairness forgetting。
2. **Steed et al. (2022)**：质疑上游去偏对下游去偏效果的保证性。本文通过 PEFT 冻结机制回应此问题，证明冻结去偏参数可有效保留上游去偏效应。
3. **Lauscher et al. (2021)、Kumar et al. (2023)**：证明 Adapters 可用于下游去偏。本文扩展至多种 PEFT 方法并探索 upstream 学习+downstream 冻结的范式。
4. **Zmigrod et al. (2019)**：CDA 反事实数据增强去偏方法。本文将其与 PEFT 结合，在 upstream 阶段用 PEFT 捕获 CDA 产生的去偏信息。
5. **Hauzenberger et al. (2023)**：通过稀疏子网络识别不同偏见轴并组合去偏。本文聚焦 PEFT 参数化去偏的通用框架。
6. **全参数去偏（Full-Debias）**：直接 fine-tune 整个模型进行去偏。本文证明 PEFT 方法能以更少参数实现同等甚至更好的去偏效果。

## 局限性与未来方向
- **模型局限**：仅在 BERT 上验证，未扩展至 GPT-3 等大模型。
- **任务局限**：仅验证分类任务，未测试生成式任务的适用性。
- **偏见轴局限**：性别采用二元定义，种族仅覆盖特定属性词列表，不够全面。
- **标注偏见**：仅关注数据偏见，未处理任务标签中的标注偏见。
- **未来方向**：扩展至生成式模型和任务；探索多偏见轴的组合适配（composable multi-axis debiasing）。

## 研究启发与可借鉴点
1. **两阶段 PEFT 去偏范式**："上游学习去偏参数+下游冻结"的设计思路可迁移到其他需要保持特定性质的微调场景（如知识保留、公平性约束）。
2. **Prompt Tuning 作为去偏优选方法**：其对模型内部扰动最小，在保持任务性能的同时去偏效果最优，可作为后续研究的默认基线。
3. **CDA + PEFT 结合策略**：反事实数据增强与参数高效微调的协同机制，可用于其他需要去偏的场景（如年龄、残疾等偏见轴）。
4. **内在指标与外在指标联合评估**：同时使用 StereoSet/CrowS-Pairs（内在）和 TPR-GAP/FPRD（外在）评估去偏效果，为全面评估提供模板。
5. **跨数据集迁移验证**：将同一轴的去偏参数迁移到不同数据集验证 task-agnostic 性，该实验设计可用于评估去偏方法的泛化能力。

## 关键术语表
- **PEFT（Parameter-Efficient Fine-Tuning）**：参数高效微调，仅微调少量参数而非全量模型，如 Adapters、Prompt Tuning、LoRA 等方法。
- **CDA（Counterfactual Data Augmentation）**：反事实数据增强，通过替换敏感属性词（如 he→she）生成对抗样本，用于去偏训练。
- **Fairness Forgetting**：公平性遗忘，指在下游微调过程中原本去偏的参数表征被覆盖或丢失的现象。
- **Bias Transfer Hypothesis**：偏见转移假设，指预训练阶段习得的偏见会传播到下游微调模型中的现象。
- **TPR-GAP**：真实正例率差距，衡量不同性别群体在职业预测中正例率的差异，用于量化性别偏见。
- **FPRD（False Positive Rate Difference）**：假正例率差异，衡量提及受保护种族属性的样本与总体假正例率的差值，用于量化种族偏见。
- **StereoSet**：内在偏见评估基准，通过填空题评估语言模型在不同偏见类别（语言、通用、领域）上的刻板关联。
- **CrowS-Pairs**：内在偏见评估基准，通过最小对比句对评估模型对劣势/优势群体刻板印象的偏向。

## 可复现要素
- **数据集**：BiasBios、GHC、MNLI、LHC、StereoSet、CrowS-Pairs（均为公开数据集）
- **代码**：论文声明已开源（https://github.com/ 未完整给出，标记为 footnote 1）
- **权重**：未提及开源
- **关键超参**：学习率 $1e^{-5}$，batch size 上游 128/32，下游 32，PEFT 参数量约为 LM 的 1%，上游 MLM 训练 10,000 步，下游 10 epochs，少数类损失加权 10×/6.7×
