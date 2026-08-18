---
title: "Large-Language-Models-Can-Self-Improve"
source: https://aclanthology.org/2023.emnlp-main.67.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:49:35"
---

# 论文速读：Large-Language-Models-Can-Self-Improve

## 一句话总结
本文证明，仅需输入问题而无须任何人工标注答案，大语言模型即可利用思维链（CoT）提示与自一致性（Self-Consistency）筛选出高置信度推理路径，并以其为监督信号对模型本身进行微调，实现推理能力的显著自我提升（LMSI），且该方法能有效泛化至域外任务并在极端低资源条件下依然成立。

## 研究问题与动机
1. **监督数据瓶颈**：现有 LLM 性能跃升高度依赖海量人工标注数据（如 FLAN、InstructGPT、Minerva），在低资源领域或特定任务中难以获取足够高质量的 ground truth 标签。
2. **无监督自我改进的空白**：人类可通过“自我思考”提升推理能力，但大模型是否能在无外部反馈的情况下仅凭输入序列完成自我训练，尚缺乏系统性验证。
3. **既有推理增强方法的局限**：CoT 与 Self-Consistency 等传统方法仅在推理阶段利用多次采样提升准确率，未将模型自生成的高质量推理轨迹反哺至训练阶段形成闭环。
4. **极端低资源场景的探索不足**：当训练问题与人工 Curated 的 Few-shot CoT 示例均极度匮乏时，模型能否自主生成足够质量的训练数据仍未知。

## 核心贡献（创新点）
1. **提出 LMSI 无监督自我改进框架**：利用无标签数据集配合 CoT 提示与自一致性生成高置信推理路径，直接用于原模型微调；与依赖人工 rationale 或真值过滤的先驱工作本质不同，本文完全摒弃 ground truth 标签。
2. **设计混合格式（Mixed-format）训练数据增强策略**：将同一条推理路径转化为四种提示-输出格式组合进行联合微调；与仅使用单一 Few-shot CoT 格式的基线相比，该设计有效防止模型过拟合特定交互风格，显著提升跨任务泛化。
3. **实现极端低资源下的全栈自生成**：提出模型自主生成新训练问题与 Few-shot CoT 提示样本的完整流程；与现有数据增强方法依赖人工种子集或真值标签不同，该方法在零真值、零手写示例条件下仍能驱动模型自我优化。
4. **揭示自改进后的关键超参迁移规律**：发现模型经 LMSI 训练后，最优多路径采样温度应从 0.7 提升至 1.2，且改进后模型用极少采样路径即可超越原模型多路径表现，为低算力部署提供实证依据。

## 方法详解
1. **多路径解码与自一致性筛选**：给定预训练 LLM $M$ 与无标签问题集 $\mathcal{D}^{\text{train}}$，对每个问题 $x_i$ 以采样温度 $T>0$ 生成 $m$ 条 CoT 推理路径 $\{r_{i_1}, \ldots, r_{i_m}\}$ 及对应答案 $\{y_{i_1}, \ldots, y_{i_m}\}$。通过多数投票选出最一致答案 $\tilde{y}_i = \arg\max_{y} \sum_{k=1}^m \mathbb{I}(y_{i_j}=y_{i_k})$，保留所有导向 $\tilde{y}_i$ 的推理路径构成训练子集 $\mathcal{D}^{\text{self-consistent}}$。
2. **置信度-准确率正交性验证**：实验表明，投票支持率（置信度）越高，$\tilde{y}_i$ 的真实准确率越高；错误答案通常仅获少数路径支持，对训练噪声的贡献有限，因此无需真值过滤即可安全用于 SFT。
3. **四种混合格式构造**：每条选中推理路径被转换为以下格式混合输入微调（防过拟合）：
   - Format 1：Few-shot CoT 示例 + 问题 + 输出 CoT 推理
   - Format 2：Few-shot 标准问答示例 + 问题 + 输出直接答案
   - Format 3：Zero-shot 问题 + “Let’s think step by step” + 输出 CoT 推理
   - Format 4：Zero-shot 问题 + 输出直接答案
4. **极端低资源扩展模块**：
   - **问题生成**：随机拼接少量示例问题作为 prompt，令模型续写生成新题，再用 Self-Consistency 过滤高置信题目。
   - **提示生成**：采用 “Step-by-Step” 零样本策略令模型生成 CoT 示例，作为 Few-shot 演示材料。
5. **微调配置**：在构造的 $\mathcal{D}^{\text{self-consistent}}$ 上以固定学习率与 batch size 微调原模型，得到 LMSI 模型。

## 实验与结果
- **数据集**：域内任务包括 GSM8K、DROP、ARC-C、OpenBookQA、ANLI-A2/A3；域外泛化任务包括 AQUA、SVAMP、StrategyQA、ANLI-A1、RTE、MNLI-M/MM。
- **基线模型**：PaLM 540B（主实验）、UL2 20B（消融）。
- **主要结果（PaLM 540B，Self-Consistency 评测）**：
  - GSM8K：74.4% → **82.1%**（+7.7%）
  - DROP：78.2% → **83.0%**（+4.8%）
  - OpenBookQA：90.0% → **94.4%**（+4.4%）
  - ANLI-A3：63.4% → **67.9%**（+4.5%）
  - 单路径 CoT-Prompting 的 LMSI 模型已接近或超越未改进模型的 Self-Consistency 性能。
- **OOD 泛化**：在全部 6 个域外任务上均取得提升，其中 ANLI-A1 大幅提升 +10.4%，MNLI-M/MM 提升 +8.2%/+9.8%。
- **极端低资源**：仅用 10 个真实问题+模型自生成问题微调，GSM8K 零样本 CoT 准确率达 **74.2%**，远超 Kojima et al. (2022) 的 43.0% 及 naive 扩展的 70.1%。
- **蒸馏验证**：将 LMSI 数据用于 62B PaLM 微调后达 57.4%，超越原始 540B 模型的 56.5%；8B 模型蒸馏后达 33.4%，超越原始 62B 模型的 29.7%。

## 相关工作脉络
1. **Self-training / Pseudo-labeling**：传统方法用分类器为无标签数据打硬伪标签；本文将其扩展至序列生成任务，以 CoT+Self-Consistency 筛选的推理轨迹替代伪标签，适用于复杂推理而非简单类别预测。
2. **CoT 与 Self-Consistency（Wei et al., 2022c; Wang et al., 2022c）**：先前工作仅将多路径采样用于推理时解码；本文将其移至训练阶段，形成“生成→筛选→微调”的闭环，实现模型能力的永久性跃升。
3. **Rationale 学习与过程监督（Lightman et al., 2023; Zelikman et al., 2022）**：STAR 等方法依赖人工或真值辅助过滤/生成 rationale；本文完全不依赖 ground truth，且证明混合多路径比单一真值路径更具泛化价值。
4. **知识蒸馏至小模型（Snell et al., 2022; dhar et al., 2023）**：本文不仅验证蒸馏可行性，还首次展示 L
