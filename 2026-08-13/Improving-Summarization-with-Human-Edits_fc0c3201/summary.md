---
title: "Improving-Summarization-with-Human-Edits"
source: https://aclanthology.org/2023.emnlp-main.158.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:29:05"
field: "文本摘要与人类反馈学习"
keywords: ["Human Edits", "SALT", "Unlikelihood Training", "序列对齐", "灾难性遗忘", "临床笔记生成", "RLHF"]
innovations: ["提出SALT框架，通过序列对齐联合likelihood和unlikelihood损失利用Human Edits训练摘要模型", "提出Imitation Edits概念，利用已有Ground Truth模拟编辑数据以降低数据成本", "提出RSALT，结合Replay机制有效缓解在增量编辑数据上训练时的灾难性遗忘"]
benchmarks: ["CC (Clinician Conversations)", "CCUser", "CNN/Daily Mail", "XSum", "ROUGE", "UMLS-F1", "SAGE"]
---

# 论文速读：Improving-Summarization-with-Human-Edits

## 一句话总结
本文针对摘要生成任务，提出了一种名为 **Sequence Alignment (un)Likelihood Training (SALT)** 的新训练框架，用于高效利用 **Human Edits（人类编辑数据）** 改进模型摘要质量。该方法通过序列对齐定位模型输出与人类编辑之间的差异，并分别施加反似然（unlikelihood）和最大似然（likelihood）损失进行训练，同时提出了 **Imitation Edits（模仿编辑）** 以降低对昂贵人工编辑数据的依赖，并有效缓解了持续学习中的灾难性遗忘问题。

## 研究问题与动机
*   **核心问题**：如何利用低成本、可持续的人类反馈（Human Edits）来进一步微调和提升已训练好的语言模型（特别是中小规模模型，如 T5）的摘要生成质量。
*   **现有方法不足**：
    1.  传统最大似然微调（ML）将所有错误（幻觉与轻微语法错误）同等对待，且对所有标注数据一视同仁，无法针对不同质量或难度的数据进行精细化优化。
    2.  主流的人类反馈强化学习（RLHF，如 DPO）主要通过奖励模型和偏好比较进行训练，获取大规模、高质量的专家偏好数据成本高昂，且在隐私敏感的垂直领域（如医疗）难以规模化实施。
    3.  直接利用用户在实际工作流中对 AI 摘要的编辑行为（Human Edits）是一种更自然、数据效率更高的反馈形式，但如何有效利用这种细粒度的 token 级编辑信息进行模型训练尚不充分探索。

## 核心贡献（创新点）
1.  **首次将人类编辑反馈探索扩展至自动临床笔记生成任务**，证明了在工作流中收集 Human Edits 的可行性与有效性。
2.  **提出 SALT 训练框架**：通过序列对齐（Needleman-Wunsch 算法）识别 AI 生成摘要（$S_{AI}$）与人类编辑摘要（$S_E$）之间的对应、插入、删除和替换关系，并分别设计 likelihood loss（强化保留/新增内容）和 unlikelihood loss（抑制被删除/修改内容）进行联合训练。与 RLHF/DPO 相比，SALT 利用了 token 级别的细粒度对齐信息，而非仅对整个序列进行惩罚。
3.  **提出 Imitation Edits 概念**：利用已有的 Ground Truth 摘要（$S_I$）模拟 Human Edits，通过与模型生成的 $S_{AI}$ 进行序列对齐构造训练数据，降低了对昂贵 Human Edits 数据的依赖。
4.  **提出 RSALT 以缓解灾难性遗忘**：结合 Rebuffi 等人的 Replay-based 持续学习方法，将历史 seen data 与新的 Human/Imitation Edits 数据混合训练，有效缓解了因分布偏移导致的性能下降。
5.  **证明了 SALT 在 Human Edit 数据上优于 DPO**：实验表明，在相同数据下，SALT 的性能（ROUGE, UMLS-F1, Reward Accuracy）均优于直接将 $S_{AI}$ 作为拒绝样本、$S_E$ 作为选择样本的 DPO 方法。

## 方法详解
*   **核心思想**：SALT 的核心是将摘要文本视为序列，通过序列对齐找出 $S_{AI}$ 和 $S_E$ 之间的差异，并对不同位置的 token 施加不同的训练目标。
*   **序列对齐**：使用 Needleman-Wunsch 算法对 $S_{AI}$ 和 $S_E$ 进行全局对齐，得到每个 token 的状态指示符：`C` (Correspondence/匹配), `I` (Inserted/插入), `D` (Deleted/删除), `S` (Substituted/替换)。
*   **损失函数**：
    *   **Likelihood Loss ($L_r$)**: 对 $S_E$ 中未被修改的 token（$E-NC$）和被插入/替换新增的 token（$E-C$）施加标准的负对数似然损失，即 $L_r(x, t) = -\log p_\theta(x_t | x_{<t}, U)$，鼓励模型生成正确内容。
    *   **Unlikelihood Loss ($L_p$)**: 对 $S_{AI}$ 中被用户删除或修改的 token（$AI-C$）施加反似然损失，即 $L_p(x, t) = -\log(1 - p_\theta(x_t | x_{<t}, U))$，抑制模型再次生成这些错误内容。
    *   **总损失**：$L_{S_{AI}}$ 和 $L_{S_E}$ 分别计算后加权求和，构成最终的 SALT 训练目标。权重参数 $w_{AI-C}, w_{AI-NC}, w_{E-C}, w_{E-NC}$ 用于调节各项损失的贡献。
*   **Imitation Edits ($S_I$)**：将已有的 Ground Truth 摘要 $S_I$ 视为对 $S_{AI}$ 的“模仿编辑”，通过同样的序列对齐流程计算 $L_{S_I}$，用于在缺乏真实 Human Edits 时的训练或数据增强。
*   **RSALT (Replay-based SALT)**：为应对在 CCUser 数据上继续训练 CC 模型时出现的灾难性遗忘，采用 replay buffer 策略。从已学习的 CC 数据中采样一部分，将其作为 Imitation Edits 与新的 CCUser Human Edits 数据一起进行 SALT 训练，损失函数为 $L = L_{SALT} + L_{RSALT}$。

## 实验与结果
*   **数据集**：
    *   **CC (Clinician Conversations)**: 63,000 条医生-患者对话及对应的 SOAP 格式人工标注摘要，用于初始模型训练。
    *   **CCUser**: 215 条 ASR 转录稿，包含 T5-small 模型生成的 AI 摘要 ($S_{AI}$) 和由医务秘书/医生编辑后的摘要 ($S_E$)，作为 Human Edits 数据集。
    *   **公开数据集**: CNN/Daily Mail, XSum 用于验证 Imitation Edits 的泛化性。
*   **评估指标**：ROUGE-1/2/L, UMLS-F1（基于 QuickUMLS 的医学术语 F1）, GPT-4 偏好排序 (MRR), 以及论文提出的 **SAGE** 指标（衡量模型生成内容与 $G_{w1}$ (AI 错误)、$G_{w2}$ (用户新增)、$G_{w3}$ (共同正确) 的匹配程度）。
*   **主要结果**：
    *   **Human Edits 实验 (Table 3)**: 在 CCUser 上，`SALT_{l+u}` (联合 likelihood 和 unlikelihood) 达到 R1=58.39, U-f=62.13，显著优于仅 likelihood 训练的 `SALT_l` (R1=57.77, U-f=61.02)。结合 RSALT 后 (`SALT_{l+u} + RSALT_{l+u}`), R1 提升至 60.43, U-f 提升至 63.44。在原始 CC 测试集上，`SALT_{l+u} + RSALT_{l+u}` (R1=36.26, U-f=48.69) 也优于或持平于基线模型 M (R1=36.07, U-f=48.97)，证明其缓解了遗忘。
    *   **Imitation Edits 实验 (Table 5, 6)**: 在 CC 和 CNN 的 seen/unseen 数据上，使用 Imitation Edits 的 `SALT_{l+u}` 均优于仅 likelihood 训练的 `SALT_l`。
    *   **SALT vs DPO (Table 7)**: 在 CCUser 上，`SALT_{l+u}` 的 R1 (0.394), R2 (0.215), Meteor (0.320) 和 Reward Acc (0.591) 均优于不同 $\beta$ 值下的 DPO。
    *   **偏好评估 (Figure 1)**: GPT-4 和人类评估者（CC 数据集，n=25）均倾向于 `SALT_{l+u}` 及其带 RSALT 的版本。

## 相关工作脉络
1.  **与传统微调对比**：传统 SFT/ML 仅最大化 $P(S_E|U)$，所有 token 权重相等。SALT 通过引入 $S_{AI}$ 和细粒度对齐，实现了对模型生成行为的“去伪存真”式引导。
2.  **与 RLHF/DPO 对比**：RLHF/DPO 依赖奖励模型或偏好对 $(S_{chosen}, S_{rejected})$ 进行序列级优化。SALT 直接利用 Human Edits 的 token 级编辑差异，信息更细粒度，且无需显式奖励模型。论文证明在 Human Edit 数据上 SALT 优于 DPO。
3.  **与 Unlikelihood Training 对比**：Welleck et al. (2019) 提出 UNILIKELY 训练用于减少重复、矛盾等。SALT 将 unlikelihood 应用于由序列对齐定义的特定 negative tokens (被用户修改/删除的部分)，目标更具体。
4.  **与自动临床笔记生成工作对比**：如 Schloss & Konam (2020), Ramprasad et al. (2023) 等聚焦于从零训练摘要模型。本文聚焦于如何利用额外的 Human Edits 进一步微调已训练好的模型。
5.  **与持续学习/灾难性遗忘缓解对比**：Rebuffi et al. (2017) 的 Replay 方法是基础。本文将其与 SALT 结合形成 RSALT，专门用于在 Human Edit 数据上微调时保留旧知识。

## 局限性与未来方向
*   **模型规模限制**：实验主要在 T5-small/large 上进行，未能在更大的 LLM 上验证 SALT 的效果。
*   **编辑动机的歧义性**：用户编辑可能源于纠正模型错误，也可能源于个人偏好，当前方法未能区分这两类编辑。
*   **对齐算法局限**：仅使用了全局序列对齐算法（Needleman-Wunsch），未来可探索局部对齐或结合医学实体识别等约束。
*   **LLM-in-the-loop 未探索**：未尝试用 LLM 来模拟或生成 Imitation Edits。
*   **隐私限制**：医疗数据隐私导致无法在公共平台完全复现（CC/CCUser 数据未公开），但提供了在 CNN/XSum 等公开数据集上的验证。

## 研究启发与可借鉴点
1.  **利用产品日志数据进行模型迭代**：对于已在实际场景中部署的模型，可以直接收集用户的修改记录（edit logs）作为高质量的、细粒度的反馈信号（类似 Human Edits），用于模型的持续优化，无需额外标注。
2.  **序列对齐作为构建负样本的通用工具**：Needleman-Wunsch 等序列对齐算法可用于任何“原始输出 vs. 修正后输出”的数据对，自动划分 token 级的正负样本，适用于文本修复、翻译编辑等多种场景。
3.  **Imitation Edits 的数据增强策略**：当缺乏真实的“编辑轨迹”数据时，可以利用已有的 ground truth 与模型当前输出进行对齐，构造模拟的编辑数据，作为数据增强或冷启动方案。
4.  **结合 Replay 与新范式**：在将新反馈（尤其是格式或来源不同的反馈）用于微调时，结合 replay buffer 和新的训练目标（如 SALT），可以有效平衡性能提升与灾难性遗忘。
5.  **SAGE 等过程导向评估**：除了 ROUGE 等表面指标，可以设计如 SAGE 般的评估方法，直接衡量模型在避免特定错误和采纳特定正确内容方面的表现，更贴合实际应用需求。

## 关键术语表
*   **Human Edits ($S_E$)**：用户在实际工作流中，基于 AI 生成的摘要 ($S_{AI}$) 进行修改后得到的最终摘要。
*   **Imitation Edits ($S_I$)**：利用已有的 Ground Truth 摘要，通过与模型输出 $S_{AI}$ 对齐而模拟出的编辑数据。
*   **SALT (Sequence Alignment (un)Likelihood Training)**：本文提出的训练方法，通过序列对齐识别差异，并联合使用 likelihood 和 unlikelihood loss 进行训练。
*   **RSALT (Replay-based SALT)**：结合 Replay 机制的 SALT，用于缓解在增量数据上训练时的灾难性遗忘。
*   **SAGE (System output Against the Generated and Edited sentence)**：一种评估指标，衡量模型在新数据上的输出与原始 AI 摘要、人类编辑摘要之间 token/concept 级别的匹配关系。
*   **Needleman-Wunsch Algorithm**：用于全局序列对齐的动态规划算法，本文用于对齐 $S_{AI}$ 和 $S_E$/$S_I$。
*   **Catastrophic Forgetting**：模型在学习新任务/数据时，对原有任务性能的严重下降现象。
*   **DPO (Direct Preference Optimization)**：一种无需显式奖励模型的 RLHF 变体，通过优化偏好对之间的概率比来训练模型。

## 可复现要素
*   **数据集**：CC 和 CCUser 数据集由于隐私原因**未公开**。附录中使用了公开的 **CNN/Daily Mail** 和 **XSum** 数据集进行 Imitation Edits 的补充实验。
*   **代码**：代码仓库地址在论文中给出：https://github.com/seasonyao/LearnFromHumanEdit。
*   **模型**：基于公开的 **T5-small** 和 **T5-large** 模型。
*   **关键超参**：训练步数 1000 步（Human Edits 实验），batch size=8；beam size=4，no-repeat-ngram-size=2，生成长度限制 (10, 100)；损失权重 $w_{AI-C}$ 设为 -1，$w_{AI-NC}$ 设为 1。
