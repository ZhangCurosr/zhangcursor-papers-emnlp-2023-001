---
title: "DeSIQ-Towards-an-Unbiased-Challenging-Benchmark-for-Social-I"
source: https://aclanthology.org/2023.emnlp-main.191.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:37:59"
field: "多模态社会智能理解"
keywords: ["Social Intelligence", "Debiasing", "Multimodal QA", "Benchmark", "Commonsense Knowledge", "Social-IQ"]
innovations: ["提出六类系统性偏差检测方法（NCAQ/MPM/RIWI/RIWA/RAWI/RAWA）量化Social-IQ答案侧偏差", "构建去偏基准DeSIQ（DeSIQd/DeSIQs）通过RIWA扰动替换错误答案", "提出T5-small_Delphi通过社会常识知识蒸馏提升去偏基准性能"]
benchmarks: ["Social-IQ", "Social-IQ-2.0", "DeSIQ", "DeSIQd", "DeSIQs"]
---

# 论文速读：DeSIQ: Towards an Unbiased, Challenging Benchmark for Social Intelligence Understanding

## 一句话总结
本文发现社会智能基准数据集 Social-IQ 存在严重偏差——即使不给上下文和问题、仅给答案选项，小模型 T5-small 也能达到 100% 准确率；为此提出通过替换错误答案的去偏方法构建 DeSIQ 数据集，并建立新的多模态基线模型与常识知识注入策略。

## 研究问题与动机
1. **Social-IQ 基准有效性存疑**：该数据集旨在评估机器社会智能，但经验证发现其偏差严重，模型可借此捷径（shortcut）达到虚假完美成绩，无法真实反映社会智能理解能力。
2. **现有去偏方法局限**：Shah et al. (2020) 仅从期望检验角度分析 MCQA 偏差，Gat et al. (2020) 从模型侧平衡多模态权重，Ye & Kovashka (2021) 仅修改验证集并做模型侧 masking 训练——均未从数据侧系统性地检测并消除 Social-IQ 答案侧偏差。
3. **需要可复现、可迁移的去偏范式**：如何构建一个既去偏又具挑战性的社会智能多模态基准，以推动更真实的模型评估成为关键问题。

## 核心贡献（创新点）
1. **提出六类系统性偏差检测方法**：首次从数据侧定义 NCAQ/MPM/RIWI/RIWA/RAWI/RAWA 六种扰动设置来量化 MCQA 数据集偏差，本质区别于已有工作仅靠模型侧正则化或单次验证的方式。
2. **构建去偏基准 DeSIQ**：通过 RIWA 扰动将错误答案替换为其他问题的正确答案，构造 DeSIQd（跨视频采样）与 DeSIQs（同视频采样）两个版本，打破正确/错误答案间的可分离性。
3. **引入常识知识蒸馏提升性能**：基于 Social Chemistry 101/ETHICS/Moral Stories 对 T5-small 进行预训练再微调得到 T5-small_Delphi，在社会智能任务上显著提升，证明常识注入对去偏基准仍有价值。
4. **揭示多模态输入的真实作用边界**：系统比较 q/a/t/v/s 五种输入组合下模型表现，发现 transcript 通常比 video 更有效，而 raw transcript 过长反而损害性能。
5. **建立更强基线并超越 GPT-3/ChatGPT**：新架构在 DeSIQ 上达到最高 76.77%（A2）/74.51%（A4），超过同等设定下 GPT-3/ChatGPT 的 few-shot 成绩。

## 方法详解
**偏差检测方法体系**（Section 2.2）：
- **NCAQ（No context and question）**：仅输入答案选项 a 和 i，不给问题和上下文，检验模型是否能凭答案本身猜对。
- **MPM（More Powerful Model）**：用更强模型（T5-small vs. LSTM）测试性能是否"不合理地"接近 100%。
- **RIWI**：$(q,a,i) \to (q,a,i')$，将错误答案替换为其他问题的错误答案，期望性能不变。
- **RIWA**：$(q,a,i) \to (q,a,a')$，将错误答案替换为其他问题的正确答案，期望性能大幅下降（接近随机）。
- **RAWI**：$(q,a,i) \to (q,i',i)$，将正确答案替换为错误答案，期望性能接近随机。
- **RAWA**：$(q,a,i) \to (q,a',i)$，将正确答案替换为其他问题正确答案，期望性能接近随机。

**DeSIQ 构造**（Section 3.1）：
- 对所有训练/开发集的 $(q,a,i)$，随机采样另一视频中的 $(q',a',i')$，令 $i \leftarrow a'$，得到 DeSIQd。
- 同视频内采样得到 DeSIQs，因其答案更可能指向同一实体，更具挑战性。

**新模型架构**（Section 4 / Figure 4）：
- 将 q/a/i/t/v/s 各特征经 LSTM/投影层映射到同维，concatenation 后送入 T5-small 骨干。
- T5-small_Delphi：先在 Social Chemistry 101、ETHICS、Moral Stories 上预训练注入社会常识，再在下游 Social-IQ/DeSIQ 微调。
- DeSIQ-2.0 使用 ViT + Wav2Vec 2.0 提取原始视频/音频特征。

**零/少样本评估**（Section 3.2）：
- 零样本提示：`Choose the correct answer option corresponding to the question: [q] A: [a] B: [i]`，答案顺序随机 shuffle。
- 少样本：用 Sentence-Transformers 语义距离在训练集选取 top-3 相似问题作为示例。

## 实验与结果
**数据集**：
- Social-IQ v1：888 视频 / 5,328 问题 / 21,312 正确答案 / 15,984 错误答案（A2 与 A4 配置）。
- Social-IQ-2.0：987 视频 / 6,159 问题（A4 配置，含原始音视频）。

**偏差证据（Table 3）**：
- NCAQ 下 T5-small 在 Social-IQ v1 A2 达到 **100%**（随机应为 50%），A4 达到 **100%**（随机 25%）；v2 在 A4 达到 63.35%。
- RAWA 扰动后 T5-small 仍达 97.25%（A2）和 100%（A4），证明模型无需问题即可区分正确答案。
- T-SNE 可视化（Figure 3）显示正确/错误答案嵌入存在清晰边界。

**去偏效果（Tables 4–6）**：
- DeSIQd 上 T5-small NCAQ A2 降至 **50.16%**（接近随机），A4 降至 34.15%。
- DeSIQs 上 T5-small NCAQ A2 降至 **48.73%**，A4 降至 33.53%。
- DeSIQd-2.0 上 T5-small A4 从 63.35% 降至 **28.07%**。

**最强模型结果（Table 4–6）**：
- T5-small_Delphi 在 DeSIQd 上 q+a+t（concat）A2 达 **76.77%**、A4 达 **74.51%**。
- DeSIQ-2.0 上 q+a+s（音频）A4 达 **74.13%**。
- GPT-3/ChatGPT 在 DeSIQ 上零样本/少样本均低于上述结果，证明基准挑战性。

## 相关工作脉络
1. **Social-IQ (Zadeh et al., 2019)**：本文分析的核心基准，提供多模态 MCQA 评测社会智能；本文与其关系为"发现问题并提出更严谨的替代基准"。
2. **Social-IQ-2.0**：在线发布的增强版（含原始音视频、新标注），本文同样对其进行偏差分析与去偏（DeSIQd-2.0）。
3. **Shah et al. (2020) Expectations for MCQA**：从模型行为期望角度分析偏差，但仅关注模型侧，本文从数据侧系统提出六类扰动方法。
4. **Gat et al. (2020) Regularization by maximizing functional entropies**：从模型侧平衡文本/图像影响，涉及 Social-IQ 但未识别其数据侧偏差。
5. **Ye & Kovashka (2021) Shortcut effects in visual commonsense reasoning**：仅在验证集做 masking 去偏，本文提出数据侧系统性替换方案。
6. **TVQA / TVQA+ / MovieQA / TGIF-QA**：多模态视频 QA 基准，侧重时空定位与情节理解；本文聚焦社会智能特有的意图/规范推理，任务粒度与评测目标不同。

## 局限性与未来方向
1. 模型架构仍以文本语言模型为主，未探索强大原生多模态 LLM（论文自述局限）。
2. 所有实验仅在单一随机种子（seed=42）下运行一次，未做统计显著性检验（论文自述局限）。
3. DeSIQ 中是否仍存在残余偏差尚需进一步研究（作者提出研究问题）。
4. 更强语言模型在 DeSIQ 上的性能表现待探索。
5. 如何将社会文化/常识知识更有效地融入大语言模型仍是开放问题。
6. 如何利用多模态 LLM 更好地利用视频/音频输入有待深入研究。

## 研究启发与可借鉴点
1. **NCAQ 检测法可迁移**：仅给答案不给问题/上下文的评估设置，是检验 MCQA 数据集是否存在答案级偏差的高效手段，可推广至其他 QA 基准（如 MovieQA、TVQA）的审计。
2. **RIWA 去偏策略适用于答案侧偏差**：当发现正确/错误答案嵌入可线性分离时，用随机正确答案替换错误答案是一种简单但有效的去偏手段，可复用于其他有类似偏差的多选题数据集。
3. **常识知识蒸馏的社会智能增益**：T5-small_Delphi 证明小规模模型通过特定领域语料（社会规范/道德故事）预训练，可在社会智能任务上显著超越更大基座模型，为低资源社会智能研究提供路径。
4. **transcript 优于 video 的发现**：在 DeSIQ 上 transcript 模态的帮助通常大于 video，提示未来多模态融合策略应重视文本模态的主导作用，避免简单堆叠多模态特征。
5. **few-shot 不改善性能**：GPT-3 在 DeSIQ 上 few-shot 与 zero-shot 表现相近，说明去偏后模型无法依赖数据分布捷径，这一现象可作为"去偏有效性"的辅助验证指标。

## 关键术语表
**Social Intelligence**：理解与推理人类意图、情感和社会互动的能力，是本文评测的核心目标。
**Social-IQ**：Zadeh et al. (2019) 提出的社会智能多模态 MCQA 基准数据集，本文分析并去偏的对象。
**DeSIQ**：本文提出的去偏社会智能基准，通过 RIWA 扰动替换错误答案构建，分为 DeSIQd（跨视频）和 DeSIQs（同视频）两版。
**NCAQ**：No context and question，仅给答案选项不给任何上下文的评估设置，用于检测数据集答案侧偏差。
**RIWA**：Replace incorrect with another question's correct answer，$(q,a,i)\to(q,a,a')$，去偏核心扰动操作。
**T5-small_Delphi**：在 Social Chemistry 101/ETHICS/Moral Stories 上预训练后微调的 T5-small，注入社会常识知识。
**Shortcut**：模型利用数据集中虚假相关（如关键词重叠）而非真正理解任务来答题的行为模式。
**A2/A4 配置**：A2 为二选一（1 正确 +1 错误），A4 为四选一（1 正确 +3 错误），本文分别报告 binary 和 multiple-choice 准确率。

## 可复现要素
- **数据集**：Social-IQ v1 和 v2（Social-IQ-2.0）可从原论文及官网获取；DeSIQ 数据集由作者构造，论文未明确声明开源链接。
- **代码/权重**：论文未提及代码是否开源，T5-small 为 HuggingFace 公开权重；Delphi 预训练语料（Social Chemistry 101、ETHICS、Moral Stories）均为公开数据集。
- **关键超参**：learning rate = 1e-4，batch size = 8，训练 100 epochs（early stopping on dev loss），random seed = 42，few-shot k = 3（受预算限制未调参）。
- **硬件**：单卡 NVIDIA A40 40GB。
