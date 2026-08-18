---
title: "Understanding-the-Effect-of-Model-Compression-on-Social-Bias"
source: https://aclanthology.org/2023.emnlp-main.161.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:34:11"
field: "AI 安全与公平性"
keywords: ["社会偏见", "模型压缩", "量化", "知识蒸馏", "大语言模型", "偏见缓解", "Bias Bench"]
innovations: ["首次系统研究 PTQ 与蒸馏对多种 LLM 社会偏见的影响，发现量化具正则化效应", "揭示预训练规模与社会偏见的正相关关系，定位量化最佳权衡点为 20% 预训练步", "通过 Pythia 中间 checkpoint 追踪偏见演化，证明量化可在保持 LM 性能的同时有效降低偏见"]
benchmarks: ["CrowS-Pairs", "StereoSet", "SEAT"]
---

# 论文速读：Understanding-the-Effect-of-Model-Compression-on-Social-Bias

## 一句话总结
本文系统研究了模型压缩方法（动态后训练量化 PTQ 和知识蒸馏）对大型语言模型（LLM）社会偏见的影响，发现量化与蒸馏均可有效降低社会偏见，且量化在保持语言建模性能的同时展现出最佳权衡效果。

## 研究问题与动机
- **核心问题**：预训练的 LLM 从网络语料中学习并放大社会偏见，而模型压缩（剪枝、量化、蒸馏）被广泛用于降低计算负担，但二者交互作用的研究仍严重不足。
- **现有方法不足**：现有偏见缓解工作多聚焦于专门的 debiasing 方法，缺乏将压缩本身作为偏见缓解策略的系统性实验验证。
- **实践需求**：重新训练模型以去除偏见成本极高（财务与环境代价），而压缩是部署阶段更实用的手段。
- **研究空白**：尽管已有少量工作表明剪枝对偏见有影响，但针对主流编码器/解码器架构的系统量化与蒸馏研究仍缺乏。

## 核心贡献（创新点）
- **首次系统研究 PTQ 与蒸馏对多种 LLM 社会偏见的影响**：覆盖 BERT、RoBERTa、Pythia 系列编码器与解码器架构，跨度 70M 至 6.9B 参数。
- **揭示预训练规模与社会偏见的正相关关系**：通过 Pythia 系列中间 checkpoint 发现，更长预训练和更大模型会导致更高的社会偏见。
- **发现量化的正则化效应并定位最佳权衡点**：量化可有效降低偏见，其最佳 LM 性能-偏见权衡出现在约 20% 的原始预训练步数处。
- **对比分析蒸馏与专门 debiasing 方法**：蒸馏能显著降低偏见但代价是 LM 分数大幅下降，而量化在保持语言能力的同时实现偏见缓解。

## 方法详解
- **Bias Bench 评估框架**：采用三个基准数据集：
  - **CrowS-Pairs**：通过最小距离句子对测量偏见，无偏模型应在刻板印象与反刻板印象选择间各占 50%。
  - **StereoSet (SS)**：含刻板/反刻板/无关三类选项，同样以 50% 为无偏目标，同时测量 LM Score 以跟踪语言能力变化。
  - **SEAT**：基于词嵌入余弦相似度测量属性词与目标词之间的偏置距离，效应值越接近 0 表示偏见越小。
- **动态后训练量化（Dynamic PTQ）**：训练完成后将权重静态映射为 int8，推理时激活动态映射，通过 PyTorch 标准实现对 Transformer 线性层进行 fp32→int8 压缩。
- **知识蒸馏**：使用 DistilBERT 和 DistilRoBERTa 作为蒸馏学生模型，训练时同时最小化软标签（教师预测）和硬标签（真实标签）的损失。
- **Pythia 系列预训练分析**：使用在去重版 The Pile（768GB 文本）上训练的 Pythia 模型及其中间 checkpoint，评估不同训练步数与社会偏见的相关性，采用 Kendall Tau C 进行统计检验（单侧 t 检验）。

## 实验与结果
- **数据集与模型**：使用 Bias Bench（CrowS-Pairs、StereoSet、SEAT），评估 BERT Base/Large（110M/345M）、RoBERTa Base/Large（123M/354M）、DistilBERT（66M）、DistilRoBERTa（82M）及 Pythia 系列（70M–6.9B）。
- **主要结果（StereoSet 刻板评分，越接近 50% 越好）**：
  - **BERT Base** 基线：GENDER 57.25、RACE 62.33、RELIGION 62.86；动态 PTQ int8 后：↓0.19/↓9.53/↑0.19（RELIGION 显著下降至 46.67）。
  - **BERT Large** 基线：GENDER 55.73、RACE 60.39、RELIGION 67.62；PTQ 后：↓6.87/↑0.78/↓7.62。
  - **DistilBERT** 较 BERT Base：GENDER ↓8.73、RACE ↓6.40、RELIGION ↓9.57，但 LM Score 下降 30.30。
  - **RoBERTa Base** 基线：GENDER 60.15、RACE 63.57、RELIGION 60.95；PTQ 后：↓6.51/↓5.04/↓10.47。
  - **Pythia 量化对比**（Table 3 vs Table 2）：全精度最佳 LM Score 出现在约 80% 预训练步，而量化后最佳 LM Score 出现在约 20% 预训练步，偏见得分同步降低。
- **最强结果**：量化在保持较高 LM Score 的同时实现最均衡的偏见降低，是所有方法中性能-偏见权衡最优的策略；DistilBERT 偏见降低幅度最大（如 RELIGION 从 62.86 降至 49.87），但以 LM Score 大幅下降为代价。

## 相关工作脉络
- **Hooker et al. (2020, 2021)**：研究剪枝对计算机视觉和多语言 Transformer 偏见的影响，本文聚焦量化与蒸馏，扩展了压缩方法的偏见影响研究范畴。
- **Xu & Hu (2022)**：证明压缩可改善模型公平性（减少毒性），本文在此基础上将焦点转向社会偏见（GENDER/RACE/RELIGION），并使用更系统的 Bias Bench 评估。
- **Meade et al. (2022)**：系统评测多种 debiasing 方法（INLP、CDA、DROPOUT、SELF-DEBIAS、SENTDEBIAS），本文将其作为基线对比，验证压缩方法可与专门 debiasing 方法竞争。
- **Blodgett et al. (2021)**：指出 CrowS 和 SS 数据集的局限性（西方偏见主导、性别二元定义），本文在局限性部分同样承认这些约束。
- **Kaneko & Bollegala (2021, 2022)**：研究预训练嵌入的社会偏见测量，本文利用 SEAT 延续该方向但扩展至更多模型规模与压缩方法。
- **Delobelle & Berendt (2022) / FairDistillation**：研究蒸馏对刻板印象的缓解，本文对比发现蒸馏可降低偏见但显著损害 LM 性能。

## 局限性与未来方向
- **数据集局限性**：Bias Bench 仅覆盖 GENDER、RACE、RELIGION 三类偏见，且偏向西方视角；GENDER 分类采用二元定义，无法覆盖非二元群体。
- **评估指标局限**：CrowS 和 SS 将无偏定义为刻板与反刻板选择各 50%，这是对无偏的有限定义；SS 存在未平衡的刻板对和部分非共识样本。
- **仅研究两种压缩方法**：未涉及剪枝、混合专家（MoE）、自适应计算等其他压缩策略，留作未来工作。
- **量化结果为初步结论**：作者明确声明量化效果的研究结果为 preliminary results，需在更多场景下验证。
- **未测试生产部署适用性**：论文强调这些模型尚不适合生产环境部署。

## 研究启发与可借鉴点
- **压缩作为偏见缓解的潜在策略**：动态 PTQ 可作为轻量级、无需额外训练的偏见缓解手段，为资源受限场景提供实用方案。
- **预训练早期 checkpoint 的偏见优势**：量化模型的最佳 LM 性能出现在约 20% 预训练步，提示对于偏见敏感任务可考虑 early stopping 策略。
- **编码器与解码器架构的系统性对比**：通过 Pythia 系列中间 checkpoint 追踪偏见演化趋势的方法，可迁移至其他模型架构的偏见演化研究。
- **LM Score 与偏见的权衡分析框架**：SS 数据集同时报告 LM Score 和偏见分数的评估范式，可有效区分"真正减少偏见"与"降低语言能力的虚假无偏"。
- **与团队方向结合机会**：可探索将 PTQ 集成到 bias-aware 的训练管线中，或在多语言/低资源场景下验证量化的偏见缓解效果。

## 关键术语表
**PTQ (Post-Training Quantization)**：后训练量化，在模型训练完成后将权重/激活精度降低（如 fp32→int8），无需重新训练即可压缩模型。
**Bias Bench**：由 Meade et al. (2022) 编译的社会偏见评估基准，整合 CrowS-Pairs、StereoSet 和 SEAT 三个数据集。
**CrowS-Pairs**：通过最小距离句子对测量 masked language model 中刻板印象偏见的基准，无偏得分为 50%。
**StereoSet (SS)**：测量预训练语言模型中刻板印象偏见的基准，含刻板/反刻板/无关三选项，同时报告 LM Score 评估语言能力。
**SEAT**：基于词/句嵌入余弦相似度测量偏见方向的基准，效应值接近 0 表示无偏。
**Kendall Tau C**：用于衡量两个变量（如 LM Score 与社会偏见）排序一致性的统计相关系数。
**The Pile**：800GB 多样性英文文本数据集，本文使用其去重版（768GB）作为 Pythia 模型的预训练语料。
**DistilBERT / DistilRoBERTa**：分别由 Sanh et al. (2019) 蒸馏得到的 BERT 和 RoBERTa 轻量版本，参数分别约 66M 和 82M。

## 可复现要素
- **数据集**：Bias Bench（CrowS-Pairs、StereoSet、SEAT）公开可用；Pythia 模型及中间 checkpoint 公开（HuggingFace）。
- **代码/权重**：论文使用 PyTorch 标准量化实现；Pythia 模型权重公开。
- **关键超参**：动态 PTQ 使用 int8 精度；Pythia 预训练语料为去重版 The Pile（768GB）；模型规模从 70M 至 6.9B 参数；共测试 21 个预训练 checkpoint 步。
- **硬件**：量化实验在双 AMD EPYC 7 662 64-Core CPU 上运行约 4 天；非量化实验在 NVIDIA A100 40GB GPU 上运行约 5 小时。
