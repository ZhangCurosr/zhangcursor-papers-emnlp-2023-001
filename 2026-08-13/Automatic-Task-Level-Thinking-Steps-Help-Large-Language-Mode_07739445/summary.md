---
title: "Automatic-Task-Level-Thinking-Steps-Help-Large-Language-Mode"
source: https://aclanthology.org/2023.emnlp-main.150.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:06:57"
field: "大语言模型提示与推理"
keywords: ["in-context learning", "chain-of-thought", "task-level thinking", "progressive revision", "classification", "LLM agents", "prompt engineering"]
innovations: ["提出task-level thinking steps概念，从任务级别消除演示偏差", "设计progressive revision framework通过教师-学生agent迭代修正思维步骤", "利用困难演示反馈自动改进思维步骤质量"]
benchmarks: ["Stance16", "Emotion20", "20news", "i2b2", "ChemProt", "DDI"]
---

# 论文速读：Automatic-Task-Level-Thinking-Steps-Help-Large-Language-Models-for-Challenging-Classification-Task

## 一句话总结
本文提出"task-level thinking steps"概念及渐进式修订框架，通过自动修正困难样本逐步优化思维步骤，消除演示数据偏差并澄清易混淆类别，在零样本、少样本和少样本-CoT设置下均取得三类挑战性分类任务的最佳性能。

## 研究问题与动机
1. **演示数据偏差严重影响分类性能**：LLMs在ICL范式下受演示数据的分布（内容、标签、顺序）严重影响，尤其是立场检测、细粒度分类、领域特定分类等挑战性任务。
2. **现有去偏方法的局限**：现有工作多聚焦于选择高质量/代表性演示（如相似度选择、类别平衡），但最优分布依赖任务，偏差不可避免。
3. **自动CoT对困难样本效果不佳**：Zero-shot-CoT对困难演示可能产生不愿解释或幻觉（hallucinations），导致性能下降甚至低于零样本和少样本设置。
4. **难以区分易混淆类别**：复杂分类任务需要区分大量相似类别（如20news的20个类别）或细微差异的关系（如医疗关系分类），需要更精细的任务理解。

## 核心贡献（创新点）
1. **提出task-level thinking steps概念**：与自动生成的简易CoT（"Let's think step by step"）不同，该步骤包含任务级分析流程（如识别领域、查找关键词、确定立场、匹配选项），能消除演示数据引入的偏差。
2. **设计progressive revision framework**：通过教师agent和学生agent的迭代交互，基于困难演示的反馈逐步修正思维步骤，本质区别在于在任务级别而非样本级别进行修正。
3. **实现自动CoT生成的质量提升**：利用task-level thinking steps自动生成更合理、更可靠的CoT，避免手动CoT的人力成本和零-shot-CoT的不忠实问题。
4. **在三种分类任务和三种设置下均验证有效性**：涵盖多面主观文本分析、细粒度分类、领域特定分类，覆盖zero-shot、few-shot、few-shot-CoT场景。

## 方法详解
**整体框架**：基于两个LLM agent（教师agent和学生会agent）的迭代修正系统。

**1. 困难演示选择**：遍历训练集，为每个类别选择LLM无法正确预测的样本构成$\mathcal{D}_{demo}$。

**2. 初始思维步骤生成**：教师agent使用困难演示生成初始task-level thinking steps，提示词为"Please generate generic thinking steps for the task."

**3. 学生agent推理**：学生对每个困难演示$x_i$应用当前思维步骤$S_i$，生成输出$R_i$。

**4. 错误分析与修正**：
- 教师agent分析错误原因："Input: $\{x_i\}$. Outputs: $\{R_i\}$. Please analyze the outputs from the student following your thinking steps."
- 教师agent修正思维步骤："Please revise the thinking steps to clarify."
- 修正通常不是大幅改动，而是添加提醒（如"If the treatment is ineffective or worsens the medical problem, consider choice (B) rather than choice (D)."）

**5. 检查机制（Checking Mechanism）**：为防止LLM仅靠扩展捷径而非可靠分析来修正：
- 教师agent生成多个候选思维步骤
- 学生会agent测试其在之前样本上的表现
- 保留最佳版本

**6. 迭代流程**：重复上述过程直到正确预测或达到最大尝试次数（实验中设为5次）。

**7. 自动CoT生成**：将生成的task-level thinking steps应用于演示，自动为每个演示生成解释。

## 实验与结果
**数据集**：
- 多面主观文本分析：Stance16（3类，立场检测）、Emotion20（4类，情感识别）
- 细粒度文本分类：20news（20类，新闻组分类）
- 领域特定分类：i2b2（8类，医疗关系）、ChemProt（5类，化学-蛋白质相互作用）、DDI（4类，药物-药物相互作用）

**主要结果（F1-score，平均值）**：

| 设置 | Auto | ITS | PRTS | 提升幅度 |
|------|------|-----|------|----------|
| Zero-shot | 46.78 | 51.37 | **62.67** | +18.75% vs Auto |
| Few-shot | 54.30 | 50.54 | **64.19** | +9.89% vs Auto |
| Few-shot-CoT | 51.83 | 62.46 | **72.47** | +20.64% vs Auto |

**关键发现**：
- PRTS在所有任务和设置下均取得最佳性能
- 对于biased demonstrations（Unbalanced），PRTS+相比原始方法提升27.71%，展现强鲁棒性
- 检查机制带来额外提升：Stance16 +5.29%，i2b2 +3.07%
- GPT-4作为教师agent显著优于text-davinci-003和gpt-3.5-turbo

## 相关工作脉络
1. **In-context Learning bias研究**：Zhao et al. (2021)提出校准方法；Liu et al. (2021)研究相似度选择；Lu et al. (2022)研究演示顺序敏感性。本文定位：不仅选择演示，更通过思维步骤消除偏差影响。
2. **Chain-of-Thought prompting**：Wei et al. (2022)提出手动CoT；Kojima et al. (2022)提出Zero-shot-CoT。本文定位：从简易推理提示升级为任务级别的系统化思维步骤。
3. **自动CoT生成**：Zhang et al. (2023b)提出auto-CoT聚类方法；Shum et al. (2023)优化潜在变量选择演示。本文定位：基于困难演示反馈的任务级修正，而非样本级选择。
4. **Prompt Engineering**：Honovich et al. (2022)、Zhou et al. (2022)探索自动提示生成。本文定位：将思维步骤作为提示的一部分，系统性地指导LLM推理。
5. **Demonstration selection**：Diao et al. (2023)基于不确定性选择；Min et al. (2022)、Webson & Pavlick (2022)揭示LLM可能忽略演示映射。本文定位：不选择演示本身，而是增强对演示的理解方式。

## 局限性与未来方向
1. **任务覆盖有限**：仅验证于传统分类任务，未扩展到NLI、多选题等。
2. **可靠性问题**：LLM对思维步骤的小变化仍敏感，存在意外挑战的可能。
3. **未标注训练集的困难**：获取困难演示需要标注，建议未来用多次零-shot提示的不确定性衡量。
4. **教师agent依赖**：需要GPT-4级别模型才能有效修正思维步骤，text-davinci-003无法生成合理初始步骤。
5. **修订顺序固定**：采用默认修订顺序，可能存在更优策略。

## 研究启发与可借鉴点
1. **Agent交互范式**：教师-学生agent的迭代修正框架可迁移至其他需要逐步改进的场景（如代码生成、数学推理）。
2. **困难样本驱动优化**：利用模型无法正确预测的样本作为优化目标，而非随机选择或相似度选择，这一思路可用于改进ICL的演示策略。
3. **检查机制设计**：生成多个候选方案并在历史样本上测试，保留最佳版本——这一机制可有效防止"捷径学习"，适用于任何需要保证泛化性的生成任务。
4. **任务级vs样本级思维**：将推理从单样本级别提升到任务级别，这一抽象层次提升值得在其他ICL应用场景中探索。
5. **自动CoT质量评估**：通过case study揭示Auto CoT的缺陷（不愿解释、幻觉），为后续CoT研究提供评估视角。

## 关键术语表
**Task-level thinking steps**：针对特定任务设计的结构化推理步骤，包含任务理解、关键要素识别、类别匹配等流程，区别于通用的"Let's think step by step"。

**Progressive Revision Framework**：基于教师-学生agent迭代的思维步骤优化框架，通过困难演示的反馈逐步修正和改进推理步骤。

**Hard Demonstrations**：LLM无法正确预测的训练样本，用于生成和修正task-level thinking steps。

**In-context Learning (ICL)**：通过提供少量演示示例让LLM适应下游任务的范式，无需更新模型参数。

**Zero-shot-CoT**：使用"Let's think step by step"提示让LLM自动生成推理过程的零样本方法。

**Checking Mechanism**：生成多个候选思维步骤并在历史样本上测试，保留最佳版本以防止模型仅靠捷径修正。

**Few-shot-CoT**：在少样本演示中自动附加CoT解释的推理方法。

**Domain-specific Classification**：需要领域知识理解的分类任务，如医疗关系分类。

## 可复现要素
- **数据集**：Stance16、Emotion20、20news、i2b2、ChemProt、DDI（均为公开数据集）
- **代码/权重**：论文未提及开源代码，使用商业API（GPT-4、GPT-3.5-turbo）
- **关键超参**：
  - 温度：0
  - 最大修订尝试次数：5
  - 检查机制候选数：3
  - 每类困难演示数：C（类别数）
- **Prompt模板**：见论文附录Table 9
