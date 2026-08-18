---
title: "CRT-QA-A-Dataset-of-Complex-Reasoning-Question-Answering-ove"
source: https://aclanthology.org/2023.emnlp-main.132.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:08:34"
field: "表格推理与大语言模型评估"
keywords: ["Table QA", "Complex Reasoning", "Large Language Models", "Tool-Augmented Reasoning", "Pandas Code Execution", "Implicit Questions"]
innovations: ["提出操作与推理分离的细粒度分类体系并构建 CRT-QA 数据集", "设计无需人工示例的 ARC 方法通过自动代码生成与执行增强 LLM 表格推理", "引入不可答/不确定问题子集及细粒度推理路径标注以系统评估 LLMs"]
benchmarks: ["CRT-QA"]
---

# 论文速读：CRT-QA-A-Dataset-of-Complex-Reasoning-Question-Answering-ove

## 一句话总结
本文构建了第一个同时包含多步操作与非形式推理的表格问答数据集 CRT-QA，提出了操作与推理的细粒度分类体系，并设计了无需人工示例的 ARC 方法以增强 LLMs 的表格计算与推理执行能力。

## 研究问题与动机
- **现有数据集侧重事实检索而非深度推理**：主流 Table QA 数据集（如 WTQ、WikiSQL、HybridQA）主要关注简单的多跳事实检索，缺乏对高阶认知任务（逻辑、数值、常识推理）的系统性评估。
- **推理定义与 LLM 研究存在错位**：已有工作将过滤等基础 Pandas/SQL 操作也称为"推理"，这与当前 LLM 研究中对非形式推理（基于直觉、经验、常识的演绎）的定义不一致。
- **现实场景中存在隐含/模糊查询**：真实用户提问常常是隐式的或信息不足的，而现有数据集仅包含有明确答案的问题，缺乏对不可答（unanswerable）和不确定（indeterminate）问题的研究。
- **LLMs 擅长规划但执行偏弱**：初步实验发现，LLMs 能够生成正确的推理计划，但在执行算术、聚合、排序等具体操作时容易出错，需要借助外部工具辅助。

## 核心贡献（创新点）
1. **建立了全面的表格分析操作与推理分类体系（Taxonomy）**：将单一 Pandas/SQL 查询可执行的步骤定义为"操作"（索引、过滤、分组、排序），将依赖常识/逻辑的步骤定义为"推理"（聚合、算术、Grounding、自动分类、时间/空间推理、量词推理等），这与以往将简单操作泛化为推理的做法有本质区别。
2. **构建了首个包含多步操作与非形式推理的表格 QA 数据集 CRT-QA**：不同于以往仅含显式问题与明确答案的数据集，CRT-QA 引入了隐式问题、细粒度注释（直接性、子问题构成类型、人工推理路径）以及不可答/不确定问题子集，更贴近真实场景。
3. **提出了无需人工示例的 ARC（Autoexemplar-guided Reasoning with Code）方法**：通过零样本生成代码示例作为上下文演示，结合外部 Python 解释器执行 Pandas 代码，并将执行结果注入提示以迭代推理，避免了手动构造演示的高成本与低灵活性，与依赖人工示例的 ReAct/PAL 形成对比。
4. **进行了系统的 LLMs 表格推理能力评估与不可答性问题研究**：在多个模型与提示方法下评测，揭示了 LLMs 在聚合、算术等操作类型上的薄弱点，并深入分析了模型识别问题可答性的能力。

## 方法详解
**ARC（Autoexemplar-guided Reasoning with Code）方法框架**：
- **自动示例生成（Auto-exemplar generation）**：从开发集中随机采样一个样本，使用指令提示让 LLM 零样本生成用于解决该样本的 Python/Pandas 代码，作为上下文演示（in-context exemplar）。
- **上下文代码生成（In-context code generation）**：对于测试集中的每个样本，将开发集生成的代码示例作为演示，引导 LLM 为当前问题生成对应的 Pandas 代码。
- **外部工具执行（Code execution with external tools）**：将生成的代码在安装了 Pandas 的 Python 解释器环境中执行，获取中间或最终计算结果，避免 LLM 自行计算导致的错误。
- **带代码输出的迭代 LLM 调用（Iterative LLM calling with code output）**：将代码执行结果注入提示，再次请求 LLM 进行最终推理与答案生成，结合代码精确计算与 LLM 的常识推理能力。

**关键超参**：Temperature=0.7，max_len（CoT/Code）=1024，max_len（Few-shot/Zero-shot）=16，top_p=1.0。

## 实验与结果
- **数据集规模**：423 个独立表格，1000 个问题（其中 744 个可答，256 个不可答/不确定），平均每问题 3.2 步推理、3.1 步操作。
- **评估基线**：Zero-shot/Few-shot（2-shot）、Zero-shot-CoT、Few-shot-CoT、PAL、ReAct，在 ChatGPT、GPT-3.5-turbo、GPT-4 上测试。
- **主要结果**：
  - CRT-QA 整体难度高，最优方法（GPT-4 + ARC）Exact Match 为 **60.11%**。
  - ARC 相比其他方法平均提升 **1.846** EM 分数，且在所有模型上无需人工示例。
  - GPT-4 的 Few-shot-CoT 达到 **56.32%**，而 ARC 在相同模型上取得 **60.11%**。
  - LLMs 在"量词推理"上表现最好，在"聚合"和"算术"上表现最差；ARC 和 PAL 能显著改善后两类任务。
  - Zero-shot-CoT 在 ChatGPT 上甚至不如 Zero-shot，说明"Let's think step-by-step"并非万能。
  - ReAct 在 GPT-3.5-turbo 和 GPT-4 上表现不佳，常因迭代过多而失控或无法输出答案。
- **不可答性问题识别**：Question Answering 方式效果最佳（F1=0.838），优于 Binary Classification（F1=0.749）。

## 相关工作脉络
- **TableQA 数据集（WTQ、WikiSQL、HybridQA、TabFact 等）**：侧重于多跳事实检索或文本-表格混合问答，缺乏对非形式推理与细粒度推理路径的标注，CRT-QA 在其基础上补充了操作与推理的区分及隐式问题。
- **数值推理数据集（FinQA、TAT-QA、TABMWP）**：聚焦财务或数学文本-表格混合推理，未引入常识与多步操作分解，CRT-QA 覆盖更广泛的日常领域与操作类型。
- **代码生成基准（ARCADE）**：使用 Pandas 进行表格分析代码生成，但标签非自然语言且缺少常识推理，CRT-QA 以自然语言问答为目标并包含更多推理维度。
- **工具增强 LLMs（Toolformer、ReAct、PAL）**：Toolformer 需微调、ReAct 依赖人工设计的 (Thought, Act, Obs) 示例且调用成本高，PAL 仅提供代码示例无推理反馈；ARC 通过自动示例生成与代码执行注入结合，兼顾灵活性与低成本。
- **隐式推理评估（StrategyQA）**：启发了 CRT-QA 中隐式/显式问题的划分方式，但 StrategyQA 基于纯文本，CRT-QA 将其拓展至表格领域。

## 局限性与未来方向
- **仅为测试集**：无梯度更新，数据规模受限于标注复杂度，难以平衡数量与质量。
- **仅关注单表问答**：未涉及多表联合推理，而现实场景多表分析常见。
- **未探索外部知识边界**：不可答/不确定问题的产生与数据生成目标相关，对隐式与不确定问题的边界研究留待未来。
- **评估指标局限**：采用 Exact Match 与人工评估结合，自由文本答案的自动化评估（如 F1、ROUGE-L）仍存在固有缺陷，推理路径的有效性评估尚未解决。

## 研究启发与可借鉴点
- **操作与推理的分离定义**：将可直接由工具（Pandas/SQL）执行的步骤归类为操作，将需常识/逻辑演绎的步骤归类为推理，为后续表格推理基准的设计提供了清晰的分类框架。
- **人机协同的数据生成流水线**：利用 LLM 生成候选问题后，由人工筛选并提供基于特定词汇特征（如"complex"、"math"）的反馈，可有效缓解 LLM 生成问题的复杂性不足、多样性差及不可答等问题，该方法可迁移至其他领域的数据构建。
- **自动示例生成替代人工演示**：ARC 通过零样本生成代码示例作为 in-context demonstration，免去了手动构造示例的成本，且在不同模型间具有更好的可迁移性，可用于其他需要代码执行的推理任务。
- **细粒度推理路径标注**：通过模板填充方式高效标注人工推理步骤（操作/推理类型及目标对象），为评估模型推理过程而非仅最终答案提供了可行路径，后续研究可借鉴此标注范式。
- **不可答性问题分类体系**：将不可答问题分为范围外、幻觉、逻辑错误、主观及其他，并研究模型识别可答性的能力，为构建更鲁棒的 QA 系统提供了评估维度。

## 关键术语表
- **CRT-QA**：Complex Reasoning QA over Tabular data，本文提出的包含多步操作与非形式推理的表格问答数据集。
- **操作（Operation）**：可通过单次 Pandas/SQL 查询执行的表格步骤，包括索引、过滤、分组、排序、聚合、算术等。
- **非形式推理（Informal Reasoning）**：依赖直觉、经验与常识进行演绎的推理过程，如 Grounding、自动分类、时间/空间推理、量词推理等。
- **ARC（Autoexemplar-guided Reasoning with Code）**：本文提出的方法，通过自动生成代码示例作为上下文演示，结合外部代码执行与迭代 LLM 调用解决表格推理任务。
- **直接性（Directness）**：问题是否隐式（implicit）或显式（explicit）的标注，隐式问题需引入问题外词汇才能描述推理过程。
- **子问题构成类型（Decomposition types）**：包括桥接（Bridging）、交集（Intersection）、比较（Comparison）三种多跳问题分解方式。
- **不可答问题（Unanswerable question）**：因信息缺失、假设无效、逻辑错误或主观性而无法回答的问题。
- **不确定问题（Indeterminate question）**：因指标、算法或标准不同可能导致不同合理答案的问题。

## 可复现要素
- **数据集**：CRT-QA，来源于 TabFact/Wikipedia 表格，**已公开**（https://github.com/zzh-SJTU/CRT-QA）。
- **代码**：实验代码与提示模板**已开源**。
- **模型**：ChatGPT (text-chat-davinci-003)、GPT-3.5-turbo、GPT-4，通过 Microsoft Azure API 调用。
- **关键超参**：Temperature=0.7，max_len(CoT/Code)=1024，max_len(Few-shot/Zero-shot)=16，top_p=1.0，best_of=1。
- **代码执行策略**：若生成的代码含语法错误，最多重试 5 次；仍不可运行则将代码保留在提示中，输出设为 "None"。
- **评估指标**：Exact Match（EM），辅以人工评估处理格式误差。
- **训练**：本工作为测试集，无梯度更新（论文未提及）。
