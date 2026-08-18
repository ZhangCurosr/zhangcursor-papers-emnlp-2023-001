---
title: "Cross-lingual-Prompting-Improving-Zero-shot-Chain-of-Thought"
source: https://aclanthology.org/2023.emnlp-main.163.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:10:54"
field: "跨语言大语言模型推理"
keywords: ["cross-lingual prompting", "zero-shot chain-of-thought", "multilingual reasoning", "self-consistency", "large language models", "prompt engineering"]
innovations: ["提出两阶段跨语言对齐与求解提示框架 CLP，无需示例实现跨语言零样本 CoT", "设计跨语言自洽提示 CLSP，通过多语言推理路径投票提升一致性", "证明 CLP 可与 ICL 正交组合并显著提升小模型跨语言推理性能"]
benchmarks: ["MGSM", "XCOPA", "XNLI", "PAWS-X"]
---

# 论文速读：Cross-lingual Prompting: Improving Zero-shot Chain-of-Thought Reasoning across Languages

## 一句话总结
论文提出了跨语言提示（CLP），通过“跨语言对齐提示”与“任务特定求解提示”两阶段交互，将任意语言的零样本思维链推理统一引导至目标语言（英语）输出，显著提升跨语言数学与常识推理准确率；进一步引入跨语言自洽提示（CLSP），聚合多语言推理路径投票，取得当前最优性能。

## 研究问题与动机
- 传统零样本 CoT（如 "Let’s think step by step!"）仅在单一语言内有效，难以泛化至其他语言，阻碍跨语言推理能力的全球普及。
- 现有跨语言 CoT 研究多依赖少数示例（few-shot）或外部翻译 API，缺乏无需额外示例的零样本跨语言对齐机制。
- 直接让多语模型以目标语言生成推理链（En-CoT）在低资源语言上表现不稳定，且未显式建模源语言与目标语言间的语义对齐过程。
- 不同语言下 LLM 的推理模式存在差异，单一语言路径难以充分利用多语言知识分布，需要设计跨语言集成机制。

## 核心贡献（创新点）
- 提出两阶段跨语言提示框架 CLP，将跨语言表示对齐与任务求解分离，区别于 En-CoT 的直接端到端生成。
- 设计跨语言对齐提示，要求模型分步理解源语言输入并映射至目标语言，显式捕获跨语言语义关系，而非依赖机械翻译。
- 提出任务特定求解提示，在已对齐语义基础上逐步推导最终答案并格式化输出，保证推理链条完整可追溯。
- 引入跨语言自洽提示 CLSP，在多目标语言上并行生成推理路径并通过一致答案投票集成，优于单语言自洽方法。
- 在 MGSM、XCOPA、XNLI、PAWS-X 等多个跨语言基准上验证 CLP/CLSP 有效性，在 GPT-3.5 上平均准确率提升 6.1%（CLSP vs CLP），并证明方法可与 in-context learning 正交组合。

## 方法详解
- **跨语言对齐提示（Cross-lingual Alignment Prompting）**  
  角色设定：让模型扮演源语言 $L_s$ 的多语言理解专家。  
  提示模板：`Please act as an expert in multi-lingual understanding in [Source Language Ls]. Request: [Given sentence X]. Let’s understand the task in [Target Language Lt] step-by-step!`  
  目标：生成中间语义对齐序列 $\{a_i\}_{i=1}^S$，优化对齐响应概率 $p(a_1,\dots,a_S|X,L_s,L_t)$。
- **任务特定求解提示（Task-specific Solver Prompting）**  
  基于第一阶段对齐文本 $C$，设定目标任务 $T$ 与目标语言 $L_t$ 专家角色。  
  提示模板：`After understanding, you should act as an expert in [Target Task T] in [Target Language Lt]. Let’s resolve the task you understand above step-by-step! Finally, format your answer as 'Answer: [num]'.`  
  目标：生成多步推理路径 $\mathcal{R}_t$ 并提取最终答案 $F_t$，优化 $p(\mathcal{R}_t|C,L_t,T)$ 与 $p(f|\mathcal{R}_t)$。
- **跨语言自洽提示（CLSP）**  
  对多个目标语言 $L_t$ 分别运行 CLP，得到答案集合 $\{F_t\}$，通过投票选取出现频率最高的答案：$\hat{F} = \arg\max \sum_{t}\sum_{f} \mathbb{1}(F_t = f)$。
- **关键超参**：top-p=1；对齐阶段温度采样区间 [0,2]，求解阶段温度采样区间 [0,1]。

## 实验与结果
- **数据集与基线**：MGSM（10 种语言数学推理）、XCOPA（11 种语言常识推理）、XNLI、PAWS-X；基线包括 Direct、Native-CoT、En-CoT、Translate-En（Google 翻译 API）及 few-shot 版本。
- **主要结果（GPT-3.5，Table 1）**：CLP 平均准确率 70.6%，超越 Translate-En（68.4%）2.2%；CLSP 平均准确率 76.7%，较 CLP 提升 6.1%，较 Translate-En 提升 8.3%，创 SOTA。
- **最强结果**：CLSP 在德语（de）达 86.8%、俄语（ru）达 87.6%、孟加拉语（bn）达 75.2%；整体 AVG 较所有基线高出 1.8% 以上。
- **小模型验证（Table 2）**：在 mT0-XXL、Bloomz-7B、Llama-2-13B 上，CLP 较 En-CoT 平均提升 6.8% 以上。
- **泛化分析（Figure 5）**：CLP 在 XNLI 上较 En-CoT 平均提升 3.1%，在 PAWS-X 上提升 4.5%。
- **ICL 组合（Table 5）**：CLP 加入 3-shot 对齐示例后平均提升 6.9%；再结合 Complex-CoT 求解示例可额外提升 1.1%。

## 相关工作脉络
- **传统 Zero-shot CoT（Kojima et al., 2022）**：仅单语言分步提示；本文将其扩展至跨语言场景，显式引入对齐步骤。
- **En-CoT（Shi et al., 2022）**：直接要求模型用英语输出推理链；本文区分“对齐”与“求解”，避免盲目端到端翻译。
- **Translate-En（Shi et al., 2022）**：依赖外部 Google 翻译 API；本文通过模型内生语义对齐实现跨语言理解，提升 2.2% AVG。
- **Self-consistency（Wang et al., 2022）**：同语言多次采样投票；本文推广至多语言并行推理，CLSP 较 VSC 平均提升 4.5%。
- **Few-shot 跨语言对齐（Winata et al., 2021; Tanwar et al., 2023）**：需构造多语言示例；本文为零样本方法，无需额外示例构建。
- **ReAct / Reflexion（Yao et al., 2023; Shinn et al., 2023）**：交互执行与反馈；本文聚焦纯推理链生成的跨语言泛化，不与上述方法冲突。

## 局限性与未来方向
- 不同跨语言对齐提示表述会导致平均 4% 以上的性能波动，提示鲁棒性仍待提升。
- 低资源语言（如特卢固语 te）性能提升有限，可能与预训练数据分布不均有关。
- 引入更多语言并非线性增益，低资源语言加入反而可能降低整体性能，需权衡语言选择策略。
- 当前方法依赖强闭源模型（GPT-3.5），在更小或开源模型上的绝对性能仍有差距。

## 研究启发与可借鉴点
- 两阶段交互提示（对齐→求解）可视为“对话式思维链”的简化范式，适合迁移至多轮推理、代码生成等任务。
- 跨语言自洽机制可推广至多工具调用或多代理协作场景，通过异构路径投票提升可靠性。
- 提示工程与 ICL 正交组合的实验设计表明，CLP 可作为即插即用模块嵌入现有推理流水线。
- 策略多样性分析（Table 7）显示，分解对齐过程为“问题重述”与“初步求解”等子策略可显著提升性能，值得在复杂指令遵循中复用。
- ROSCOE 跨语言扩展评估框架为 CoT 质量分析提供了可复用的多语言 faithful/informative 度量。

## 关键术语表
- **Chain-of-Thought (CoT)**：通过分步提示引导大模型生成中间推理过程的技术。
- **Zero-shot CoT**：仅依赖自然语言指令（如 "Let’s think step by step!"）而无需示例的推理提示方法。
- **Cross-lingual Alignment Prompting**：引导模型将源语言输入逐步映射至目标语言的语义对齐提示步骤。
- **Task-specific Solver Prompting**：在已对齐语义基础上，以目标语言完成具体任务推理并格式化输出的提示步骤。
- **Cross-lingual Self-consistent Prompting (CLSP)**：在多目标语言上并行执行推理并通过投票集成一致答案的自洽方法。
- **ROSCOE**：评估思维链忠实度与信息量的多语言适配度量框架。
- **En-CoT**：直接将非英语输入以英语输出思维链的跨语言推理基线。
- **Translate-En**：先通过外部 API 将输入翻译为英语再进行推理的基线方法。

## 可复现要素
- **数据集**：MGSM、XCOPA、XNLI、PAWS-X（均为公开学术数据集）。
- **代码**：论文声明代码开源，地址为 Cross-Lingual-Prompting（未提供具体仓库链接）。
- **权重**：使用商业模型 GPT-3.5 (gpt-3.5-turbo)、GPT-3、PaLM-540B；小模型实验涉及 mT0-XXL、Bloomz-7B、Llama-2-13B（开源权重）。
- **关键超参**：top-p=1；对齐阶段 temperature ∈ [0,2]；求解阶段 temperature ∈ [0,1]；CLSP 投票聚合所有目标语言答案。
