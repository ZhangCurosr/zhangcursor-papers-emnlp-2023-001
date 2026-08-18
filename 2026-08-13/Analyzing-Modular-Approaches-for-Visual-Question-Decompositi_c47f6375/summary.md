---
title: "Analyzing-Modular-Approaches-for-Visual-Question-Decompositi"
source: https://aclanthology.org/2023.emnlp-main.157.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:37:39"
---

# 论文速读：Analyzing-Modular-Approaches-for-Visual-Question-Decompositi

## 一句话总结
本文对近期热门的无训练模块化视觉问答系统 ViperGPT 进行可控消融分析，发现其相对于底层 SOTA 模型 BLIP-2 的性能增益主要来源于人工预设的任务特定模块组合，而非符号化程序本身；同时提出一种基于自然语言逐步提示的分解方法（Successive Prompting），在多项 VQA 基准上达到甚至超越了等价的代码生成方案。

## 研究问题与动机
- **性能归因模糊**：ViperGPT 等最新模块化方法同时引入了 SOTA 底座模型（BLIP-2）与复杂的符号程序/多模块架构，其实际性能提升究竟来自“更强的基础模型”还是“额外的工程与符号组件”尚未被剥离验证。
- **任务特定设计的公平性存疑**：已有工作通常针对特定数据集（如 GQA、OK-VQA）手工筛选模块子集与 prompt，缺乏与“任务无关（task-agnostic）”设置的对照，难以评估模块化设计的泛化价值。
- **代码分解是否必要**：ViperGPT 依赖 Codex 生成 Python 程序来实现问题分解，但 NLP 领域已证明自然语言逐步推理（如 CoT、decomposed prompting）同样有效，两者在视觉任务上的优劣关系缺乏直接对比。
- **OOD 泛化与工程鲁棒性未知**：模块化方法宣称具备更好泛化能力，但在分布外数据集（A-OKVQA、ScienceQA）上的实际表现、in-context 示例的有效性以及生成代码的运行稳定性均未被系统评估。

## 核心贡献（创新点）
1. **揭示 ViperGPT 增益的真实来源与任务特定模块选择强相关，在 task-agnostic 设置下其相对 BLIP-2 的优势从 +8.7% 降至 -0.8%**；本质区别在于首次对复杂神经符号流水线进行归因解耦，而非单纯报告新分数或提出更大规模模型。
2. **提出 Successive Prompting 自然语言逐步提示分解方法，在直接回答任务中达到等价 ViperGPT 变体 92% 的性能，在多选题任务上平均超越 +12%**；本质区别在于用隐式语言推理替代显式 Python 程序生成，证明代码接口并非视觉问答分解的必要条件。
3. **系统量化模块化 VQA 在分布外数据上的泛化边界与代码执行失败模式，发现添加 few-shot 示例反而使 OOD 性能下降 2%-11%，且运行时错误率显著上升**；本质区别在于关注训练自由方法在实际部署中的鲁棒性与错误剖析，填补了该类系统可迁移性评估的空白。

## 方法详解
- **End-to-end 基线（BLIP-2）**：使用 EVA-CLIP ViT-g/14 图像编码器 + FlanT5-XXL 编码器-解码器语言模型，8-bit 推理，beam search（width=5, length penalty=-1）。VQA 提示格式为 `Question: {} Short answer: []`，直接回答模式直接生成文本，多选题模式选取 log-likelihood 最高的选项。
- **Modular 方法（ViperGPT 及其变体）**：原始 ViperGPT 调用 Codex 生成 Python 程序，程序通过 `ImagePatch` API 串联 GLIP（目标检测）、MiDaS（深度估计）、BLIP-2（简单查询）、X-VLM、InstructGPT 等模块。本文实现三个变体以进行归因：
  - `task-agnostic`：提供完整 API 与全部模块，不针对特定任务筛选；
  - `without BLIP-2`：删除 `simple_query` 及相关演示，测试模块冗余性；
  - `only BLIP-2`（zero-shot/few-shot）：仅保留 BLIP-2，重写演示以保证公平对比。
  多选题结果通过 InstructGPT 映射到最近选项。
- **Prompting 方法（Successive Prompting）**：联合 LLM（InstructGPT text-davinci-002）与 VLM（BLIP-2）。LLM 每次仅提出一个 follow-up 问题，BLIP-2 独立回答；LLM 基于历史问答生成下一步问题或最终答案。提示末尾附加 `Follow-up:` / `Answer to the original question:` 前缀作为停止条件。与 ViperGPT 使用相同 in-context 示例与底座，确保对比公平。
- **评估指标**：直接回答使用数据集原有指标及 InstructGPT-eval（用 InstructGPT 判断候选答案
