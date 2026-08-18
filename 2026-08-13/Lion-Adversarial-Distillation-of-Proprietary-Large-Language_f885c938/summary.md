---
title: "Lion-Adversarial-Distillation-of-Proprietary-Large-Language"
source: https://aclanthology.org/2023.emnlp-main.189.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:50:44"
---

# 论文速读：Lion: Adversarial Distillation of Proprietary Large Language Models

## 一句话总结
本文提出了一种面向黑盒专有大语言模型的对抗性知识蒸馏框架（Lion），通过“模仿-判别-生成”三阶段对抗循环，仅用70K自生成指令即可将ChatGPT的知识高效蒸馏至开源学生模型，在开放生成与零样本推理任务上显著超越现有SOTA基线。

## 研究问题与动机
- 现有指令微调/知识蒸馏方法多为单向知识传递（教师生成答案→学生模仿），缺乏对学生薄弱环节的针对性反馈机制，难以高效突破能力瓶颈。
- 传统对抗知识蒸馏（AKD）需访问教师模型的权重或梯度以训练生成器，无法直接应用于ChatGPT等仅提供API的黑盒专有模型。
- 静态训练数据分布容易固化，学生在迭代学习中会将原有难样本逐渐掌握，导致后续训练收益递减；亟需动态生成新难样本以维持训练挑战度。

## 核心贡献（创新点）
1. **首次将对抗知识蒸馏思想引入大语言模型蒸馏**。与Alpaca、Vicuna等单向静态微调方法本质不同，本文引入动态反馈闭环，使数据生成受学生学习进度驱动。
2. **设计无需权重访问的三阶段对抗循环框架**。通过Prompt角色适配让同一黑盒LLM兼任Teacher、Referee与Generator。与依赖梯度反演的传统AKD相比，完全绕过黑盒限制。
3. **极高的数据效率与推理性能提升**。仅用70K无人工标注数据完成3轮迭代，Lion-13B在BBH上较Vicuna-13B提升55.4%，在AGIEval上提升16.7%，并在部分子任务上反超ChatGPT。
4. **双池机制与多样性约束策略**。提出Train Pool与Cache Pool协同更新机制，结合ROUGE-L去重与Hard/Easy样本比例控制，有效缓解灾难性遗忘并保障生成指令分布多样性。

## 方法详解
- **角色与数据池初始化**：使用同一黑盒LLM（gpt-3.5-turbo）通过不同Prompt模板扮演四个角色：教师$\mathcal{T}$、学生$\mathcal{S}$、裁判$\mathcal{R}$、生成器$\mathcal{G}$。维护两个数据池：Train Pool $X^A$（用于微调）与Cache Pool $X^B$（用于评估性能差距），初始均从Alpaca 52K指令构建。
- **阶段一：模仿（Imitation）**：将$X^A$中指令输入$\mathcal{T}$获取标准响应，以$\{x_i^A, \mathcal{T}(x_i^A)\}$为监督信号对学生模型进行自回归语言建模微调。
- **阶段二：判别（Discrimination）**：将$X^B$中指令同时输入$\mathcal{T}$与$\mathcal{S}$，由$\mathcal{R}$评估两者质量差。公式表达为 $d _ { i } = \mathcal { R } ( \mathcal { T } ( x _ { i } ^ { B } ) , S ( x _ { i } ^ { B } ) \mid x _ { i } ^ { B } )$。为消除位置偏差，交换答案顺序运行两次取平均。设定阈值$\tau=1.0$，$d_i \geq \tau$标记为难样本（Hard），否则为易样本（Easy）。
- **阶段三：生成（Generation）**：$\mathcal{G}$从Hard池随机采样指令，生成同领域/同任务类型的新难指令；同时从Easy池采样指令生成同领域但分布更Long-tail的新易指令。Hard与Easy生成数量保持1:1比例（$r=1$）。新指令与Cache Pool中已有指令的ROUGE-L重叠度需低于0.7方视为有效。随后用新指令替换Train Pool，并将有效新指令加入Cache Pool，开启下一轮迭代。
- **理论视角**：该过程可解释为Min-Max博弈——学生模型在难样本上最小化师生响应差距，裁判/生成器根据学生学习进度最大化该差距，形成正反馈循环，直至系统收敛至师生响应难以区分的平衡态。

## 实验与结果
- **数据集与评估设置**：开放生成采用Vicuna-Instructions（80题，9类任务，GPT-4自动评估）；推理采用AGIEval（8任务，2,546样本）与BIG-Bench Hard（23任务，5,511样本）。均采用Zero-shot无CoT设置，解析首字母大写选项计算准确率。
- **基线模型**：LLaMA、Alpaca、WizardLM、Vicuna、ChatGPT。
- **主要结果**：
  - 开放生成：Lion-13B在Setting1/Setting2下平均相对ChatGPT质量达98.38%，超越Vicuna-13B（92.61%）约5.77个百分点；在数学与代码类
