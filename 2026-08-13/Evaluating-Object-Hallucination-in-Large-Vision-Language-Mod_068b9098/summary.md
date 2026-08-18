---
title: "Evaluating-Object-Hallucination-in-Large-Vision-Language-Mod"
source: https://aclanthology.org/2023.emnlp-main.20.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:39:38"
field: "多模态大模型评估"
keywords: ["物体幻觉", "大视觉语言模型", "POPE", "多模态评估", "视觉指令微调"]
innovations: ["提出POPE将幻觉评估转化为稳定的二分类任务", "揭示视觉指令数据分布对幻觉的影响机制"]
benchmarks: ["MSCOCO", "A-OKVQA", "GQA"]
---

# 论文速读：Evaluating-Object-Hallucination-in-Large-Vision-Language-Models

## 一句话总结
本文系统评估了大视觉语言模型（LVLM）中的物体幻觉问题，发现主流LVLM普遍存在严重的幻觉现象，并提出了一种基于投票的二分类评估方法POPE，以更稳定、灵活地度量幻觉程度。

## 研究问题与动机
- **LVLM的幻觉问题被忽视**：尽管LVLM在复杂多模态任务上表现优异，但其生成的描述中常包含图像中不存在的物体，严重影响实际部署的安全性（如自动驾驶场景）。
- **现有评估方法不稳定**：CHAIR等评测指标对指令设计敏感、依赖人工解析规则，且受生成文本长度影响大，难以公平比较不同模型。
- **幻觉成因不明**：不清楚LVLM幻觉是否与训练数据的物体分布特征（如频繁出现或共现的物体）相关。
- **评估方法需改进**：需要一种更稳定、可扩展（可应用于无标注数据）、与指令无关的幻觉评估方法。

## 核心贡献（创新点）
1. **系统性幻觉评估**：首次在多个代表性LVLM（mPLUG-Owl、LLaVA、MultiModal-GPT、MiniGPT-4、InstructBLIP）上系统评估物体幻觉，发现幻觉比小规模VLPM更严重。
2. **揭示幻觉数据成因**：首次实证表明LVLM倾向于幻觉视觉指令数据中频繁出现（HR_A@10≈0.5）或与图像中物体共现（HR_C@10≈0.6）的物体。
3. **提出POPE评估方法**：将幻觉评估转化为二分类任务（Yes/No问答），避免了解析规则和指令敏感性问题，显著提升评估稳定性（F1标准差从CHAIR的3.22降至0.78）。
4. **扩展性验证**：结合SEEM自动分割工具，证明POPE可迁移至无标注数据集（A-OKVQA、GQA），且结果与人工标注一致。

## 方法详解
**POPE（Polling-based Object Probing Evaluation）** 核心设计：

1. **问题构建**：将幻觉评估转化为二元分类任务，使用模板问句"Is there a/an <object> in the image?"让模型输出Yes/No。

2. **物体采样策略**：
   - **Random Sampling**：随机采样图像中不存在的物体
   - **Popular Sampling**：从全数据集中采样top-k最常见但不存在于当前图像的物体
   - **Adversarial Sampling**：采样与图像中已有物体共现频率最高的top-k不存在物体

3. **评估指标**：Accuracy、Precision、Recall、F1 Score，以及Yes回答比例（衡量过度自信倾向）。

4. **可扩展设计**：支持两种物体来源——人工标注或自动分割工具（如SEEM），使方法可应用于无标注数据集。

**假设验证方法**：
- 定义HR_A@k（高频物体命中率）和HR_C@k（共现物体命中率）量化统计：
$$HR_A@k = \frac{1}{n}\sum_{i=1}^{n}\frac{Hit@k(i)}{Hallucinated(i)}$$
- 分析表明约50%的幻觉物体属于COCO top-10高频物体，>50%属于与已有物体共现的top-10物体。

## 实验与结果
**数据集**：MSCOCO验证集（2000张图像），及A-OKVQA、GQA用于扩展验证。

**评估模型**：
- LVLMs：mPLUG-Owl、LLaVA、MultiModal-GPT、MiniGPT-4、InstructBLIP
- VLPMs基线：OSCAR、VinVL、BLIP、OFA

**CHAIR评测结果（Table 1）**：
- LLaVA在I_1指令下CHAIR_I=14.8，CHAIR_S=25.4；改用I_2指令后CHAIR_I=18.8，CHAIR_S=62.7（几乎翻倍）
- InstructBLIP表现最佳：CHAIR_I=2.6，CHAIR_S=3.7，归因于其使用短指令数据
- LLaVA、MultiModal-GPT、mPLUG-Owl幻觉严重程度远超小规模VLPM（如OSCAR_CHAIR_S=13.0）

**POPE评测结果（Table 3）**：
- **最佳模型**：InstructBLIP在Random设置下F1=88.73，Adversarial设置下F1=74.37
- **最差模型**：MultiModal-GPT在所有设置下F1≈66.67（接近随机）
- LLaVA、MultiModal-GPT、mPLUG-Owl的Yes回答比例>95%，表明过度自信
- 性能趋势一致：Random > Popular > Adversarial，验证了假设

**稳定性对比（Table 4）**：
- POPE F1标准差：0.78 vs CHAIR_I标准差：3.22
- 四种不同指令模板下POPE结果波动小，CHAIR波动大

**SEEM扩展验证（Table 5）**：
- 无标注数据集上的POPE结果趋势与人工标注一致
- 性能差距归因于SEEM分割粒度更细，挑战更大

**VQA相关性（Table 6）**：
- InstructBLIP在POPE和VQA上均表现最佳
- MiniGPT-4 POPE优于LLaVA，但VQA表现较差，说明幻觉程度与VQA能力不完全正相关

## 相关工作脉络
1. **CHAIR基准（Rohrbach et al., 2018）**：最早的图像描述幻觉评估方法，基于精确匹配统计幻觉物体比例；本文指出其对指令敏感、依赖解析规则，不适合LVLM。
2. **VLPM幻觉研究（Biten et al., 2022; Dai et al., 2023b）**：研究小规模VLPM的幻觉问题；本文发现LVLM因指令微调反而加剧幻觉，且从数据分布角度解释成因。
3. **LLM幻觉（Ji et al., 2022; Bang et al., 2023）**：综述和评估LLM的事实不一致问题；本文聚焦于视觉模态引入的物体幻觉，强调多模态场景的特殊性。
4. **视觉指令微调（Liu et al., 2023 - LLaVA）**：提出使用LLM生成合成指令数据；本文发现此类数据包含潜在幻觉信息，会误导LVLM。
5. **SEEM分割工具（Zou et al., 2023）**：通用物体分割模型；本文创新性地将其与POPE结合，实现无标注数据集的幻觉评估。
6. **VQA评估方法**：传统VQA依赖开放答案和人工评判；本文使用ChatGPT辅助解析，并对比POPE与VQA性能的相关性。

## 局限性与未来方向
- **仅关注物体级幻觉**：未考虑属性、数量、位置等细粒度幻觉
- **计算资源限制**：仅在部分验证集上评估，结果可能受数据分布影响
- **答案匹配局限**：使用基于规则的匹配判断Yes/No，可能遗漏模型未明确输出的情况
- **自动分割工具偏差**：SEEM标注的物体类别可能与人工标注不一致
- **评估模型数量有限**：未包含近期发布的模型（如GPT-4V、Qwen-VL等）
- **评估不等于整体能力**：POPE高分不代表模型整体性能更强

## 研究启发与可借鉴点
1. **二分类评估范式**：将生成任务转化为判别任务（Yes/No问答）可大幅提升评估稳定性，适用于其他生成模型的可靠性评估。
2. **对抗性采样策略**：Popular/Adversarial采样可构造不同难度测试集，量化模型在"陷阱"上的表现，值得推广至其他评估场景。
3. **数据分布归因分析**：通过HR指标量化训练数据统计特性与模型行为的关联，为幻觉成因提供可解释视角。
4. **工具辅助扩展性**：结合自动分割工具实现无标注数据的评估，降低了评测成本，可扩展至更多领域。
5. **幻觉与下游任务解耦分析**：揭示幻觉程度与VQA性能不完全正相关，提醒评估需多维度考量。

## 关键术语表
**LVLM（Large Vision-Language Model）**：整合强大LLM与视觉编码器的大规模多模态模型，支持图像描述、VQA等任务。

**Object Hallucination**：模型在描述图像时生成图像中不存在或不一致的物体。

**POPE（Polling-based Object Probing Evaluation）**：本文提出的基于投票的物体探测评估方法，将幻觉评估转化为二分类任务。

**CHAIR（Caption Hallucination Assessment with Image Relevance）**：基于精确匹配的幻觉评估指标，计算描述中包含但未出现在图像中的物体比例。

**HR@k（Hit Ratio at k）**：衡量前k高频/共现物体在幻觉物体中的占比，用于量化数据分布与幻觉的关联性。

**Visual Instruction Tuning**：在图像-文本指令对上微调已预训练的VL模型，使其能遵循自然语言指令完成任务。

**SEEM（Segment Everything Everywhere All at Once）**：通用物体分割模型，可自动标注图像中的物体。

**Adversarial Sampling**：选择与图像中已有物体共现频率最高的非存在物体进行提问的采样策略。

## 可复现要素
- **数据集**：MSCOCO验证集（已公开），A-OKVQA、GQA（已公开）
- **代码**：论文未提及代码开源（注：后续工作POPE已有开源实现，可参考https://github.com/pohang-ai-lab/POPE）
- **模型权重**：使用的LVLM（LLaVA、MiniGPT-4、InstructBLIP等）均为开源模型
- **关键超参**：每图像问题数l=6，正负样本比例1:1，Top-k分别取10/20/30
