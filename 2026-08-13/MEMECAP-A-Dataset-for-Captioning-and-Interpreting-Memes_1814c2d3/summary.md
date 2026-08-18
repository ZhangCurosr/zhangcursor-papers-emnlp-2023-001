---
title: "MEMECAP-A-Dataset-for-Captioning-and-Interpreting-Memes"
source: https://aclanthology.org/2023.emnlp-main.89.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:50:51"
field: "多模态理解与视觉隐喻"
keywords: ["meme captioning", "visual metaphor", "vision-language models", "multimodal understanding", "dataset"]
innovations: ["提出梗图 captioning 任务及首个专用数据集 MEMECAP", "系统性基准测试显示 SOTA VL 模型在视觉隐喻理解上远逊于人类", "发现 CoT 隐喻显式提示在生成任务中未必有效，揭示图文依赖差异"]
benchmarks: ["MEMECAP", "BLEU-4", "ROUGE-L", "BERTScore", "Human Evaluation (Correctness/Completeness/Faithfulness)"]
---

# 论文速读：MEMECAP: A Dataset for Captioning and Interpreting Memes

## 一句话总结
本文提出"梗图 captioning"任务并发布首个专用数据集 MEMECAP（6.3K 条标注数据），测试 SOTA 视觉-语言模型在该任务上的表现，发现当前模型在视觉隐喻理解上显著低于人类水平，常犯错误包括字面化处理图像内容、复制梗图文本或幻觉生成。

## 研究问题与动机
- **视觉隐喻理解是梗图解读的核心难点**：梗图通过现有图像附加新文本产生意义，需同时理解视觉隐喻和文字内容，现有 VL 模型在多为字面描述的数据集上训练，缺乏隐喻推理能力。
- **缺少面向梗图 captioning 的专用数据集**：既有隐喻数据集（如 MultiMET、Met-Meme）仅标注隐喻存在与否及类型，不含 meme caption；New Yorker Cartoon 基准为判别式任务而非生成式 captioning。
- **SOTA VL 模型在真实用户梗图上表现不明**：尽管模型在 image captioning 和 VQA 上进展迅速，但在需要背景知识和常识推理的梗图解读任务上是否有能力尚待评估。
- **人类与模型的差距值得系统分析**：模型倾向于复制梗图内文字、忽略重要视觉元素或将视觉内容字面化处理，需定量揭示此类错误模式。

## 核心贡献（创新点）
1. **提出梗图 captioning（meme captioning）任务**：输入为梗图及其帖子标题，输出为描述发帖人意图的简洁 caption，要求模型识别并解释视觉隐喻、忽略字面图像元素——与已有工作（MultiMET 等仅标注隐喻类型）的本质区别在于这是生成式任务而非判别式。
2. **发布 MEMECAP 数据集（6,384 条梗图）**：每条样本含 meme caption、literal image caption（去文字后的纯图像描述）和 visual metaphor 注解，同时提供互补性的元数据分析（隐喻类型分布）——与现有数据集（MetaCLUE 使用合成图像、IRFL 为四选一判别任务）的本质区别在于面向真实用户发布的梗图且支持生成式评测。
3. **系统性基准评测三模型 × 多输入配置 × 多种训练设置**：覆盖 zero-shot、few-shot（4/8/12 shots）、fine-tuning 和 CoT 提示，揭示各模态贡献与模型依赖差异——与已有工作的本质区别在于首次全面评估 VL 模型在真实梗图隐喻解读任务上的能力边界。
4. **通过人类评估定量刻画模型错误模式**：从 Correctness、Appropriate Length、Visual Completeness、Textual Completeness、Faithfulness 五个维度评估，发现模型与人类相差 18~36 个百分点，并归纳出"字面化""复制文本""幻觉"三大错误类型。

## 方法详解
- **数据收集**：从 Reddit `/r/memes` 子版块抓取含标题的帖子，人工筛选排除无文字/文字过多内容，使用 Google banned word list 过滤脏话、用 NudeNet Classifier 过滤色情图像。
- **两轮注释流程**：第一轮：使用 LaMa inpainting 工具移除梗图文字后，由标注者提供 literal image caption（如"Tom cat 和两只小猫握手微笑"）；第二轮：标注者先逐术语判断字面描述中哪些是隐喻载体（vehicle）及隐喻目标（target），再撰写 meme caption（不含隐喻载体名称），训练集每梗 1 条 caption，测试集 2~4 条。
- **质量控制**：标注者限定英语国家、5000 HIT 通过率≥98%、通过资格考试；每轮均剔除被标注者标记为冒犯/色情/仇恨/无法解读的梗图。
- **数据集划分**：使用 OPT-2.7B 对 meme caption 向量聚类，每簇采样 10% 入 test set，确保训练/测试主题多样性（训练+验证 5,828 条，测试 559 条）。
- **模型与输入配置**：使用 OpenFlamingo-9B（LLaMA-7B + CLIP ViT/L-14）、MiniGPT4（Vicuna-13B + BLIP-2）、LLaMA-7B；输入组合包括 meme+title、meme+img cap、meme+title+img cap、meme+title+img cap+OCR text（EasyOCR 提取）、以及 CoT 格式下额外提供隐喻 rationale。
- **CoT 提示模板**：给定图像、title、img cap、OCR text，要求 LLM 先以"X 是 Y 的隐喻"格式逐条写出隐喻解释（rationale），再给出最终 answer。
- **评估指标**：自动指标 BLEU-4、ROUGE-L、BERTScore（microsoft/deberta-xlarge-mnli）；人类评估五维度（Correctness、Appropriate Length、Visual Completeness、Textual Completeness、Faithfulness），30 条样本、3 名标注者多数投票。

## 实验与结果
- **数据集统计**：训练+验证 5,828 条（平均每梗 1.0 条 M-Cap、1.0 条 I-Cap、2.1 条 Mph），测试 559 条（3.4 条 M-Cap、1.0 条 I-Cap、3.1 条 Mph）；44% 梗图为互补型（需图文联合理解），其余分布为文字主导 19%、图像主导 19%、无隐喻 19%。
- **最佳自动评测结果**（few-shot，4 shots，全输入 + rationale）：
  - **Flamingo CoT**：BLEU-4 = **27.02**，ROUGE-L = **43.46**，BERTScore-F1 = **74.32**
  - **LLaMA few-shot**（无图像输入，仅 title+img cap+OCR+rationale）：BLEU-4 = **26.63**，ROUGE-L = **43.41**，BERTScore-F1 = **74.71**
  - Flamingo 在多数设置下优于 MiniGPT4（BLEU/ROUGE/BERTScore 分别高出约 15/12/6 分点）
- **Human Evaluation 差距**（模型 vs 人类，全输入 few-shot 设置）：Correctness 低 36.6 分点、Textual Completeness 低 29.3 分点、Visual Completeness 低 24.5 分点、Faithfulness 低 18.4 分点；仅 Appropriate Length 与人类接近。
- **CoT 提示反效果**：Flamingo CoT 的 BLEU-4 从 26.73（无 rationale）到 27.02，提升有限；但 LLaMA CoT 从 26.63 降至 26.40，幻觉明显增加（BERTScore 下降），说明显式提供隐喻信息不一定有帮助。
- **Ablation 结论**：Flamingo 更依赖视觉模态（去掉 meme 比去掉 title 下降更多），MiniGPT4 更依赖文本模态（去掉 title 导致 BLEU 下降 8.2），说明不同模型对多模态信息的利用策略存在本质差异。
- **Fine-tuning 负面效应**：MiniGPT4 fine-tune 最后一层后 BLEU 从 12.46 降至 7.50，所有人类评估维度均下降，推测冻结的预训练模型缺乏梗图解读知识，仅微调投影层不足。

## 相关工作脉络
1. **MultiMET（Zhang et al., 2021）/ Met-Meme（Xu et al., 2022）**：标注图像-文本对的隐喻存在性和类型，但不含 meme caption 生成目标；MEMECAP 在其基础上增加了生成式 caption 和 literal image caption 的配套标注。
2. **MetaCLUE（Akula et al., 2023）**：面向合成图像的视觉隐喻检索/captioning 任务；MEMECAP 的差异在于使用真实用户发布的梗图，更具现实挑战性。
3. **IRFL（Yosef et al., 2023）**：四选图像匹配隐喻表达（判别式）；MEMECAP 提供生成式 captioning 任务，评测视角不同。
4. **New Yorker Cartoon Caption Contest（Hessel et al., 2023）**：针对漫画的 caption 匹配和幽默解释（判别+解释）；MEMECAP 定位为生成式任务，且基于 meme 而非原创插画，文化背景更贴近网络用语。
5. **WHOOPS（Bitton-Guetta et al., 2023）**：测试模型对违反常识的合成图像的理解；MEMECAP 聚焦真实场景中的视觉隐喻解读，两者互补。
6. **Chakrabarty et al.（2023）**：使用 LLM 解释视觉隐喻以提升图像生成质量；本文发现将隐喻 rationale 作为输入并不能显著提升 meme captioning 性能，与图像生成场景形成对比。

## 局限性与未来方向
- **隐喻注释质量不一致**：标注者能解释梗图含义但未必善于将视觉载体映射为文字目标，导致提供的 metaphor keywords 质量参差，这可能是 CoT 未带来显著提升的原因。
- **主观性与背景知识依赖**：不同标注者对 meme 含义的理解存在差异，评测标准本身具有一定主观性，尽管有内部标注员澄清机制仍无法完全消除。
- **内容过滤可能导致偏差**：使用脏话列表和 NudeNet 过滤虽保护了数据安全，但可能系统性移除某些类型的梗图（如讽刺/政治类），影响数据多样性。
- **MiniGPT4 fine-tuning 效果差**：仅微调最后一层投影无法有效提升模型理解梗图的能力，需探索更合适的微调策略（如全参数微调、LoRA 等）。
- **未来方向**：改进视觉隐喻识别与解释方法、探索多模态融合机制以更好地利用图文互补信息、开发更强的评测指标（超越 n-gram 重叠）、扩展至多语言梗图理解。

## 研究启发与可借鉴点
1. **双轮注释策略值得迁移**：第一轮去除文字获得 literal caption 防止文本 bias，第二轮结合完整信息标注隐喻——此范式可复用于其他含隐喻的跨模态任务（如广告图像理解、漫画解读）。
2. **CoT 提示设计可借鉴**：将"隐喻 vehicle→target"的映射作为中间推理步骤显式给出，是一种新颖的 multimodal CoT 设计，可推广至其他需要隐喻推理的任务。
3. **消融实验设计启示**：系统性消融视觉/文本模态并对比不同模型的依赖偏好（Flamingo 偏视觉 vs MiniGPT4 偏文本），揭示了模型架构差异对多模态融合的深刻影响，可作为后续工作的评测范式。
4. **人类评估多维框架**：Correctness、Completeness（视觉/文本分开）、Faithfulness 的三维评估体系可复用于其他生成式跨模态任务的评测，比单一自动指标更能反映实际问题。
5. **LLaMA 无图像输入的强表现**：仅靠 title + img cap + OCR text 即达到与 Flamingo 相近的自动指标，提示在资源受限场景下可考虑"纯文本蒸馏视觉信息"的轻量方案。

## 关键术语表
- **MEMECAP**：本文发布的梗图 captioning 数据集，含 6,384 条梗图及 meme caption、literal image caption、visual metaphor 三类标注。
- **Visual metaphor（视觉隐喻）**：用一个视觉元素（vehicle）比喻另一个概念（target）的表达方式，是梗图意义生成的核心机制。
- **Meme caption（梗图 caption）**：描述梗图发帖人试图传达的含义的简洁文本，不包含隐喻载体（如人物/角色名）的字面提及。
- **Literal image caption（字面图像 caption）**：忽略梗图文字、仅描述图像视觉内容的 caption。
- **Metaphor vehicle（隐喻载体）**：承载隐喻意义的视觉元素（如人物、角色、手势、表情）。
- **Metaphor target（隐喻目标）**：隐喻所指向的实际含义（如"发帖人本人"、"某种态度或行为"）。
- **Chain of Thought (CoT) prompting**：通过逐步推理提示激发大语言模型多步推理能力的提示策略，本文将其用于隐喻解释。
- **In-context learning（上下文学习）**：在 prompt 中提供若干示例而不更新模型参数，让模型根据示例进行推理。

## 可复现要素
- **数据集**：MEMECAP 通过 ACL Anthology 提供（论文 DOI: 10.18653/v1/2023.emnlp-main.89），6,384 条梗图，标注成本：image caption $0.03/条、meme caption $0.16/条，标注者平均时薪 $13。
- **代码/权重**：论文未明确声明开源代码仓库；使用的基线模型均为开源模型（OpenFlamingo-9B、MiniGPT4、LLaMA-7B）。
- **关键超参**：few-shot 设置取 4/8/12 shots；CoT prompt 格式固定；OCR 使用 EasyOCR；图像去文字使用 LaMa inpainting；聚类使用 OPT-2.7B；NudeNet 安全阈值 >0.9。
- **硬件/训练**：论文未详细报告训练耗时与 GPU 配置，仅说明 MiniGPT4 微调仅更新最后一层投影。
