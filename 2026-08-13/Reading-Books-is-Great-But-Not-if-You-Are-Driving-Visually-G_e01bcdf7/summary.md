---
title: "Reading-Books-is-Great-But-Not-if-You-Are-Driving-Visually-G"
source: https://aclanthology.org/2023.emnlp-main.57.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:08:53"
field: "多模态社会常识推理"
keywords: ["visually grounded reasoning", "commonsense norms", "defeasible reasoning", "multimodal benchmark", "moral judgment", "knowledge distillation", "human-AI collaboration"]
innovations: ["提出 NORMLENS 多模态基准，首次系统性评估视觉接地下可废止常识规范推理；提出基于大语言模型蒸馏的低成本对齐方法，无需额外人工标注即可显著提升模型与人类判断的一致性；按人类共识度分层评估（HA/MA），更稳健地衡量主观性任务的模型对齐程度"]
benchmarks: ["NORMLENS"]
---

# 论文速读：Reading-Books-is-Great-But-Not-if-You-Are-Driving-Visually-Grounded-Reasoning-About-Defeasible-Commonsense-Norms

## 一句话总结
本文提出多模态基准数据集 NORMLENS，用于评估模型在视觉接地（visual grounding）情境下对可废止常识规范（defeasible commonsense norms）进行道德判断与解释的能力，并展示了通过大语言模型蒸馏知识可有效缩小模型与人类对齐的差距。

## 研究问题与动机
1. **核心问题**：常识规范具有可废止性（defeasibility），即同一行为在不同情境下可能被判定为"对"或"错"；现有工作主要基于纯文本情境研究，缺乏视觉情境下的可废止常识规范推理研究。
2. **现实场景缺口**：真实世界的情境往往以视觉形式呈现（如图像），而非显式的语言描述，人类能直接从视觉场景做出直觉性判断，但机器在这类任务上严重不足。
3. **评估空白**：缺乏涵盖多模态情境、道德判断与自由形式解释的综合基准。
4. **道德判断主观性**：道德判断存在天然分歧，需系统性地捕捉并建模这种不一致性。

## 核心贡献（创新点）
1. **构建 NORMLENS 多模态基准**：包含 2K 个多模态场景（图像-动作对），由 5 名人工标注者提供道德判断（Wrong/Okay/Impossible）及自由形式解释，共 10K 条标注。
2. **设计两项新任务**：判断任务（judgment）评估模型与人类判断的对齐度；解释任务（explanation）评估模型生成解释与人类解释的相似度（基于 BLEU-2、Rouge-L、METEOR）。
3. **首次系统性评估视觉接地常识规范推理**：测试了 LLM、Socratic Models（SM）、视觉语言模型（VLM）等多类模型，揭示了 VLM 在视觉感知利用上的不足及 SM 结合强大 LLM 后的优势。
4. **提出基于 LM 蒸馏的无额外人工标注对齐方法**：利用 ChatGPT 生成 90K 个多模态情境标注，对模型微调后可实现最高 31.5% 的判断精度提升。
5. **数据集按人类共识度分层**：将数据集分为 NORMLENS^HA（高共识）与 NORMLENS^MA（中共识），使评估更具鲁棒性。

## 方法详解
1. **数据生成流水线**：
   - 从 Sherlock、COCO Captions、Localized Narratives 三个视觉-语言数据集随机采样 15K 图像描述。
   - 使用 GPT-3.5-turbo（ChatGPT）生成三元组（图像描述 d → 动作 a、上下文 c^T），要求动作在图像描述下道德上可接受，但在生成上下文下道德上不可接受。
   - 两轮 LM 过滤：第一轮过滤"物理上不可能"的动作；第二轮过滤"LM 认为在生成上下文中仍道德可接受"的动作。
   - 通过 FAISS 向量检索为文本上下文匹配对应图像描述和图像，再经多样性过滤（关键词限制如 funeral、hospital ≤30次），最终得到 18K 多模态场景。
   - 从 18K 中抽样 2.2K 场景进行人工标注，每场景 5 名标注者，提供三项判断（Wrong/Okay/Impossible）及自由形式解释，并通过复核排除不合理解释。

2. **判断任务评估**：
   - 给定动作 a_i 和图像 x_i，模型输出判断 ŷ_i ∈ {Wr., Ok., Im.}。
   - 精度（precision）衡量模型是否与至少一名人类标注者的判断一致。
   - 采用宏平均（macro average）处理类别不平衡。

3. **解释任务评估**：
   - 评估公式：E_i = max_{0≤j≤n-1} δ(ŷ_i, y_i^j) · f(ê_i, e_i^j)，其中 δ 为判断一致性指示函数，f 为生成解释与人类解释的相似度（BLEU-2、Rouge-L、METEOR）。
   - 仅在 NORMLENS^HA 上评估（因 MA 集合存在多种标签对应多种解释）。

4. **LM 蒸馏微调**：
   - 用 ChatGPT 基于 30K 图像描述生成三种动作（道德不当、道德适当、无关情境），各带判断与解释，共 90K 样本。
   - 将生成数据按 8:1 划分为训练集与验证集，对 SM（仅微调 LM 部分）和 VLM 进行监督微调（一个 epoch），学习率 1e-5~2e-5，batch size 16~256。

## 实验与结果
1. **数据集统计**：
   - NORMLENS^HA：934 场景，含 Wrong（187）、Okay（350）、Impossible（397）三类。
   - NORMLENS^MA：1049 场景，三类两两组合的混合场景。
   - 约 50% 场景存在标注者分歧；10% 场景四种判断均出现，已排除但不影响可用性声明。

2. **判断任务（NORMLENS^HA）**：
   - Random：33.3%，Majority Vote（以 Im. 为主）：42.5%。
   - 纯文本 LM：Vicuna-13B（39.9%）、GPT-3 Curie（33.7%）、GPT-3 Davinci（38.6%）、ChatGPT（42.2%）、GPT-4（43.2%）。
   - Socratic Model（SM + BLIP-2）：Vicuna-13B（42.1%）、GPT-3 Curie（36.4%）、GPT-3 Davinci（36.6%）、ChatGPT（63.9%）、**GPT-4（74.7%）**。
   - VLM（直接输出）：LLaVA（34.3%）、BLIP-2（39.8%）、InstructBLIP（41.9%）。
   - **最强结果**：SM with GPT-4 达 74.7%（HA），85.9%（MA）。
   - **关键发现**：① 视觉输入重要（SM 普遍优于纯 LM）；② 推理能力关键（VLM 分数低于 Majority Vote，表明视觉-语言模型未能有效利用视觉感知）；③ 使用 ChatGPT 生成数据后，其在 Wrong 类上得分最高（71.1%），但改用 BLIP-2 生成描述后降至 71.1%，使用 ground-truth 描述后升至 80.2%，说明视觉理解能力制约 SM 表现。

3. **解释任务（NORMLENS^HA）**：
   - SM with GPT-4：BLEU-2=18.7、Rouge-L=16.6、METEOR=19.7，显著领先其余基线。

4. **微调后提升（NORMLENS^HA）**：
   - SM with Vicuna-13B：判断精度 55.6%（+13.5），解释 METEOR 12.2（+2.4）。
   - SM with GPT-3 Curie：判断精度 56.2%（+19.8）。
   - SM with GPT-3 Davinci：判断精度 58.0%（+21.4）。
   - VLM with LLaVA Vicuna-13B：判断精度 49.7%（+15.4），解释 METEOR 10.7（+5.4）。
   - **最大提升**：GPT-3 Davinci 在 Impossible 类上提升 73.3%（从 0.0 到 73.3），总体精度提升 21.4%。
   - **副作用**：Okay 类分数普遍下降（如 Vicuna-13B 从 99.1 降至 64.0），说明生成数据导致模型更保守。

## 相关工作脉络
1. **视觉接地推理**：Zellers et al. (2019) 的 VCR、Hessel et al. (2022) 的 Sherlock 数据集关注视觉常识推理和 abduction 推理；本文补充了视觉接地下**可废止道德规范**这一未被探索的子领域。
2. **常识规范数据集**：Jiang et al. (2021) 的 Delphi（纯文本道德判断）、Pyatkin et al. (2022) 的 ClarifyDelphi（澄清问题生成）、Ziems et al. (2023) 的 NormBank（情境化规范知识库）均以文本为中心，本文引入**视觉情境**与自由形式解释，对比基准具有更强的 multimodal grounding 挑战。
3. **Socratic Models**：Zeng et al. (2022) 提出先将视觉输入转为文本再用 LM 推理的两阶段框架；本文验证该框架在视觉接地规范推理上的有效性，并发现视觉理解能力是瓶颈。
4. **VLM 直接推理**：LLaVA、BLIP-2、InstructBLIP 等端到端模型在通用视觉任务表现优异，但本文揭示它们在道德规范推理上反而不及纯文本 LM，暴露了**视觉表征与语义推理之间的gap**。
5. **知识蒸馏对齐**：West et al. (2022)、Kim et al. (2022) 等通过 LM 生成知识蒸馏到小模型；本文将该思路拓展到**多模态社会常识规范**领域，证明无需额外人工标注即可提升模型对齐。

## 局限性与未来方向
1. **文化偏差**：NORMLENS 由美英加英语母语者标注，未能覆盖不同社会文化背景下的道德规范多样性。
2. **平均化掩盖少数观点**：评估以人类共识平均值为准，可能忽略有效的少数派观点；虽然区分 HA/MA 子集，但未显式建模分歧。
3. **生成数据的保守倾向**：微调后模型在 Okay 类上得分下降，表明 AI 生成数据可能诱导更严格的道德判断，未来需设计更精细的数据平衡策略。
4. **数据规模有限**：仅 2.2K 场景，难以覆盖所有情境类型，计划扩展到更大规模。
5. **视觉理解瓶颈**：SM 在依赖 BLIP-2 生成描述时性能显著下降，未来需改进视觉-语言对齐或引入更强 VLM。

## 研究启发与可借鉴点
1. **人类-AI 协作的数据构建范式**：先由 LM 批量生成候选情境，再通过人类标注筛选高质量样本，兼顾规模与质量，可复用到其他多模态基准构建。
2. **分歧驱动的分区评估**：按人类共识度将数据集拆分为 HA/MA 两类，使评估更稳健，值得在多主观性任务中推广。
3. **LM 蒸馏作为低成本对齐手段**：在无额外人工标注的前提下，用 ChatGPT 生成 90K 多模态训练数据微调模型，显著提升与人类对齐度（最高 +31.5%），可迁移到类似的社会常识推理任务。
4. **Socratic Model 架构的启示**：将视觉感知与语言推理解耦，让强 LLM 专注推理，VLM 专注感知，既保留推理能力又适配视觉输入，可作为基线架构设计参考。
5. **错误分析视角**：指出 GPT-4 在 Wrong 类（71.1% vs 80.2%）因 BLIP-2 视觉描述误差导致的性能下降，提示在多模态推理中**感知模块的质量瓶颈**需优先优化。

## 关键术语表
- **Defeasible Commonsense Norms（可废止常识规范）**：指根据具体情境可被加强或削弱的日常行为规范，同一行为在不同语境下可能被判定为道德上恰当或不恰当。
- **Visual Grounding（视觉接地）**：将抽象概念或语言描述与视觉场景建立联系的能力，本文指模型根据图像情境进行推理。
- **NORMLENS**：本文提出的多模态基准数据集，包含 2K 场景、10K 标注，用于评估视觉接地常识规范推理。
- **NORMLENS^HA / NORMLENS^MA**：按人类共识度划分的子集，HA 为所有标注者一致判断，MA 为标注者存在分歧。
- **Socratic Model（SM）**：两阶段视觉-语言模型，先用 VLM 将图像转为文本描述，再用 LLM 进行推理判断。
- **Macro Average Precision（宏平均精度）**：先对每个类别计算精度再均匀平均，用于处理多类别不平衡问题。
- **Judgment and Explanation Alignment（判断与解释对齐）**：评估模型输出与人类判断一致性，以及解释与人类解释的语言相似度。
- **LM Distillation（语言模型蒸馏）**：用大语言模型生成带标注的合成数据，用于微调更小或特定领域的模型。

## 可复现要素
- **数据集**：NORMLENS（2.2K 多模态场景，10K 标注），公开链接：https://seungjuhan.me/normlens
- **代码**：公开（同链接）
- **图像来源**：Sherlock、COCO Captions、Localized Narratives（公开数据集）
- **关键超参**：
  - 数据生成：GPT-3.5-turbo，temperature=0.1，top-p=0.95，frequency/presence penalty=0
  - 微调学习率：Vicuna-13B 2e-5、GPT-3 通过 OpenAI API、LLaVA 2e-5、InstructBLIP 1e-5
  - Batch size：Vicuna-13B 256（gradient accumulation=8）、LLaVA 32、InstructBLIP 16
  - Epoch：全部 1 epoch
- **评估指标**：Precision（判断）、BLEU-2 / Rouge-L / METEOR（解释）
