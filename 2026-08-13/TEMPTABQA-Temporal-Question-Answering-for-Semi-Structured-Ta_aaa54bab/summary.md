---
title: "TEMPTABQA-Temporal-Question-Answering-for-Semi-Structured-Ta"
source: https://aclanthology.org/2023.emnlp-main.149.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:31:29"
field: "表格推理与时间问答"
keywords: ["时间问答", "半结构化表格", "Wikipedia Infobox", "大语言模型", "链式思维", "head-tail 泛化"]
innovations: ["首个半结构化表格时间问答基准 TEMPTABQA，覆盖 90+ 领域 11,454 QA 对", "建立 Head/Tail 领域划分评估稀有领域泛化能力", "多维度细粒度误差分析框架（问题类型/推理操作/显式隐式/实体类型/领域）"]
benchmarks: ["TEMPTABQA Head Test", "TEMPTABQA Tail Test"]
---

# 论文速读：TEMPTABQA: Temporal Question Answering for Semi-Structured Tables

## 一句话总结
本文提出首个针对半结构化实体中心表格（Wikipedia Infoboxes）的时间问答基准任务 TEMPTABQA，包含 11,454 个问答对覆盖 90+ 领域；实验表明即使最强的 LLM（GPT-4）在此任务上仍落后人类 13.5 F1 分，揭示了当前 NLP 系统在表格时间推理方面的显著不足。

## 研究问题与动机
- **核心问题**：现代 NLP 系统能否有效对半结构化表格中的时间信息进行推理？现有方法为何不足？
- **表格信息的动态性**：实体事实信息随时间演变，半结构化表格（如 Infobox）中时间信息以显式（日期、年份）和隐式（排名先后、前后关系）两种形式存在，需要模型理解时间范围和时间区间。
- **现有数据集的局限**：WIKITABLEQUESTIONS、SQUALL、FINQA、TAT-QA 等现有表格问答数据集在时间问题的数量和复杂度上严重不足，多数未聚焦时间推理。
- **半结构化表格的独特性**：Infobox 不同于结构化 SQL 表和知识图谱，时间信息往往隐含在文本片段中（如"Born: July 30, 1983 (age 38)"），需要常识推理而非简单查找。

## 核心贡献（创新点）
- **首创时间问答数据集**：构建了 TEMPTABQA，包含 11,454 个问答对和 1,208 个 Wikipedia Infobox 表格，覆盖 90+ 领域，是首个专注于半结构化表格时间推理的数据集。
- **多层次的时间问题分类体系**：将问题分为显式/隐式时间、过去/现在/未来时间区间、简单/复杂问题，并标注了所需的数学操作（max/min/count/difference/comparison 等）。
- **全面的人机对比评测**：评估了从 BART、T5 到 GPT-4、PaLM 等多种模型在零样本、微调、少样本、CoT 提示下的表现，发现最强模型 GPT-4 与人类仍存在 13.19-20.61 F1 分的差距。
- **细粒度误差分析框架**：从问题类型、推理操作、显式/隐式区分、答案实体类型、Head/Tail 领域五个维度对模型错误进行系统性分析。
- **数据集质量保证机制**：采用多轮审核流程（MTurk 标注 + 三重独立标注验证 + NLP 专家手动审核），确保答案准确性和问题复杂性，人类准确率达 86%。

## 方法详解
- **数据构建流程**：
  - **表格选择**：分析 1,208 个 Infobox 模板，优先选择包含大量时间值（日期、年份）的热门高浏览量文章中的长表格。
  - **标注过程**：使用 MTurk  crowdsourcing 生成 QA 对，要求标注者使用不同疑问词开头（What/When/How many 等），避免 yes/no 问题和直接提取型问题，鼓励跨行推理和数学计算。
  - **偏差控制**：多样化表格类别（每批不超过 3 个同域表格）、移除热门行（如 Born/Died）、打乱表格顺序、移除热门子章节（如 Olympics）。

- **数据集划分**：
  - Train: 784 表格, 7,680 QA 对, 73 领域
  - Dev: 97 表格, 885 QA 对, 67 领域
  - Head Test: 202 表格, 1,851 QA 对, 73 领域（流行常见领域）
  - Tail Test: 125 表格, 1,038 QA 对, 19 领域（罕见领域）

- **问题复杂度分类**：
  - 简单问题占 57.81%，复杂问题占 42.19%
  - 单实体问题占 52.10%，多实体问题占 47.90%
  - 复杂问题定义：需同时进行至少两个时间推理步骤（before/after/inbetween 类）

- **时间推理类型分布**：
  - 隐式时间问题占 63.23%，显式时间问题占 36.76%
  - 现在时间区间占 66.64%，未来占 8.48%，过去占 3.08%

- **数学操作分布**：
  - Count 操作最多（3,564 个），其次为 max（402）、sum（312）、min（377）、difference（98）、compare（133）、average（40）

- **表格表示方式**：
  - 将 HTML 表格转换为 JSON 结构后线性化，使用 "tab"/":" 分隔列，";"/换行分隔行，"##" 分隔子章节，例如："Title: Petya Nedelcheva # Personal Information # Country: Bulgaria; Born: July 30, 1983 (age 38)"

- **评估指标**：F1、EM、Rouge-1/2、MET，评估时设定当前日期为 2022 年 12 月，并在表格中新增"Current Date: December, 2022"行

## 实验与结果
- **最强模型表现**：
  - GPT-4 + CoT 在 Head 集上 F1=74.30，在 Tail 集上 F1=67.21
  - 最佳微调模型 Flan-T5-XL 在 Head 集上 F1=55.74，在 Tail 集上 F1=55.24
  - 人类在 Head 集上 F1=87.49，在 Tail 集上 F1=87.82

- **关键结论**：
  - TEMPTABQA 极具挑战性，所有模型均显著落后于人类
  - GPT-4 与人类差距：Head 集 13.19 F1，Tail 集 20.61 F1
  - 微调帮助中等规模模型提升：Flan-T5-XL 微调后超越未微调的 Flan-T5-XXL（Head: 55.74 vs 43.29，Tail: 55.24 vs 38.68）
  - 提示策略排序：Few-shot + CoT > Few-shot > Zero-shot
  - GPT 系列模型在所有设置下 consistently 优于 Flan-T5 和 T5
  - 将表格转为知识图谱（+KG）略优于线性化方法

- **Head vs Tail 性能差距**：
  - GPT-4 在 Head 比 Tail 高约 7 F1 分（CoT 设置下）
  - 人类在两个集合上表现几乎相同（~87.5 F1）
  - 性能差距主要源于知识迁移能力不足

- **细粒度分析发现**：
  - 模型在"Where"和"How Much"问题上表现较好，"What"/"Who"/"When"问题表现较差
  - 模型在 max/count 操作上较强，在 min/difference/comparison 操作上较弱
  - 隐式时间问题比显式时间问题上模型表现更好
  - 模型在 age gap、boolean、person、place 相关问题上表现较弱，count 相关问题表现较好
  - 百分比和序数词是所有模型和人类的共同难点

## 相关工作脉络
- **表格问答数据集**：WIKITABLEQUESTIONS (Pasupat & Liang, 2015)、SQUALL (Shi et al., 2020)、FINQA (Chen et al., 2021d)、TAT-QA (Zhu et al., 2021)、HYBRIDQA (Chen et al., 2020c)、FETAQA (Nan et al., 2022) — 这些数据集主要关注非时间性问题，时间相关问题数量极少（Table 32 显示大多低于 40%）
- **半结构化表格推理**：INFOTABS (Gupta et al., 2020) 聚焦表格 NLI 而非时间 QA；TEMPTABQA 扩展至时间推理场景
- **时间问答数据集**：TIME-SENSITIVE-QA (Chen et al., 2021c)、TORQUE (Ning et al., 2020) 基于维基百科段落而非表格；TEMPQA-WD、CRONQUESTIONS、TEMPQUESTIONS 基于知识图谱嵌入；TEMPTABQA 填补了表格场景空白
- **表格预训练模型**：TAPAS (Herzig et al., 2020)、TaBERT (Yin et al., 2020)、TABBIE (Iida et al., 2021)、TabGCN (Pramanick & Bhattacharya, 2021) — 本文使用这些预训练模型的微调变体作为基线
- **时间感知语言模型**：Dhingra et al. (2022)、Iv et al. (2022) 探索在预训练中融入时间知识；本文建议将此方向应用于表格场景

## 局限性与未来方向
- **数据集局限**：
  - 仅使用 Wikipedia Infobox 表格，未涵盖电商属性、医疗记录、金融报告等其他半结构化表格
  - 英文单语限制，未探索多语言场景
  - 潜在标注偏差：尽管采取措施，但可能仍存在选择偏差和流行度偏差

- **实验局限**：
  - 计算资源有限，未能对 XXL 模型进行微调
  - 仅测试了少数开源 LLM，未全面覆盖所有开源模型

- **未来方向**（论文自述）：
  - 扩展到混合结构（文本+表格+图像）的时间查询
  - 研究随时间演变的动态表格
  - 探索开放域查询（检索+提取+理解+时间推理一体化）
  - 使用参数高效微调（PEFT）技术微调更大模型
  - 将时间预训练和知识注入方法应用于表格场景

## 研究启发与可借鉴点
- **数据集构建设计**：多轮验证机制（Crowdsourcing → 三重独立标注 → 专家审核）值得借鉴，可将此质量保障流程迁移至其他表格 QA 数据集构建
- **Head/Tail 划分策略**：按领域流行度划分 Head 和 Tail 测试集的设计，可有效评估模型在稀有领域的泛化能力，适合迁移至其他 benchmark 设计
- **多维细粒度分析框架**：从问题类型、推理操作、显式/隐式、实体类型、领域五个维度进行误差分析的方法论，可复用于其他基准的任务拆解诊断
- **隐式时间推理的挑战性**：实验表明模型在隐式时间问题上表现反而优于显式，这挑战了直觉认知，提示我们在设计时间推理任务时应更多关注隐性时间关系的建模
- **CoT 提示的增量收益有限**：即使 GPT-4 + CoT 也仅比零样本提升约 2 F1，说明时间推理的核心瓶颈在于模型对表格结构和时间语义的理解，而非推理链长度，这为后续工作指明改进方向

## 关键术语表
- **TEMPTABQA**：首个专为半结构化实体中心表格设计的时间问答数据集，包含 11,454 个 QA 对和 1,208 个 Wikipedia Infobox 表格
- **半结构化表格（Semi-structured Tables）**：介于非结构化文本和结构化数据库之间的表格形式，如 Wikipedia Infobox，具有不规则布局和隐式时间信息
- **显式时间问题（Explicit Temporal Questions）**：直接在问题中提及具体时间/日期的时间问答，如"When was X born?"
- **隐式时间问题（Implicit Temporal Questions）**：不直接提及具体时间，但需通过时间关系推断的问题，如"Who was the predecessor of X?"
- **Head/Tail 领域划分**：Head 指流行常见领域（如体育），Tail 指稀有领域（如小众运动、历史事件），用于评估模型的领域泛化能力
- **Chain of Thought Prompting（CoT）**：在 few-shot 示例中加入逐步推理过程，引导模型输出中间推理步骤的提示技术
- **时间区间推理（Temporal Interval Reasoning）**：涉及 before/after/present 时间区间判断的推理任务

## 可复现要素
- **数据集**：TEMPTABQA 已在 Zenodo 公开（https://zenodo.org/records/10022927），包含 train/dev/head test/tail test 四部分
- **代码**：分析脚本和建模代码托管在 https://temptabqa.github.io
- **模型权重**：微调模型权重未明确公开，但提供了完整的超参数设置（Appendix C）
- **关键超参**：训练 1-3 epochs，sequence length 1024-4096，warmup 500 steps，weight decay 0.01，learning rate 2e-5，gradient accumulation 8 steps
- **评估设置**：当前日期设为 December 2022，评估指标 F1/EM/Rouge-1-2/MET
