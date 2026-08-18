---
title: "Don-t-Get-Too-Technical-with-Me-A-Discourse-Structure-Based"
source: https://aclanthology.org/2023.emnlp-main.76.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:46:45"
field: "学术文本简化与新闻生成"
keywords: ["科学新闻生成", "话语结构", "内容规划", "风格迁移", "低资源摘要"]
innovations: ["提出结合元数据与discourse rhetoric role的两阶段BART生成框架", "构建多领域公开科学新闻数据集SCITECHNEWS", "验证结构化内容规划可显著提升新闻风格可读性与信息丰富度"]
benchmarks: ["SCITECHNEWS", "PLOS Lay Summarisation", "eLife Lay Summarisation", "ARXIV", "CNN/DailyMail"]
---

# 论文速读：Don't Get Too Technical with Me: A Discourse Structure-Based Framework for Science Journalism

## 一句话总结
本文构建了面向自动化科学新闻写作的多领域数据集SCITECHNEWS，并提出一种结合论文元数据（作者/机构）与话语修辞结构（如背景、方法、结论）的两阶段生成框架，通过在BART架构上学习内容规划策略，使生成结果在可读性、风格匹配度和信息丰富度上均优于多个零样本大模型基线。

## 研究问题与动机
- **科学文献体量爆炸**：2022年arXiv收录185,692篇论文，公共卫生领域（如COVID-19相关）PubMed已有约345,332篇，传统人工写作科学新闻难以应对信息过载。
- **现有数据集覆盖不足**：已有的Science Daily（不公开）和PLOS/eLife（仅限生物医学领域）难以支持跨学科的科学新闻生成任务；同时这些数据集多为"扁平"seq2seq建模，忽略了科学论文的元数据和话语结构信息。
- **科学摘要 vs 新闻写作的差异显著**：科学摘要偏好背景→方法→结果→结论的线性结构，而以ACM TechNews为代表的科技新闻更强调结论前置、作者/机构实体突出，且具更高的抽象性与风格化特征，两者在可读性、n-gram新颖度、命名实体分布上存在系统性差异。
- **大模型在此任务上表现不佳**：GPT-3.5、Alpaca等指令微调LLM在零样本条件下对风格迁移和信息压缩任务效果有限，提示需要更结构化的引导机制。

## 核心贡献（创新点）
- **多领域科学新闻数据集**：构建SCITECHNEWS（含2,431个对齐样本与18,933个非对齐样本），覆盖CS、物理、工程、生物医学等跨学科领域，弥补了现有数据集在跨域覆盖与公开可用性上的空白。
- **元数据融合框架**：将作者姓名与机构作为结构化的元数据token显式注入编码器输入，而非依赖隐式学习；与仅使用正文的Bart_SciT相比，Bart_meta在ROUGE-1上提升约1.6个百分点，验证了元数据在新闻写作风格中的重要价值。
- **基于话语结构的内容规划生成**：首次将科学论文各句子的discourse rhetoric role（背景/方法/结果/结论）编码进BART架构，在解码端以[s; y]拼接形式联合优化内容与计划，使模型不仅生成摘要还同步产出每句对应的修辞角色标签，实现结构可控的叙事规划。
- **实证对比大模型的局限性**：在自动指标与人工评测双维度上系统展示零样本LLM（Alpaca、FlanT5、GPT-3.5）在可读性（CLI=16.36 vs Bart_plan=13.55）与新闻风格（Sty=0.25/0.10/0.81 vs Bart_plan=0.98）上的明显劣势，提出结构化引导路径的必要性与有效性。

## 方法详解
- **任务形式化**：给定科学论文文本D及其元数据M（作者-机构对），模型首先生成内容计划s，再由s和D生成最终摘要y，即联合建模 p(s|D,M)·p(y|s,D,M)。
- **编码器输入构造**：输入序列为 D' = [m, m_0,...,m_|M|, t_0,x_0,...,t_N,x_N]，其中m为元数据起始符，m_i 为作者名与机构的拼接token，t_j 为第j个句子的话语修辞角色标签（由Li et al., 2021在PubMed-RCT上训练的discourse tagging模型标注），x_j 为对应句子。
- **解码策略（Bart_plan）**：采用BART架构，训练时以[s; y]作为目标序列，其中计划部分 s = [[PLAN], s_0,...,s_|y|]，每个 s_k 为摘要中第k个句子对应的话语角色标签；模型学习同时输出计划标签与对应文本，实现"规划-写作"一体化。
- **对比模型设定**：Bart_arx（在ARXIV数据集上预训练）；Bart_SciT（仅输入摘要+引言正文，无元数据与话语标签，直接生成摘要）；Bart_meta（加入元数据但不生成计划，仅生成摘要）；三者形成逐层递进的消融链条，验证各组件贡献。
- **可控生成扩展**：推理阶段允许用户指定任意修辞角色序列作为自定义内容计划，模型按顺序逐角色生成内容，实现对叙事结构的人机协同控制。

## 实验与结果
- **数据集统计**：测试集1,000个样本（全为开源论文），验证集1,431个样本（开源+机构访问）；主要来源包括Nature（320/188）、arXiv（231/263）、IEEE（14/126）、ACM（64/67）等。
- **自动指标（表4/5）**：最优模型Bart_plan在ROUGE-1达38.84（较Bart_SciT提升2.42）、ROUGE-L达33.50（+1.79）、BertScore达84.78（+0.66）；可读性CLI=13.55（优于GPT-3.5的16.36与FlanT5的16.36）；Faithfulness（QuestEval）38.16，显著高于Bart_SciT的36.54。
- **LLM基线劣势**：GPT-3.5 ROUGE-1仅35.67、CLI高达16.36、风格得分Sty仅0.81；Alpaca ROUGE-1仅21.24、Sty低至0.25；FlanT5同样表现较差，说明零样本指令遵循难以替代结构化的风格迁移训练。
- **人工评测（表6，Best-Worst Scaling）**：Bart_plan在Informativeness (+0.08)、Readability (+0.22)、Style (+0.30)、Usefulness (+0.02) 四个维度上均居首位，综合表现优于Bart_meta与GPT-3.5；Gold PR Summary在各维度得分最高（Inf=0.58, Sty=0.91），显示任务仍有提升空间。
- **事实错误分析（表7）**：Bart_plan共61处错误，以extrinsic实体错误（34%，多为虚构国家名称）和intrinsic实体混序为主；GPT-3.5共50处错误，以extrinsic名词短语错误（18%，常编造出版venue）为典型；两者均未出现world knowledge错误，但GPT-3.5在输入较长时出现"续写"模式导致prompt指令被忽略的现象。

## 相关工作脉络
- **Dangovski et al. (2021)**：构建Science Daily数据集，使用标准seq2seq模型生成科学新闻摘要，但该数据集因版权限制不公开；本文在公开可用性与跨域覆盖上作了关键补充。
- **Goldsack et al. (2022)**：提出PLOS与eLife两个生物医学期刊的lay summarization数据集，聚焦简化写作风格；本文强调press release新闻风格与lay summary在话语结构上的差异（结论前置vs背景先行）。
- **Cao et al. (2020)**：针对医学期刊做expertise style transfer与句子简化；本文将其思路扩展至多领域科学新闻，并引入结构化内容规划而非单纯句子级简化。
- **Vadapalli et al. (2018)**：用pointer-generator网络生成科学博客标题；本文工作进一步深入到完整的摘要生成与叙事结构控制。
- **Dernoncourt & Lee (2017)**：PubMed-RCT discourse tagging数据集及Li et al. (2021)的标注模型为本文话语角色建模提供了技术基础，本文的创新在于将其引入自动生成管线。
- **Narayan et al. (2021)**：提出entity prompt-based planning用于摘要生成；本文采用话语修辞角色而非实体作为规划单位，在科学新闻场景更具可解释性与人机协同价值。

## 局限性与未来方向
- **语言限制**：数据集与模型仅支持英语，未来需扩展至多语言科学新闻场景。
- **数据集规模偏小**：对齐样本仅2,431个，跨领域覆盖（如工程、物理）仍不够均衡，可能制约模型泛化能力。
- **元数据利用有限**：当前仅纳入作者名与机构，未充分利用发表时间、期刊/会议名称、DOI等辅助信息；虽然数据已完整公开供后续研究使用，但现有框架尚未挖掘其潜力。
- **大模型prompt工程尚待优化**：GPT-3.5等LLM在零样本下的表现提示可能需要更有针对性的prompt设计与in-context学习策略，以激发其风格迁移与压缩能力。

## 研究启发与可借鉴点
- **结构化话语标签可替代entity prompt作为规划信号**：在科学文档场景中，修辞角色（背景/方法/结论）比实体名更具叙事逻辑引导价值，可迁移至其他需要叙事控制的生成任务（如技术报告、专利摘要）。
- **元数据作为结构token显式注入编码器是低成本高回报策略**：仅需在输入序列头部添加[m, author_1-affil_1, ...]即可显著提升新闻风格生成质量，该设计可推广至法律、医疗等领域的结构化文档生成。
- **Best-Worst Scaling用于多任务评测具有统计稳健性**：本文选用BWS而非简单打分制，并结合ANOVA+Tukey-HSD检验维度间显著性，为科学计算领域的系统评测提供了方法论参考。
- **QuestEval在高技术域事实一致性评估中需谨慎解读**：实验发现gold PR摘要的QuestEval分数反而偏低，提示依赖预训练模型的事实性度量在跨风格域迁移中存在偏差，需结合人工或领域知识验证。
- **内容规划阶段支持用户自定义标签序列是人机协同写作的关键接口**：本框架在推理时允许人类指定修辞角色序列，这一设计为交互式写作助手（如journalist copilot）提供了可落地的交互范式。

## 关键术语表
- **SCITECHNEWS**：本文构建的多领域科学新闻数据集，包含科学论文与对应ACM TechNews新闻摘要的对齐与非对齐样本。
- **Discourse Rhetoric Role**：科学文本中句子所承担的修辞功能角色，如background（背景）、methods（方法）、results（结果）、conclusions（结论）等。
- **Content Plan**：解码过程中先生成的每句摘要对应话语角色标签序列，用于引导后续文本生成结构。
- **QuestEval**：基于问答形式的摘要事实一致性评估指标，通过提问-回答链路验证生成内容与源文档的一致性。
- **Best-Worst Scaling (BWS)**：一种心理学与NLP评测方法，要求 annotator 从多选项中同时选择"最好"与"最差"，相比Likert量表具有更高区分度与统计稳健性。
- **Sty Score（风格得分）**：本文训练的二分类风格识别器输出的概率均值，用于量化生成文本符合科技新闻风格而非科学摘要风格的程度。
- **Extractivity / Abstractivity**：分别指摘要从原文直接复制片段的程度与 paraphrase/fusion 新造内容的程度，是衡量生成策略的重要维度。
- **ALPACA / FlanT5 / GPT-3.5-Turbo**：本文用于zero-shot对比的三个主流指令微调大语言模型。

## 可复现要素
- **数据集**：SCITECHNEWS代码与数据已公开于 https://github.com/ronaldahmed/scitechnews；测试集（1,000个对齐样本）已直接发布，验证集提供下载指引。
- **代码**：完整代码开源，使用HuggingFace transformers库。
- **模型架构**：基于BART-LARGE预训练checkpoint微调。
- **训练超参**：2×NVIDIA A100 80GB GPU；Adam优化器，learning rate=1e-6，batch size=128，最大5,000步；LLM实验使用单张A100 40GB。
- **推理超参**：FlanT5-Large max_length=256，beam size=5，temperature=0.9，top_k=100，early stopping；Alpaca使用默认参数。
