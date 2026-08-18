---
title: "Temporal-Knowledge-Graph-Forecasting-Without-Knowledge-Using"
source: https://aclanthology.org/2023.emnlp-main.36.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:32:05"
field: "时序知识图谱预测"
keywords: ["Temporal Knowledge Graph", "In-Context Learning", "LLM Forecasting", "Zero-shot Prediction", "Symbolic Pattern Learning"]
innovations: ["首次系统性验证LLM零样本ICL可用于TKG预测且性能媲美监督SOTA", "发现LLM依赖符号模式而非语义先验，数字映射后性能仅降0.4%", "证明ICL能学习超越频率/近期偏置的复杂时序模式，超启发式基线10%-28%"]
benchmarks: ["WIKI", "YAGO", "ICEWS14", "ICEWS18", "ACLED-CD22"]
---

# 论文速读：Temporal-Knowledge-Graph-Forecasting-Without-Knowledge-Using-In-Context-Learning

## 一句话总结
本文探索将时序知识图谱（TKG）预测任务转化为大语言模型（LLM）的上下文学习（ICL）问题，仅需零样本提示即可实现与监督学习方法相当的预测性能；核心发现是LLM依赖符号模式而非语义知识，即使将实体/关系映射为纯数字也能维持相近表现。

## 研究问题与动机
1. **现有TKG预测方法的局限性**：主流监督方法（如RE-Net、TANGO、CyGNet等）依赖图神经网络或强化学习架构，需要大量历史数据训练，且不同数据集的最优模型各不相同，模型选择成本高。
2. **LLM的ICL能力是否适用于TKG预测**：尚未有研究系统探索大语言模型在不微调的情况下，能否通过上下文学习完成时序知识图谱的链接预测任务。
3. **语义先验知识的必要性存疑**：现有方法假设模型需要理解实体和关系的语义信息，但LLM是否真正依赖这些语义先验仍需实证检验。
4. **缺乏统一的预测范式**：TKG预测领域缺少一个无需定制架构、可直接复用于多数据集的统一框架。

## 核心贡献（创新点）
1. **首次将TKG预测系统性地转化为LLM的零样本ICL任务**：设计了三阶段流水线（历史建模→提示构建→概率解码），无需微调即可在多个基准上达到与监督SOTA相当的性能（在Median Hits@1上下±3.6%~1.5%范围内）。
2. **发现LLM不依赖语义先验知识，而是利用符号模式**：将实体/关系替换为随机数字索引后，性能仅下降0.4%，颠覆了"TKG预测需要语义理解"的直觉假设。
3. **验证LLM能学习超越频率/近期偏置的复杂模式**：ICL在ICEWS14/18上比最优启发式规则基线（frequency/recency）提升10%~28%的Hits@1，证明模型确实从历史上下文中学习到了不规则模式。
4. **系统性分析了ICL各设计维度的影响**：涵盖了历史建模策略（Entity/Pair、Unidirectional/Bidirectional）、提示类型（Index/Lexical）、时间戳必要性、模型规模与历史长度等6个关键变量的消融分析。

## 方法详解
**整体框架**：三阶段零样本ICL流水线，将TKG预测转化为LLM的next-token prediction任务。

1. **历史建模（History Modeling）**：
   - 给定查询 $q = (s, p, ?, t)$，从历史快照 $\mathcal{G}_{1:t-1}$ 中筛选相关事实构建上下文 $\mathcal{E}_q$
   - **Entity vs Pair**：Entity使用包含 $s$ 的所有历史事实；Pair仅使用同时包含 $s$ 和 $p$ 的事实
   - **Unidirectional vs Bidirectional**：Unidirectional要求 $s$（或 $(s,p)$）在历史事实中处于与查询相同的位置；Bidirectional允许出现在任意位置，并在反向时交换主客体并替换为逆关系
   - 历史长度固定为100（主要实验）

2. **提示构建（Prompt Construction）**：
   - **Index提示**：每个事实表示为 $f_t : [\mathcal{I}(f_s), \mathcal{I}(f_p), n_{f_o}.\mathcal{I}(f_o)]$，其中 $\mathcal{I}(\cdot)$ 为实体/关系的数字索引映射，$n_{f_o}$ 为递增的唯一数字标签（作为间接logit）
   - **Lexical提示**：使用实体/关系的原始文本名称替代数字索引
   - 查询 $q$ 表示为 $t : [\mathcal{I}(s), \mathcal{I}(p), ''$] 拼接至提示末尾

3. **响应解码与预测**：
   - 将提示输入LLM，获取首个生成token的概率分布
   - 从top-100概率token中筛选与数字标签 $n$ 对应的token，作为候选实体的分数
   - 按概率排序生成实体rank列表
   - **Single-Step**：测试时使用真实历史事实，每步预测后插入ground truth继续
   - **Multi-Step**：测试时不使用ground truth，用top-1预测插入历史继续，模拟真实场景

## 实验与结果
**数据集**（5个基准，含新构建的ACLED-CD22）：
- WIKI：12,554实体，24关系，539K训练事实，年度间隔
- YAGO：10,623实体，10关系，161K训练事实，年度间隔
- ICEWS14：6,869实体，230关系，74K训练事实，日间隔
- ICEWS18：23,033实体，256关系，373K训练事实，日间隔
- ACLED-CD22：243实体，6关系，1.8K训练事实，日间隔（新构建）

**评测指标**：Hits@k (k=1,3,10)，Time-aware filter设置

**主要结果**：
- **GPT-NeoX（20B参数）**在Single-Step设置下，Entity历史在YAGO上Hits@1=0.784、WIKI=0.694、ICEWS14=0.324、ICEWS18=0.192、ACLED-CD22=0.324，与监督SOTA相当
- 相对各数据集监督Median性能：在(-3.6%, +1.5%) Hits@1范围内波动
- **超越启发式规则**：在ICEWS14/18上，ICL比frequency/recency基线提升10%~28% Hits@1
- **零语义依赖**：Index与Lexical提示性能差异仅≈4e-3（平均），移除语义后性能仅降0.4%
- **时间戳必要**：移除时间戳后性能下降，打乱顺序后进一步恶化
- **Scaling law成立**：模型规模与历史长度均呈现正向scaling趋势
- **Instruction-tuned模型（gpt-3.5-turbo）**：Lexical提示（0.1858）优于Index提示（0.1615），与基础模型相反，说明指令微调模型能更好地利用语义先验

## 相关工作脉络
1. **TKG预测的监督学习方法**（RE-Net、RE-GCN、TANGO、xERTE、TimeTraveler、CyGNet、TLogic）：基于GNN/RL/逻辑规则，需大量训练数据和定制化架构；本文定位为"无需训练的零样本替代方案"。
2. **ICL机制研究**（Wei et al., 2023; Garg et al., 2022）：关注LLM如何通过输入-标签映射学习函数；本文将这一机制扩展到时序图预测任务，并发现符号模式足以支撑预测。
3. **语义先验对ICL的影响研究**（Min et al., 2022; Razeghi et al., 2022; Hahn & Goyal, 2023）：探讨LLM是否依赖预训练语义知识；本文通过Index/Lexical对比提供了新的实证证据——基础模型不依赖语义。
4. **TKG评估方法论**（Gastinger et al., 2022）：提出统一评估框架；本文在其基准上验证ICL的有效性，并指出部分数据集（如YAGO/WIKI）存在recency偏置过强的问题。
5. **事件预测与LLM结合**（Jin et al., 2021; Zou et al., 2022）：使用LLM分析非结构化文本进行预测；本文聚焦结构化TKG场景，探索纯符号模式利用能力。

## 局限性与未来方向
1. **模型规模受限**：受计算资源限制，仅在小规模开源模型（最高20B）上实验，更大模型的潜力未充分探索。
2. **tokenizer限制**：对于词汇表仅含个位数字token的模型（如LLaMA），性能显著低于同规模其他模型，方法适配性受限。
3. **仅限Inductive设置**：当前方法要求预测实体必须出现在历史中，无法处理Transductive场景下的 unseen entity 预测。
4. **未来方向**：探索Transductive extrapolation链接预测；研究微调对性能的进一步提升；深入分析ICL的其他潜在能力。

## 研究启发与可借鉴点
1. **"去语义化"实验设计**：将实体/关系映射为随机数字索引以检验模型是否依赖语义先验，这一思路可迁移至其他NLP/图学习任务中验证模型的真实学习能力。
2. **间接Logit解码策略**：使用递增数字标签作为间接logit来估计token概率，有效解决了多token实体名称的概率聚合问题，可复用至其他LLM输出解析场景。
3. **历史建模的Entity/Pair维度设计**：对比"仅实体相关"与"实体-关系对相关"的历史上下文，发现不同数据集最优策略不同，为后续研究提供了精细化的上下文选择思路。
4. **零样本TKG预测的新范式**：证明了无需训练、无需定制架构的ICL方法可与监督SOTA竞争，为低资源/快速部署场景提供了实用替代方案。
5. **多步预测的噪声累积分析**：Single-Step与Multi-Step的对比揭示了ICL在误差传播下的退化行为，为研究LLM在序列决策中的稳定性提供了参考。

## 关键术语表
**Temporal Knowledge Graph (TKG)**：在知识图谱中引入时间维度，用四元组$(s, p, o, t)$表示实体-关系-实体在特定时间的事实序列。
**In-Context Learning (ICL)**：大语言模型在不更新参数的情况下，通过在prompt中提供少量输入-输出示例来诱导模型执行特定任务的能力。
**Index Prompt**：将实体和关系映射为数字索引的提示格式，用于剥离语义信息检验模型是否依赖知识语义。
**Lexical Prompt**：使用实体和关系的原始文本名称的提示格式，保留完整语义信息。
**Single-Step Prediction**：在测试时提供 ground truth 历史事实的预测设置，每步预测后插入真实答案继续推理。
**Multi-Step Prediction**：在测试时不提供 ground truth，使用模型自身预测插入历史的更困难设置，模拟真实部署场景。
**Hits@k**：链接预测评估指标，表示正确答案排在前k个候选中的比例。
**Inductive Setting**：要求预测的实体必须在历史图中出现过的设置，与Transductive（允许预测未见实体）相对。

## 可复现要素
- **数据集**：WIKI、YAGO、ICEWS14、ICEWS18为标准公开基准；ACLED-CD22为新构建数据集，来源为ACLED项目公开数据
- **代码/权重**：论文未明确提供代码仓库链接；使用了GPT-NeoX（20B开源）、GPT-J（6B开源）、GPT-2系列（开源）及gpt-3.5-turbo（API访问）
- **关键超参**：历史长度=100；top-100 token筛选；Multi-Step中k=1（top-1预测插入历史）；使用GPT-2 byte-level BPE tokenizer
- **实现框架**：PyTorch + Huggingface Transformers
