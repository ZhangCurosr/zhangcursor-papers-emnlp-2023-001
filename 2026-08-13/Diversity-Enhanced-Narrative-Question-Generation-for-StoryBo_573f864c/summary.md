---
title: "Diversity-Enhanced-Narrative-Question-Generation-for-StoryBo"
source: https://aclanthology.org/2023.emnlp-main.31.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:38:29"
field: "自然语言生成与问答"
keywords: ["question generation", "diversity", "narrative QA", "recursive generation", "fairytaleQA", "answerability evaluation"]
innovations: ["提出L_MQS损失通过参考问题相似度约束驱动多样化生成", "递归生成框架将历史问题作为输入持续扩展问题集合", "基于SQuAD2.0微调的可答性评估模型实现三分类质量筛选"]
benchmarks: ["FairytaleQA", "TellMeWhy", "SQuAD1.1"]
---

# 论文速读：Diversity-Enhanced-Narrative-Question-Generation-for-StoryBooks

## 一句话总结
本文提出 **mQG** 模型，通过引入**最大问题相似度损失（L_MQS）**和**递归生成框架**，在无答案条件下从故事书中自动生成多样化且可回答的多个叙事问题，并在 FairytaleQA、TellMeWhy、SQuAD1.1 三个数据集上验证了其有效性。

## 研究问题与动机
- 现有 QG 研究多依赖语义相似度与金标准问题做自动评估，**忽视问题多样性**这一重要维度；
- 已有方法（如 QAG、EQG）生成的问题偏重显性（explicit）问题，隐含推理类问题（implicit）生成不足；
- 教育场景中，让儿童自问自评叙事问题是培养阅读理解的有效方式，**系统自动生成多样化问题**具有明确应用价值；
- 如何在**多样性与语义正确性**之间取得平衡是关键挑战。

## 核心贡献（创新点）
- **提出 mQG 模型**：基于 BART，通过 L_MQS 损失学习"同类问题表征相似"，并以递归方式持续生成差异化问题。
- **最大问题相似度损失 L_MQS**：利用同上下文真实问题作为参考，最大化参考问题与生成问题句级表示的相似度，从而在语义正确范围内驱动多样性。
- **递归生成框架**：将上一轮生成的问题重新作为历史输入，逐轮扩展问题集合，避免一次性生成多题时的冗余。
- **答案可答性评估模型**：基于 SQuAD2.0 微调的 DeBERTa-base，可将生成问题分类为 explicit / implicit / unanswerable，为自动质量评估提供依据。

## 方法详解
- **基础架构**：以 BART-large（406M 参数）为骨干，编码器同时接收问题类型（QT）、上下文（C）和参考问题（ground-truth 问题），三者以 `[SEP]` 分隔拼接；解码器自回归生成目标问题。
- **L_MQS 损失**：对参考问题集合 $\{Q_1, \dots, Q_m\}$ 和生成目标问题句级表示 $TQ$（mean pooling），计算：
  $$\mathcal{L}_{MQS} = \frac{1}{m}\sum_{i=1}^{m}\max(0,\,1 - s(Q_i, TQ))$$
  其中 $s(\cdot,\cdot)$ 为余弦相似度；该损失鼓励所有参考问题都与生成问题语义接近，从而保证生成问题在正确语义范围内多样化。
- **总体训练目标**：$\mathcal{L} = \mathcal{L}_{CE} + \beta \cdot \mathcal{L}_{MQS}$，其中 $\beta$ 最优取 0.4。
- **递归生成**：首轮以 QT + C 生成第一问；后续每一轮将上一轮生成问题与原始输入拼接，重复解码，直至达到设定数量（实验中每类问题生成 4 问，共 28 问/节）。
- **答案可答性分类模型**：在 DeBERTa-base 上预测答案起止位置，并引入特殊 token `[IMP]` 标记隐含问题；按阈值规则分为 explicit / implicit / unanswerable。

## 实验与结果
- **数据集**：FairytaleQA（故事书叙事 QA，含 explicit/implicit 标签，80/10/10 split 重采样）；零样本迁移至 TellMeWhy 与 SQuAD1.1。
- **基线**：E2E（BART-large 端到端）、CB（BART/GPT2 + contrastive framework）、QAG（Yao et al., 2022）、EQG（Zhao et al., 2022）。
- **主要指标**：Rouge-L F1、BERTScore F1、BLEURT、Self-BLEU（越低越多样）；人工多样性与质量评估。
- **FairytaleQA 结果**：mQG 在所有自动指标上最强，Rouge-L F1 = **58.90**（+3.13 vs QAG top10）、BERTScore = **0.9394**、BLEURT = **0.5698**、Self-BLEU = **0.6389**；每节平均可回答问题 **23.08** 个，显著领先。
- **TellMeWhy 零样本**：mQG 可回答问题 2.10/节（vs EQG 0.63、QAG 0.45），Rouge-L F1 = **56.17**，Self-BLEU = **0.3191**（多样性最高）。
- **SQuAD1.1 零样本**：mQG 可回答问题 20.15/节（vs QAG 14.40），BLEURT = **0.5508**；Rouge-L 略低于 QAG（45.38 vs 46.75），归因于 QAG 使用答案-aware 生成对显性问题有利。
- **消融**：去掉 L_MQS 或参考问题均导致 Self-BLEU 显著升高（多样性下降）与可回答问题数减少。

## 相关工作脉络
- **Su et al. (2022) CB**：对比学习 + contrastive search 解码；定位差异——CB 训练成本高且针对开放文本生成，mQG 面向叙事 QA 并引入参考问题驱动的相似度约束。
- **Yao et al. (2022) QAG**：启发式先生成答案再生成问题；定位差异——QAG 强于显性问题但隐含问题生成极少（约 5%），mQG 无需答案输入仍能生成均衡的显/隐性问题。
- **Zhao et al. (2022) EQG**：先生成语义银摘要再转问题；定位差异——EQG 依赖规则型银摘要构造，生成题数受限且易重复，mQG 通过递归框架直接生成多样化问题。
- **Zhu et al. (2018) Self-BLEU**：用于 NLG 多样性评估；本文沿用该指标衡量同上下文中多问题间的词级重叠度。
- **Rajpurkar et al. (2018) SQuAD2.0**：引入不可答题；本文借此构建三分类可答性评估模型，用于过滤生成结果。

## 局限性与未来方向
- 递归生成存在**质量传递风险**：若早期问题生成质量差，可能污染后续轮次输出。
- 生成数量受**最大 token 长度**约束，可能无法充分覆盖一个段落的所有语义维度。
- 可答性评估模型存在**误分类风险**（将不可答判定为可答），尤其在边界案例上。
- 未来可探索：（1）质量感知截止策略，动态停止低质量递归；（2）跨领域大规模零样本适配；（3）结合人工反馈做交互式多样性优化。

## 研究启发与可借鉴点
- **参考问题驱动的相似度约束**：用已知高质量样本的表示作为目标锚点，可在不引入额外标签的情况下同时提升生成质量与多样性，该思路可迁移到摘要、对话等其他生成任务。
- **递归生成框架**：将历史输出递归喂回模型，以"自我对比"方式避免重复，可作为通用的多样化解码策略。
- **可答性评估模型**：用 SQuAD 类问答模型扩展为三分类器，为生成质量提供可微的自动筛选信号，适合需要"正确性保障"的生成流程。
- **$\beta$ 敏感性分析**：实验中系统扫描 $\beta \in [0, 5]$ 并报告 Rouge-L 与 Self-BLEU 的权衡曲线，可作为后续工作的标准消融维度。

## 关键术语表
- **mQG**：multi-Question Generation，本文提出的多问题生成模型，基于 BART 并引入 L_MQS 与递归框架。
- **L_MQS（Maximum Question Similarity Loss）**：最大问题相似度损失，通过余弦相似度约束使生成问题与参考问题在语义空间邻近。
- **Recursive Generation Framework**：递归生成框架，将上一轮生成问题重新拼接输入，逐轮扩展问题集合。
- **Self-BLEU**：用同一批次生成句子互相作为参考计算 BLEU，越低表示多样性越高。
- **Answerability Evaluation Model**：基于 SQuAD2.0 微调的分类模型，将问题判为 explicit / implicit / unanswerable。
- **Explicit / Implicit Question**：显性问题（答案可直接从上下文提取）与隐含问题（需推理得出）。
- **FairytaleQA**：面向 4-14 岁儿童的故事书叙事 QA 数据集，含 8,548 条训练问答对。
- **TellMeWhy**：包含自由形式 why 问题的叙事理解数据集，约 28.82% 为隐含问题。

## 可复现要素
- **数据集**：FairytaleQA、TellMeWhy、SQuAD1.1、SQuAD2.0 均已公开；FairytaleQA 用于训练的 QA 对已清洗（移除跨段问题）。
- **代码/权重**：论文未提供开源代码链接，模型基于开源 BART-large / DeBERTa-base 微调。
- **关键超参**：learning rate = 5e-6；batch size = 8；epoch = 15；$\beta = 0.4$；beam search（beam size = 5）为主解码策略；评估模型 lr = 5e-6、batch = 16、epoch = 8。
- **硬件与时间**：RTX A6000 GPU，主模型训练约 3 小时，评估模型约 1 小时。
