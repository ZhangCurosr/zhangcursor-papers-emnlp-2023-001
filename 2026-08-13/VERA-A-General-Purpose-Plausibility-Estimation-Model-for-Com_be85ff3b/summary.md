---
title: "VERA-A-General-Purpose-Plausibility-Estimation-Model-for-Com"
source: https://aclanthology.org/2023.emnlp-main.81.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:35:11"
field: "常识推理与语言模型验证"
keywords: ["commonsense verification", "plausibility estimation", "contrastive learning", "language model calibration", "knowledge filtering"]
innovations: ["三目标联合训练（二元分类+多类别+监督对比）提升细粒度常识区分与泛化", "两阶段训练（KB 大规模噪声数据→QA 高质量数据）实现广度与精度兼顾", "LM 增强虚假语句构造与后验温度校准相结合提供可校准的连续可塑性分数"]
benchmarks: ["CommonsenseQA", "PIQA", "SIQA", "Winogrande", "HellaSwag", "StrategyQA", "CREAK", "SKD_anno", "I2D2_anno"]
---

# 论文速读：VERA-A-General-Purpose-Plausibility-Estimation-Model-for-Com

## 一句话总结
VERA 是一个通用常识语句可塑性估计模型，通过将 19 个 QA 数据集和 2 个常识知识库中的 700 万语句自动转换并微调 T5/LLaMA 骨干网络，实现对自包含宣言式语句的常识正确性打分；在验证格式下求解常识问题时，VERA-T5（5B）在已见基准上达到 85.51% 准确率，超越 Flan-T5 6 个百分点，并在未见基准、过滤 LLM 生成知识及检测 ChatGPT 常识错误等应用中均取得 SOTA 效果。

## 研究问题与动机
- 当前大语言模型（LM）在各类任务上表现优异，但仍会产出包含 trivial 常识错误的文本（如 ChatGPT 关于"大理石密度小于水银却会下沉"的错误推论），缺乏可靠的常识错误检测工具。
- 既有事实验证（fact verification）方法依赖从语料库中检索证据，而常识判断往往需要隐式、模糊的内部知识推理，难以获取显式证据。
- 已有常识验证器（SKD Critic、I2D2 Critic、Entailer 等）训练数据规模小、领域特定，泛化能力有限；基于 prompt 的大模型方案因无法获取原始 token logits 而难以精确排序。
- 需要一个"开箱即用"、仅依靠模型参数中存储的常识知识、无需检索组件、且输出校准良好的通用常识语句验证器。

## 核心贡献（创新点）
- **大规模多源常识语句训练数据构建**：将 19 个 QA 数据集与 2 个常识知识库（Atomic2020、GenericsKB）自动转换为 700 万条正确/错误声明（含 statement group 结构），覆盖科学、物理、社会、数量与定性常识等多领域；与先前小范围、单领域标注数据相比，数据来源更广、规模更大。
- **三目标联合训练框架**：同时使用二元分类损失（$\mathcal{L}_{\text{bin}}$）、多类别损失（$\mathcal{L}_{\text{mc}}$）与监督对比损失（$\mathcal{L}_{\text{ctr}}$），其中多类别损失利用同组语句间的相对排序增强对表面相似但正误不同的语句的区分能力，对比损失提升对输入变化的鲁棒性与跨数据集泛化；这与仅依赖单一交叉熵的既有验证器形成本质区别。
- **两阶段训练策略**：先用大规模但噪声较多的 KB 数据（Stage A）进行预训练，再用高质量 QA 数据（Stage B）微调；实验证明该顺序优于单阶段或混合训练，体现了"先广覆盖、后精调"的数据质量分层思想。
- **LM 增强虚假语句（LM-augmented falsehoods）**：用小 LM 对多选题采样生成低概率错误选项作为训练负样本，缓解人工标注数据的 annotation pattern 过拟合，显著提升对未见基准的泛化；不同于传统的人工构造或规则扰动方式。
- **后验温度校准（post hoc temperature scaling）**：在推理阶段对 logit 除以温度 $T$ 使输出概率校准，ECE 控制在 ≤3%（已见/未见基准），且不改变相对排序从而影响任务准确率；为常识验证器的置信度输出提供了可操作的校准方案。

## 方法详解
- **模型架构**：以 T5-v1.1-XXL（编码器约 5B 参数，称 VERA-T5）或 LLaMA-7B（称 VERA-LLaMA）为骨干。对输入语句 $x$，取最后 EOS token 的隐藏状态 $\mathbf{h}=f_{\text{LM}}(x)$，经线性层得到 logit $z=f_{\text{linear}}(\mathbf{h})$，再经 sigmoid 映射为 $s=\sigma(z)\in[0,1]$；阈值 0.5 对应 $z=0$ 用于二值预测。
- **语句组（statement group）**：来自同一多选题或 KB 条目的语句构成一组，多选型每组有且仅有一个正确语句；布尔型每组仅一条。训练时将同组语句置于同一 batch 以便计算组内对比与多类别损失。
- **三元损失组合**：$\mathcal{L}=\alpha\mathcal{L}_{\text{bin}}+\beta\mathcal{L}_{\text{mc}}+\gamma\mathcal{L}_{\text{ctr}}$，实验设定 $\alpha=1.0,\beta=1.0,\gamma=0.1$。
  - $\mathcal{L}_{\text{bin}}$：标准 BCE，按同标签语句数归一化以缓解组内正负不平衡。
  - $\mathcal{L}_{\text{mc}}$：对每组内唯一正确语句的 logit 做 softmax 交叉熵，迫使模型在同组相似语句中选出正确项；不适用于布尔单语句组。
  - $\mathcal{L}_{\text{ctr}}$：监督对比损失，在表示空间 $\mathbf{h}$ 上拉近同标签语句、推远异标签语句，温度 $\tau=0.05$。
- **两阶段训练**：Stage A 使用 1.6M 语句组（600 万语句）的 KB 数据，Stage B 使用 200K 语句组（40 万语句）的 QA 数据；每阶段 50k 步，Adam 优化，T5 学习率 $1\times10^{-5}$、LLaMA $2\times10^{-6}$，语句截断至 128 token，每组最多 4 条。
- **校准**：在验证集上搜索温度 $T$ 使 ECE 最小，推理时计算 $\tilde{z}=z/T,\ s=\sigma(\tilde{z})$；校准不改变排序，故准确率、AUROC 等指标不受影响。

## 实验与结果
- **评测协议**：分为已见（seen，训练所用 16 个多选+1 个布尔基准）与未见（unseen type 1 含 WSC/COPA/NumerSense/PROST/SpatialCS/Rainier；type 2 含 SWAG/HellaSwag/CODAH/SCT/αNLI/StrategyQA/CREAK）；多选/平衡布尔报 accuracy，非平衡布尔报 AUROC/AP，校准报 ECE。
- **问题求解**：VERA-T5（5B）在已见基准 accuracy 85.51%、AUROC 较 Flan-T5（11B）提升 9%；未见 type 1 accuracy 81.65%（+4%）、AUROC +5%；未见 type 2 accuracy 83.37%（+4%）、AUROC +6%；ECE ≤3%。VERA-T5 全面优于 VERA-LLaMA（7B，82.99%/75.51%/82.56%）。
- **过滤 LLM 生成常识知识**：在 SKD_anno、I2D2_anno 上 AUROC/AP 均优于全部基线；在 Generated Knowledge Prompting 流水线中，结合 UnifiedQA-large，VERA 过滤使 GPT-3（davinci）知识的有用性提升 46%、Rainier-large 提升 233%。
- **检测 ChatGPT 真实错误**：收集 27 条互联网 anecdote 改写为 54 条语句，VERA 在检测 ChatGPT 常识错误上 precision 91%、recall 74%、F1 82%；典型例中 marble-mercury 密度错误获 0.04 分（判错），修正后获 0.96 分（判对）。
- **消融**：移除对比损失主要损害未见基准；移除 Stage A 全局下降；LM 增强虚假语句对未见提升显著；多类别损失对多选有益，二元损失对布尔关键。缩放实验显示 5B 尚未饱和，更大模型有望更好。

## 相关工作脉络
- **SKD Critic（West et al., 2021）**：基于 RoBERTa-large 在 8k GPT-3 生成知识上微调的分类器，规模小、领域受限；VERA 使用 7M 多源语句且无需微调即可 zero-shot 使用。
- **I2D2 Critic（Bhagavatula et al., 2022）**：在 12k I2D2 生成语句上训练的 critic，对 I2D2_anno 有优势但不泛化；VERA 在三损失与两阶段训练下跨基准更稳定。
- **Entailer（Tafjord et al., 2022）**：多角度的科学假设验证模型，需生成前提链；VERA 无需检索/推理链，直接输出标量可塑性分数。
- **UnifiedQA-v2（Khashabi et al., 2022）**：通用 QA 模型，通过 yes/no  logits 改造为验证器；验证格式比 QA 格式在多选上低约 1.5%，但优势在于可直接评判独立声明与提供置信度。
- **GPT-3.5/ChatGPT/GPT-4**：通过 prompt 复用为大模型验证基线；因 API 不提供 logits 导致性能被低估，且缺乏可校准的连续分数；VERA 提供端到端可微、可校准的评分。
- **Flan-T5（Chung et al., 2022）**：指令微调的强基线，部分未见基准存在数据污染；VERA 在干净评估下仍实现绝对提升，体现专项常识验证的必要性。

## 局限性与未来方向
- 仅针对单句自包含声明，无法处理多句、长文本或含上下文/可废止条件的常识语句。
- 对输入句法变换（如 paraphrase、negation）鲁棒性不足。
- 未训练/评估道德常识数据，道德判断能力未知；训练数据可能含偏见或毒性。
- 缺乏范围门控（scope guard），对超出常识范围的输入仍会给出预测。
- 未来方向：扩展到多句/长形式验证、上下文依赖常识、加入范围检测器、结合生成式 QA 模型的比较优势进行互补融合。

## 研究启发与可借鉴点
- **三损失联合训练的可迁移设计**：二元分类+组内多类别+监督对比的组合可有效提升细粒度二分类任务的区分度与泛化，适用于任何存在"相似表面形式但正误不同"样本对的任务。
- **两阶段"粗到精"训练范式**：先用大规模噪声数据建立广泛常识覆盖，再用高质量数据精调，可作为大模型下游适配的数据调度启发。
- **LM 增强负样本构造**：用自身或小型 LM 采样低概率错误答案作为训练负例，是一种低成本、可扩展的数据增强手段，值得在其他推理/验证任务中尝试。
- **后验温度校准对验证器的价值**：校准后的连续分数可直接用于阈值筛选（如知识过滤），且不影响排序指标，为需要可解释置信度的应用提供实用方案。
- **验证格式与 QA 格式的互补性分析**：本文系统比较了 verification vs. QA 两种 problem-solving 格式的性能与适用场景，为后续研究选择交互范式提供了实证依据。

## 关键术语表
**Statement group**：由同一 QA 问题或 KB 条目派生的一组声明，多选组含恰好一个正确项，布尔组仅一条。
**Plausibility score**：模型输出的 [0,1] 连续分数，1.0 表示完全确信正确，0.0 表示完全确信错误，阈值 0.5 对应二值判定。
**Supervised contrastive loss**：在表示空间中拉近同标签样本、推远异标签样本的对比损失，提升模型对输入扰动的鲁棒性。
**Post hoc temperature scaling**：推理时对 logit 除以温度 $T$ 以校准输出概率，不改变样本间相对排序。
**LM-augmented falsehoods**：用小 LM 对多选题采样生成低概率选项作为人工标注之外的错误训练样本。
**ECE（Expected Calibration Error）**：衡量模型预测置信度与实际准确率之间偏差的校准指标， bins 数 $M=10$。
**Seen / Unseen benchmarks**：已见基准指训练数据中包含的测试集；未见 type 1 任务类型相近，type 2 任务性质差异较大（如含上下文故事或多实体推理）。
**Generated Knowledge Prompting**：先用知识生成模型产生相关常识语句，再由 QA 模型基于这些语句作答的流水线；VERA 可用于过滤生成语句。

## 可复现要素
- **数据集**：所有 21 个来源（Atomic2020、GenericsKB 及 19 个 QA 数据集）均为公开数据；转换代码与语句组构建方法在附录 A.2 详细说明。
- **代码/权重**：论文未明确声明代码与模型权重开源链接（URL 指向 ACL Anthology PDF）。
- **关键超参**：最大 token 数 128，每 batch 64 个语句组、最多 256 条语句、每组最多 4 条；训练步数 50k；T5 学习率 $1\times10^{-5}$、LLaMA $2\times10^{-6}$；损失权重 $\alpha=1.0,\beta=1.0,\gamma=0.1$；对比温度 $\tau=0.05$；校准温度 $T$ 在验证集上通过最小化 ECE 选取。
