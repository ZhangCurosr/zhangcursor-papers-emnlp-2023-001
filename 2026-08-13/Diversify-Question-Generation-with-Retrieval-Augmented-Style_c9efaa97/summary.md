---
title: "Diversify-Question-Generation-with-Retrieval-Augmented-Style"
source: https://aclanthology.org/2023.emnlp-main.104.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:37:57"
field: "自然语言生成"
keywords: ["Question Generation", "Retrieval-Augmented Generation", "Reinforcement Learning", "Text Diversity", "Style Transfer"]
innovations: ["提出RAST框架，通过外部风格模板检索提升QG多样性", "设计RL双奖励（一致性+多样性）联合训练检索器与生成器", "引入多样性驱动聚类采样策略避免检索器退化到局部最优"]
benchmarks: ["SQuAD", "NewsQA"]
---

# 论文速读：Diversify-Question-Generation-with-Retrieval-Augmented-Style

## 一句话总结
本文提出 RAST（Retrieval-Augmented Style Transfer）框架，通过从外部语料检索问题风格模板并利用强化学习联合训练检索器与生成器，在保证问答一致性的前提下显著提升问题生成（QG）的表达多样性。

## 研究问题与动机
- 现有 QG 系统普遍存在**一致性不足**（生成问题与上下文/答案无关）和**多样性不足**（同一上下文-答案对只能生成单一表达）两大问题。
- 内部方法（内容规划、隐变量、解码策略）依赖语言模型黑盒，**可控性差**；外部方法（如模板改写）需要人工标注类型或改写样本，**数据依赖强**。
- 现有检索增强方法多关注一致性而非多样性，检索器通常单独训练，与生成模型协同优化不足。
- QG 的"一问多答"特性决定了需要探索**外部风格知识**来驱动表达多样性，同时不牺牲答案一致性。

## 核心贡献（创新点）
1. 提出 RAST 框架，从外部语料库检索问题风格模板并用于生成多样化问题——与依赖内部知识或手动模板的方法不同，无需人工标注问题类型。
2. 设计基于强化学习（RL）的两阶段训练方法，联合优化检索器和风格生成器——无需问题改写样本，直接以一致性奖励（QA模型）和多样性奖励（Jaccard相似度）联合训练。
3. 引入多样性驱动的采样策略（聚类抽样），避免检索器退化到局部最优——与单纯按检索分数采样的方法相比，显著提升风格探索能力。
4. 在 NewsQA 和两个 SQuAD 划分上实现最优多样性指标，且在一致性上与强基线相当——优于 CVAE、Composition 等方法。

## 方法详解
**整体架构**：pipeline 包含 Vanilla QG（初始问题生成）和 RAST 模型（风格检索+风格迁移生成）两部分。

**问题风格模板构建**：
- 以训练集问题为语料，使用 Spacy 识别实体（NER）和名词短语（NP），将上下文敏感信息替换为 `[MASK]`，保留停用词和疑问词。
- 计算 pairwise Jaccard 相似度，去除近重复模板，形成外部模板库 Z。

**风格检索模型（DPR）**：
- 使用两个 BERT 编码器分别编码查询模板 $z_0$ 和候选模板 $z$，计算内积相似度 $p_\phi(z|z_0) \propto \exp[q(z)^T q(z_0)]$。
- 支持 MIPS 实现亚线性时间检索。

**风格迁移模型（T5）**：
-  autoregressive 生成最终问题：$p_\theta(y|x,z) = \prod_{i=1}^{T} p_\theta(y_i|x,z,y_{1:i-1})$。
- 输入包含上下文 $x$ 和检索到的风格模板 $z$。

**两阶段训练**：
1. **监督学习初始化**：对模板 $\tilde{z}_0$ 进行主动噪声注入（随机替换 MASK、添加名词、删除 MASK、随机选模板），以 cross-entropy loss 训练风格迁移模型。
2. **强化学习联合优化**：
   - 策略网络：检索模型 $\phi$ + 生成模型 $\theta$
   - 损失函数：$L^{RL}(\theta,\phi) = -\mathbb{E}_{y^s,z^s}[r(y^s,z^s)]$
   - 使用 REINFORCE + SCST baseline 降低方差，并加入 KL 散度约束防止策略漂移。

**奖励设计**：
- 一致性奖励：$r_{cons}(y^s,z^s) = \exp(-L_{qa})$，其中 $L_{qa}$ 为 T5 生成的 QA 模型交叉熵损失。
- 多样性奖励：$r_{divs}(y^s,z^s) = \text{Jaccard}(z^s, y^s)$，衡量生成问题与模板的字面匹配程度。
- 总奖励：$r = r_{cons} + \lambda \cdot r_{divs}$，$\lambda \in [0,1]$ 为多样性系数。

**多样性驱动采样（Algorithm 1）**：
- 训练时：将检索到的模板按 Jaccard 相似度聚类，每簇随机采样一个，促进风格探索。
- 推理时：直接选取检索得分最高的模板。

## 实验与结果
**数据集**：SQuAD/1（Zhou et al. split）、SQuAD/2（Du et al. split）、NewsQA，数据统计见论文 Table 1。

**评估指标**：Top-1 BLEU（最佳单句一致性）、Oracle BLEU（top-N 最优一致性）、Pairwise BLEU（句间多样性，越低越多样）、Overall BLEU（综合指标）。

**基线方法**：Mixture-Decoder、Mixture-Selector、CVAE、Composition（内容规划类）；Nucleus-T5（解码采样类）。

**主要结果**（Table 2）：
- SQuAD/1：RAST 取得最优 Overall BLEU（9.14）、最优 Pairwise BLEU（48.91，最低即最多样）、Top-1 BLEU（19.25）仅次于 Nucleus-T5。
- SQuAD/2：RAST 取得最优 Top-1 BLEU（19.36）、最优 Pairwise BLEU（56.42）、Overall BLEU（7.75）。
- NewsQA：RAST 取得最优 Pairwise BLEU（23.16）和 Overall BLEU（2.82），Top-1 BLEU（11.02）略低于 Mixture-Selector（10.90†）。

**消融实验**（Table 3, SQuAD/1）：
- w/o e2e（冻结 DPR）：Overall BLEU 从 9.14 降至 9.02，验证联合训练价值。
- w/o cluster：Overall BLEU 从 9.14 降至 5.88，Pairwise BLEU 从 48.91 升至 61.06，验证聚类采样的重要性。
- w/ question（用完整问题而非模板检索）：多样性下降（Pairwise BLEU 54.09 vs 48.91），验证 MASK 模板的泛化优势。

**超参**（Table 6）：$\lambda=0.5$（SQuAD）、$\lambda=0.4$（NewsQA），$\beta=0.1$（SQuAD）、$\beta=0.05$（NewsQA）。

**人类评估**（Table 4, SQuAD/1）：RAST 一致性评分 3.36、多样性评分 2.36，均高于 Nucleus（3.00、1.78）。

## 相关工作脉络
- **RetGen (Zhang et al., 2022)**：同样使用 RL 联合训练检索与生成，但仅优化检索器，且奖励基于 ground truth 似然，而非本文的一致性+多样性双奖励。
- **CVAE / Mixture-Decoder / Composition**：基于内部知识的内容规划方法，通过隐变量或混合专家控制多样性，但可控性不如外部模板方法。
- **Nucleus Sampling (Holtzman et al., 2019)**：解码层面采样方法，虽能提升多样性但代价是 Top-1 BLEU 显著下降。
- **Retrieve-and-edit 方法**：如 Cai et al. (2019) 的 skeleton 方法，利用骨架构建多样性，但需要针对查询特化的模板。
- **Text Style Transfer**：本文风格转移无预定义风格标签，与有标签的风格转移方法（情感、正式度等）本质不同。
- **Paraphrase Generation**：现有改写数据集未关联上下文，本文通过 QA 一致性奖励允许基于不同上下文线索生成多样化问题。

## 局限性与未来方向
- QG 评估困难："一问多答"特性使得自动指标无法完全反映多样性质量，需要更多人类评估，但之前工作的代码/数据不可得。
- RL 训练耗时：采样轮次多、收敛慢，两阶段训练虽有缓解但仍需进一步优化效率。
- Transformer 最大上下文长度限制，难以处理超长文档。
- 作者指出未来可探索**少量 paraphrase 样本辅助训练**的高效训练方案。

## 研究启发与可借鉴点
- **RL 双奖励设计**：一致性奖励（QA模型）+ 多样性奖励（模板匹配）的组合思路可迁移至其他检索增强生成任务（如对话生成、文档摘要）。
- **多样性驱动的采样策略**：基于相似度聚类后随机采样，避免检索器退化到局部最优，可推广至其他需要探索多样性的 RL 生成任务。
- **主动噪声注入初始化**：监督阶段对模板施加多种噪声（替换、添加、删除、随机选择），提升模型对噪声模板的鲁棒性，类似思想可用于其他模板辅助生成任务。
- **检索-生成联合端到端训练**：与单独训练检索器不同，本文验证了联合训练的优越性，为 Retrieval-Augmented Generation 提供了新的训练范式。
- **风格模板构建方法**：通过 MASK 上下文敏感信息提取通用模板的策略，可迁移到其他需要风格多样性的文本生成任务（如改写、摘要）。

## 关键术语表
- **RAST (Retrieval-Augmented Style Transfer)**：本文提出的检索增强风格迁移框架，利用外部问题模板生成多样化问题。
- **Pairwise BLEU (P-BLEU)**：衡量 top-N 生成问题之间的句间相似度，值越低表示多样性越高。
- **SCST (Self-Critical Sequence Training)**：使用贪婪解码作为 baseline 的序列生成 RL 训练方法，降低梯度方差。
- **Reinforcement Learning (RL)**：本文用于联合训练检索器和生成器的方法，直接优化一致性+多样性奖励。
- **DPR (Dense Passage Retrieval)**：基于 BERT 的稠密检索模型，用于风格模板的语义匹配检索。
- **Consistency Reward**：基于生成式 QA 模型计算的奖励，衡量生成问题能否正确回答答案。
- **Diversity-driven Sampling**：训练时按聚类随机采样模板的策略，促进风格探索避免退化。
- **Vanilla QG**：基础问题生成模型，用于推理时生成初始问题模板 $z_0$。

## 可复现要素
- **数据集**：SQuAD/1、SQuAD/2、NewsQA（均为公开数据集）。
- **代码**：已开源，https://github.com/gouqi666/RAST。
- **模型权重**：论文未明确说明是否公开权重。
- **关键超参**：$\lambda=0.5$（SQuAD）、$\lambda=0.4$（NewsQA），$\beta=0.1$（SQuAD）、$\beta=0.05$（NewsQA），batch size 12/8/2，训练轮数 7 epochs（RL 阶段），学习率 generator $1e-6$、DPR $1e-7$。
