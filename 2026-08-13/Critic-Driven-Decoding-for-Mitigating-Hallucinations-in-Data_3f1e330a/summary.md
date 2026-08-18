---
title: "Critic-Driven-Decoding-for-Mitigating-Hallucinations-in-Data"
source: https://aclanthology.org/2023.emnlp-main.172.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:11:24"
field: "知识 grounded 自然语言生成"
keywords: ["幻觉缓解", "数据到文本生成", "评审器驱动解码", "语言模型解码", "事实一致性", "合成负样本"]
innovations: ["提出将文本评审分类器实时融入自回归解码的贝叶斯联合框架，无需修改底层LM架构", "设计了五种基于已有数据合成负样本的训练策略，实现评审器零额外数据训练", "引入前缀长度感知的线性warmup策略和top-k token裁剪以平衡效率与效果"]
benchmarks: ["WebNLG", "OpenDialKG"]
---

# 论文速读：Critic-Driven-Decoding-for-Mitigating-Hallucinations-in-Data-to-text-Generation

## 一句话总结
本文提出了一种**评审器驱动解码（Critic-Driven Decoding）**方法，通过联合条件语言模型的生成概率与一个专用文本评审分类器的判定分数，在自回归解码过程中实时引导生成，从而在不修改底层模型架构、不收集额外数据的前提下，有效缓解数据到文本生成中的幻觉问题。

## 研究问题与动机
1. **幻觉问题严重**：神经数据到文本生成中，模型常输出与输入数据不一致的内容（幻觉），严重影响输出质量与可靠性。
2. **已有方法局限**：现有去幻觉方法通常需要修改模型架构或收集额外训练数据，难以直接应用于已训练好的现成模型。
3. **判别模型的潜力未被充分利用**：主流幻觉评估指标（如NLI-based metric、BLEURT）本身基于文本分类器，说明分类器具备识别数据-文本不一致性的能力，但以往仅用于**事后评估**或**候选重排序**，未用于**解码过程中的实时引导**。
4. **需要轻量、即插即用的方案**：提出一种无需额外数据、可与任意基于词概率的解码算法兼容的去幻觉机制。

## 核心贡献（创新点）
1. **提出了首个将文本评审分类器融入自回归解码过程的框架**：通过贝叶斯公式推导，将评审器输出 $P(c|y_{\leq i}, x)$ 与语言模型概率 $P(y_i|y_{\leq i-1}, x)$ 结合，实现实时解码引导，区别于仅在事后选择候选结果的方法。
2. **无需修改底层LM架构或训练流程**：评审器与LM条件独立，可单独训练，任何基于词概率的解码算法（贪婪/beam search）均可直接接入，与需要重写训练目标（如unlikelihood training）或添加控制码的方法形成本质区别。
3. **仅需合成负样本训练评审器，无需额外标注数据**：提出五种合成负样本策略（随机替换token/句子、基于vanilla/fine-tuned LM采样等），从已有数据到文本数据集自动生成，解决了评审器训练数据的获取难题。
4. **设计了实用的计算加速策略**：仅在top-$k$（$k=5$）高概率token上运行评审器，并引入基于前缀长度的线性warmup策略（前5个token逐步增大λ），兼顾效率与效果。
5. **在WebNLG和OpenDialKG两个基准上验证有效性**：不仅提升了幻觉相关指标（NLI、BLEURT），且在人工评估中证实了输出质量的改善，代码已公开。

## 方法详解
**概率推导**：在标准自回归语言模型 $P(y|x) = \prod_i P(y_i|y_{\leq i-1}, x)$ 的基础上引入评审变量 $c$（$c=1$表示文本与数据一致，$c=0$为不一致），应用贝叶斯变换得到：

$$P(y_i|y_{\leq i-1}, x, c) \propto P(c|y_{\leq i}, x) \cdot P(y_i|y_{\leq i-1}, x)$$

即下一步token的概率由语言模型概率和评审器判断概率相乘得到。

**实际解码公式**（对数空间，引入权重λ调节评审器影响力）：

$$\ln P(y_i|y_{\leq i-1}, x, c) \propto \lambda \ln P(c|y_{\leq i}, x) + \ln P(y_i|y_{\leq i-1}, x)$$

- $\lambda$ 超参：WebNLG设为0.25，OpenDialKG设为1.0
- 引入**线性warmup**：前5个token的$\lambda_i = \min(\frac{i}{5}, 1) \cdot \lambda$，因为短前缀时评审器准确率较低（图1显示prefix长度约5时准确率趋于稳定）
- 计算优化：仅对LM输出的top-$k$（$k=5$）概率最高的token运行评审器

**评审器架构与训练**：
- 骨干：XLM-RoBERTa-base + 分类头（SELU激活全连接层 + sigmoid输出层）
- 输入：数据表示 $x$ 与当前生成前缀 $y_{\leq i}$，以分隔token拼接
- 损失：二元交叉熵（binary cross-entropy）
- 正样本：从训练数据参考文本中提取所有前缀 $(x, y_1), (x, y_{\leq 2}), \ldots, (x, y_{\leq n})$
- 负样本（五种策略）：
  1. **base**：将最后一个token替换为随机token（来自同数据的其他参考或随机句）
  2. **base with full sentences**：用随机句替换参考文本中某句，再从第一个偏离处生成所有前缀
  3. **vanilla LM**：从无条件LM的top-5候选token中随机选一个替换
  4. **fine-tuned LM**：从数据条件LM的top-5候选token中随机选一个替换
  5. **fine-tuned LM with full sentences**：用条件LM生成整句描述，再从中提取偏离后的前缀作为负样本

## 实验与结果
**实验设置**：
- 数据集：**WebNLG**（RDF三元组→文本，测试集约46%样本为out-of-domain）和 **OpenDialKG**（对话 utterances 作为RDF三元组的文本化）
- 基线模型：在WebNLG上fine-tuned的 **BART-base** 编码器-解码器
- 评审器：基于 **XLM-RoBERTa-base** 的五种变体
- 解码：默认贪婪解码，补充beam search（beam size=5）实验
- 评估指标：BLEU、METEOR、BERTScore（文本质量）；NLI-based metric、BLEURT（幻觉检测）
- 硬件：NVIDIA Quadro P5000 16GB，BART以8-bit模式运行（bitsandbytes）

**WebNLG自动评估结果**（Table 2）：
| 方法 | BLEU | NLI (all) | BLEURT (all) |
|---|---|---|---|
| Baseline | 45.09 | 0.841 | 0.128 |
| **Critic (base)** ★ | **45.48** | **0.855 (+1.4pp)** | **0.155 (+2.7pp*)** |
| Critic (base w/full sent.) | 44.90 | 0.868* (+2.7pp) | 0.153* (+2.5pp) |

- 幻觉指标（NLI、BLEURT）提升最高达**2.5个百分点绝对值**，且达到统计显著性（α=0.05）
- 文本质量指标（BLEU、METEOR、BERTScore）基本持平或略有提升
- **out-of-domain**泛化：两种critic变体在ood子集上均持续提升NLI和BLEURT
- 使用fine-tuned LM训练的评审器（变体4、5）未能提升NLI，表明有效评审器可与NLG系统分离训练

**OpenDialKG自动评估结果**（Table 3）：
- Critic (base)：NLI从0.748→**0.796（+4.8pp）**，BLEURT从-0.933→**-0.905（+2.8pp）**
- Critic (base w/full sentences)：在标准文本质量指标上得分最高，同时幻觉指标有所改善

**人工评估**（100样本，5名标注员，Table 5）：
- Critic (base) **平均排名优于baseline 0.23位**，复杂样本（≥3个triples）提升达**0.33位**，极复杂样本（≥5个triples）提升达**0.53位**
- **重大幻觉率从0.40降至0.30（-10个百分点绝对值）**，复杂样本中降幅达15.3%~20%
- 同时在省略（omissions）、重复（repetitions）、不流畅（disfluencies）等维度均有改善

**修改幅度分析**（Table 4）：
- 30%~70%的输出未被修改，平均每句仅增减约2~5个词，说明评审器仅在必要时做微调

**Beam search补充实验**（Table 7，Appendix F）：
- 在beam size=5的更强基线上，Critic (base) 达到NLI=**0.886**、BLEURT=**0.202**，进一步提升

## 相关工作脉络
1. **Filippova (2020) Controlled hallucinations**：通过特殊控制码控制幻觉，但需重新训练带控制码的LM；本文方法无需修改LM架构，仅改动解码过程。
2. **Rashkin et al. (2021)**：结合控制码与多次重采样选优；需要生成多个完整文本并额外计算资源，本文只需单遍解码+轻量评审器。
3. **Cao et al. (2020) Generate & Refine**：两步生成-修正流程，需要训练额外的修正LM且需双遍解码；本文单次解码即完成。
4. **Chen et al. (2021) Contrast Candidate Selection**：使用文本分类器从对比候选中选择最佳输出，但仅在**解码后**选择，而非解码过程中实时引导。
5. **FUDGE (Yang & Klein, 2021)**：形式上与本文相似（贝叶斯公式引入判别器），但目标是控制文本书面语体程度，而非缓解幻觉。
6. **Welleck et al. (2020) Unlikelihood Training**：通过降低不合理token概率抑制重复生成，需要修改训练目标并训练新模型；本文评审器训练与LM解耦，且目标直指幻觉。
7. **Class-conditional LMs (Cohn-Gordon et al., 2018; Dathathri et al., 2020)**：本文方法的理论灵感来源，利用贝叶斯规则引入额外条件，但本文将其创新性地应用于幻觉检测场景。

## 局限性与未来方向
1. **幻觉无法完全消除**：所有设置下仍存在一定比例的幻觉，受限于base LM自身能力及每个解码步骤仅考虑有限候选token（k=5）。
2. **计算开销增加**：评审器在每步解码时运行，导致解码速度下降；未来可探索更高效的评审器架构或更大的k值权衡。
3. **对噪声数据的鲁棒性存疑**：作者明确指出，若底层训练数据噪声过多，评审器效果可能下降。
4. **需进一步验证**：当前仅在WebNLG和OpenDialKG两个数据集上验证，需扩展到更多数据到文本及知识 grounded 生成任务。
5. **λ超参需按数据集调整**：WebNLG用0.25、OpenDialKG用1.0，缺乏统一的自动调参策略。

## 研究启发与可借鉴点
1. **"生成+判别"联合解码框架的可迁移性**：将贝叶斯公式应用于实时解码引导的思路可迁移到其他生成任务（如摘要、对话生成）中，用于嵌入事实一致性约束或风格控制。
2. **合成负样本策略的通用设计**：五种负样本生成方式（尤其是基于LM采样构造"看似合理但实际错误"的负例）为训练判别模型提供了低成本范式，可推广至其他需要监督信号但标注稀缺的场景。
3. **前缀长度感知的warmup策略**：发现短前缀时判别器不可靠、需渐进增强的现象，这一设计思路（对早期步骤降低外部信号权重）可推广至任何在线引导生成的场景。
4. **评审器与生成器解耦训练**：证明了外部判别器可与主模型完全独立训练，避免了联合优化带来的不稳定问题，为模块化NLP系统设计提供了实践参考。
5. **与团队方向的结合机会**：本方法可直接与知识图谱增强生成、事实一致性校验等方向结合；亦可将评审器替换为更先进的NLI模型（如DeBERTa-xnli）以进一步提升判别精度。

## 关键术语表
**Hallucination（幻觉）**：生成的文本中包含输入数据未支持或与之矛盾的信息，是数据到文本生成的核心质量问题。
**Critic-Driven Decoding（评审器驱动解码）**：在自回归解码每一步，将语言模型概率与文本评审分类器输出相乘来调整next-token概率分布的解码策略。
**Data-to-text Generation（数据到文本生成）**：将结构化数据（如RDF三元组、知识图谱片段）转换为自然语言文本的生成任务。
**NLI-based Metric（基于自然语言推理的评估指标）**：利用NLI模型判断生成文本与输入数据之间是否为"蕴含"关系，以量化事实一致性。
**BLEURT**：基于BERT预训练模型的上下文感知文本评估指标，比传统BLEU更好地关联人工评分，对幻觉敏感。
**Conditional Language Model（条件语言模型）**：给定输入数据x，自回归地生成文本y的概率模型 $P(y|x)$。
**Unlikelihood Training（非似然训练）**：通过在训练目标中加入对不合理token的负对数概率惩罚，来抑制模型生成重复或无意义内容的训练方法。
**FUDGE（Future Discriminator-Guided Decoding）**：利用贝叶斯公式结合未来判别器控制文本风格（如正式程度）的解码方法，与本文方法形式相似但目标不同。

## 可复现要素
- **数据集**：WebNLG（公开）、OpenDialKG（公开）
- **代码**：已公开（论文摘要提及，原文链接见 footnote 1）
- **基线模型**：BART-base（HuggingFace默认架构），在WebNLG上fine-tuned（AdamW, lr=2e-5, β=(0.9, 0.997), ε=1e-9, polynomial scheduler with 10% warmup, 20 epochs, patience=10, batch size=8, label smoothing=0.1）
- **评审器模型**：XLM-RoBERTa-base + 分类头（SELU激活全连接层 + sigmoid），AdamW优化，lr=1e-5，训练至验证损失收敛
- **关键超参**：k=5（参与评审的top token数），λ=0.25（WebNLG）/ λ=1（OpenDialKG），linear warmup前5个token
- **硬件**：NVIDIA Quadro P5000 16GB
- **加速**：BART以8-bit模式加载（bitsandbytes库）
