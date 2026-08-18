---
title: "Discourse-Structures-Guided-Fine-grained-Propaganda-Identifi"
source: https://aclanthology.org/2023.emnlp-main.23.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:37:48"
field: "细粒度宣传检测与话语结构分析"
keywords: ["propaganda detection", "discourse structure", "knowledge distillation", "fine-grained analysis", "PDTB", "news discourse role", "misinformation identification"]
innovations: ["首次将局部PDTB话语关系与全局新闻话语角色双重结构引入宣传识别任务", "设计响应式与特征关系式双重知识蒸馏机制迁移话语结构知识", "证明话语结构可显著提升细粒度宣传识别的精确率与召回率"]
benchmarks: ["NLP4IF-2019 propaganda dataset", "PDTB 2.0", "News Discourse dataset"]
---

# 论文速读：Discourse-Structures-Guided-Fine-grained-Propaganda-Identifi

## 一句话总结
论文提出利用局部PDTB式话语关系和全局新闻话语角色两种话语结构来识别政治新闻中的细粒度宣传内容，通过构建两个教师模型并结合知识蒸馏或特征拼接的方式，将话语结构知识迁移到句子级和token级宣传识别任务中，显著提升了精确率和召回率，达到当前最优性能。

## 研究问题与动机
- **细粒度宣传识别的必要性**：现有工作多集中在文章级检测，无法定位具体哪句话或哪个token包含宣传内容，缺乏可解释性；而宣传常通过论证策略嵌入文本，需要理解更大上下文。
- **局部话语关系的启发**：宣传内容更可能嵌入与邻近句子存在因果关系（contingency）或对比关系（comparison）的句子中，用于传递虚假推论或质疑可信度。
- **全局话语角色的启发**：表达主观评价（Evaluation）、推测未来期望（Expectation）或虚构历史背景（Historical Event）的句子比事实性描述（Main Event、Context）更易携带宣传内容。
- **两种话语结构的互补性**：局部关系捕捉句间逻辑，全局角色把握叙事功能，二者结合可更全面地理解宣传策略。

## 核心贡献（创新点）
- **提出双结构框架**：首次将局部PDTB式话语关系与全局新闻话语角色同时引入宣传识别任务，建立统一的细粒度宣传检测框架。
- **设计知识蒸馏方案**：提出响应式蒸馏损失（模仿教师预测概率）与特征关系式蒸馏损失（学习教师嵌入空间关系）相结合，两者互补增强。
- **实现feature concatenation基线**：将教师模型预测概率直接拼接为额外特征，提供简洁有效的对比方法。
- **验证话语结构有效性**：通过统计分析揭示宣传句子与特定话语关系（causal/comparison占比更高）和话语角色（Evaluation/Expectation占比更高）的强相关性。

## 方法详解
- **基础模型**：采用Longformer编码整篇新闻（最大长度4096），句子嵌入取自首token `<s>` 的隐藏状态，token嵌入为对应隐藏状态。
- **局部教师模型**（话语关系）：基于PDTB 2.0训练，输入相邻句对，预测四种关系（Comparison/Contingency/Temporal/Expansion）概率分布 $P_i^{local}$。
- **全局教师模型**（话语角色）：基于News Discourse数据集训练，采用actor-critic架构，预测八个话语角色（M1/M2/C1/C2/D1/D2/D3/D4）概率分布 $P_i^{global}$。
- **特征拼接模型**：将 $P_i^{local}$ 和 $P_i^{global}$ 拼接至句子/token嵌入后输入两层分类头，使用cross-entropy损失。
- **知识蒸馏模型**：构建propaganda学习层、学生话语关系层、学生话语角色层；总损失 = 主任务损失 + 响应式蒸馏损失（KL散度） + 特征关系式蒸馏损失（MSE空间矩阵）。
- **响应式蒸馏**：$Loss = \sum P^{teacher} \log(P^{teacher}/Q^{student})$，迫使学生学习教师预测分布。
- **特征关系式蒸馏**：计算教师与学生生成的句子嵌入余弦相似度矩阵，最小化MSE差异，传递空间结构知识。

## 实验与结果
- **数据集**：NLP4IF-2019 propaganda dataset（Da San Martino et al., 2019b），提供句子级和token级人工标注，train/dev/test划分固定。
- **评估指标**：宣传类的Precision、Recall、F1-score。
- **最强结果**（句子级）：Knowledge Distillation + both structures → Precision=61.22%, Recall=69.75%, F1=65.21%。
- **最强结果**（token级）：Knowledge Distillation + both structures → Precision=37.22%, Recall=46.86%, F1=41.48%。
- **提升幅度**：相比longformer基线，knowledge distillation使句子级recall提升约9.25%，F1提升约4.8%；优于此前NLP4IF-2019挑战赛冠军报告结果。
- **消融结论**：局部关系与全局角色均有效，联合使用效果最佳；两种蒸馏损失均贡献显著；四种话语关系中expansion贡献相对较小。

## 相关工作脉络
- **文章级宣传检测**（Rashkin et al., 2017; Barron-Cedeno et al., 2019）：仅判读整篇新闻，无法定位具体宣传片段。
- **细粒度宣传检测开山之作**（Da San Martino et al., 2019b, a）：引入句子级和token级标注，本文在其基础上利用话语结构提升性能。
- **多视图/多模态宣传检测**（Vijayaraghavan & Vosoughi, 2022; Dimitrov et al., 2021）：侧重特征融合，未利用话语结构知识。
- **媒体偏见检测**（Baly et al., 2020; Fan et al., 2019）：关注意识形态偏向，与宣传的任务定义和标注规范存在差异。
- **谣言/假新闻检测**（Pérez-Rosas et al., 2018; Wei et al., 2021）：侧重事实性验证，而宣传重在论证策略与说服意图。
- **PDTB与新闻话语结构研究**（Prasad et al., 2008; Choubey et al., 2020）：提供本文两个教师模型的预训练基础。

## 局限性与未来方向
- 仅针对宣传这一类 misinformation，对假新闻、阴谋论等其他类型推广效果未知（论文自述）。
- 实验仅在单一英语新闻数据集上验证，跨语言/跨平台泛化性待考察。
- 教师模型依赖预训练语料（PDTB 2.0、News Discourse），在低资源场景下可能受限。
- ChatGPT作为对比基线表现仍有较大差距，说明大模型在细粒度定位任务上仍需领域适配。

## 研究启发与可借鉴点
- **知识蒸馏用于结构知识迁移**：响应式+特征关系式双重蒸馏可有效将外部语言学知识融入下游任务，可迁移至其他需要上下文理解的任务。
- **话语结构作为先验信号**：PDTB关系和新闻话语角色可视为文本结构先验，对其他叙事分析任务（如立场检测、论证挖掘）有借鉴价值。
- **教师-学生双阶段范式**：先训练专用教师模型再蒸馏到学生，避免端到端训练中对大规模标注的依赖，适合标注数据稀缺场景。
- **统计分析验证假设**：论文先用统计验证话语结构与宣传的相关性再设计模型，这种"观察→验证→建模"的研究路径值得借鉴。
- **长文本建模基础**：采用Longformer处理整篇新闻，对长文档理解任务具有参考价值。

## 关键术语表
- **Propaganda（宣传）**：带有政治目的的误导性叙事，通过论证策略影响公众信念。
- **PDTB discourse relation**：Penn Discourse Treebank定义的四类句间话语关系（对比/因果/时间/扩展）。
- **News discourse role**：新闻叙事中句子的八种功能角色（主事件/后果/先前语境/当前语境/历史事件/轶事/评价/期望）。
- **Knowledge Distillation（知识蒸馏）**：将教师模型的知识迁移到学生模型的训练范式。
- **Response-based Distillation**：通过KL散度使学生学习教师预测概率分布的蒸馏方式。
- **Feature Relation-based Distillation**：通过余弦相似度矩阵匹配师生嵌入空间关系的蒸馏方式。
- **Fine-grained Propaganda Identification**：在句子级或token级定位宣传内容的任务。
- **Actor-Critic Model**：结合强化学习与模仿学习的训练框架，用于新闻话语角色预测。

## 可复现要素
- **数据集**：NLP4IF-2019 propaganda dataset（Da San Martino et al., 2019b），公开可用。
- **代码**：论文未提供开源声明（截至ACL Anthology版本），需联系作者获取。
- **教师模型权重**：论文未公开，需在PDTB 2.0和News Discourse数据集上自行训练。
- **关键超参**：最大序列长度4096，训练轮数6 epoch，AdamW优化器，weight decay=1e-2，线性学习率调度。
- **基础模型**：Longformer（Beltagy et al., 2020）。
