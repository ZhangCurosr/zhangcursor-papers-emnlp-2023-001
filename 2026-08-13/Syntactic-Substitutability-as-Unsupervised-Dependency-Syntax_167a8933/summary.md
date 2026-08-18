---
title: "Syntactic-Substitutability-as-Unsupervised-Dependency-Syntax"
source: https://aclanthology.org/2023.emnlp-main.144.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:32:21"
field: "计算句法学与 NLP 可解释性"
keywords: ["无监督依存解析", "句法可替换性", "LLM 可解释性", "注意力分布", "最大生成树", "BERT", "Universal Dependencies"]
innovations: ["提出 SSUD 方法，通过句法可替换性约束从 BERT 注意力分布中无监督诱导依存树", "证明使用替换句集合的平均注意力可显著提升解析精度，尤其在长距离主谓一致构式上召回率达 79.5%", "发现诱导树更贴合 SUD 标注体系而非 UD，揭示模型内部表征与标注形式的系统性差异"]
benchmarks: ["WSJ10", "EN-PUD", "Marvin & Linzen 主谓一致数据集"]
---

# 论文速读：Syntactic-Substitutability-as-Unsupervised-Dependency-Syntax

## 一句话总结
本文提出 SSUD（Syntactic Substitutability as Unsupervised Dependency Syntax），一种无需额外参数或监督信号、直接从 BERT 注意力分布中提取句法依存结构的无监督依存句法分析方法；核心思想是利用"句法可替换性"——同一句法类内的词可以互换而不影响句法合格性——对多组替换句子求平均注意力矩阵，再通过 MST 诱导依存树，显著提升了解析精度。

## 研究问题与动机
1. **能否不依赖监督训练从 LLM 内部机制中直接提取句法依存结构？** 已有方法（如 Hewitt & Manning, 2019 的 probe）需额外训练参数，难以区分模型固有知识与 probe 注入的外在知识。
2. **直接利用注意力分布进行句法解析会引入非句法信号**（如 co-reference、词汇相似度），Clark et al.（2019）已指出 BERT 注意力头不仅捕捉句法信息，还混杂其他关系，直接 MST 诱导会产生误边。
3. **现有无监督方法缺乏对"句法结构不变性"的显式约束**；作者认为句法关系的核心特征是"句法类内可互换性"，引入此约束可帮助从注意力中纯化句法信息。
4. **长距离主语-动词一致等复杂句法构式上，现有无监督方法召回率极低**，需要更鲁棒的句法信号提取机制。

## 核心贡献（创新点）
1. **首次将"句法可替换性"作为形式中立的形式化约束引入无监督依存解析**——不同于依赖特定标注框架（如 UD）的方法，本文从替换不变性出发定义句法关系，不预设任何具体句法理论。
2. **提出 SSUD 方法：通过 BERT MLM 生成替换句集合，对每个位置的字词在其句法类别内采样替换，再对多组替换句的注意力分布取平均后施加 MST，从而诱导依存树**——区别于直接对单句注意力矩阵做 MST（Raganato & Tiedemann, 2018; Htut et al., 2019），SSUD 利用替换不变性过滤非句法噪声。
3. **在长距离主语-动词一致构式上实现突破性提升**——SSUD 召回率达 79.5%，相比 Zhang & Hashimoto（2021）的 conditional MI 方法（8.9%）提升超过 70 个百分点。
4. **证明 SSUD 可迁移至不同解析框架**——将其应用于 Limisiewicz et al.（2020）的监督式 head selection 算法，UAS 从 52.8 提升至 54.5（+1.7），LAS 从 22.5 提升至 26.3（+3.8）。
5. **揭示诱导树更贴合 SUD（Surface-Syntactic UD）而非 UD 标注体系**——在 EN-PUD 上，SUD 评分提升幅度（+3.0）大于 UD（+2.1），说明模型内部表征可能更偏好以功能词为 head 的句法形式。

## 方法详解
1. **句法可替换性定义（Modified quasi-Kunze property）**：若 $w_{(i)} \to w_{(j)}$ 是句法关系，则存在一个词类集合 X，使得将 $w_{(j)}$ 替换为任意 $x \in X$ 均不影响句子的句法合格性。即对于满足 $Dep_{syn t}(s, i, j)=1$ 的词对，其依存关系在替换后保持不变。
2. **替换句生成**：使用 Stanza 的 Universal POS tagger 识别词性，对开放类词（形容词、名词、动词、副词）和部分封闭类词（介词、限定词）进行同范畴替换；利用 BERT 自身的 MLM 预测合理替换词，确保替换结果受 subcategorization 约束（如 "thought" 只能被同样要求小句补语的动词替换）。
3. **平均注意力矩阵构建**：对目标句 s 的每个位置 i，生成替换句集合 $S_{sub}(s, i, X)$，分别计算各替换句的注意力矩阵 $Att(s')$，然后按行取平均：$Att_{sub}(s)[i] = avg(\{Att(s')[i] \mid \forall s' \in S_{sub}(s, i, X)\})$。
4. **MST 树诱导**：在平均注意力矩阵 $Att_{sub}(s)$ 上应用 Prim 算法求最大生成树，得到无向非投影依存树：$t_s = MST(Att_{sub}(s))$。
5. **超参 k（替换数）**：每个位置用 k 个额外替换句参与平均；实验表明 UUAS 随 k 单调递增，但在使用监督 head selection 的 Setup 中 k=3 最优（过多替换对封闭类词产生负面影响）。

## 实验与结果
- **数据集**：WSJ10（Penn Treebank 第23节，389句，Stanford Dependencies 标注）、EN-PUD（1000句，UD 标注）、Marvin & Linzen（2018）长距离主谓一致数据集（1000句，两种模板）。
- **模型**：bert-base-uncased（110M 参数）、bert-large-uncased（336M 参数）。
- **Eval 指标**：UUAS（无标签无向附加分）、UAS/LAS。
- **Experiment 1.1**（层选择）：bert-base 最优层为 Layer 10（UUAS: WSJ10 55.7→56.8，Δ=+1.1；EN-PUD 44.3→44.7，Δ=+0.4）；bert-large 最优层为 Layer 17。
- **Experiment 1.2**（k 效应）：bert-base Layer 10，WSJ10 UUAS 从 55.7 单调升至 k=10 时的 57.6（+1.9）；EN-PUD 从 44.3 升至 46.4（+2.1）。bert-large Layer 17 同步提升（WSJ10: 56.1→57.2，EN-PUD: 45.5→47.0）。
- **Experiment 2**（长距离主谓一致）：object relative clause 召回率 79.5%（k=10），subject relative clause 召回率 63.0%（k=10）；对比 Zhang & Hashimoto（2021）conditional MI 方法仅 8.9% / 1.9%，提升分别达 +70.6pt 和 +61.1pt。
- **Experiment 3**（迁移到 Limisiewicz et al. 的有向解析框架）：UAS 从 52.8 提升至 54.5（+1.7），LAS 从 22.5 提升至 26.3（+3.8）；head selection 准确率在绝大多数依赖关系上均有提升（仅 aux/amod/nummod 三处轻微下降）。
- **SUD 对齐实验**：EN-PUD 上用 SUD 标注重评分，SSUD 获得 59.0 UUAS（UD 为 46.4），且 SUD 上的提升幅度（+3.0）大于 UD（+2.1）。

## 相关工作脉络
1. **Clark et al.（2019）**：发现 BERT 部分注意力头对应依存关系，但也捕获 co-reference 等非句法信息——本文用可替换性约束来解决此混杂问题。
2. **Hewitt & Manning（2019）**：训练 probe 从 BERT 表示中提取句法结构——本文方法完全无监督、无额外参数，避免 probe 注入外在知识的争议。
3. **Raganato & Tiedemann（2018）; Htut et al.（2019）**：直接对单句注意力矩阵做 MST 诱导依存树——本文的核心改进即引入替换句集合的平均注意力，而非仅用目标句单一分布。
4. **Zhang & Hashimoto（2021）**：基于 MLM 条件 MI 进行句法解析——在标准数据集上相当，但在长距离主谓一致构式上大幅落后于 SSUD（79.5% vs 8.9%）。
5. **Limisiewicz et al.（2020）**：监督式 head selection 提取有向依存树——本文将其作为基线框架，证明 SSUD 作为注意力预处理可带来额外增益，且本文本身完全无监督。
6. **Kim et al.（2020）**：基于注意力的成分解析方法——本文聚焦依存解析，但层选择结果与其发现一致（base 模型的 Layer 9-10 富含句法信息）。

## 局限性与未来方向
1. **仅使用英语数据**——英语语序相对严格、形态标记较少；在形态丰富的语言中替换一词可能影响其他词的形态标记，需更细粒度的句法类别定义。
2. **仅研究 BERT**——不同模型架构可能以不同方式存储句法信息；方法理论上是模型无关的，但在其他模型上的表现仍需验证。
3. **领域受限**——仅覆盖 WSJ 新闻语料和 EN-PUD（新闻+Wikipedia），未涉及其他文体或低资源场景。
4. **固定 k 值**——对封闭类词（如 det），过多的替换反而降低性能；动态调整 k 值可能是改进方向。
5. **未来方向**：扩展到多语言模型验证跨语言泛化；将句法结构诱导与实际模型行为（如主谓一致判断）关联评估；探索动态 k 策略和更细粒度的类别定义。

## 研究启发与可借鉴点
1. **"不变性约束"可用于从预训练模型表示中纯化特定类型的 linguistic 信息**——句法可替换性是一种强大的归纳偏置，可推广至其他 linguistic 属性（如语义角色、短语结构）的无监督提取。
2. **多上下文变体的注意力平均是一种简单而有效的信号增强策略**——无需额外训练，只需利用模型自身的 MLM 能力生成变体，即可显著提升下游结构诱导质量。
3. **评估框架的选择会显著影响方法比较结论**——本文发现 SSUD 诱导的树更贴合 SUD 而非 UD，提示在比较无监督解析方法时，应考虑标注体系本身的主观性偏差，不宜单一依赖某一标注框架的分数。
4. **可替换性生成策略（MLM + POS tagger）可作为通用的句法数据增强工具**——对低资源语言，可结合子树替换（ subtree swapping）思路扩展该方法。
5. **SSUD 可直接作为 attention 预处理器嵌入其他 parsing pipeline**——与 Limisiewicz et al.（2020）监督 head selection 的兼容性好，可作为通用模块提升各类基于注意力的解析框架性能。

## 关键术语表
- **Syntactic Substitutability（句法可替换性）**：句法关系的一个本质属性，指处于同一句法关系两端的词可以被同类词替换而不影响句子的句法合格性。
- **SSUD（Syntactic Substitutability as Unsupervised Dependency Syntax）**：本文提出的方法名，通过生成替换句集合并平均其注意力分布来诱导无监督依存句法结构。
- **Modified quasi-Kunze property（修正 quasi-Kunze 性质）**：本文采用的形式化定义，表述为：若 $w_{(i)} \to w_{(j)}$ 是句法关系，则存在词类 X，将 $w_{(j)}$ 替换为任意 $x \in X$ 不改变句子的句法合格性。
- **UUAS（Unlabeled Undirected Attachment Score）**：无标签无向附加分，衡量诱导树中与 gold 标注匹配的边缘比例，用于评估无监督依存解析质量。
- **SUD（Surface-Syntactic Universal Dependencies）**：与 UD 相似的标注体系，主要区别在于将功能词（如 complementizer、preposition）视为依赖关系的 head。
- **MST（Maximum Spanning Tree）**：最大生成树算法，用于从词间分数矩阵中诱导无向依存树结构。
- **Conditional MI（条件互信息）**：Zhang & Hashimoto（2021）提出的基于 MLM 目标的句法解析方法，通过计算词对的 conditioned mutual information 构建依存分数。
- **Head Selection（Head 选择）**：Limisiewicz et al.（2020）方法中的监督步骤，根据 gold 标注为每个 UD 关系选择最具信息量的注意力头。

## 可复现要素
- **数据集**：WSJ10（需 LDC 授权许可）、EN-PUD（公开，Parallel Universal Dependencies treebanks）、Marvin & Linzen（2018）主谓一致数据集。
- **代码/权重**：论文未明确声明开源仓库，仅提到 SSUD 实验可用 GPU 2GB / CPU 24GB 复现；使用了 transformers、Stanza、networkx 等公开包。
- **关键超参**：k（每位置替换句数量，实验取 1/3/5/10）、BERT 层数（base 用 Layer 10，large 用 Layer 17）、MST 算法（Prim）。
- **硬件**：RTX 8000 GPU（24GB），每次实验约 7-10 小时。
- **软件版本**：Stanza 1.4.0, networkx 2.2.4, numpy 1.22.4, transformers 4.19.2, torch 1.11.0。
