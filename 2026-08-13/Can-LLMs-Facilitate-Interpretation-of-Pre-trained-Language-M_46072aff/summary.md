---
title: "Can-LLMs-Facilitate-Interpretation-of-Pre-trained-Language-M"
source: https://aclanthology.org/2023.emnlp-main.196.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:08:27"
---

# 论文速读：Can-LLMs-Facilitate-Interpretation-of-Pre-trained-Language-M

## 一句话总结
本文提出将 ChatGPT 作为零样本自动标注器，对预训练语言模型（pLM）通过层次聚类发现的潜在概念进行大规模语义标注，突破传统人工标注与预定义词表的规模瓶颈，实现细粒度、可扩展的模型解释性分析，并开源了包含 39,000 个概念的 Transformers Concept Net (TCN) 数据集。

## 研究问题与动机
- **预定义概念限制解释范围**：现有 pLM 解释性研究多依赖预定义的 linguistics ontology（如词性 POS、命名实体 NER），只能覆盖极泛化的语言类别，无法揭示模型实际学到的细粒度或领域特定概念。
- **人工标注缺乏可扩展性**：human-in-the-loop 方法（如 Geva et al., 2021 每层仅 100 keys，Dalvi et al., 2022 最终仅 269 概念）虽能提供精准标签，但成本高、规模受限，难以系统性地扫描整个潜空间。
- **LLM 具备自动化标注潜力**：GPT 系列模型历经海量文本训练，具备强大的语义理解与归纳能力，但尚未被系统探索用于标注 pLM 内部表征空间中自然涌现的隐式概念。
- **验证 LLM 标注器能否赋能下游解释性方法**：需实证检验 ChatGPT 标注在精度、语义丰富度上是否优于人工基线，并证明其能为探针分类器（probing）与神经元分析（neuron analysis）提供高质量监督信号。

## 核心贡献（创新点）
1. **首次系统性验证 LLM 作为 pLM 潜概念自动标注器**：与仅依赖预定义标签或有限人工标注的工作本质不同，本文利用 ChatGPT 零样本提示实现全量潜概念自动标注，将标注规模从数百级跃升至数万级。
2. **构建并开源大规模可解释性基准 TCN**：发布包含 39,000 个概念标注的 Transformers Concept Net，覆盖 BERT、RoBERTa、XLNet、ALBERT、XLM-R 五种主流架构，为社区提供开箱即用的解释性研究资源。
3. **展示 GPT 标注对细粒度解释性方法的赋能效应**：通过探针分类器与神经元分析两项下游任务，证明自动标注不仅能驱动传统语言学类别探测，还可揭示性别、宗教、地域、副词子类等超细粒度概念的表征分布与神经元定位。

## 方法详解
- **概念发现（Concept Discovery）**：给定预训练模型 $M$ 与输入数据集 $\mathbb{D}$，对每一层 $l$ 进行前向传播获取上下文向量序列 $\mathbf{z}^l$；采用 Ward 最小方差准则的凝聚层次聚类（Agglomerative Hierarchical Clustering），以平方欧氏距离为相异度度量，将词汇逐步合并为 $K$ 个簇（本文设 $K=600$），每个簇视为一个潜在编码概念。
- **概念标注（Concept Annotation）**：面向每个词簇，构造零样本自然语言提示通过 Azure OpenAI 调用 ChatGPT，要求模型输出简洁标签描述词间共性（如 “Terms related to ice hockey”）；设置 $temperature=0$ 保证输出确定性。
- **概念探针（Concept Probing）**：基于标注数据提取对应词的特征向量，训练二分类线性探测器，最小化交叉熵损失 $\mathcal{L}(\theta) = -\sum_i \log P_\theta(\mathbf{c}_i|\mathbf{z}_i)$，以探针准确率与选择性（selectivity）评估概念在表征空间中的可分性与纯度。
- **神经元分析（Concept Neurons）**：采用 Probeless 方法计算每个神经元 $n$ 对概念 $\mathcal{C}$ 的激活排名得分 $R(n, \mathcal{C}) = \mu(\mathcal{C}) - \mu(\hat{\mathcal{C}})$（$\mu$ 为激活均值，$\hat{\mathcal{C}}$ 为随机概念集），独立对每层神经元排序，从而定位捕获特定细粒度概念的神经元群。

## 实验与结果
- **数据集与配置**：使用 WMT News 2018 子集（250K 随机句，5M token），词频阈值设为 10，每词最多保留 10 次上下文，最终保留 25,000 个词型；在 5 种模型（BERT-base-cased, RoBERTa, XLNet, ALBERT, XLM-R）的 12 层上聚类生成 39K 概念。
- **人工评估结果**：3 位标注员对比 ChatGPT 与 BCN 标注；90.7% 的 ChatGPT 标签被评定为可接受（Fleiss' Kappa=0.71，substantial agreement），其中 75.1% 为精确标签（Kappa=0.34）；75.5% 情况下 ChatGPT 优于或等同于 BCN（Kappa=0.56，moderate agreement）。
- **错误分析**：10 例因内容策略拦截失败；8 例属句法/形态/词汇关系（默认提示偏语义，可通过定制提示改善）；11 例可通过补充上下文句子纠正；21/26 个被人类判定为“不可解释”的簇成功获得准确标签。
- **下游实验**：探针分类器在 ALBERT/XLNet 跨模型对比中验证了细粒度概念（如 “Spanish Male Names”, “Middle East Conflict”, “Islamic Terminology”）的可学习性（测试准确率普遍 >0.90）；神经元分析揭示 BERT-POS 模型中副词超概念的子概念（频率/方式/程度等 17 类）与其顶层神经元存在显著重叠（平均对齐率 0.36），名词/形容词/数字同理。
- **最强结果**：ChatGPT 标注覆盖率与语义丰富度全面超越 BCN 人工基准；39K 概念数据集支持跨架构、跨层的细粒度解释性研究。

## 相关工作脉络
- **Michael et al. (2020) / Dalvi et al. (2022)**：早期在 pLM 潜空间中使用聚类发现概念，但依赖预定义标签或人工注解（如 BCN，仅 269 个概念），本文将其扩展至全量自动化标注。
- **Geva et al. (2021) / Kádár et al. (2017)**：人工介入的神经元/key 级解释工作（每层限 100 keys），受限于规模，本文用 LLM 标注实现大规模神经元-概念映射。
- **Tenney et al. (2019) / Belinkov et al. (2017a)**：经典探针研究多聚焦预定义语言学属性，本文证明自动标注可驱动超越传统 Ontology 的细粒度探针（如按地域/宗教/性别细分的命名实体）。
- **Ding et al. (2022) / Wang et al. (2021) / Gilardi et al. (2023)**：LLM 作为数据标注器的已有评估工作，聚焦于文本分类/NER 等下游任务；本文的独特定位在于将 LLM 标注对象转向 *模型内部表征空间* 的隐式概念，而非外部监督信号。
- **Antverg & Belinkov (2022) Probeless**：神经元分析方法的基础，本文将其与大规模 GPT 标注概念结合，实现从粗粒度词性到细粒度语义子类的神经元定位。

## 局限性与未来方向
- **计算成本与延迟**：大规模聚类后逐个调用 LLM API 仍存在算力与响应延迟开销，若需引入上下文提升准确率，成本将进一步上升；但作者指出这是一次性成本，未来 LLM 运行效率有望提升。
- **内容安全策略拦截**：ChatGPT 内置的内容过滤器会阻止对涉及种族、仇恨言论、敏感偏见概念的标注，限制了其对模型偏见与不公平性的深度解释。
- **知识时效性局限**：LLM 训练数据存在截止期，对新闻摘要等反映动态现实世界状态的概念难以给出准确标签。
- **提示泛化依赖**：默认提示偏向语义相似性，对形态/句法/词汇类概念的识别需针对性优化提示模板（
