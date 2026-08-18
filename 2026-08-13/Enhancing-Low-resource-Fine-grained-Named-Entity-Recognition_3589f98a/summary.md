---
title: "Enhancing-Low-resource-Fine-grained-Named-Entity-Recognition"
source: https://aclanthology.org/2023.emnlp-main.197.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:38:32"
field: "低资源命名实体识别"
keywords: ["细粒度命名实体识别", "低资源学习", "跨粒度映射", "不一致性过滤", "CoFiNER", "Few-NERD"]
innovations: ["提出F2C映射矩阵显式建模细粒度与粗粒度实体类型的层级关系以实现跨粒度联合训练", "设计不一致性过滤方法剔除跨数据集标注噪声提升训练质量"]
benchmarks: ["Few-NERD", "CoNLL'03", "OntoNotes"]
---

# 论文速读：Enhancing-Low-resource-Fine-grained-Named-Entity-Recognition

## 一句话总结
论文提出 CoFiNER 模型，通过显式建模细粒度（fine-grained）与粗粒度（coarse-grained）实体类型间的层级关系，利用大规模粗粒度数据集提升低资源细粒度 NER 的性能，并结合不一致性过滤方法剔除噪声标注。

## 研究问题与动机
- **细粒度 NER 面临严重的标注数据稀缺问题**：细粒度实体类型（如 Few-NERD 的 66 类）标注成本高昂，需要领域专家，导致训练数据量少。
- **现有 K-shot 学习方法存在性能早饱和现象**：当标注数量超过几十个时，基于 episode 的 few-shot 方法性能提升有限，需要切换到监督学习。
- **预训练微调策略未能显式利用层级关系**：prefinetuning（如 Muppet）仅利用粗粒度数据进行表示学习，没有利用细粒度类型与粗粒度类型之间通常是子类型关系的先验知识。
- **不同标注体系存在语义不一致性**：粗粒度数据集（如 OntoNotes）与细粒度数据集由不同标注者创建，存在标签不匹配（如"Microsoft"在财务文档中可能是 Company 或 Stock），直接混合训练会引入噪声并降低性能。

## 核心贡献（创新点）
- **提出 F2C（Fine-to-Coarse）映射矩阵**：显式建立细粒度实体类型到粗粒度实体类型的映射，使得单一模型可以直接同时利用多粒度数据集进行联合训练，与 prefinetuning 仅用于表征学习的本质区别在于实现了跨粒度标签空间的直接通信。
- **设计不一致性过滤方法**：利用训练好的细粒度模型预测粗粒度标签，过滤掉预测结果与原始标签不一致的粗粒度样本，避免噪声实体干扰模型训练；与简单数据拼接或预训练的本质区别在于主动识别并屏蔽跨数据集标注冲突。
- **提出交替联合训练策略**：在单个训练循环中交替使用粗粒度和细粒度数据批次，结合 F2C 映射和过滤后的粗粒度损失函数，实现端到端的协同优化，与两阶段预训练微调的本质区别在于一次性完成多粒度知识融合。

## 方法详解
**CoFiNER 训练流程分为四个步骤：**

1. **训练细粒度 NER 模型**：将细粒度数据集送入 PLM（如 RoBERTa）生成 token 级上下文表示 $\mathbf{H}$，通过分类头 $\mathbf{W}$ 和偏置 $\mathbf{b}$ 计算细粒度标签概率分布 $\mathbf{p}^{\mathcal{F}}_i = \text{softmax}(\mathbf{W}\mathbf{h}_i + \mathbf{b})$，以交叉熵损失 $L_{\mathcal{F}}$ 训练。

2. **生成 F2C 映射矩阵**：矩阵 $\mathbf{M} \in \mathbb{R}^{|E^{\mathcal{F}}| \times |E^{\mathcal{C}}|}$ 中元素 $\mathbf{M}_{\ell,s} = p(y^{\mathcal{C}}=s | y^{\mathcal{F}}=\ell)$ 表示给定细粒度标签 $\ell$ 时粗粒度标签 $s$ 的条件概率。通过训练粗粒度模型对细粒度数据集重新标注，统计共现矩阵 $\mathbf{C}$ 并采用 top-$k$ 滤波（$k=1$ 最优）后归一化得到。

3. **不一致性过滤**：冻结细粒度模型，用 $\tilde{y}^{\mathcal{C}}_i = \arg\max(\mathbf{p}^{\mathcal{F}}_i \cdot \mathbf{M})$ 预测粗粒度标签，若预测值与原始标签 $y^{\mathcal{C}}_i$ 一致则保留，否则标记为不一致并从后续训练中排除。

4. **联合训练**：交替使用粗粒度和细粒度数据批次。细粒度批次使用标准交叉熵损失；粗粒度批次仅对过滤后的一致样本计算损失 $L_{\mathcal{C}} = -\frac{1}{m}\sum_{i=1}^{m} \log \mathbf{p}^{\mathcal{C}}_i[y^{\mathcal{C}}_i] \cdot \mathbb{I}[y^{\mathcal{C}}_i = \tilde{y}^{\mathcal{C}}_i]$，其中 $\mathbf{p}^{\mathcal{C}}_i = \mathbf{p}^{\mathcal{F}}_i \cdot \mathbf{M}$。

## 实验与结果
- **数据集**：细粒度 Few-NERD（SUP）含 66 类实体；粗粒度 CoNLL'03（4 类）和 OntoNotes（18 类）。细粒度数据采用 K~(K+5)-shot 重采样（K=10, 20, 40, 80, 100）。
- **评估指标**：span-level F1。
- **最强结果**：在 100-shot 设置下，CoFiNER（RoBERTa-LARGE）取得 **57.18** F1，相比监督基线 RoBERTa-LARGE（54.17）提升 **+3.01**，相比 SOTA 监督方法 PL-Marker（53.06）提升 **+4.12**；在 40-shot 和 80-shot 下也分别以 56.41 和 56.85 的最佳成绩显著优于所有对比方法（除 10-shot 略低于 LSFS 外）。
- **消融结论**：移除不一致性过滤（w/o filtering）导致 F1 下降 2.05~7.27；单独使用任一粗粒度数据集仍显著优于纯细粒度训练（w/o coarse）；F2C 矩阵中 top-$k$ 取 $k=1$ 效果最佳。

## 相关工作脉络
- **LSFS（Ma et al., 2022）**：采用 prefinetuning 策略从 OntoNotes 学习先验知识再微调 Few-NERD，但仅利用粗粒度数据做表征学习，未显式建模粒度间层级关系；CoFiNER 在此基础上进一步通过 F2C 矩阵实现跨粒度联合训练。
- **PL-Marker（Ye et al., 2022）与 PIQN（Shen et al., 2022）**：监督学习的 SOTA 方法，但依赖全量细粒度标注，在低资源场景下性能受限；CoFiNER 通过引入外部粗粒度数据突破标注数量瓶颈。
- **Muppet（Aghajanyan et al., 2021）**：大规模多任务预训练工作，展示了 prefinetuning 的有效性，但未针对 NER 的粒度层级结构做专门设计；CoFiNER 将该思想适配到细粒度 NER 并解决跨数据集标注不一致问题。
- **Few-NERD 数据集（Ding et al., 2021）**：提出细粒度 NER  benchmark，包含 66 类实体及对应的粗粒度标签，为本研究提供实验基础。
- **episode-based few-shot NER**：如 Das et al. (2021)、Huang et al. (2022) 等基于 N-way K-shot 的方法，适用于类别数少的场景；本文指出当细粒度类别数（66）远大于粗粒度类别数时，此类方法不适用，转而利用现有粗粒度数据。

## 局限性与未来方向
- **不适用于嵌套 NER**：论文明确指出 token-level 的 F2C 映射和过滤机制难以直接推广到嵌套实体识别任务，因为嵌套结构涉及重叠 span，超出 token-level 假设。
- **依赖粗粒度模型的标注质量**：F2C 矩阵的构建需要先用粗粒度模型对细粒度数据重新标注，若粗粒度模型性能不佳会引入误差。
- **仅测试了两个粗粒度数据集**：虽然提到方法可扩展到多级层次结构，但实验仅验证了 CoNLL'03 和 OntoNotes 的组合，多数据集融合效果有待探索。

## 研究启发与可借鉴点
- **显式建模层级映射的通用思路**：F2C 映射矩阵的设计可迁移到其他存在粒度层次的信息提取任务（如事件提取、关系抽取），通过构建类型间映射实现多源数据联合训练。
- **不一致性过滤作为数据清洗策略**：利用目标域模型预测辅助域标签并过滤冲突样本的思想，可应用于跨领域自适应、弱监督学习等场景，提升辅助数据的信噪比。
- **K~(K+5)-shot 采样策略**：针对 NER 句子内多实体特性提出的柔性采样方案，比严格 K-shot 更贴近实际分布，可作为 few-shot NER 实验设置的参考。
- **非可学习映射矩阵的稳定性优势**：实验表明固定的 top-$k$ 共现矩阵比可学习矩阵效果更好，提示在跨粒度对齐任务中，基于统计的确定性映射可能比端到端学习更稳健。

## 关键术语表
- **Fine-grained NER**：细粒度命名实体识别，使用更细致实体类别（如 Few-NERD 的 66 类）进行实体分类的任务。
- **Coarse-grained NER**：粗粒度命名实体识别，使用较少通用实体类别（如 CoNLL'03 的 4 类：PER, ORG, LOC, MISC）的任务。
- **F2C 映射矩阵（Fine-to-Coarse Mapping Matrix）**：矩阵 $\mathbf{M}$，元素表示给定细粒度标签时粗粒度标签的条件概率，用于将细粒度预测投影到粗粒度空间。
- **不一致性过滤（Inconsistency Filtering）**：利用细粒度模型预测粗粒度标签，剔除预测值与原始标注不一致的样本，以过滤跨数据集标注噪声的方法。
- **K~(K+5)-shot**：每个实体类型至少包含 K 个、最多包含 K+5 个示例的采样设置，适应 NER 句子内多实体的特性。
- **Prefinetuning**：先在大规模辅助数据集上微调预训练模型，再在目标数据集上微调的策略，用于缓解数据稀缺问题。
- **Few-NERD**：包含 66 类细粒度实体和 18 类粗粒度实体的 few-shot NER 数据集 benchmark。

## 可复现要素
- **数据集**：Few-NERD（SUP）、CoNLL'03、OntoNotes 均为公开数据集；细粒度数据采用 K~(K+5)-shot 重采样。
- **代码**：已开源，地址为 https://github.com/sue991/CoFiNER。
- **关键超参**：PLM 使用 RoBERTa-LARGE；序列最大长度 256；AdamW 优化器，学习率 2e-5，batch size 16；dropout 0.1；F2C 矩阵构建中 top-$k$ 取 $k=1$；细粒度模型训练 30 epoch，粗粒度模型 50 epoch，过滤模型根据 shot 数变化（10/20-shot 为 150 epoch，40-shot 为 120 epoch，80/100-shot 为 50/30 epoch）；硬件为 NVIDIA RTX 3090 GPU。
