---
title: "Three-Stream-Based-Multi-level-Event-Contrastive-Learning-fo"
source: https://aclanthology.org/2023.emnlp-main.103.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:33:14"
field: "多模态信息抽取"
keywords: ["多模态事件提取", "光流特征", "对比学习", "文本-视频融合", "跨模态对齐"]
innovations: ["首次将光流特征引入文本-视频事件提取任务", "提出多级事件对比学习模块对齐运动特征与事件语义", "设计双查询文本模块实现细粒度跨模态交互"]
benchmarks: ["TVEE", "VM2E2"]
---

# 论文速读：Three-Stream-Based-Multi-level-Event-Contrastive-Learning-for-Text-Video-Event-Extraction

## 一句话总结
本文针对文本-视频事件提取（TVMEE）任务，首次将光流特征作为运动表示引入多模态融合，提出**TSEE**（Three Stream Multimodal Event Extraction）框架，通过多级事件对比学习（MECL）和双查询文本模块（DQT）实现文本、视频外观、光流三者的高效对齐与交互，在TVEE和VM2E2数据集上刷新了SOTA。

## 研究问题与动机
- **现有方法忽视视频运动信息**：已有TVMEE方法仅利用文本序列特征（TSF）和视频外观特征（VAF），忽略了视频中承载动作语义的运动表示。
- **RGB帧背景噪声干扰对比学习目标**：VAF包含复杂场景背景噪声，易误导基于对比学习的跨模态对齐；而光流特征仅刻画物体运动轨迹，对背景噪声鲁棒。
- **同类触发词具有相似运动模式**：作者观察到相同事件触发词（trigger）对应的视频运动轨迹高度相似，该特性可被用于约束特征空间对齐。
- **多模态交互不足**：现有方法在融合文本与视频特征时，缺乏细粒度的跨模态查询机制来强化语义关联。

## 核心贡献（创新点）
1. **首次将光流特征引入TVMEE任务**：利用预训练PWC网络提取OFF作为运动表示，与TSF、VAF构成三模态输入，本质区别于仅依赖RGB帧的现有方法。
2. **提出多级事件对比学习模块（MECL）**：在事件类型层和触发词层分别定义对比目标，对齐OFF↔触发词、触发词↔事件类型的嵌入空间，与单级对比学习形成显著差异。
3. **设计双查询文本模块（DQT）**：VAF和OFF分别作为Query检索TSF中的相关token，实现细粒度跨模态注意力交互，优于简单的拼接或加权融合策略。
4. **在两个基准数据集上刷新SOTA**：TVEE上Trigger F1达81.5%（+3.0% vs CoCoEE），VM2E2上Trigger F1达51.6%（+4.1% vs CoCoEE）。

## 方法详解
- **特征提取**：
  - TSF：预训练T5-base编码文本token序列，维度$d_t=768$。
  - VAF：I3D（Kinetics预训练）提取视频外观特征，维度$d_v=1024$。
  - OFF：PWC-Net（Sintel预训练）提取光流特征，维度$d_o=1024$。
  - VAF/OFF经线性投影统一至768维后与TSF相加聚合为视频级表示。

- **多级事件对比学习（MECL）**：
  - 仅保留单事件样本参与对比损失计算（过滤多事件样本避免OFF标签歧义）。
  - 事件类型层：以事件类型$e^i$为anchor，同类触发词$z^i$为正样本，其余为负样本：
    $$\mathcal{L}_{type} = -\sum_{i=1}^{B} \log \frac{\exp(x^i \cdot z^i / \tau)}{\sum_{z^l \in W_c \setminus z^i} \exp(x^i \cdot z^l / \tau)}$$
  - 触发词层：以触发词$z^i$为anchor，对应OFF $F_O^i$为正样本：
    $$\mathcal{L}_{trig} = -\sum_{i=1}^{B} \log \frac{\exp(z^i \cdot F_O^i / \tau)}{\sum_{F_O^u \in F_{O_c} \setminus F_O^i} \exp(z^i \cdot F_O^u / \tau)}$$
  - 总对比损失：$\mathcal{L}_{multi} = \mathcal{L}_{type} + \mathcal{L}_{trig}$，温度系数$\tau=0.3$。

- **双查询文本模块（DQT）**：
  - VAF和OFF分别投影为Query，TSF投影为Key/Value：
    $$A_v = \text{softmax}\left(\frac{F_V H_{q_1} H_{k_1}^\top F_T^\top}{\sqrt{d_t}}\right) F_T H_{v_1}$$
    $$A_o = \text{softmax}\left(\frac{F_O H_{q_2} H_{k_2}^\top F_T^\top}{\sqrt{d_t}}\right) F_T H_{v_2}$$
  - 聚合交叉模态注意力：$F_A = A_v \cdot F_T + A_o \cdot F_T$。

- **解码器**：CRF解码器联合预测触发词和论元。

## 实验与结果
- **数据集**：
  - **TVEE**：7598对文本-视频，8个上级事件类型+33个子类型，基于ACE2005扩展。
  - **VM2E2**：562对文本-视频，16个事件类型，每样本仅含单一事件。
- **评估指标**：Precision、Recall、F1（Trigger/Argument分开报告）。
- **主要结果**（Table 1）：
  - TVEE多模态Trigger F1：**81.5%**（vs JMMT 77.1%、CoCoEE 78.5%）。
  - VM2E2多模态Trigger F1：**51.6%**（vs CoCoEE 47.5%）。
  - TSEE在多数指标上超越所有基线，论证了OFF和MECL的有效性。
- **消融实验**（Table 2）：
  - 加入OFF：Trigger F1提升约2-3%。
  - 加入MECL：TVEE Trigger F1提升2.8%，VM2E2提升3.7%。
  - 加入DQT：VM2E2 Trigger Recall从50.6%升至53.5%，Argument Recall从25.3%升至27.4%。
- **可视化**：t-SNE显示MECL使同类OFF更紧凑聚类；DQT热力图验证了VAF关注外观相关token、OFF关注运动相关token。

## 相关工作脉络
- **CoCoEE (Wang et al., 2023)**：现有TVMEE最强基线，采用监督对比学习对齐VAF与事件类型，但未利用运动特征，且对比目标仅停留在单级。
- **JMMT (Chen et al., 2021)**：早期文本-视频联合事件提取工作，使用Transformer联合编码两模态，但未引入对比学习。
- **DEEPSTRUCT (Wang et al., 2022)**：纯文本事件提取SOTA，提出结构预训练，本文作为单模态基线对比。
- **Supervised Contrastive Learning (Khosla et al., 2020)**：MECL的理论基础，本文将其扩展至多级跨模态对齐场景。
- **光流估计（PWC-Net, Sun et al., 2018）**：特征提取 backbone，本文首次将其迁移至TVMEE领域。
- **定位差异**：本文与CoCoEE的核心区别在于引入OFF运动表征与多级对比目标，有效缓解背景噪声干扰并增强跨模态语义对齐。

## 局限性与未来方向
- **离线训练限制**：受GPU资源限制，VAF和OFF需预提取，无法端到端优化视频预训练模型。
- **封闭域假设**：数据集仅标注固定事件类型集，无法处理开放域事件提取。
- **未来方向**：作者计划探索大语言模型（LLM）在多模态特征融合中的应用，以提升TIMEE性能。

## 研究启发与可借鉴点
- **运动特征作为鲁棒视觉信号**：光流对背景噪声不敏感，在噪声敏感的多模态对齐任务中可作为高效辅助模态复用。
- **多级对比学习目标设计**：将对比学习从样本级扩展至"类型-触发词-运动"多级粒度，可迁移至其他跨模态理解任务（如视频-文本检索）。
- **双查询注意力机制**：用不同模态分别Query共享文本表示，可实现细粒度跨模态交互，适用于多模态信息抽取场景。
- **单事件样本筛选策略**：针对多事件样本的标签歧义问题，通过过滤样本简化对比学习训练，是处理复杂标注的有效启发。

## 关键术语表
- **TVMEE（Text-Video Multimodal Event Extraction）**：从文本-视频对中联合提取事件触发词和论元的多模态任务。
- **OFF（Optical Flow Features）**：通过光流网络提取的视频帧间像素运动向量，表征物体运动轨迹。
- **VAF（Video Appearance Features）**：基于RGB帧提取的视频外观特征，编码颜色、纹理、形状等静态视觉信息。
- **TSF（Text Sequence Features）**：预训练语言模型编码的文本token序列表示。
- **MECL（Multi-level Event Contrastive Learning）**：在事件类型层和触发词层分别定义对比损失，对齐跨模态特征嵌入空间。
- **DQT（Dual Querying Text）**：VAF和OFF分别作为Query检索TSF token的两路交叉注意力模块。
- **Supervised Contrastive Learning**：利用样本标签构造正负对，拉近同类样本、推远异类样本的对比学习目标函数。

## 可复现要素
- **数据集**：TVEE和VM2E2均为公开数据集（论文引用来源公开）。
- **代码/权重**：论文未明确声明开源，但使用了公开预训练模型（T5-base、I3D-Kinetics、PWC-Sintel）。
- **关键超参**：batch size=16，训练epoch=15，学习率=1e-5（Adam），对比学习温度τ=0.3，视频采样间隔16帧。
- **环境**：PyTorch + 2080 Ti GPU。
