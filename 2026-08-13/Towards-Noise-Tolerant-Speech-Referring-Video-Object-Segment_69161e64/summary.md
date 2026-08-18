---
title: "Towards-Noise-Tolerant-Speech-Referring-Video-Object-Segment"
source: https://aclanthology.org/2023.emnlp-main.140.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:33:40"
field: "多模态视频理解"
keywords: ["Referring Video Object Segmentation", "Speech-Visual Grounding", "Noise-Tolerant Representation", "Cross-Modal Alignment", "Audio-Guided Video Understanding"]
innovations: ["提出 STBridge 框架实现冻结 R-VOS 模型的语音输入适配", "设计 NSA 与 SJS 模块分别实现噪声自适应语义提取和抖动抑制", "在 AVOS 基准上建立含噪语音参考分割新 SOTA"]
benchmarks: ["AVOS", "Ref-YoutubeVOS", "A2D-sentences", "JHMDB-sentences"]
---

# 论文速读：Towards Noise-Tolerant Speech-Referring Video Object Segmentation: Bridging Speech and Text

## 一句话总结
本文提出 STBridge，通过噪声感知语义调整（NSA）与语义抖动抑制（SJS）两个模块，使冻结的文本条件 R-VOS 模型能够直接接受含噪语音作为参考查询，在 AVOS 基准上显著优于基于 ASR 转写的基线方法。

## 研究问题与动机
- **语音比文本更脆弱**：真实场景中背景噪声、口音等会导致语音信息失真或缺失，而现有 R-VOS 方法仅在干净文本上验证，缺乏对语音输入的研究。
- **直接 ASR 转写不够用**：将语音经 ASR 转为文本后再送入 R-VOS，会因识别错误（如"cat"→"cap"）导致目标指代偏差，且在含噪条件下性能下降严重。
- **语义空间存在鸿沟**：语音嵌入与文本嵌入分布不同，需要一种机制对齐两者语义，同时保持 R-VOS 模型本身不被重新训练以适应实际部署需求。
- **现有语音增强不适用**：传统语音增强关注波形恢复，而本文只需提取可靠的语义表示，无需重建低层声学特征。

## 核心贡献（创新点）
- **提出 STBridge 框架**：以可插拔方式将噪声容错能力注入冻结的文本 R-VOS 模型，无需重新训练主干网络。与以往仅依赖 ASR+R-VOS 级联方案相比，直接利用语音嵌入进行分割，避免了识别错误的级联放大。
- **噪声感知语义调整（NSA）模块**：通过双向交叉注意力和噪声引导调制，生成对噪声类型自适应的语音语义表示。与通用语音增强方法不同，NSA 专注于语义层面的去噪而非波形恢复。
- **语义抖动抑制（SJS）模块**：对文本嵌入施加随机掩码和噪声扰动（语义抖动），并训练 SJS 从含扰动的查询中恢复稳定的对象 query。这使 R-VOS 模型具备容忍不完整/失真参考的能力。
- **松弛语义对齐损失**：采用池化后的 L2 距离对齐文本与语音嵌入，而非严格的序列到序列对齐，更适配句子级语义匹配需求。
- **在 AVOS 基准上建立新 SOTA**：在干净语音和多种噪声强度（10/20/30 dB SNR）下均优于 ASR-based 基线，噪声鲁棒性显著提升。

## 方法详解
- **整体架构**：冻结预训练的 R-VOS 模型（本文使用 ReferFormer），在其上方添加可训练的 STBridge 模块，包括 Wav2Vec2 语音编码器、NSA、SJS 及语义对齐约束。训练时使用视频-文本-含噪语音三元组，推理时仅保留语音和视频分支。
- **编码器**：视频编码器 $ \mathcal{E}_v $ 和文本编码器 $ \mathcal{E}_t $ 冻结；语音编码器采用 Wav2Vec2，并在其最后隐藏状态之上附加两个线性层用于预测噪声类型，得到噪声嵌入 $ g_n $。
- **语义抖动抑制（SJS）**：对文本嵌入 $ g_t $ 施加扰动 $ g'_t = m \circ g_t + \delta $，其中 $ m $ 是词级或通道级二元掩码，$ \delta $ 为随机噪声。随后通过由 Transformer 编码器和全局平均池化组成的 $ \varphi(\cdot) $ 生成对象 query $ q = \varphi(g'_t) $，迫使模型学会从抖动中恢复。
- **噪声感知语义调整（NSA）**：包含两部分：① 双向交叉注意力（BCA）实现噪声-语音相互引导，得到融合后的 $ g'_{ns} $ 和 $ g'_n $；② 噪声引导调制（NGM）利用 $ g'_{ns} $ 生成动态滤波器 $ \Theta $，通过通道注意力调制语音特征：$ g_{s|n} = \Theta \circ g'_{ns} $，最终输出 $ g_s = g'_{ns} + g_{s|n} $。
- **冻结掩码解码器**：直接使用 R-VOS 原有的 query-based mask decoder $ \mathcal{D}(q, f) $，输入为对象 query $ q $ 和视频特征 $ f $，输出分割掩码、边界框和置信度，该部分全程冻结。
- **训练损失**：总损失 $ \mathcal{L} = \lambda_{align}\mathcal{L}_{align} + \lambda_{noise}\mathcal{L}_{noise} + \mathcal{L}_{match} $。其中 $ \mathcal{L}_{align} = \| \text{pool}(g_t) - \text{pool}(g_s) \|_2 $ 为松弛语义对齐损失；$ \mathcal{L}_{noise} = -\log p[c] $ 为噪声分类损失；$ \mathcal{L}_{match} $ 为标准分割损失（Dice+BCE、Focal、GIoU+L1）。

## 实验与结果
- **数据集**：AudioGuided-VOS（AVOS），包含 Ref-YoutubeVOS、A2D-sentences、JHMDB-sentences 三个基准的语音增强版本，共 18,811 对视频-语音样本。测试时合成 10/20/30 dB SNR 的含噪语音。
- **评估指标**：区域相似度 $ \mathcal{J} $、轮廓精度 $ \mathcal{F} $ 及其均值 $ \mathcal{J}\&\mathcal{F} $。
- **主要结果**（Table 1 & 2）：
  - 干净语音：STBridge 达到 65.5 $ \mathcal{J}\&\mathcal{F} $，接近 ReferFormer 文本版本的 67.2，大幅超越 ASR+ReferFormer（58.4）。
  - 30 dB SNR：STBridge 63.5，较 ASR+ReferFormer（55.3）提升 8.2 点；20 dB 时 62.1 vs 53.3；10 dB 时 59.9 vs 50.3，噪声鲁棒性优势随噪声增强而扩大。
- **消融实验**：NSA 和 SJS 均对性能有贡献；语义抖动中词级掩码、通道级掩码和随机噪声三者组合效果最佳；对齐损失 L2 优于 L1 和 Cosine；掩码比率 0.1、噪声振幅 0.5 为最优设置。
- **可视化**：在含噪条件下 STBridge 能正确指代目标物体，而 ASR 辅助方法易因识别错误导致指代失败。

## 相关工作脉络
- **Referring Video Object Segmentation (R-VOS)**：ReferFormer、MTTR、R²-VOS 等基于 Transformer 的文本参考分割方法，本文在其冻结权重基础上扩展至语音输入。
- **Audio-guided VOS**：WNet 等利用音频辅助视频分割的工作，但多依赖干净或同步音频，缺乏对噪声语音的专门建模。
- **Speech-Visual Grounding**：AVOS 数据集及相关研究（如 LAVISH、VisualVoice）探索语音与视觉的对齐，但未聚焦于分割任务中的噪声容忍。
- **ASR-based 方法的局限**：将 ASR 输出直接作为文本查询的级联方案在噪声下性能衰减严重，本文证明端到端语音接入更有效。
- **噪声容忍表示学习**：DropPath、Masked Autoencoding 等数据增强思想与本文语义抖动策略相通，但本文首次将其应用于语音参考分割领域。
- **跨模态对齐**：CLIP 式对比学习、松弛对齐等技术与本文的池化对齐损失有相似动机，但本文聚焦语音-文本嵌入对齐而非图文对齐。

## 局限性与未来方向
- **仅支持同语言**：当前方法假设语音与文本查询语言相同，跨语言场景语义空间差异更大，尚未探索。
- **噪声类型依赖 AudioSet**：训练时噪声从 AudioSet 合成，实际开放环境中的未知噪声类型可能导致泛化下降。
- **冻结主干的限制**：虽然避免了对 R-VOS 的重训练，但也限制了模型对语音输入的适配能力；表 A 显示更新 R-VOS 参数反而导致过拟合。
- **未考虑说话人差异**：不同口音、语速、音色可能影响语音编码，文中未对此进行专门讨论。
- **未来方向**：作者明确指出跨语言语音-文本桥接是下一步重点；也可扩展至多说话人、实时在线分割等场景。

## 研究启发与可借鉴点
- **冻结主干+可训练适配模块**的设计范式：在保持预训练视觉-语言模型能力的同时，通过轻量附加模块适应新模态（语音），这一策略可迁移到其他多模态迁移任务。
- **语义抖动作为正则化手段**：对文本嵌入施加随机扰动并训练模型恢复，类似 Dropout/Masked Autoencoder 思想，可用于提升任何文本查询模型对噪声输入的鲁棒性。
- **噪声感知调制机制**：NSA 中的双向交叉注意+通道注意力调制，为其他音频-视觉融合任务提供了可复用的噪声自适应模块设计。
- **松弛对齐替代严格对齐**：采用池化后 L2 距离而非序列级对齐，降低了跨模态匹配的复杂度，对语音-文本、图像-语音等松耦合对齐任务具有参考价值。
- **可结合本团队方向**：若团队从事语音驱动的视频理解、机器人语音交互或无障碍视频分析，STBridge 的噪声容错机制可直接应用于改善语音命令下的目标追踪性能。

## 关键术语表
**R-VOS**：Referring Video Object Segmentation，基于语言描述的视频对象分割与跟踪任务。
**AVOS**：AudioGuided-VOS，融合语音指导的视频对象分割大规模基准数据集。
**Wav2Vec2**：Facebook 提出的自监督语音表征学习模型，本文用作语音编码器。
**NSA**：Noise-Aware Semantic Adjustment，通过双向注意力和噪声引导调制生成噪声自适应语音语义表示的模块。
**SJS**：Semantic Jitter Suppression，通过对文本嵌入施加抖动扰动并学习抑制，提升 R-VOS 模型对含噪查询的容忍度。
**SNR**：Signal-to-Noise Ratio，信号噪声比，用于量化语音噪声强度的分贝指标。
**ASR**：Automatic Speech Recognition，自动语音识别，将语音转换为文本的技术。
**L2 alignment**：采用 L2 范数距离对池化后的文本与语音嵌入进行松语义对齐的损失函数。

## 可复现要素
- **数据集**：AVOS（公开可用，含 Ref-YoutubeVOS、A2D-sentences、JHMDB-sentences）；噪声通过从 AudioSet 随机采样叠加合成（0–40 dB SNR 训练，10/20/30 dB 测试）。
- **代码/权重**：论文未明确声明开源，但提到使用 ReferFormer 作为冻结基线（其代码公开）；STBridge 模块为新增可训练部分。
- **关键超参**：学习率 1e-4，batch size 8，训练 2 epochs；掩码比率 0.1，噪声振幅 Uniform(-0.5, 0.5)；损失权重 $ \lambda_{align}=\lambda_{noise}=1 $，$ \lambda_{conf}=\lambda_{giou}=\lambda_{dice}=2 $，$ \lambda_{L1}=\lambda_{focal}=5 $；Transformer 层数 3；图像裁剪最长边 640、最短边 360；优化器 AdamW，weight decay 5e-4。
- **硬件**：8× NVIDIA V100 GPU。
