---
title: "Outlier-Suppression-Accurate-quantization-of-large-language"
source: https://aclanthology.org/2023.emnlp-main.102.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:52:06"
field: "模型压缩与量化"
keywords: ["post-training quantization", "outlier suppression", "large language models", "channel-wise shifting", "equivalent transformation", "low-bit quantization"]
innovations: ["发现outliers通道间不对称性并提出channel-wise shifting消除偏斜", "设计基于输出变化MSE的scaling优化目标与阈值网格搜索", "统一迁移模式将shifting/scaling等价融合到后续模块无推理开销"]
benchmarks: ["GLUE", "PIQA", "Winogrande", "HellaSwag", "LAMBADA", "WikiText2"]
---

# 论文速读：Outlier-Suppression-Accurate-quantization-of-large-language

## 一句话总结
本文提出 Outlier Suppression+ (OS+) 框架，通过通道级平移（shifting）消除激活值通道间不对称的outliers，再通过通道级缩放（scaling）集中抑制高幅值通道，并将操作等价迁移到后续模块，从而在标准与细粒度量化设置下实现LLMs的高精度低比特量化。

## 研究问题与动机
1. **核心问题**：Transformer语言模型（BERT、OPT、LLaMA等）进行后训练量化（PTQ）时，激活值中存在极端outliers（范围可达140），导致离散表示无法准确逼近连续值，引发灾难性性能下降。
2. **现有方法不足**：Fine-grained方法（如PEG）额外分配bit位给outlier通道，损害加速效果；现有scaling方法（OS、SmoothQuant）未考虑迁移和量化引起的输出变化最小化，且仅从token或channel单一维度处理，忽略通道间不对称性。
3. **新发现**：作者观察到outliers在通道间呈**不对称分布**（如某通道负轴范围-97~-58，另一通道正轴5.7~43），即使各通道范围较小（~40），整体张量范围也会膨胀到140，严重阻碍量化。
4. **目标**：在保持FP等价的条件下，通过shifting和scaling使激活更"量化友好"，支持INT8/INT6/INT4标准量化及per-token/per-group细粒度量化。

## 核心贡献（创新点）
1. **发现outliers通道间不对称性并提出channel-wise shifting**：与已有工作（OS、SmoothQuant仅考虑channel集中性）的本质区别在于，首次识别并消除通道分布的偏斜，将张量范围压缩至最大通道范围而非全张量极差。
2. **提出统一迁移模式（unified migration pattern）**：将shifting和scaling的逆操作无缝融合到后续线性层权重/偏置及残差连接中，保持FP等价且无额外推理开销，区别于仅作用于activation的已有方法。
3. **设计基于输出变化的量化目标优化scaling值**：通过最小化缩放+量化后激活与权重的矩阵乘积输出MSE（而非单独最小化activation/weight量化误差），并引入异常值阈值t的网格搜索，实现激活与权重量化负担的平衡。
4. **广泛验证与SOTA提升**：在BERT、OPT、BLOOM、LLaMA等多模型、多任务（GLUE、zero-shot）上验证，INT4 BERT提升15.5%，INT6 OPT-175B接近FP，4-bit LLaMA per-token超越SOTA 9.41%。

## 方法详解
**整体框架**：对LayerNorm输出（problematic activation）依次应用channel-wise shifting和channel-wise scaling，再通过等价变换将逆操作迁移到后续模块。

1. **Channel-wise Shifting（消除不对称）**：
   - 公式：$\widetilde{X}' = X - z$，其中$z_j = \frac{\max(X_{:,j}) + \min(X_{:,j})}{2}$为第j通道中心对齐到0的偏移量。
   - 效果：去除通道间偏斜，张量范围从140降至最大通道范围（如40）。

2. **Channel-wise Scaling（集中抑制）**：
   - 公式：$\widetilde{X} = (X - z) \oslash s$，其中$s_j = \max(1.0, \frac{\max(X_{:,j} - z_j)}{t})$，t为异常值阈值。
   - 优化目标：最小化迁移+量化后输出的MSE变化：
     - 单线性层：$\min_s \mathbb{E}[\|Q(W \odot s)^\top Q((X-z) \oslash s) + \tilde{b} - W^\top X - b\|_F^2]$
     - 多线性层（如Attention的Q/K/V）：通过softmax(QK^T)V封装信息后计算MSE。
   - 搜索策略：仅缩放超过阈值t的通道，将多维s优化转化为单变量t的网格搜索，稳定高效。

3. **统一迁移模式（Equivalence）**：
   - **线性层**：权重和偏置吸收缩放和平移，$\widetilde{W} = W \odot \mathbf{s}_{repeat}$，$\widetilde{b} = zW^\top + b$。
   - **残差连接（Post-LN）**：identity函数替换为channel-wise乘法与加法，开销可忽略。
   - **实现**：以Pre-LN LayerNorm为例，参数更新为$\widetilde{\beta} = (\beta - z) \oslash s^*$，$\widetilde{\gamma} = \gamma \oslash s^*$。

4. **校准流程**：随机采样128样本计算z和s，迁移参数后执行MinMax或相关calibration，无额外训练开销。

## 实验与结果
**数据集与模型**：GLUE benchmark（BERT-base/large）、零样本任务PIQA/Winogrande/HellaSwag/LAMBADA（OPT-13B/30B/66B/175B、BLOOM/BLOOMZ 176B、LLaMA-1 7B/13B/30B/65B），校准数据来自PILE或训练集。

**基线**：MinMax、Percentile、OMSE、PEG、OS（Outlier Suppression）、ZeroQuant、SmoothQuant。

**主要结果**：
- **BERT（标准量化）**：INT8*下avg 83.5（+0.5 vs FP32），INT6下82.8（+1.6 vs OS），**INT4下78.2（+15.5 vs OS）**，建立新SOTA。
- **OPT/BLOOM（标准量化）**：INT8下接近FP16；INT6下OPT-175B PIQA提升32.5%（vs SmoothQuant的47.2），HellaSwag提升27.4%；BLOOM在6-bit下优于最佳基线约2个百分点。
- **LLaMA（细粒度量化）**：INT4 per-token下Winogrande提升10.58%、WikiText2 PPL降低10.04；INT6 lossless；per-group量化（group=512/1024）在4-bit OPT上无损。

**最强结果**：4-bit BERT-base avg 78.2（相比OS提升15.5%）；INT6 OPT-175B在PIQA上达76.0（接近FP16的79.7）。

## 相关工作脉络
1. **Outlier Suppression (OS, Wei et al. 2022)**：利用LayerNorm的γ参数作为scaling向量迁移到后续权重，但未考虑通道间不对称性，且scaling固定，本文通过输出变化目标优化scaling并引入shifting。
2. **SmoothQuant (Xiao et al. 2022)**：启发式均衡激活与权重range，忽略迁移和量化对输出的联合影响，本文直接优化输出MSE，在INT6/INT4下显著超越。
3. **PEG (Bondarenko et al. 2021)**：per-embedding-group fine-grained量化，为outlier通道分配额外bit，本文聚焦标准量化下等价变换，保持硬件加速友好。
4. **ZeroQuant (Yao et al. 2022)**：per-token动态量化，属token维度细粒度方法，本文聚焦channel维度，可与之结合（附录提及）。
5. **LLM.int8() (Dettmers et al. 2022)**：对信号>6的通道使用FP16，牺牲加速效果，本文通过等价变换保持整数计算。
6. **Q-drop (Wei et al. 2022a)**、**Olive (Guo et al. 2023)**：前者随机drop量化，后者丢弃normal邻近值，本文通过连续等价变换处理，无硬件定制需求。

## 局限性与未来方向
1. **outliers成因未深入分析**：作者指出对outliers涌现机制和属性缺乏深刻理解，需结合训练流程、超参进行耗时分析，有助于FP和量化双场景。
2. **LLaMA特殊结构挑战**：LLaMA FFN最后一层采用两激活element-wise乘法，信号可超600，4-bit量化仍存在较大性能差距（Table 10），需针对模型设计改进。
3. **仅处理channel维度outliers**：token维度outliers依赖配套技术（如Token-Wise Clipping），未统一建模。
4. **未探索与其他压缩技术结合**：如混合精度、稀疏化、蒸馏等的协同优化。

## 研究启发与可借鉴点
1. **通道中心对齐shifting设计**：将张量范围压缩至最大通道范围而非全张量极差，思路简洁且有效，可迁移至视觉Transformer、多模态模型的量化优化。
2. **输出变化MSE作为量化目标**：避免分别优化activation/weight量化误差的次优性，直接对齐下游任务输出，适用于任何可微分近似或校准场景。
3. **异常值阈值t的网格搜索替代多维优化**：将scaling向量求解降维为单变量搜索，兼顾稳定性和效率，可推广至其他需要校准的压缩方法。
4. **统一等价迁移模式**：channel-wise操作的逆操作融合到后续权重/残差连接，无推理开销，为后续research提供可复用的transform范式。
5. **与per-token/per-group细粒度量化兼容**：OS+可作为预处理模块与动态量化结合，启发跨维度outlier联合处理的研究方向。

## 关键术语表
**Post-Training Quantization (PTQ)**：仅用少量校准数据对预训练模型进行低比特量化的技术，无需重新训练。
**Outlier**：激活值中极端偏离正常分布的高幅值元素，集中在特定通道，破坏量化精度。
**Channel-wise Shifting**：对每个通道独立平移，对齐其中心到0，消除通道间分布不对称性。
**Channel-wise Scaling**：对每个通道独立缩放，将超过阈值的异常值压缩到指定范围内。
**Unified Migration Pattern**：将shifting和scaling的逆操作等价迁移到后续线性层权重/偏置及残差连接，保持FP输出不变。
**Per-token Quantization**：为每个token动态计算独立的量化参数，属细粒度量化策略。
**Per-group Quantization**：将张量元素分组，每组独立量化参数，粒度介于per-tensor与per-token之间。
**Outlier Threshold (t)**：控制哪些通道需要缩放的超参，网格搜索优化输出MSE，替代直接优化scaling向量。

## 可复现要素
- **数据集**：GLUE benchmark、PILE（零样本校准）、WikiText2（LLaMA校准），均为公开数据集。
- **代码**：已开源，https://github.com/ModelTC/Outlier_Suppression_Plus。
- **权重**：使用公开模型（BERT、OPT、BLOOM、LLaMA-1系列）。
- **关键超参**：校准样本数128；batch size 32；网格搜索迭代K（未明确数值，算法1提及）；LLaMA省略shifting；Token-Wise Clipping比例（BLOOM: 0.5%/1.5% for INT8/INT6）。
- **硬件**：校准约20分钟（OPT-175B），推理无额外开销。
