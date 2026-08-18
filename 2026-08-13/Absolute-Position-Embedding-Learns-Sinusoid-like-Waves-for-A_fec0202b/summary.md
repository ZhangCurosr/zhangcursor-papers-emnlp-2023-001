---
title: "Absolute-Position-Embedding-Learns-Sinusoid-like-Waves-for-A"
source: https://aclanthology.org/2023.emnlp-main.2.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:06:40"
field: "Transformer可解释性"
keywords: ["位置嵌入", "自注意力", "可解释性", "相对位置", "Transformer", "RoBERTa"]
innovations: ["揭示APE自发习得类正弦波周期性成分", "证明attention通过query-key相位偏移实现相对位置依赖", "提出基于SVD的Query-Key分解框架显式提取周期性成分"]
benchmarks: ["wikitext-2"]
---

# 论文速读：Absolute-Position-Embedding-Learns-Sinusoid-like-Waves-for-A

## 一句话总结
本文通过分析RoBERTa中self-attention的注意力权重，揭示了**可学习的绝对位置嵌入（APE）在预训练过程中自发习得了类正弦波周期性成分**，且attention head通过查询（query）和键（key）中的**相位偏移**来集中注意力于邻近token，从而实现了基于相对位置的上下文依赖推理。

## 研究问题与动机
- **核心问题**：Transformer本身对称且不内置相对位置约束，但实践中某些attention head却会集中注意力于邻近token；这种"相对位置依赖"的内在机制尚不清楚。
- **现有方法不足**：BERT/RoBERTa使用可学习绝对位置嵌入（APE），而Vaswani等提出正弦位置编码（SPE）的理论假设是"模型可通过旋转矩阵轻松学习相对位置"，但APE是否也能实现类似功能、以及如何实现，缺乏直接证据。
- **已有工作局限**：Clark等和Kovaleva等仅观察到attention权重受相对位置影响（图1），Ravishankar和Søgaard发现APE某些列呈周期性，但未解释周期性成分如何被attention用于相对位置推理。

## 核心贡献（创新点）
1. **首次揭示APE自发习得类正弦波成分**：通过对RoBERTa的APE做DFT分析，证明可学习位置嵌入在预训练后自然形成有限频率的周期性分量（图3），而非人工设计的SPE。
2. **提出基于SVD的Query-Key分解框架**：通过对$W^Q(W^K)^T$做SVD，将query和key重新定义为$Q=XU^Q$、$K=XU^K$（式13-14），显式提取出periodic components并建立$K=QR$的旋转关系（式15）。
3. **证明相位偏移决定注意力方向**：通过cross-covariance/cross-correlation分析（式19-20），揭示query和key中类正弦波成分的相位差$\Delta$直接决定注意力集中方向的相对位置（式21-23）。
4. **从特征值角度给出理论解释**：证明当特征值幅角$\theta_i$与频率$f_i$之比为常数时（式27），所有类正弦波成分的相位偏移量$\Delta$相同，解释了为何多频率成分能协同作用（第4.4节）。

## 方法详解

**1. t-offset trace与聚类分析（第3节）**
定义$t$-offset trace $\mathrm{tr}_t(A)$为所有相对位置为$t$的注意力权重之和（式7），将每个head的注意力模式压缩为21维向量$\boldsymbol{a}_{lh}$（式8），对100个wikitext-2句子（长度512）用k-means聚类（6类），发现存在始终关注左侧/右侧/紧邻token的head（图2）。

**2. DFT分析APE的周期性（第4.1.1节）**
对APE矩阵$E_{pos}\in\mathbb{R}^{T\times d}$的每一列做DFT：$\mathrm{spec}_i=\mathrm{abs}(\mathrm{DFT}(E_{pos}\boldsymbol{e}_i))$（式9），发现768条时间序列的振幅谱在特定频率处出现峰值，表明APE隐式包含了周期性（图3a）；作为对比，词嵌入序列无此峰值。

**3. PCA与CCA分析位置表示维度（第4.1.2节）**
- PCA：APE前4个主成分贡献50.51%，前12个贡献92.23%，说明位置信息集中在低维子空间。
- CCA：计算hidden state $Z^l$与位置嵌入的相关系数$\rho_k$（式12），发现高层hidden state与位置嵌入的相关性逐渐降低（图4），印证Lin等"BERT在第3-4层丢弃位置信息"的结论。

**4. SVD分解与Query-Key重定义（第4.2.1节）**
对$W^Q(W^K)^T$做SVD：$W^Q(W^K)^T=U^Q S (U^K)^T$（式13），重新定义$Q=XU^Q$、$K=XU^K$（式14）。此时attention score可写为$\sum_i s_i q_ik_i^T$（式16），即top奇异值对应的$q_i,k_i$对注意力分布贡献最大。定义正交矩阵$R=(U^Q)^TU^K$，则$K=QR$（式15），表明key是query经正交变换的结果。

**5. DFT分析Q和K的频谱（第4.2.2节）**
对每层head的query列做DFT：$\mathrm{spec}_{lhi}=\mathrm{abs}(\mathrm{DFT}(Q_{lh}\boldsymbol{e}_i))$（式17），取各层最大频谱$\mathrm{max\text{-}spec}_l$（式18），发现query/key的峰值频率与APE一致（图6），且高层峰值消失。

**6. Cross-covariance与相位偏移（第4.3节）**
定义cross-covariance $\mathrm{xcov}_j(t)=\sum_i q_{i,j}k_{i+t,j}$（式19）和归一化cross-correlation（式20）。证明t-offset trace可表示为$\mathrm{tr}_t(QSK^T)=\sum_js_j\mathrm{xcov}_j(t)$（式21-23），即注意力在相对位置$t$的集中程度等于各周期性成分的cross-covariance按奇异值加权求和。图8-9显示top奇异值对应的cross-correlation峰值在同一$t$处叠加，决定注意力方向。

**7. 相位偏移宽度与频率无关的理论证明（第4.4节）**
设$R$的特征值为$\lambda_i=e^{j\theta_i}$、特征向量为$\boldsymbol{p}_i$，则有$K\boldsymbol{p}_i=Q\boldsymbol{p}_i\cdot e^{j\theta_i}$（式24）。若$Q\boldsymbol{p}_i$为单一频率$f_i$的正弦波，且$\theta_i/f_i$为常数，则相位偏移$\Delta=(T/2\pi)\cdot(\theta_i/f_i)$与$i$无关（式27，附录D定理1），即所有频率成分的偏移量相同，协同聚焦于同一相对位置。

**8. 消融实验：位置嵌入vs词嵌入的作用（第5节）**
- 仅保留位置嵌入（词嵌入置零）：cross-correlation无尖峰，说明位置嵌入不足以产生紧邻聚焦。
- 仅保留词嵌入（位置嵌入置零）：cross-correlation平坦，但$t=1$处仍有相对较大的covariance。
- Shuffle词嵌入顺序：尖峰仍在$t=1$，说明相邻token的方向由位置嵌入决定，而非共现频率。

**9. APE再学习实验（附录E）**
在wikitext-2上冻结除APE外的所有参数，将每token间插入3个UNK token后重新训练APE。结果（表1）：重新学习后，模型能正确预测MASK处的名词；注意力集中于每隔4个token的位置（图22），证明相对位置依赖可从少量数据中学得。

## 实验与结果

**数据集**：wikitext-2（Merity et al., 2017），生成100条长度512的句子用于聚类分析，200条用于CCA分析。

**模型**：以RoBERTa-base（Liu et al., 2019）为主要分析对象，另对比GPT-2（Radford et al., 2019）和BART（Lewis et al., 2020）。

**主要结果**：
- **聚类分析**（图2）：12层×12 head共1728个head中，存在"leftward""next-to-left""rightward""next-to-right"等稳定聚类的head，部分head对所有输入始终关注同一方向。
- **DFT频谱**（图3a）：RoBERTa的APE在特定低频带出现明显峰值，而词嵌入无峰值。
- **CCA相关系数**（图4）：高层hidden state与位置嵌入相关性逐层下降，第8-10层后趋近于0。
- **SVD分解后的Query-Key频谱**（图6）：query/key的峰值频率与APE一致，高层衰减。
- **Cross-correlation**（图7-9）：head(1,1)在$t=1$处cross-correlation峰值叠加；head(8,9)在$t=-1$处出现窄峰（周期约8+token，非纯正弦机制）。
- **消融实验**（图11）：仅位置嵌入时无尖峰；仅词嵌入时$t=1$仍有响应；shuffle词嵌入后尖峰仍在$t=1$。
- **APE再学习**（表1）：冻结参数后仅训练APE，模型即可从"this _ _ _ mask"中预测出合理名词，注意力集中于非UNK token（图22）。

**最强结果**：head(8,9)在仅关注紧邻左侧token（$t=-1$）时达到极窄的cross-correlation尖峰，展示了精确的相对位置聚焦能力。

## 相关工作脉络

1. **Vaswani et al. (2017)** 提出SPE并假设其可通过旋转矩阵（式6）支持相对位置学习——本文证明此机制在APE+MLM预训练下同样被自发习得，但习得的频率数量有限（远少于SPE的$d/2$个）。
2. **Clark et al. (2019)、Kovaleva et al. (2019)** 观察到BERT attention权重受相对位置影响——本文进一步解释了其内在机制（相位偏移）。
3. **Ravishankar和Søgaard (2021)** 发现APE某些列呈周期性——本文首次将周期性与attention的相对位置依赖建立因果关系。
4. **Chang et al. (2022)** 用LDA证明hidden representation中存在周期性位置表示——本文用SVD+DFT更直接地从参数层面提取周期性。
5. **Su et al. (2022) RoPE** 作为RPE变体，通过显式旋转角随相对位置变化实现相对位置建模——本文揭示APE模型通过"位置嵌入频率+attention旋转角"的耦合自发实现类似效果。
6. **Lin et al. (2019)** 发现BERT在第3-4层丢弃位置信息——本文的CCA结果与此一致，并补充解释了高层attention为何不再依赖周期性成分。

## 局限性与未来方向

- **不适用于decoder-only模型**：GPT-2的APE频谱与encoder差异显著（低频为主），且 causal mask 改变了注意力机制，无法用相同框架解释（第9节、附录B/C）。
- **仅分析邻近token**：对于依存句法分析等需要远距离相对顺序的任务，本文未分析位置嵌入是否能提供跨长距离的相对位置信息（第9节）。
- **softmax非线性未纳入**：分析基于softmax前的attention score分解，未考虑softmax非线性对下游Value聚合的影响（第9节）。
- **未来方向**：将框架推广至causal decoder模型；分析远距离相对位置信息的编码；量化理解APE的经济性（有限频率vs完整SPE）。

## 研究启发与可借鉴点

1. **SVD分解$W^Q(W^K)^T$分析注意力机制**：将query/key重新定义为$Q=XU^Q$、$K=XU^K$，使周期性成分显式可见，此方法可迁移至其他位置编码研究（如分析RoPE的旋转结构）。
2. **t-offset trace + k-means聚类**：将每head的注意力模式压缩为21维向量并聚类，可快速识别具有特定相对位置偏好的head类型，适合用于模型可解释性分析流水线。
3. **消融实验设计**：分别置零位置嵌入/词嵌入、shuffle词嵌入顺序，隔离各组件对相对位置依赖的贡献，实验设计简洁有力，可复用于其他位置编码研究。
4. **APE再学习实验**：冻结除APE外所有参数，仅用少量数据微调位置嵌入，可验证位置表示的学习效率，为位置编码的高效微调提供思路。
5. **跨模型对比框架**：本文同时分析RoBERTa、GPT-2、BART，揭示了encoder/decoder架构在位置编码学习上的本质差异，为位置编码选型提供理论依据。

## 关键术语表

- **Absolute Position Embedding (APE)**：为序列中每个位置分配独立可学习向量的位置编码方式，如BERT/RoBERTa所用。
- **Sinusoidal Position Embedding (SPE)**：Vaswani等提出的确定性正弦/余弦位置编码，通过三角函数公式为每个位置生成固定编码。
- **t-offset trace**：将所有相对偏移为$t$的注意力权重求和，用于刻画attention模式随相对位置的变化趋势。
- **Cross-covariance / Cross-correlation**：衡量query和key在偏移$t$时的协方差/相关系数，用于量化相位偏移量。
- **Singular Value Decomposition (SVD) 框架**：将$W^Q(W^K)^T$分解为$U^Q S (U^K)^T$，使query和key的周期性成分显式可见。
- **Phase shift (相位偏移)**：query和key中同频正弦成分的相位差，决定注意力集中方向的相对位置。
- **Canonical Correlation Analysis (CCA)**：寻找两组变量最大相关性的方法，本文用于量化hidden state与位置嵌入的相关程度。
- **Masked Language Modeling (MLM)**：BERT/RoBERTa的预训练目标，随机掩码token后预测其内容。

## 可复现要素

- **数据集**：wikitext-2（公开）。
- **模型**：RoBERTa-base、GPT-2、BART-base（均从Hugging Face获取，公开）。
- **代码/权重**：论文未明确提供开源代码，但模型权重可通过Hugging Face Transformers获取。
- **关键超参**：序列长度512，聚类数k=6（图2），k-means使用t-offset trace向量（式8）；CCA使用200个句子、$m=200\times512$个token响应；APE重学习时冻结除位置嵌入外所有参数，新增训练数据8645条（附录E）。
