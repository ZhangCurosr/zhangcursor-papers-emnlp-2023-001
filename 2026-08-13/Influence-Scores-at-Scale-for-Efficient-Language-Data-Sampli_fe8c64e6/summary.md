---
title: "Influence-Scores-at-Scale-for-Efficient-Language-Data-Sampli"
source: https://aclanthology.org/2023.emnlp-main.152.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:47:27"
field: "语言模型数据效率"
keywords: ["influence scores", "data pruning", "variance of gradients", "language model", "data sampling", "NLU", "data efficiency"]
innovations: ["系统benchmark 5种影响分数在语言任务的剪枝效果，确立VoG为最优one-shot方案", "在商业NLU系统验证VoG可削减50%数据而不退化性能", "揭示归一化策略（dataset vs. class）对多域不平衡数据的决定性影响"]
benchmarks: ["SNLI", "internal Alexa NLU system"]
---

# 论文速读：Influence-Scores-at-Scale-for-Efficient-Language-Data-Sampling

## 一句话总结
本文系统评估了影响分数（influence scores）在语言数据处理中的有效性，发现梯度方差（VoG）分数可在不降低性能的前提下剪除约50%的SNLI训练数据，并在商业语音助手NLU系统中验证了其在大规模真实场景中的实用性。

## 研究问题与动机
- 影响分数方法主要在计算机视觉领域开发，其在预训练语言模型下游任务中的数据选择效果尚不明确
- 现代ML系统从合成数据、弱信号、真实用户交互等多源混合数据中训练，需要可扩展的影响评估方法支持数据筛选
- 商业场景中用户语音模式持续演化（受突发事件、用户交互影响），需验证影响分数在动态噪声环境中的鲁棒性
- 生产环境通常只允许单次训练（无法构建ensemble），要求影响力评估方法具备"one-shot"计算能力且无需大量超参调优

## 核心贡献（创新点）
1. **首次系统 benchmark 语言任务影响分数**：在SNLI数据集上评估5种影响分数（VoG、EL2N、Forgetting Score、PVI、TracIn）的数据剪枝效果，填补了CV领域之外语言任务影响分数应用的空白
2. **VoG作为scalable one-shot方案的确立**：发现VoG仅需单次训练运行、无需超参调优即可在45%剪枝率下保持测试准确率，显著优于其他需复杂调优的分数
3. **规模化NLU系统实证**：在亚马逊Alexa商用语音助手的多层NLU架构（DC+IC+NER）中部署VoG，实现约50%训练数据削减且未引发统计显著的UER/PDR退化
4. **归一化策略的量化影响**：首次清晰揭示 dataset-normalization vs. class-normalization 对多域不平衡数据的巨大影响差异，发现前者在商业场景显著优于后者

## 方法详解
**VoG（Variance of Gradients）计算流程**：
- 输入：BERT模型在训练过程中保存的 $N_c$ 个checkpoint
- 梯度计算：对每个示例 $i$，在ground-truth标签位置计算预softmax输出的关于embedding层的梯度：$G_i^{(c)} = \frac{\partial A_i^{(c)}}{\partial E_i^{(c)}}$
- 方差聚合：计算跨checkpoint的梯度均值 $\mu_i$ 和方差 $V_i = \frac{1}{\sqrt{N_c}}(G_i^{(c)} - \mu_i)^2$
- 最终得分：$v_i = \text{mean}(V_i)$，再经归一化得到 $\mathrm{VoG}_i$

**两种归一化方案**：
- class-normalization：按类别均值和标准差归一化（$\frac{v_i - \mu_{\mathrm{class}}}{\sigma_{\mathrm{class}}}$）
- dataset-normalization：按全数据集均值和标准差归一化（$\frac{v_i - \mu_{\mathrm{dset}}}{\sigma_{\mathrm{dset}}}$）

**采样策略**：
- 硬截断（hard cut-off）：保留得分高于阈值的示例
- 概率采样（probabilistic sampling）：保留概率与得分成正比（线性变换至$[\epsilon, 1]$后归一化）
- 对于NLU系统采用复合得分：$\text{final score} = \text{mean}(\mathrm{VoG}_{\mathrm{DC}}, H_{\mathrm{intent}})$，其中 $H_{\mathrm{intent}}$ 是基于slot-label-trail的Shannon熵估计

## 实验与结果
**SNLI实验**：
- 模型：$\text{BERT}_{\text{SMALL}}$（L=4, H=512, 29.1M参数），10 epoch训练
- VoG-class-norm 在45%剪枝时达到 85.04±0.20% 准确率（全量数据：85.52±0.14%）
- 添加噪声实验：VoG-easy剪枝在5%-30%各噪声水平下均优于随机基线
- 关键发现：剪 hardest 示例普遍导致性能下降（破坏决策边界），剪 easy 示例更安全

**内部NLU系统实验**：
- 数据规模：10.9M去标识化历史用户数据，涵盖Music、Video、Shopping等7个domain
- 52%剪枝率下：VoG-dset-norm实现 $\Delta\text{DCER}=2.94\%$，显著优于 Random（6.04%）和 Stratified（5.23%）
- 46%剪枝率下：VoG-dset-norm实现 $\Delta\text{DCER}=1.52\%$，效率指标 $\sigma_{\text{DCER}}=-3.25$（最优）
- 14%保守剪枝：$\Delta\text{DCER}=0.53\%$，证明小幅度数据削减几乎无损

**用户研究（A/B测试，11天）**：
- 剪枝约50%历史数据 + 40%总训练数据削减
- UER：相对变化 -1.18%（统计显著改善，CI: [-2.00%, -0.35%]）
- PDR：相对变化 +0.45%（p=0.16，不显著）
- PDR-tail：相对变化 +0.56%（显著退化，CI: [0.003%, 1.09%]）
- 结论：整体性能持平，但长尾请求（tail-PDR）存在轻微退化风险

## 相关工作脉络
- **Paul et al. (2021) EL2N**：基于confidence margin的difficulty score，但需多次训练run取平均且在lang任务中易收敛至零，本文定位其适用于早期fine-tuning阶段
- **Toneva et al. (2019) Forgetting Score**：衡量示例学习-遗忘动态，但fine-tuning收敛过快导致大部分示例score为0，需高频记录（每50步），生产部署困难
- **Ethayarajh et al. (2022) PVI**：信息论难度度量，可识别少量误标数据但整体剪枝效果有限（>10%后不优于随机）
- **Garima et al. (2020) TracIn**：梯度点积追踪影响，self-influence变体与EL2N-hard示例高度重叠（similarity index=0.37 vs 0.11随机）
- **Agarwal & Hooker (2022) VoG**：原始在CV提出，本文首次将其适配语言模型（替换pixel为embedding）并验证其在NLU生产环境的可行性
- **Sorscher et al. (2022)**：支持"剪hard示例破坏决策边界"的发现，与本文t-SNE可视化证据一致

## 局限性与未来方向
- 内部NLU实验因计算开销限制无法复现同等采样技术数量和数据削减比例
- 客户数据动态变化（数据过期、删除请求）难以严格控制跨实验的训练数据一致性
- 论文明确表示目标非寻找绝对最优分数，而是证明简单可扩展的one-shot方法存在性
- 未探索迭代剪枝（iterative pruning，基于前一轮剪枝数据重新计算分数）
- 主要聚焦BERT架构的supervised fine-tuning，未覆盖LLM pretraining或decoder-only架构
- 多域数据中仍观察到部分domain（如Shopping）的slotting错误上升

## 研究启发与可借鉴点
1. **归一化策略优先级**：数据集不平衡场景下dataset-normalization显著优于class-normalization，这对多域/多标签NLU系统的数据筛选具有直接指导意义
2. **one-shot实用主义**：生产环境中应优先选择无需ensemble、无需频繁checkpoint记录的方法（VoG vs. Forgetting Score），即使精度略低也更具部署价值
3. **复合得分设计**：将DC模型VoG与下游任务复杂度估计（如slot-label-trail熵）结合，可有效缓解层级NLU系统中单模块剪枝导致的下游退化
4. **噪声鲁棒性验证**：在公开数据集上引入可控标签噪声（5%-30% isotropic noise）是验证影响分数缺陷检测能力的有效实验范式
5. **效率指标 $\sigma_{\text{ER}}$**：相对性能变化与相对数据变化的比值提供了跨不同剪枝率的统一比较基准，值得推广到数据采样研究

## 关键术语表
**VoG (Variance of Gradients)**：衡量模型输出关于输入的梯度方差，表征示例对参数更新的影响程度
**Class-normalization vs. Dataset-normalization**：前者按类别归一化消除类别不平衡，后者按全局归一化保留跨域影响关系
**Relative ER (Error Rate)**：剪枝后模型相对基线模型的错误率变化百分比
**$\sigma_{\text{ER}}$ (Relative Data-Score Efficiency)**：单位数据削减带来的性能退化率，值越接近0表示数据效率越高
**PDR (Predicted Defect Rate)**：基于系统响应和用户行为信号预测的用户感知缺陷概率
**Slot-label-trail entropy**：标注序列的Shannon熵，用于量化intent的NER复杂度
**Hard cut-off vs. Probabilistic sampling**：前者按阈值硬性筛选，后者按得分分布概率采样
**Composite NLU model**：由DC（domain classifier）、IC（intent classifier）、NER（named entity recognition）组成的层级模型，配合确定性规则组件使用

## 可复现要素
- **数据集**：SNLI (Bowman et al., 2015) 公开可用；内部NLU数据未公开（去标识化历史用户语音）
- **代码/权重**：论文未提供开源代码，VoG实现细节见附录B.1 Algorithm 1
- **关键超参**：
  - $\text{BERT}_{\text{SMALL}}$：L=4, H=512, 29.1M参数，batch size=128，encoder lr=$10^{-4}$，head lr=$10^{-3}$
  - VoG checkpoints：$N_c=10$（每500步记录一次）
  - Forgetting score cadence：每50步记录（首次2 epoch）
  - PVI：null model和full model各训练2 epoch
  - 归一化：class vs. dataset（论文明确对比）
