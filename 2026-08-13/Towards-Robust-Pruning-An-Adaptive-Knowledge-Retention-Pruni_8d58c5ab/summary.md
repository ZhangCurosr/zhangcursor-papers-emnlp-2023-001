---
title: "Towards-Robust-Pruning-An-Adaptive-Knowledge-Retention-Pruni"
source: https://aclanthology.org/2023.emnlp-main.79.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:33:45"
field: "语言模型鲁棒压缩"
keywords: ["adversarial robustness", "model pruning", "post-training pruning", "Hessian-based pruning", "knowledge retention", "language model compression"]
innovations: ["提出鲁棒性与预训练知识保留量成正比的假设，从知识保持视角重构鲁棒剪枝问题", "设计自适应层间Hessian更新与密集权重修正机制，解决累积误差传播问题", "采用Weight Averaging构建鲁棒密集模型作为后训练剪枝起点，消除捷径学习偏差"]
benchmarks: ["SST-2", "IMDB", "AG News"]
---

# 论文速读：Towards-Robust-Pruning-An-Adaptive-Knowledge-Retention-Pruni

## 一句话总结
本文针对语言模型剪枝过程中鲁棒性下降的问题，提出一种后训练自适应知识保持剪枝策略（Ada-Pruning），通过保留密集模型的嵌入空间与特征空间，在无需重新训练的前提下显著提升了稀疏模型的对抗鲁棒性。

## 研究问题与动机
- **鲁棒性与稀疏度的权衡困境**：现有鲁棒剪枝方法在提升模型稀疏度时，对抗攻击下的准确率迅速下降，难以同时兼顾精度、稀疏度与鲁棒性。
- **预训练知识流失严重**：传统剪枝方法加剧了模型对数据集虚假特征（spurious features）的依赖，导致预训练阶段习得的深层语义知识大量丢失。
- **对抗样本的语义替代威胁**：NLP对抗样本通常通过替换句子中关键词为语义相似词构造，若稀疏模型未能保持原始词的嵌入表示，则无法抵御此类攻击。
- **重训练成本高昂**：现有方法多依赖对抗训练或联合优化，计算成本大，不适用于大规模语言模型的高效部署。

## 核心贡献（创新点）
1. **提出"鲁棒性与预训练知识保留量成正比"的核心假设**：首次从知识保持视角系统分析语言模型对抗脆弱性的根源，与以往仅关注剪枝掩码优化的方法形成本质区别。
2. **设计基于Hessian矩阵的自适应后训练剪枝框架（Ada-Pruning）**：在不重新训练的前提下，逐层迭代剪枝并动态修正累积误差，区别于一次性全局剪枝或需端到端微调的方法。
3. **引入自适应密集权重更新机制**：通过最小二乘估计 $\bar{W}_l = (\hat{X}_l^T \hat{X}_l)^{-1} \hat{X}_l^T Y_l$ 修正因前层稀疏化导致的输入偏移，使当前层剪枝目标更接近原始密集模型输出。
4. **构建鲁棒密集模型作为剪枝起点（Weight Averaging）**：通过不同超参数训练多个模型并贪心平均有效权重，消除底层密集模型本身的捷径学习偏差，与直接对预训练模型剪枝的策略不同。

## 方法详解
**整体流程**：先通过Weight Averaging生成鲁棒密集模型，再采用Ada-Pruning逐层剪枝。

**Step 1 — Weight Averaging构建鲁棒密集模型（§4.2）**
- 使用不同超参数组合（学习率、优化器、epoch数、是否对抗训练等）微调预训练模型，得到多个知识异构的模型。
- 按对抗准确率降序排序，贪心选择对最终性能有贡献的模型进行权重平均，形成鲁棒的密集模型作为剪枝基础。

**Step 2 — 自适应Hessian矩阵更新（§4.3.2）**
- 对第 $l$ 层，初始Hessian矩阵 $H_l = X_l X_l^T$。
- 完成第 $l$ 层剪枝后，其稀疏输出 $\hat{Y}_l$ 成为第 $l{+}1$ 层的输入 $\hat{X}_{l+1}$，导致原Hessian失效。
- 自适应更新下一层的Hessian：$H_{l+1}^{-1} \leftarrow (\hat{X}_{l+1} \hat{X}_{l+1}^T)^{-1}$，确保剪枝决策基于当前真实输入分布。

**Step 3 — 自适应密集权重修正（§4.3.3）**
- 前层累积误差使输入由 $X_l$ 变为 $\hat{X}_l$，原始权重 $W_l$ 不再最优。
- 通过正则化最小二乘更新当前层目标权重：
$$\bar{W}_l = (\hat{X}_l^T \hat{X}_l + \lambda I)^{-1} \hat{X}_l^T Y_l, \quad \lambda = 1e{-}4$$
- 以 $\bar{W}_l$ 为基准计算剪枝损失，避免在错误目标上浪费稀疏预算。

**Step 4 — Hessian感知的逐权剪枝（Algorithm 1）**
- 每步选择要剪除的权重 $p$：
$$p = \arg\min_{p} \frac{[W_l]_p^2}{[H_l^{-1}]_{pp}}$$
- 更新剩余权重：$W_r \leftarrow W_r - \frac{[W_l]_p}{[H_l^{-1}]_{pp}} H_{:,p}^{-1}$
- 递推更新Hessian逆矩阵（Gauss消元）：
$$H_{-p}^{-1} = \left(H^{-1} - \frac{1}{[H^{-1}]_{pp}} H_{:,p}^{-1} H_{p,:}^{-1}\right)_{-p}$$
- 循环至达到目标稀疏度，每层完成后将稀疏输出经post-process传入下一层，触发新一轮自适应更新。

## 实验与结果
- **基线模型**：$\mathbf{BERT}_{base}$（110M参数，剪枝后60M/43M/11M）与 $\mathbf{BERT}_{large}$（330M参数，剪枝后155M）。
- **数据集**：SST-2（二分类情感）、IMDB（二分类情感）、AG News（四分类新闻）。
- **评估指标**：Clean Accuracy (Acc%)、Attack Accuracy (Aua%)、Attack Success Rate (Asr%)。
- **对抗攻击**：TextFooler（主实验）、BERT-Attack、TextBugger（附录扩展）。
- **主要结果（Table 1，sparsity=50%）**：
  - SST-2：Ours Aua%=43.1%，Asr%=51.2%，显著优于RobusT（Aua=24.8%，Asr=73.9%）与RMC（Aua=9.7%，Asr=89.3%）。
  - AG News：Ours Aua%=48.5%，Asr%=48.1%，优于所有基线。
  - IMDB：Ours Aua%=53.2%，Asr%=43.6%，在三种稀疏度下 consistently 最优。
- **高稀疏度优势更突出**：在87.5%稀疏度下，Ours仍保持Aua≈37-41%，而RobusT/Aucia低于10%，表明知识保持策略在高压缩比下尤为关键。
- **Ablation（Table 2）**：替换为IMP/LTH+FreeLB后鲁棒性大幅下降，验证Ada-Pruning组件不可替代。
- **结构化剪枝扩展（Table 5）**：在N:M结构化模式下同样优于EarlyRobust。
- **最强提升**：在IMDB 50%稀疏度下，Aua相对RobusT提升约+21.7个百分点，Asr相对RobusT降低约-22.1个百分点。

## 相关工作脉络
- **RobustT（Zheng et al., 2022）**：联合优化剪枝掩码与输入扰动，但重训练成本高且高稀疏度下鲁棒性骤降；本文在后训练框架下无需重训练即实现更优Trade-off。
- **Bag-of-Tricks（Xu et al., 2021）**：基于知识蒸馏与后训练量化的鲁棒压缩，依赖额外蒸馏损失；本文直接从嵌入/特征空间对齐角度出发，不引入蒸馏开销。
- **RMC（Du et al., 2023）**：利用样本难度防止稀疏模型过拟合简单样本；本文聚焦知识保留而非数据过滤，两者正交可互补。
- **SuperTicket（Liang et al., 2021）**：寻找超级掩码以保留偏置、降低方差；本文的自适应权重修正提供了更精细的层间误差传递机制。
- **EarlyRobust（Xi et al., 2022）**：鲁棒Early-Bird Ticket思路，需提前识别鲁棒子网；本文无需搜索阶段，一次性后训练剪枝即可。
- **OWD/OBS类Hessian剪枝（Frantar & Alistarh, 2022）**：提供单权剪枝的数学基础；本文将其扩展至层间自适应修正，解决累积误差传播问题。

## 局限性与未来方向
- **Hessian计算开销限制模型规模**：$O(d_{in}^3)$ 的复杂度使方法难以直接应用于数十亿参数的LLM，作者明确承认此局限。
- **校准数据量存在阈值效应**：超过一定数量后鲁棒性不再提升，但过少则Hessian估计不准（Figure 3），需精细调参。
- **仅面向Transformer线性层**：未覆盖LayerNorm、Attention权重等结构，结构化扩展需额外设计。
- **未来方向**：发展更高效的知识空间近似方法（如低秩Hessian近似、one-shot稀疏化），以扩展到百亿参数级别模型。

## 研究启发与可借鉴点
- **"鲁棒性←预训练知识保持"的归因视角**：可将此假设迁移至模型蒸馏、LoRA微调、量化等压缩场景，作为鲁棒性分析的统一解释框架。
- **层间自适应Hessian更新机制**：对任何层序网络（如CNN、Transformer）的多层压缩均有参考价值，尤其是误差累积敏感的场景。
- **Weight Averaging作为鲁棒初始化策略**：贪心权重平均思想可独立于剪枝使用，作为对抗鲁棒模型的高效构建手段。
- **后训练无重训练范式**：为LLM部署提供了一条低成本鲁棒化路径，特别适合API模型微调后的二次压缩需求。
- **特征空间距离可视化分析**：通过sentence embedding距离定量评估知识保留程度，可作为后续工作的通用诊断工具。

## 关键术语表
**Adversarial Robustness**：模型在面对恶意构造的对抗样本时保持预测正确性的能力。
**Post-training Pruning**：在模型完成训练（或微调）后，无需重新训练即可生成稀疏版本的压缩技术。
**Hessian Matrix**：损失函数二阶导数矩阵，反映参数扰动对损失的局部曲率，用于指导最优剪枝决策。
**Shortcut Learning**：模型倾向于利用数据集中的表面统计规律（虚假特征）而非深层语义进行预测的现象。
**Accuracy Under Attack (Aua%)**：模型在遭受对抗攻击后的预测准确率，衡量鲁棒性的核心指标。
**Attack Success Rate (Asr%)**：对抗攻击成功篡改模型预测的比例，越低表示防御效果越好。
**Weight Averaging / Model Soup**：将多个不同超参数训练得到的模型权重进行平均，以提升泛化与鲁棒性。
**Layer-wise Pruning**：逐层独立执行剪枝操作，每层完成后将输出作为下一层输入的梯度/误差来源进行修正。

## 可复现要素
- **数据集**：SST-2、IMDB、AG News（均为公开数据集）。
- **代码开源**：论文未提供代码仓库链接，附录含详细Algorithm与超参表（Table 8），可据此复现。
- **关键超参**：校准数据量 256–1024 句；正则化项 $\lambda = 1e{-}4$；稀疏度 30%/50%/87.5%；训练10个超参配置（Table 8）。
- **硬件**：单张 NVIDIA 3090 GPU。
- **模型**：$\mathbf{BERT}_{base}$、$\mathbf{BERT}_{large}$（HuggingFace公开权重）。
