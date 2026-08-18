---
title: "Compressing-and-Debiasing-Vision-Language-Pre-Trained-Models"
source: https://aclanthology.org/2023.emnlp-main.34.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:10:21"
field: "视觉-语言预训练模型压缩与鲁棒性"
keywords: ["视觉问答", "去偏", "模型压缩", "VLP", "OOD泛化", "稀疏子网", "lxmert"]
innovations: ["首次联合研究VLP压缩与去偏，证明稀疏子网可同步提升效率与OOD鲁棒性", "提出两阶段训练压缩流水线，发现不同去偏方法在各阶段应差异化使用", "提出模态特定稀疏度分配策略，语言/跨模态模块高压缩、视觉模块低压缩"]
benchmarks: ["VQA-CP v2", "VQA-VS"]
---

# 论文速读：Compressing and Debiasing Vision-Language Pre-Trained Models for Visual Question Answering

## 一句话总结
本文首次系统研究了视觉-语言预训练模型（VLP）在视觉问答（VQA）任务中**同时实现压缩与去偏**的可能性，通过设计两阶段训练压缩流水线并结合去偏目标，搜索出稀疏且对分布外（OOD）数据更鲁棒的子网络，在更少参数下取得了优于现有去偏方法的性能。

## 研究问题与动机
1. **数据集偏差导致OOD泛化失败**：VQA模型倾向于利用训练集中的语言偏置（language bias），在答案分布不同的OOD测试集上性能急剧下降（如lxmert在VQA-CP v2上ID到OOD准确率下降23.26%）。
2. **VLP参数量大导致部署低效**：尽管大规模VLP在ID基准上取得提升，但参数量庞大（如lxmert约202M），增加计算与存储开销。
3. **现有工作割裂处理压缩与去偏**：去偏方法主要基于小规模模型设计，压缩工作集中在NLP领域的PLM压缩，两者均未联合考虑。
4. **缺乏系统性探索VLP压缩与OOD鲁棒性的关系**：压缩是否可能去除偏置相关参数、提升鲁棒性，以及何时何模块适合更高压缩率等问题尚未被系统研究。

## 核心贡献（创新点）
1. **首次系统研究VLP稀疏性与OOD鲁棒性的联合优化**：揭示了VLP中存在比全模型更鲁棒的稀疏子网，证明压缩与去偏可同步实现。
2. **提出两阶段训练压缩流水线并整合去偏目标**：设计了"全模型微调→压缩（掩码训练或OMP）→可选进一步微调"的三段式流程，强调在各阶段引入去偏损失的重要性。
3. **发现不同去偏方法在流水线各阶段应差异化使用**：若两种去偏方法对全模型有效，则应在Stage1使用较差的方法、Stage2使用较好的方法（如"lpf+lmh"优于"lmh+lmh"）。
4. **提出模态特定稀疏度分配策略**：语言模块和跨模态模块应被更高度压缩（如sL=60%, sX=60%, sR=4%），视觉模块应保留更多参数，超越均匀稀疏度方案。

## 方法详解
**整体框架**：以lxmert为基础VLP（也可扩展到visualBERT、mPLUG），目标是寻找满足目标稀疏度s的稀疏子网络$ f(\mathbf{m} \odot \pmb{\theta}_{ft}) $，最大化OOD评估指标：
$$\max_{\mathbf{m}, \pmb{\theta}_{ft}} \left( \mathcal{E}_{\mathrm{OOD}} \left( f(\mathbf{m} \odot \pmb{\theta}_{ft}) \right) \right), \quad \text{s.t.} \quad \frac{\|\mathbf{m}\|_0}{|\pmb{\theta}_{pr}|} = (1-s)$$

**剪枝方法**：
- **OMP（One-shot Magnitude Pruning）**：基于权重绝对值进行一次性剪枝，可对剪枝后子网进行进一步微调恢复性能。
- **Mask Training**：直接优化二值掩码$\mathbf{m}$，引入实值掩码$\hat{\mathbf{m}}$并通过Binarization函数得到二值掩码，使用Straight-Through Estimator估计梯度。

**去偏方法**：
- **LMH（Learned-Mixin+H）**：引入偏置模型$p_b$，通过$g(h) = \text{softplus}(w \cdot h)$学习可信度，混合主模型与偏置模型的预测分布，并添加熵惩罚项$R$防止$p_b$被忽略。
- **LPF（Language-Prior Feedback）**：以$\alpha_k = p_b[a_k]$衡量样本偏置程度，用$(1-\alpha_k)^\gamma$加权交叉熵损失。
- **RUBi**：采用$\hat{p}_{deb} = \text{softmax}(p_m \cdot \delta(p_b))$进行去偏。

**两阶段流水线设计**：
- **Stage 1**：使用去偏损失$\mathcal{L}$对预训练lxmert进行微调，得到$f(\pmb{\theta}_{ft})$。
- **Stage 2**：对微调后的模型进行压缩（Mask Training或OMP），输出掩码$\mathbf{m}$。
- **Stage 3（可选）**：对子网进行进一步微调，建议使用与Stage 2不同的去偏损失。

**模态特定稀疏度约束**：将模型分为语言模块$\theta_{Lan}$、视觉模块$\theta_{Vis}$、跨模态模块$\theta_{Xenc}$，分别指定不同稀疏度$s_L, s_R, s_X$，满足加权平均等于目标稀疏度$s$。

## 实验与结果
**数据集与模型**：
- **VQA-CP v2**：标准OOD基准，考察问句类型偏置。
- **VQA-VS**：更具挑战性的基准，包含关键词、关键对象等多种偏置类型。
- 基础模型：lxmert-base-uncased（~202M参数）、visualBERT、mPLUG-base（~350M参数）。
- 实验结果基于4个随机种子取平均。

**关键结果（VQA-CP v2）**：
| 方法 | 参数量 | All | Y/N | Num | Other |
|------|--------|-----|-----|-----|-------|
| 全lxmert(bce) | 202M | 48.01 | 48.24 | 20.04 | 55.57 |
| 全lxmert(lmh) | 202M | 63.55 | 81.84 | 55.00 | 56.32 |
| lp f+lmh (10%) | 24M | 59.05 | 75.08 | - | 51.17 |
| **lp f+lmh (30%)** | **64M** | **64.02** | **79.99** | **57.12** | - |
| **lp f+lmh (50%)** | **103M** | **66.07** | **84.70** | **63.38** | **56.35** |

- **30% lxmert** 超越所有非VLP去偏SoTA（如LPF-UpDn: 55.34, LMH+MMBS: 56.44, CGE: 57.32），参数相似或更少。
- **50% lxmert** 超越去偏全lxmert(lmh)（66.07 vs 63.55），参数仅为前者的51%。
- 最佳配置（modality-specific sparsity）：50%整体稀疏度时，sL=60%, sR=4%, sX=60%。

**关键结果（VQA-VS）**：
- "bce+bce" 70% lxmert在ID和OOD-mean上均优于多数基线，ID: 71.85, OOD-mean: 53.51。
- "lmh+lmh" 90% lxmert达到ID: 71.97, OOD-mean: 54.75，OOD优势明显。
- 模态特定稀疏度在50%~70%整体稀疏度下持续优于均匀稀疏度。

**提升幅度**：
- 相比全lxmert(bce)，mask train(lmh)子网在70%稀疏度下提升14.02%（VQA-CP v2 All准确率从48.01到62.03+）。
- 相比去偏SoTA方法（非VLP），30% lxmert提升约5.79个百分点。

## 相关工作脉络
1. **去偏VQA基线**：LMH (Clark et al., 2019)、RUBi (Cadene et al., 2019)、LPF (Liang et al., 2021b) 等基于小模型（S-MRL、UpDn）的去偏方法，本文将其扩展至VLP并联合压缩。
2. **模型压缩**：LMH、RUBi等方法未考虑压缩；NLP领域BERT压缩（如Chen et al., 2020b的Lottery Ticket、Prasanna et al., 2020的Super Tickets）主要关注精度保持，本文将其引入VQA OOD鲁棒性场景。
3. **VQA压缩**：Gan et al. (2022) 的Playing Lottery Tickets with VLPs仅研究压缩，未涉及去偏；本文首次联合两者。
4. **鲁棒性vs压缩**：Sehwag et al. (2020) 的HYDRA、Fu et al. (2021) 等研究对抗鲁棒性与压缩的关系；本文关注的是数据集偏置导致的OOD鲁棒性。
5. **多模态预训练模型**：OFA、Florence等SOTA VLPs专注于ID性能；本文关注其OOF泛化与部署效率的平衡。

## 局限性与未来方向
1. **最优稀疏度分配策略缺乏自动求解方法**：论文依赖手动搜索找到最佳模态特定稀疏度配置，未提供高效自动优化方案。
2. **极端高稀疏度（90%）下表现下降**：当稀疏度过高时，子网推理能力受损，尤其是"其他"类问题（需要更强推理）。
3. **仅考察语言偏置**：LMH等去偏方法主要针对语言偏置，对视觉偏置（如OOD-vis）改善有限（VQA-VS实验中OOD-vis提升不明显）。
4. **扩展到生成式VLP的挑战**：如mPLUG以文本生成方式处理VQA，LMH和RUBi无法直接应用，需适配新去偏方法。

## 研究启发与可借鉴点
1. **去偏目标贯穿压缩全流程**：即使在Stage 1已完成去偏微调，Stage 2的掩码训练中仍应继续使用去偏损失（如"lmh+lmh"或"bce+lmh"），这比单纯使用BCE更有效。
2. **差异化使用去偏方法可提升性能**："lpf+lmh"优于"lmh+lmh"，表明不同去偏目标在微调与压缩阶段的角色互补性——较差方法负责"热身"全模型，较好方法负责精细子网搜索。
3. **模态特定稀疏度分配是提升鲁棒性的关键**：语言模块最易压缩（捕获更多偏置），视觉模块最难压缩（承载关键视觉推理），这一发现可迁移至其他多模态模型的压缩设计。
4. **去偏-压缩联合搜索可作为通用范式**：本文证明压缩可以去除偏置相关参数，这一思路可推广至其他具备偏置问题的多模态任务（如图像描述、视觉推理）。

## 关键术语表
- **VLP（Vision-Language Pre-trained Model）**：视觉-语言预训练模型，如lxmert、visualBERT、mPLUG等，通过大规模预训练学习跨模态表示。
- **OOD（Out-of-Distribution）**：分布外，指测试数据的分布与训练数据不同，常由数据集偏置导致，用于评估模型泛化能力。
- **VQA-CP v2**：Visual Question Answering - Counterfactual Prototype v2，通过重采样改变答案分布构建的OOD基准，主要考察语言偏置。
- **VQA-VS**：Visual Question Answering - Visual Shortcut，包含多种偏置类型（关键词、关键对象等）的挑战性OOD基准。
- **LMH（Learned-Mixin+H）**：一种去偏方法，通过学习函数$g(h)$动态混合主模型与偏置模型的预测，减少语言偏置影响。
- **LPF（Language-Prior Feedback）**：另一种去偏方法，以偏置模型预测概率作为样本权重，降低高偏置样本的-loss贡献。
- **Mask Training**：一种剪枝方法，直接优化二值掩码$\mathbf{m}$而非权重，通过Straight-Through Estimator处理不可微的二值化操作。
- **Modality-Specific Sparsity**：模态特定稀疏度，指对不同模态模块（语言/视觉/跨模态）分配不同压缩率，以实现性能-效率的最优平衡。

## 可复现要素
- **数据集**：VQA-CP v2、VQA-VS（论文引用Si et al., 2022b，VQA-VS可能未公开或需额外申请）。
- **代码/权重**：基于huggingface transformers库实现，lxmert-base-uncased从HuggingFace获取；visualBERT使用coco-pre版本。**论文未明确说明代码是否开源**。
- **关键超参**：学习率5e-5、batch size 128（V100）/256（A100）、训练20 epoch、mask训练阈值$\phi=0.01$、稀疏度搜索步长20%/5%/2%。
