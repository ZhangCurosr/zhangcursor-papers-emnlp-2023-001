---
title: "Improving-Chinese-Pop-Song-and-Hokkien-Gezi-Opera-Singing-Vo"
source: https://aclanthology.org/2023.emnlp-main.200.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:46:58"
field: "歌唱语音合成"
keywords: ["Singing Voice Synthesis", "Local Attention", "Adaptive Loss", "Chinese Pop Song", "Hokkien Gezi Opera"]
innovations: ["最近邻局部注意力机制增强音素级局部建模", "音素级局部自适应权重损失动态优化难合成区域"]
benchmarks: ["PopCS", "Gezi Opera"]
---

# 论文速读：Improving-Chinese-Pop-Song-and-Hokkien-Gezi-Opera-Singing-Voice-Synthesis-by-Enhancing-Local-Modeling

## 一句话总结
本文针对Transformer声学模型在歌唱语音合成（SVS）中出现的局部不协调问题（如发音抖动、噪音），提出两种局部建模增强方法：最近邻局部注意力机制和音素级局部自适应权重损失函数，在中文流行歌曲和闽南语歌仔戏数据集上均取得客观与主观评测的最佳结果。

## 研究问题与动机
- **核心问题**：现有基于Transformer的SVS声学模型使用全局自注意力处理整个序列，导致局部注意力分散，合成音频出现局部发音抖动或噪音等"局部不协调"现象。
- **动机1**：可视化全局自注意力矩阵发现，虽然注意力主要集中于相邻音素区域，但仍存在部分音素关注远距离区域的现象，说明局部注意力不足。
- **动机2**：传统L1损失对mel频谱图各部分等权重优化，难以充分优化难以合成的局部区域。
- **动机3**：歌手在实际表演中通常专注于当前正在演唱的词语而非同时关注整首歌词，因此局部注意力更符合演唱规律。

## 核心贡献（创新点）
- **最近邻局部注意力机制**：在解码器中添加仅关注当前音素前后相邻音素的局部注意力层，通过门控单元（gated unit）与全局自注意力表示融合；与已有工作相比，本文专门针对SVS任务设计了音素级的局部注意力掩码，而非通用的序列局部建模。
- **音素级局部自适应权重损失**：基于预测与真实mel频谱图在音素区域的差异计算自适应置信度，再通过softmax归一化得到动态权重替代传统L1损失；与已有工作相比，本文首次将此类思路应用于SVS的音素级局部优化。
- **广泛的实验验证**：在公开中文流行歌曲数据集（PopCS）和自建闽南语歌仔戏数据集上验证了方法的有效性和通用性，目标与主观评测均优于强基线。

## 方法详解
- **模型架构**：基于FastSpeech2，编码器为Transformer block，解码器为Conformer block（卷积增强的Transformer），配合长度调节器将音素级序列扩展为帧级序列。
- **最近邻局部注意力**：
  - 构建音素级局部注意力掩码矩阵 $M$，仅对当前音素前后 $l$ 个和 $r$ 个音素位置置0，其余为 $-\infty$。
  - 局部表示：$R_l = \mathrm{softmax}(M + \frac{QK^\top}{\sqrt{d_k}})V$
  - 全局表示：$R_g = \mathrm{softmax}(\frac{QK^\top}{\sqrt{d_k}})V$
  - 门控融合：$\alpha = \mathrm{sigmoid}(W([R_l; R_g]))$，$R_f = \alpha R_l + (1-\alpha)R_g$，其中 $\alpha \in [0,1]$ 为可学习系数。
- **音素级局部自适应权重损失**：
  - 自适应置信度：$m_k = \mathrm{Ave}(|M_p'(i,j) - M_p(i,j)|)$，其中 $M_p'$ 为预测mel频谱，$M_p$ 为真实mel频谱，对音素 $k$ 对应区域取平均。
  - 自适应权重：$\omega_k = \frac{e^{m_k}}{\sum_{z=1}^{n} e^{m_z}}$（softmax归一化）
  - 最终损失：$F = \frac{1}{MN}\sum_{i=0}^{M-1}\sum_{j=0}^{N-1} \omega_k |M_p'(i,j) - M_p(i,j)|$，其中 $M$ 为帧数，$N$ 为mel bin数。

## 实验与结果
- **数据集**：PopCS（约5.89小时中文流行歌曲，女歌手，24kHz）；Gezi Opera（约4.5小时闽南语歌仔戏，5位专业歌手，48kHz）。
- **评估指标**：客观指标MCD（倒谱畸变）、MSD（频谱畸变）、GPE（基频错误率）、VDE（清浊音决策错误率）、FFE（基频帧错误率）；主观指标MOS（意见得分，1-5分）。
- **主要结果（PopCS）**：Baseline-T+C+A+L模型取得最优MCD=2.8735 dB、GPE=0.65%、FFE=3.67%，MOS=3.71±0.11，均优于基线及其他对比方法。
- **主要结果（Gezi Opera）**：最佳模型MCD=2.931 dB、GPE=1.34%、FFE=4.57%，MOS=3.61±0.12。
- **参数选择**：最佳局部注意力范围为 $l=1, r=1$（关注前后各一个音素），兼顾MCD与FFE、MOS表现。
- **泛化性**：方法可灵活应用于DiffSinger（扩散模型SVS），在PopCS和Gezi Opera上分别提升MOS 0.05和0.04分。
- **GAN基线表现**：N-Singer和Baseline+GAN在客观指标上GPE/FFE较高，主观上效果不及本文方法，说明单纯GAN训练无法有效解决局部不协调问题。

## 相关工作脉络
- **FFT-Singer（Liu et al., 2022）**：本文基线模型，基于FastSpeech2的Transformer SVS系统，使用全局自注意力和L1损失。
- **N-Singer（Lee et al., 2021）**：针对韩语SVS局部不协调问题，采用postnet和后处理策略结合声带感知判别器；本文不使用后处理网络或对抗训练，从注意力与损失函数角度解决。
- **HiFiSinger（Chen et al., 2020）**：采用GAN对抗训练的SVS模型；本文通过实验表明GAN方法在基频准确性上存在不足。
- **DiffSinger（Liu et al., 2022）**：基于扩散概率模型的SVS系统；本文方法可灵活融入DiffSinger的声学模型条件输入中。
- **TTS局部建模增强（Yang et al., 2020；Watzel et al., 2021）**：在语音合成和语音识别中增强局部相对位置感知或诱导局部注意力；本文将这些思路迁移至SVS任务并针对音素级局部优化进行设计。

## 局限性与未来方向
- 最近邻局部注意力仅验证了与全局自注意力的融合方式，未探索其他局部表示的独立使用。
- 局部注意力引入额外的计算开销，增加了GPU资源需求。
- 方法不能完全解决局部不协调问题，仍有改进空间。
- **未来方向**：通过损失函数进一步控制mel频谱图中高、中、低频段的局部特征。

## 研究启发与可借鉴点
- **注意力机制设计**：将全局自注意力与局部注意力通过门控单元融合的思路，可迁移至其他序列生成任务（如TTS、ASR）中缓解局部不一致问题。
- **动态加权损失**：基于预测误差的动态权重损失设计可用于需要重点优化困难区域的任务，避免等权重优化导致的难点被忽略。
- **实验设计**：通过消融实验（Baseline→+C→+A→+L→+A+L）清晰验证各模块贡献；参数选择同时考虑主客观指标而非仅依赖单一目标函数。
- **跨模型兼容性**：方法仅依赖全局自注意力和L1损失的前提，可与DiffSinger等不同架构结合，体现了良好的通用性。

## 关键术语表
- **Singing Voice Synthesis (SVS)**：根据乐谱和歌词合成自然逼真的人声歌唱的语音合成任务。
- **Mel-spectrogram**：梅尔频谱图，将音频转换为对humans听觉感知更友好的时频表示。
- **Local incongruity**：局部不协调，指合成音频中发音抖动或噪音等局部质量问题。
- **Nearest neighbor local attention**：最近邻局部注意力，仅关注当前音素前后相邻音素的注意力机制。
- **Local adaptive weights loss**：局部自适应权重损失，根据音素区域预测误差动态调整优化权重的损失函数。
- **Conformer**：卷积增强Transformer，结合CNN和Self-Attention的序列建模架构。
- **GPE/VDE/FFE**：基频错误率/清浊音决策错误率/基频帧错误率，用于评估合成音频基频轨迹准确性的客观指标。
- **MOS**：Mean Opinion Score，主观意见得分，用于评估音频自然度和质量的1-5分量表。

## 可复现要素
- **数据集**：PopCS公开数据集；Gezi Opera数据集由作者团队自建（论文未明确说明是否公开）。
- **代码**：已开源，地址 https://github.com/baipeng1/SVSELM。
- **权重**：论文未提及预训练权重是否公开。
- **关键超参**：采样率24kHz，FFT窗口512，hop长度128，mel bins 80，表示维度256，attention head数4，encoder/decoder各4层block，训练步数160k，单卡A40 GPU，AdamW优化器。
