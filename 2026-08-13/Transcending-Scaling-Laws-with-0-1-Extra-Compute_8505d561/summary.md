---
title: "Transcending-Scaling-Laws-with-0-1-Extra-Compute"
source: https://aclanthology.org/2023.emnlp-main.91.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:34:00"
field: "大语言模型预训练与缩放"
keywords: ["scaling laws", "continued training", "UL2", "mixture-of-denoisers", "emergent abilities", "infilling", "PrefixLM", "U-PaLM"]
innovations: ["提出 UL2R 方法，以 0.1% 额外计算显著改善大模型缩放曲线", "在 540B 规模实现约 2x 计算效率提升，节省 440 万 TPUv4 小时", "通过目标多样性在更小规模（62B/8B）上激发 BIG-Bench 涌现能力"]
benchmarks: ["BIG-Bench Emergent Suite", "MMLU", "SuperGLUE", "TydiQA", "GSM8K", "MGSM", "BBH"]
---

# 论文速读：Transcending-Scaling-Laws-with-0-1-Extra-Compute

## 一句话总结
论文提出 **UL2R（UL2 Restore）** 方法，通过在已有 PaLM 大语言模型基础上以仅约 0.1% 额外计算的代价继续训练 UL2 混合去噪器目标，显著改善模型缩放曲线并提升下游性能，同时能在更小模型规模上激发涌现能力。

## 研究问题与动机
- 当前大多数百亿级大语言模型（如 GPT-3、PaLM）几乎完全以左到右因果语言建模（causal LM）目标训练，缩放曲线存在性能瓶颈。
- 单纯扩大模型规模或增加训练数据会带来巨大计算成本，如何以更小代价持续改进模型成为关键问题。
- UL2 混合去噪器目标（mixture-of-denoisers）融合了前缀语言建模与 span 填充等多种预训练范式，能够为模型引入更强归纳偏置。
- 如何在已有成熟预训练模型基础上，以极低成本"修复/增强"缩放曲线，同时不引入新数据源，是本文试图回答的问题。

## 核心贡献（创新点）
1. **提出 UL2R 继续训练方法**：以仅 0.1%-1% 额外 FLOPs 在已有 PaLM 上追加训练 UL2 混合去噪器目标，无需新数据；与从头预训练或使用额外指令数据的适配方法本质不同。
2. **构建 U-PaLM 模型系列（8B/62B/540B）并证明约 2x 计算效率提升**：540B 规模下 U-PaLM 只需约一半计算预算即可达到最终 PaLM 540B 的性能（节省约 440 万 TPUv4 小时）；核心区别在于利用多样化目标替代单纯延长因果 LM 训练。
3. **证明 UL2R 可在更小规模上激发涌现能力**：部分 BIG-Bench 涌现任务从仅在 540B 才解锁，提前至 62B 甚至 8B 出现性能跃升；与纯规模扩展激发涌现的路径不同。
4. **赋予模型双向填充（infilling）新提示能力**：通过模式切换 token（如 `[S2S]`、`[NLU]`、`[NLG]`）可在推理时灵活访问不同预训练模式的知识，无需修改模型结构或解码算法。

## 方法详解
- **架构**：使用 PrefixLM（非因果 decoder-only）架构，序列总长 2048，拆分为 1024 输入 + 1024 目标；采用 prefix padding 优化策略（先拼接再 padding）以提升样本效率。
- **混合去噪器目标（Mixture-of-Denoisers）**：由三种 denoiser 组成，比例配置为 50% PrefixLM（S-denoiser）+ 25% 长 span 极端去噪（X-denoiser）+ 25% 标准 span 去噪（R-denoiser）。
  - **Regular denoising**：span 平均长度 3，腐败率 15%，模拟 T5 标准 span corruption。
  - **Extreme denoising**：span 平均长度 32 或腐败率最高 50%，引入"极端"噪声。
  - **Sequential denoising（PrefixLM）**：噪声始终从文本起始采样到随机位置，等价于非因果前缀语言建模。
- **模式 token**：`[S2S]` 对应 S-denoiser（PrefixLM），`[NLU]` 对应 R-denoiser，`[NLG]` 对应 X-denoiser，推理时可通过 prepend 控制激活模式。
- **继续训练设置**：以 PaLM checkpoint 为初始化，追加训练 20k 步、batch size=32，额外 token 数约 13 亿（占原始 780B 的 0.16%）；学习率余弦衰减从 $10^{-4}$ 降至 $10^{-6}$（低恒定 lr 也表现相近）。
- **Sentinel tokens**：沿用 T5/UL2 做法，将最后 100 个 subword 作为 sentinel 替换被 corruption 的 span。

## 实验与结果
- **数据集**：沿用 PaLM 原始预训练语料（Common Crawl、Wikipedia、Books、Stories 等），未引入新数据源；下游评测涵盖 26 个零/少样本 NLP 任务、BIG-Bench 涌现集、MMLU、SuperGLUE、TydiQA、GSM8K、MGSM、BBH 等。
- **基线**：PaLM 8B / 62B / 540B、Gopher、Chinchilla、GPT-3、Minerva 540B 等。
- **核心结果**：
  - **缩放曲线改进**：540B 规模下 U-PaLM 约 2x 计算效率（达到同等性能仅需约一半 FLOPs）。
  - **NLP 基准**：U-PaLM 540B 在 26 个任务中 21 项取得 SOTA。
  - **MMLU 5-shot**：70.7%（相对 PaLM 540B 的 69.3% 提升 +2.0%）。
  - **BIG-Bench 涌现集（21 任务）**：平均得分 67.7% vs PaLM 64.3%（+5.3% 相对提升）；navigate 从 55.3%→67.0%（+21.2%），snarks 从 69.1%→86.1%（+24.6%），physics_questions 从 7.6%→12.5%（+64.5%）。
  - **涌现能力提前**：crass_ai、vitaminc、identify_odd_metaphors 等任务在 62B 即出现性能跃升；部分 8B U-PaLM 超越 PaLM 62B（如 snarks、understanding_fables）。
  - **微调**：SuperGLUE 8B 从 83.4→86.1（+3.2%），TydiQA 8B EM/F1 从 75.7/85.2→77.5/86.7。
  - **多语言/推理**：MGSM 540B 从 45.9→49.9（+8.7%），GSM8K 540B CoT 从 54.9→58.5（+6.6%），BBH 从 44.8→49.6（+10.7%）。
- **最强结果**：540B 规模 2x 计算效率提升；BIG-Bench 涌现任务物理问题 +64.5% 提升；MMLU 相对 +2.0%。

## 相关工作脉络
- **Scaling Laws（Kaplan et al., 2020; Hoffmann et al., 2022）**：预测模型质量随计算预算持续增长；本文聚焦下游 few-shot 性能的缩放曲线改进而非仅 upstream cross-entropy。
- **PaLM（Chowdhery et al., 2022）**：百亿级因果 LM 基座；本文在其 checkpoint 上追加训练，展示"续训优于重新扩展计算"。
- **UL2（Tay et al., 2022b）**：提出混合去噪器统一生成与理解目标；本文将其思想迁移至已有 causal LM 的继续训练（UL2R）。
- **Emergent Abilities（Wei et al., 2022a）**：定义大规模模型才解锁的新行为；本文证明通过目标多样性而非纯规模扩展同样可触发涌现。
- **MLM↔CLM 互相适配（Wang et al., 2022a）**：探讨因果与掩码目标的相互适应；本文扩展为多样化混合目标并验证大规模场景下的显著收益。
- **Prompt Tuning/Finetuning（Lester et al., 2021; Ouyang et al., 2022）**：依赖额外参数或人类标注数据；UL2R 无需新数据与参数修改，仅通过继续训练+模式提示实现能力提升。

## 局限性与未来方向
- 仅在 PaLM 及其预训练语料上验证，未探索应用于更弱基础模型或已训练至饱和的模型时的效果。
- 未研究小于 8B 参数规模的模型表现。
- 未系统分析不同续训阶段（checkpoint 选择）对 UL2R 收益的影响规律。
- 未来方向：探索更小模型上的 UL2R、研究最佳续训时机与比例、扩展到非 English 或多模态场景。

## 研究启发与可借鉴点
- **极低成本续训策略**：0.1% 额外计算即可获得显著缩放曲线改善，为既有大模型"二次开发"提供了高 ROI 范式。
- **混合目标替代纯因果 LM**：在已有 causal LM 上引入 PrefixLM/span corruption 可作为通用正则化/能力增强手段，适用于其他百亿级基座。
- **涌现能力的低尺度激发**：证明归纳偏置（inductive bias）多样性可提前解锁涌现，为计算受限场景下追求复杂能力提供新思路。
- **模式提示（mode prompting）的工程价值**：无需修改模型即可通过 `[S2S]`/`[NLU]`/`[NLG]` 切换推理行为，对开放域生成与多任务适配具有实用意义。

## 关键术语表
**UL2R (UL2 Restore)**：在已有大语言模型基础上以少量额外计算继续训练 UL2 混合去噪器目标的方法。
**PrefixLM**：前缀语言建模架构，对输入部分使用双向注意力、对输出部分保持因果掩码，融合理解与生成能力。
**Mixture-of-Denoisers**：UL2 提出的多种去噪目标混合训练策略，包含 Regular、Extreme、Sequential 三类 span corruption。
**Infilling**：模型根据上下文填充输入中间空白片段的能力，区别于传统左到右续写。
**Emergent Abilities**：随模型规模增长而突然出现的、小模型不具备的新能力。
**Scaling Laws**：描述模型性能随计算量/参数量/数据量变化的经验规律。
**S2S / NLU / NLG**：UL2 的模式切换 token，分别对应序列到序列（PrefixLM）、自然语言理解（标准去噪）、自然语言生成（极端去噪）。
**Span Corruption**：将文本中连续片段替换为 sentinel token 的预训练任务，源自 T5。

## 可复现要素
- **数据集**：使用 PaLM 原始预训练语料（Common Crawl、Wikipedia、Books、Stories 等），未引入额外数据；具体下游评测包括 SuperGLUE、TydiQA、MMLU、BIG-Bench、GSM8K、MGSM、BBH 等（多数为公开 benchmark）。
- **代码/权重**：论文未提及开源代码或模型权重。
- **关键超参**：序列长度 2048（输入 1024 + 目标 1024）；续训步数 20k（8B/62B 为 50k steps, batch size 128）；batch size 32；学习率余弦衰减 $10^{-4} \rightarrow 10^{-6}$；denoiser 比例 50% S / 25% R / 25% X；额外 token 约 13 亿（占原始 780B 的 0.16%）。
