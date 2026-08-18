---
title: "Query-as-context-Pre-training-for-Dense-Passage-Retrieval"
source: https://aclanthology.org/2023.emnlp-main.118.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:30:06"
---

# 论文速读：Query-as-context-Pre-training-for-Dense-Passage-Retrieval

## 一句话总结
本文针对稠密段落检索中上下文监督预训练依赖弱相关同文档段落对的问题，提出了一种基于查询生成（Query-as-context）的预训练技术，通过将段落与其生成的预测查询配对进行对比或生成式预训练，在显著提升检索性能的同时加快了训练速度。

## 研究问题与动机
- **弱相关正样本噪声被忽视**：现有上下文监督预训练（如coCondenser、CoT-MAE）默认同一文档内的任意两个段落具有强相关性并构建为正样本对，但作者的人工标注统计显示，训练数据中仅有35.5%的段落对具有高相关性，大量弱相关或无关对会对预训练产生负面影响。
- **Query Prediction在稠密预训练中的空白**：查询生成（Query Prediction）技术已在稀疏检索（如docT5query）中验证有效，能显著缓解词项不匹配问题，但尚未被探索用于稠密检索的预训练阶段。
- **生成查询具有更高的语义对齐度**：统计表明，由段落生成的预测查询与原文段落的强相关性比例达到56.6%，显著高于随机抽取的同文档段落对，证明生成查询是更可靠的上下文信号。
- **效率与分布对齐优势**：段落-查询对的序列长度通常短于段落-段落对，可降低预训练的token处理量与时间开销；同时，预训练与下游微调均使用段落-查询对，有助于缩小两者之间的分布差距。

## 核心贡献（创新点）
- **揭示并量化了上下文预训练中的弱相关段落对噪声问题**。与以往默认同文档段落必然相关的假设不同，本文首次通过人工标注提供了统计证据，明确了预训练数据构造中语义对齐度的瓶颈。
- **提出Query-as-context预训练方法，用生成查询替代随机段落构建正样本对**。与现有方法仅依赖文档结构共现的本质区别在于，本文引入生成模型为每个段落定制高相关性的语义上下文，从根本上提升了预训练正样本的质量。
- **实现了该技术在对比特定（coCondenser）与生成式特定（CoT-MAE）两类预训练框架中的统一适配**。与单一优化某类预训练损失的工作不同，本文证明了该数据构造策略对不同预训练范式的通用迁移能力。
- **以纯预训练改进刷新了单向量稠密检索器的SOTA性能并提升训练效率**。区别于依赖重排序蒸馏、多向量表示或复杂数据增强的SOTA方法，本文仅通过优化预训练样本相关性即实现MS-MARCO MRR@10达40.2的新记录，且将CoT-MAE预训练步数从1200k降至50k。

## 方法详解
- **预训练数据构造**：从MS-MARCO语料（约320万文档）中按句子切分提取段落（最大长度144 tokens）。对每个段落x_i，使用微调过的T5模型结合nucleus sampling（top_p=0.95, top_k=25）生成C个候选查询。训练时随机采样一个查询y_i与段落组成训练对{x_i, y_i}，替代原有同文档段落对。
- **对比式预训练（适配coCondenser）**：将段落与生成的查询分别输入共享编码器，取最后一层[CLS]位置的隐藏状态h_x和h_y作为向量表示。构建mini-batch后，通过InfoNCE对比损失拉近正样本对、推远批次内负样本：`L_co = -log(exp(sim(h_x, h_y)/τ) / Σ exp(sim(h_x, h')/τ))`。同时保留辅助解码器的MLM损失，总损失为`L = L_mlm + L_mlm^aux + L_co`。
- **生成式预训练（适配CoT-MAE）**：采用非对称编码器-解码器架构。段落x输入深层编码器，查询y输入浅层解码器。解码器将段落句子表征h_x与掩码查询词表征拼接为初始输入，经过N层Transformer后进行上下文感知的语言建模，损失为`L_ctx_mlm`。编码器侧保持标准MLM损失，总损失为`L = L_mlm + L_ctx_mlm`。
- **两阶段微调**：预训练结束后丢弃辅助模块/解码器，使用标准双编码器结构在MS-MARCO上进行两阶段微调。第一阶段用BM25挖掘hard negatives，第二阶段用第一阶段训练好的检索器挖掘hard negatives，两阶段均基于InfoNCE损失优化。

## 实验与结果
- **数据集与基线**：在MS-MARCO Passage Ranking、TREC DL 2019/2020及BEIR跨域零样本基准（14个公开子集）上评测。稀疏基线包含BM25、DeepCT、docT5query、GAR；稠密基线涵盖ANCE、SEED、TAS-B、COIL、ColBERT、Condenser、RocketQA、PAIR、SimLM、RetroMAE、LED及本文对比的coCondenser与CoT-MAE。
- **MS-MARCO与TREC结果**：coCondenser + query-as-context在MS-MARCO MRR@10提升0.6pp至39.4，TREC DL 19/20 NDCG@10分别提升2pp和3.4pp。CoT-MAE + query-as-context在MS-MARCO MRR@10大幅提升1.4pp至40.2（相比原版1200k模型仍提升0.8pp），刷新了无需重排序蒸馏的单向量预训练稠密检索器SOTA；TREC DL 19/20分别提升0.8pp和3pp。Retriever 1与Retriever 2的所有核心指标（MRR@10、R@50、R@1k）均同步提升。
- **跨域泛化（BEIR）**：coCondenser在9个数据集提升，CoT-MAE
