---
title: "BioT5-Enriching-Cross-modal-Integration-in-Biology-with-Chem"
source: https://aclanthology.org/2023.emnlp-main.70.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:07:32"
field: "生物信息学与计算生物学"
keywords: ["多模态预训练", "分子表征", "蛋白质语言模型", "SELFIES", "跨模态整合", "药物发现"]
innovations: ["使用SELFIES替代SMILES实现100%有效分子生成", "分离式词汇表避免跨模态语义冲突", "六任务多模态预训练框架整合文本-分子-蛋白质知识"]
benchmarks: ["MoleculeNet", "PEER", "BioSNAP", "BindingDB", "ChEBI-20"]
---

# 论文速读：BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations

## 一句话总结
本文提出BioT5，一个融合文本、分子（SELFIES）和蛋白质（FASTA）的多模态预训练框架，通过化学知识与自然语言关联的跨模态整合，在15项下游生物学任务中取得10项SOTA性能。

## 研究问题与动机
1. **SMILES生成无效分子问题**：现有方法多依赖SMILES表示分子，但深度学习生成的SMILES常出现化学结构非法问题（如价键错误）。
2. **上下文信息利用不足**：生物实体（分子/蛋白质名称）周围的科学文献语境蕴含交互关系与属性信息，但现有模型未有效提取。
3. **结构化与非结构化知识同等对待**：数据库中的结构化数据（如PubChem分子描述）与文献中的非结构化文本被同等处理，未能充分发挥结构化知识的价值。
4. **多模态语义空间混淆**：共用词典（如原始T5词表）导致跨模态token语义混淆（如"C"同时代表字母C、碳原子、半胱氨酸）。

## 核心贡献（创新点）
1. **引入SELFIES替代SMILES实现100%有效分子生成**：与MolT5/MolXPT等使用SMILES的方法本质不同，SELFIES保证任意字符排列均对应合法分子结构，彻底解决生成无效分子问题。
2. **分离式词汇表与生物特异性标记化**：首次为分子（SELFIES）、蛋白质（FASTA）和文本分别建立独立词表，避免跨模态语义冲突（如"Br"溴原子被错误拆分为硼原子"B"和字母"r"）。
3. **六任务多模态预训练框架**：设计三类预训练任务（单模态掩码重建、包装文本重建、结构文本双向翻译），系统性建模文本-分子-蛋白质三元模态关联。
4. **区分结构化与非结构化知识利用策略**：结构化数据（PubChem/Swiss-Prot对）采用翻译任务，非结构化数据（PubMed包装文本）采用掩码重建任务，差异化处理提升信息利用率。

## 方法详解
**数据收集与处理**：
- **单模态数据**：ZINC20（SMILES转SELFIES）、Uniref50采样27M蛋白质FASTA、C4通用文本。
- **包装文本（Wrapped Text）**：使用BERN2对33M PubMed文章进行命名实体识别与链接，将分子名替换为SELFIES、基因名附加FASTA序列。
- **结构化对**：339K分子SELFIES-描述对（来自PubChem，排除ChEBI-20测试集分子防泄露）、569K蛋白质FASTA-描述对（来自Swiss-Prot）。

**标记化设计**：
- 分子：SELFIES token按化学基团分词，如`[C][=C][Br]`保持完整基团。
- 蛋白质：氨基酸前加特殊前缀`<p>`，如`<p>M<p>K<p>R`区分文本字母。
- 文本：保留原始T5 SentencePiece词典（35,073词表）。

**预训练任务（T5目标函数）**：
1. **Task #1-3（单模态掩码重建）**：分别对SELFIES、FASTA、通用文本应用标准T5掩码目标。
2. **Task #4（包装文本掩码重建）**：对包含生物实体的包装文本应用T5目标，同时掩码文本、FASTA、SELFIES token。
3. **Task #5-6（双向翻译）**：结构化分子-文本对与蛋白质-文本对进行双向翻译（SELFIES↔文本、FASTA↔文本），使用特殊token锚定（如`<molecule_name>`、`<description>`）。

**微调策略**：采用prompt-based微调，统一各类下游任务为序列生成格式，缩小预训练-微调差距。二分类任务通过归一化Yes/No token概率获取软标签分布：$P(positive) = \frac{p_{pos}}{p_{pos} + p_{neg}}$。

## 实验与结果
**评估任务**：15项下游任务，涵盖分子属性预测、蛋白质属性预测、药物-靶标相互作用（DTI）、蛋白质-蛋白质相互作用（PPI）、分子描述生成、文本生成分子。

**分子属性预测（MoleculeNet，AUROC）**：
- BioT5（252M参数）在BBBP（77.7）、Tox21（77.9）、ClinTox（95.4）、HIV（81.0）、BACE（89.4）、SIDER（73.2）六任务平均82.4，超越MolXPT（81.9）和GEM（79.0）。
- 最强提升：ClinTox达95.4（+0.1超越MolXPT的95.3）。

**蛋白质属性预测（PEER基准，准确率）**：
- 溶解度预测：BioT5达74.65%，超越ESM-1b（70.23%）和ProtBert（68.15%），即使参数量仅为其1/3。
- 定位预测：BioT5达91.69%，仅次于ESM-1b（92.40%）。

**DTI预测（AUROC/AUPRC）**：
- BioSNAP：0.937/0.937，超越DrugBAN（0.903/0.902）。
- BindingDB：0.963/0.952，超越DrugBAN（0.960/0.948）。
- Human：0.989/0.985，三项指标均SOTA。

**PPI预测（准确率）**：
- Yeast：64.89%，Human：86.22%，超越全参数微调的ESM-1b（66.07%/88.06%）。

**分子 captioning（ChEBI-20）**：
- BLEU-2: 0.635，BLEU-4: 0.556，Text2Mol: 0.603（接近地面真值0.609）。

**文本生成分子**：
- Exact Match: 0.413，超越MolT5-Large（0.311）32.8%。
- Validity: **1.000**（100%有效），对比MolXPT为0.983。

## 相关工作脉络
1. **MolT5（Edwards et al., 2022）**：基于T5联合训练SMILES与文本，但使用SMILES导致生成无效分子；BioT5改用SELFIES并增加蛋白质模态。
2. **MolXPT（Liu et al., 2023b）**：GPT架构在SMILES、生物文本、包装文本上预训练；BioT5采用Transformer encoder-decoder架构，支持双向生成与翻译任务。
3. **Galactica（Taylor et al., 2022）**：大规模科学GPT模型，共用生物序列与自然语言词典；BioT5采用分离词典避免语义混淆。
4. **DeepEIK（Luo et al., 2023）**：多模态特征融合（药物+蛋白质+文本）；BioT5采用统一T5架构通过预训练任务隐式建模跨模态关联。
5. **KV-PLM（Zeng et al., 2022）**：BERT架构桥接分子结构与生物文本；BioT5扩展至双向生成且支持蛋白质模态。
6. **ProtBert/ESM系列**：纯序列蛋白质语言模型；BioT5证明引入文本知识可在更小参数量下实现更强预测能力。

## 局限性与未来方向
1. **指令微调泛化性不足**：未观察到不同下游任务间的泛化能力，需针对每个任务全参数微调。
2. **指令数据合并存在泄露风险**：如BindingDB训练集与BioSNAP/Human测试集存在重叠。
3. **模态覆盖有限**：仅验证文本、分子、蛋白质三元模态，未涵盖DNA/RNA、细胞等多模态。
4. **序列表示局限**：聚焦生物实体序列格式，未利用2D/3D结构信息。
5. **未来方向**：融入基因组学/转录组学数据、探索多模态指令微调、评估模型可解释性。

## 研究启发与可借鉴点
1. **化学基团感知的分离标记化策略**：可迁移至其他科学领域（如材料科学），避免共用词典导致的语义冲突。
2. **结构化-非结构化知识差异化利用**：翻译任务vs掩码重建任务的设计思路可用于其他跨模态场景（如图文、代码-文档）。
3. **Prompt-based微调统一多任务格式**：将分类任务转化为序列生成，便于单一模型处理异构下游任务。
4. **包装文本（Wrapped Text）构建范式**：BERN2实体识别+数据库链接的处理流程可复用于其他领域知识抽取。
5. **小参数+强预训练 vs 大参数弱预训练**：BioT5（252M）超越ESM-1b（652M）证明多模态知识融合可显著减少参数需求。

## 关键术语表
**SELFIES**：Self-referencing Embedded Strings，100%稳健的分子字符串表示，任意字符排列均对应合法化学结构。
**SMILES**：Simplified Molecular-Input Line-Entry System，常用的分子线性字符串表示，但缺乏语法鲁棒性。
**FASTA**：蛋白质序列的标准格式，使用单字母代码表示20种氨基酸。
**包装文本（Wrapped Text）**：将科学文献中的生物实体名称替换为对应的SELFIES/FASTA序列后的文本。
**T5 Objective**：标准T5预训练目标，掩码连续token span并用sentinel token替代进行重建。
**Prompt-based Fine-tuning**：将下游任务转换为自然语言提示格式，统一为序列生成任务进行微调。
**ChEBI-20**：分子描述生成与文本生成分子的评估数据集，每条描述包含20+词。
**Text2Mol Score**：衡量生成分子与文本描述匹配度的相似度指标。

## 可复现要素
- **数据集**：ZINC20（公开）、Uniref50（公开）、C4（公开）、PubMed（公开）、PubChem（公开）、Swiss-Prot（公开）、ChEBI-20（公开）
- **代码**：https://github.com/QizhiPei/BioT5（已开源）
- **模型权重**：论文未提及具体权重下载链接，但代码已开源
- **关键超参**：预训练350K步，batch size 96/GPU，8×A100 80GB，学习率1e-2（cosine annealing至1e-5），warmup 10K步，最大输入长度512，dropout 0.0
