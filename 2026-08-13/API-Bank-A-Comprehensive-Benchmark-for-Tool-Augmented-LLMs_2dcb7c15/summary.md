---
title: "API-Bank-A-Comprehensive-Benchmark-for-Tool-Augmented-LLMs"
source: https://aclanthology.org/2023.emnlp-main.187.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:06:00"
field: "工具增强语言模型"
keywords: ["工具增强LLM", "API调用基准", "Multi-agent数据生成", "大模型工具使用", "LLM评估基准"]
innovations: ["提出首个覆盖Call/Retrieve+Call/Plan+Retrieve+Call三级能力的工具增强LLM综合基准API-Bank", "设计Multi-agent方法实现低成本自动化生成高质量工具使用训练数据（成本降低98%，可用率94%）"]
benchmarks: ["API-Bank"]
---

# 论文速读：API-Bank-A-Comprehensive-Benchmark-for-Tool-Augmented-LLMs

## 一句话总结
本文提出了API-Bank，第一个专为工具增强型LLM设计的综合基准，涵盖Call/Retrieve+Call/Plan+Retrieve+Call三种能力等级，并设计了Multi-agent自动化数据生成方法来构建高质量训练集，最终训练出接近GPT-3.5的工具增强LLM Lynx。

## 研究问题与动机
1. **核心问题**：当前LLM利用外部API工具的有效性如何？如何提升其工具使用能力？面临的根本障碍是什么？
2. **已有基准不足**：Toolformer、APIBench、ToolAlpaca等要么仅评估单一API调用能力，要么缺乏多轮对话与多API规划能力评估，缺少真实可运行的评估系统。
3. **数据构建成本高**：人工标注每个对话成本约8美元，难以大规模构建多样化、高保真的训练数据。
4. **用户真实需求未被覆盖**：作者访谈500名用户发现，工具增强LLM需同时支持"少量API单调用""大量API检索+单调用""大量API检索+多步规划调用"三种场景，已有工作未系统覆盖。

## 核心贡献（创新点）
1. **提出首个综合工具增强LLM基准API-Bank**：定义了Call/Retrieve+Call/Plan+Retrieve+Call三级能力体系，覆盖1008个领域、2211个API，全面对比现有基准（如ToolBench、ToolAlpaca等）在多样性、真实性和覆盖度上的不足。
2. **构建可运行的真实评估系统**：实现73个真实API（天气、图片生成、搜索等），采用硬编码检索结果保证可复现性，支持API Search检索模块，实现端到端可交互评估。
3. **设计Multi-agent自动化数据生成方法**：5个Agent协作生成domain→API→query/API call/response→质检，自动生成1888个对话/4149次API调用，成本仅0.1美元/对话（相比人工降低98%），可用率达94%（self-instruct仅5%）。
4. **训练工具增强LLM Lynx-7B**：基于Alpaca-7B在API-Bank训练集上微调3个epoch，Call准确率超Alpaca提升26 pts，接近GPT-3.5水平，验证了高质量训练数据的有效性。

## 方法详解
1. **能力分级设计**：
   - **Call**：API已知，给定API描述后直接调用（类似槽位填充任务）。
   - **Retrieve+Call**：API未知，需通过API Search检索匹配API后再调用单次API。
   - **Plan+Retrieve+Call**：API未知，需规划多步API调用链，每步前均需先进行API Search检索。
   
2. **Multi-agent数据生成流程**（五个Agent协作）：
   - Agent 1：生成领域（Domain），如healthcare、fitness等。
   - Agent 2：根据领域生成模拟API，引入Public APIs真实示例保证真实性。
   - Agent 3：随机选择1个或多个API及能力等级，生成匹配的user query。
   - Agent 4：输入domain/API/ability/query，生成API call序列并模拟执行，生成最终response。
   - Agent 5（Tester）：自动验证生成数据是否符合设计原则，丢弃约35%不合格样本。
   
3. **Lynx训练配置**：基于LLaMA-7B/Alpaca-7B，训练3个epoch，batch size=256，learning rate=2e-5。

4. **评估指标**：
   - API调用正确率（Accuracy）：预测调用与标注调用在数据库查询/修改结果上一致即视为正确。
   - ROUGE-L：评估LLM生成响应的文本质量。

## 实验与结果
**数据集规模**：训练集1000领域/2138 API/1888对话/4149次调用；评测集8领域/73 API/314对话/753次调用（Call: 363, Retrieve+Call: 50, Plan+Retrieve+Call: 50）。

**最强结果**：
- GPT-4在Plan+Retrieve+Call上达70.00%准确率（Correctness），显著优于GPT-3.5的22.00%，提升约48 pts。
- GPT-4总准确率60.24%，GPT-3.5总准确率47.16%。
- Lynx-7B在Call上达49.87%准确率，较Alpaca-7B（24.06%）提升25.81 pts；总体39.58%，差距GPT-3.5约7.6 pts。

**关键发现**：
- 小规模模型（Alpaca-7B、ChatGLM-6B）具备约20%基础API调用能力，但规划和检索能力几乎为零。
- GPT-3 Davinci（175B）准确率仅0.57%，说明API调用能力需instruction tuning才能解锁。
- Lynx以仅6184条训练样本超过ToolAlpaca（10366条）的微调Alpaca（Table 7），证明数据质量优于数量。

## 相关工作脉络
1. **Toolformer (Schick et al., 2023)**：通过自训练让LLM学会调用API，但仅评估单一API调用，无检索和规划能力，且评估非真实可运行系统。
2. **ToolAlpaca (Tang et al., 2023)**：自动生成3938个工具使用实例，但无检索/规划能力，评测领域单一（仅50类），且GPT-3.5在其数据集上达80-90%，说明难度过低。
3. **APIBench (Patil et al., 2023)**：覆盖1645个API，但仅评估Call能力，无多轮对话和响应质量评估。
4. **ToolBench (Qin et al., 2023b; Xu et al., 2023)**：ToolBench1覆盖16464个真实API，但无响应质量评估；ToolBench2仅8领域，覆盖不全。
5. **ToolQA (Zhuang et al., 2023)**：仅13个API、6个领域，专注于QA任务，无法评估复杂规划能力。
6. **API-Bank定位差异**：唯一同时覆盖三种能力等级、支持多轮多调用、包含响应质量评估、具备真实可运行系统的基准。

## 局限性与未来方向
1. **仅限英文**：当前仅支持英语，未来可扩展多语言版本。
2. **仅训练7B模型**：未探索更大规模模型，内部商用大模型结果因匿名性无法公开。
3. **错误分析揭示的挑战**：GPT-4在Plan+Retrieve+Call中67.86%错误为API检索失败；Lynx最大错误为API Hallucination（61.38%），表明检索可靠性和参数严格校验是核心瓶颈。
4. **未来方向**：改进API调用方法（减少幻觉）、增强解码算法（严格遵循参数定义）、扩大训练数据规模。

## 研究启发与可借鉴点
1. **Multi-agent数据生成范式**：将复杂数据生成任务分解为多个独立Agent依次执行，每个Agent承担单一职责，可有效解决自指导（self-instruct）在多约束任务上的失效问题，可迁移至其他需要结构化数据的领域。
2. **能力分级评估框架**：将工具使用能力分解为Call/Retrieve+Call/Plan+Retrieve+Call三个递进等级，为评估其他复杂能力（如代码生成、Agent任务）提供了分层评估思路。
3. **可运行评估系统的重要性**：强调评估系统必须具备真实执行环境（硬编码检索结果保证复现性），而非仅检查格式正确性，这一设计理念对Tool Learning领域具有普适参考价值。
4. **数据质量优先于数量**：Lynx以6184条高质量样本超越ToolAlpaca的10366条，证明通过严格质检和多Agent协作生成的数据，可显著降低数据规模需求。
5. **Zero-shot prompt设计**：评测prompt保持极简（仅说明API call格式），避免额外信息干扰模型基础能力评估，值得在对比实验中借鉴。

## 关键术语表
**Tool-augmented LLM**：通过调用外部API工具来扩展能力的语言模型。
**Call**：在API已知情况下直接调用API的能力，类似槽位填充任务。
**Retrieve+Call**：在API未知时先通过检索找到合适API再调用的能力。
**Plan+Retrieve+Call**：最复杂能力，需连续规划、检索和调用多个API完成复合任务。
**Multi-agent数据生成**：利用多个LLM Agent协作完成数据生产的方法，各环节解耦以降低生成难度。
**API Hallucination**：模型生成了训练数据中存在但不在当前测试API列表中的人造API名称。
**No API Call**：模型未生成任何API调用请求，是最常见的错误类型之一。
**False API Call Format**：模型生成的API调用格式无法被系统解析。

## 可复现要素
- 数据集：API-Bank，包含训练集和评测集，论文提供了running demo和样本（见Appendix）
- 代码：论文未明确提及代码开源链接，但提到附录包含样本和demo
- 权重：Lynx-7B模型权重，论文未提及是否开源
- 关键超参：fine-tuning 3 epochs，batch size=256，learning rate=2e-5
- 评估checkpoint：GPT-3.5-turbo-0613、GPT-4-0613
