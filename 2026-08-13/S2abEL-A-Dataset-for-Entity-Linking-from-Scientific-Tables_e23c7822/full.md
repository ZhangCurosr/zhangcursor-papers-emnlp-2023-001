# S2abEL: A Dataset for Entity Linking from Scientific Tables

Yuze Lou<sup>1</sup> ∗, Bailey Kuehl<sup>2</sup>, Erin Bransom<sup>2</sup>,

Sergey Feldman<sup>2</sup>, Aakanksha Naik<sup>2</sup>, Doug Downey<sup>2</sup>

<sup>1</sup>Univserty of Michigan, <sup>2</sup>Allen Institute for AI

yuzelou@umich.edu, {baileyk,erinbransom,sergey,aakankshan,dougd}@allenai.org

## Abstract

Entity linking (EL) is the task of linking a textual mention to its corresponding entry in a knowledge base, and is critical for many knowledge-intensive NLP applications. When applied to tables in scientific papers, EL is a step toward large-scale scientific knowledge bases that could enable advanced scientific question answering and analytics. We present the first dataset for EL in scientific tables. EL for scientific tables is especially challenging because scientific knowledge bases can be very incomplete, and disambiguating table mentions typically requires understanding the paper’s text in addition to the table. Our dataset, Scientific Table Entity Linking (S2abEL), focuses on EL in machine learning results tables and includes hand-labeled cell types, attributed sources, and entity links from the PaperswithCode taxonomy for 8,429 cells from 732 tables. We introduce a neural baseline method designed for EL on scientific tables containing many out-of-knowledge-base mentions, and show that it significantly outperforms a state-of-the-art generic table EL method. The best baselines fall below human performance, and our analysis highlights avenues for improvement. Code and the dataset is available at: https://github.com/ allenai/S2abEL/tree/main.

## 1 Introduction

Entity Linking (EL) is a longstanding problem in natural language processing and information extraction. The goal of the task is to link textual mentions to their corresponding entities in a knowledge base (KB) (Cucerzan, 2007), and it serves as a building block for various knowledge-intensive applications, including search engines (Blanco et al., 2015), question-answering systems (Dubey et al., 2018), and more. However, existing EL methods and datasets primarily focus on linking mentions from free-form natural language (Gu et al., 2021; De Cao et al., 2021; Li et al., 2020; Yamada et al., 2022). Some consider tabular data, but focus on tables from the general domain (Deng et al., 2020; Tang et al., 2021b; Iida et al., 2021; Yu et al., 2019). Despite significant research in EL, there is a lack of datasets and methods for EL in scientific tables. Linking entities in scientific tables holds promise for accelerating science in multiple ways: from augmented reading applications that help users understand the meaning of table cells without diving into the document (Head et al., 2021) to automated knowledge base construction that unifies disparate tables, enabling complex question answering or hypothesis generation (Hope et al., 2022).

EL in science is challenging because the set of scientific entities is vast and always growing, and existing knowledge bases are highly incomplete. A traditional "closed world" assumption often made in EL systems, whereby all mentions have corresponding entities in the target KB, is not realistic in scientific domains. It is important to detect which mentions are entities not yet in the reference KB, referred to as outKB mentions. Even for human annotators, accurately identifying whether a rarely-seen surface form actually refers to a rarely-mentioned long-tail inKB entity or an outKB entity requires domain expertise and a significant effort to investigate the document and the target KB. A further challenge is that entity mentions in scientific tables are often abbreviated and opaque, and require examining other context in the caption and paper text for disambiguation. An example is shown in Figure 1.

In this paper, we make three main contributions. First, we introduce S2abEL, a high-quality humanannotated dataset for EL in machine learning results tables. The dataset is sufficiently large for training and evaluating models on table EL and relevant sub-tasks, including 52,257 annotations of appropriate types for table cells (e.g. method, dataset),

![](images/fc77faff3c22c5ab0c8d7375a3526946ea321101edc1fbc597cf977fc28c6539.jpg)  
Figure 1: Part of a table in OPT: Open Pre-trained Transformer Language Models (Zhang et al., 2022) showing relevant context that must be found for entity mentions in the table, and a portion of EL results mapping table mentions to the PapersWithCode KB.

9,565 annotations of attributed source papers and candidate entities for mentions, and 8,429 annotations for entity disambiguation including outKB mentions. To the best of our knowledge, this is the first dataset for table EL in the scientific domain. Second, we propose a model that serves as a strong baseline for each of the sub-tasks, as well as end-to-end table EL. We conduct a comprehensive comparison between our approach and existing approaches, where applicable, for each sub-task. Our method significantly outperforms TURL (Deng et al., 2020), a state-of-the-art method closest to the table EL task, but only designed for generaldomain tables. We also provide a detailed error analysis that emphasizes the need for improved methods to address the unique challenges of EL from scientific tables with outKB mentions.

## 2 Related Work

## 2.1 Entity Linking

In recent years, various approaches have been proposed for entity linking from free-form text, leveraging large language models (Gu et al., 2021; De Cao et al., 2021; Li et al., 2020; Yamada et al., 2019). Researchers have also attempted to extend EL to structured Web tables, but they solely rely on table contents and do not have rich surrounding text (Deng et al., 2020; Zhang et al., 2020; Bhagavatula et al., 2015; Mulwad et al., 2023; Tang et al., 2020; Iida et al., 2021). Most of these works focus on general-purpose KBs such as Wikidata (Vrandeciˇ c and Krötzsch´ , 2014) and DBPedia (Auer et al., 2007) and typically test their approaches with the assumption that the target KB is complete with respect to the mentions being linked (e.g., De Cao et al., 2021; Deng et al., 2020; Hoffart et al., 2011; Tang et al., 2021a; Yamada et al., 2019).

There is a lack of high-quality datasets for table EL in the scientific domain with abundant outKB mentions. Recent work by Ruas and Couto (2022) provides a dataset that artificially mimics an incomplete KB for biomedical text by removing actual referent entities but linking concepts to the direct ancestor of the referent entities. In contrast, our work provides human-annotated labels of realistic missing entities for scientific tables, without relying on the target KB to contain ancestor relations. Our dataset offers two distinct advantages: first, it provides context from documents in addition to original table mentions, and second, it explicitly identifies mentions referring to outKB entities.

## 2.2 Scientific IE

The field of scientific information extraction (IE) aims to extract structured information from scientific documents. Various extraction tasks have been studied in this area, such as detecting and classifying semantic relations (Jain et al., 2020; Sahu et al., 2016), concept extraction (Fu et al., 2020), automatic leaderboard construction (Kardas et al., 2020; Hou et al., 2019), and citation analysis (Jurgens et al., 2018; Cohan et al., 2019).

Among these, Kardas et al., 2020; Hou et al., 2019; Yu et al., 2019, 2020 are the closest to ours. Given a set of papers, they begin by manually extracting a taxonomy of tasks, datasets, and metric names from those papers. Whereas our data set maps each entity to an existing canonical external KB (PwC), they target a taxonomy manually built from the particular papers and surface forms they extract from. Notably, this taxonomy emphasizes lexical representations, with entities such as "AP" and "Average Precision" treated as distinct entities in the taxonomy despite being identical metrics in reality, due to different surface forms appearing in the papers. Such incomplete and ambiguous entity identification makes it difficult for users to interpret the results and limits the practical applicability of the extracted information. In contrast, we propose a dataset and baselines for the end-to-end table EL task, beginning with a table in the context of a paper and ending with each cell linked to entities in the canonicalized ontology of the target KB (or classified as outKB).

## 3 Entity Linking in Scientific Tables

Our entity linking task takes as input a reference KB (the Papers with Code<sup>1</sup> taxonomy in our experiments), a table in a scientific paper, and the table’s surrounding context. The goal is to output an entity from the KB for each table cell (or "outKB" if none). We decompose the task into several subtasks, discussed below. We then present S2abEL, the dataset we construct for scientific table EL.

## 3.1 Task Definition

Cell Type Classification (CTC) is the task of identifying types of entities contained in a table cell, based on the document in which the cell appears. This step is helpful to focus the later linking task on the correct type of entities from the target KB, and also excludes non-entity cells (e.g. those containing numeric values used to report experimental results) from later processing. Such exclusion removes a substantial fraction of table cells (74% in our dataset), reducing the computational cost.

One approach to CTC is to view it as a multilabel classification task since a cell may contain multiple entities of different types. However, our initial investigation found that only mentions of datasets and metrics co-appear to a notable degree (e.g., "QNLI (acc)" indicates the accuracy of some method evaluated on the Question-answering NLI dataset (Wang et al., 2018)). Therefore, we introduce a separate class for these instances, reducing CTC to a single-label classification task with four positive classes: method, dataset, metric, and dataset&metric.

Attributed Source Matching (ASM) is the task of identifying attributed source(s) for a table cell within the context of the document. The attributed source(s) for a concept in a document p is the reference paper mentioned in p to which the authors of p attribute the concept. ASM is a crucial step in distinguishing similar surface forms and finding the correct referent entities. For example, in Figure 1, ASM can help clarify which entities "BlenderBot 1" and "R2C2 BlenderBot" refer to, as the first mention is attributed to Roller et al., 2021 while the second mention is attributed to Shuster et al., 2022. Identifying these attributions helps a system uniquely identify these two entities despite their very similar surface forms and the fact that their contexts in the document often overlap. In this work, we consider the documents listed in the reference section and the document itself as potential sources for attribution. The inclusion of the document itself is necessary since concepts may be introduced in the current document for the first time.

Candidate Entity Retrieval (CER) is the process of identifying a small set of entities from the target KB that are most likely to be the referent entity for a table cell within the context of the document. The purpose of this step is to exclude unlikely candidates and pass only a limited number of candidates to the next step, to reduce computational cost.

Entity Disambiguation (ED) with outKB Identification is the final stage. The objective is to determine the referent entity (or report outKB if none), given a table cell and its candidate entity set. The identification of outKB mentions significantly increases the complexity of the EL task, as it requires the method to differentiate between e.g. an unusual surface form of an inKB entity versus an outKB mention. However, distinguishing outKB mentions is a critical step in rapidly evolving domains like science, where existing KBs are highly incomplete.

## 3.2 Dataset Construction

Obtaining high-quality annotations for S2abEL is non-trivial. Identifying attributed sources and gold entities requires a global understanding of the text and tables in the full document. However, asking annotators to read every paper fully is prohibitively expensive. Presenting the full list of entities in the target KB to link from is also not feasible, while showing annotators short auto-populated candidate entity sets may introduce bias and miss gold entities. We address these challenges by designing a special-purpose annotation interface and pipeline, as detailed below.

In the construction process, we used two inhouse annotators with backgrounds in data analytics and data science, both having extensive experience in reading and annotating scientific papers. In addition, one author of the paper (author A) led and initial training phase with the annotators, and another author of the paper (author B) was responsible for evaluating the inter-annotator agreement (IAA) at the end of the annotation process.

Bootstrapping existing resources — We began constructing our dataset by populating it with tables and cell type annotations from SegmentedTables<sup>2</sup> (Kardas et al., 2020), a dataset where table cells have been extracted from papers and stored in an array format. Each cell is annotated according to whether it is a paper, metric, and so on; and each paper is classified into one of eleven categories (e.g., NLI and Image Generation). To gather data for the ASM task, we fine-tuned a T5-small (Raffel et al., 2022) model to extract the last name of the first author, year, and title for each paper that appears in the reference section of any papers in our dataset from the raw reference strings. We then used the extracted information to search for matching papers in Semantic Scholar (Kinney et al., 2023), to obtain their abstracts. Since the search APIs do not always return the matching paper at the top of the results, we manually verified the output for each query.

Target KB — Papers with Code (PwC)<sup>34</sup> is a free and open knowledge base in the scientific domain with a total of 304,611 papers, 6,550 datasets, and 1,942 methods entities as of this writing. PwC includes basic relations between entities, such as relevant entities for a paper (denoted as Paper-RelatesTo-Entity in the rest of the paper), the introducing paper for an entity, etc. Its data is collected from previously curated results and collaboratively edited by the community. While the KB has good precision, its coverage is not exhaustive — in our experiments, 42.8% of our entity mentions are outKB, and many papers have empty Paper-RelatesTo-Entity relations.

Human Annotation — We developed a web interface using the Flask<sup>5</sup> library for the annotation process. It provides annotators with a link to the original paper, an indexed reference section, and annotation guidelines. The detailed annotation interface with instructions can be found at Appendix C.

For the CTC sub-task, we asked annotators to make necessary modifications to correct errors in SegmentedTables and accommodate the extra dataset&metric class. During this phase, 15% of the original labels were changed. For the ASM subtask, annotators were asked to read relevant document sections for each cell and identify attributed sources, if any. This step can require a global understanding of the document, but candidate lists are relatively small since reference sections usually contain just tens of papers. For the EL sub-task, the web interface populates each cell with entity candidates that are 1) returned from PwC with the cell content as the search string, and/or 2) associated with the identified attributed paper(s) for this cell via the Paper-RelatesTo-Entity relation in PwC. Automatic candidate population is designed to be preliminary to prevent annotators from believing that gold entities should always come from the candidate set. Annotators were also asked to search against PwC using different surface forms of the cell content (e.g., full name, part of the cell content) before concluding that a cell refers to an outKB entity.

To ensure consistency and high quality, we conducted a training phase led by author A, where the two annotators were given four papers at a time to perform all annotation tasks. We then calculated the IAA between author A and each annotator for the four papers using Cohen’s Kappa (McHugh, 2012), followed by disagreement discussion and guideline refinement. This process was repeated until the IAA score achieves "substantial agreement" as per (McHugh, 2012). Afterward, the remaining set of papers was given to the annotators for annotation.

<table><tr><td></td><td>CTC</td><td>ASM</td><td>EL</td></tr><tr><td># papers</td><td>327</td><td>316</td><td>303</td></tr><tr><td># tables</td><td>886</td><td>790</td><td>732</td></tr><tr><td># cells</td><td>52,257</td><td>9,564</td><td>8,429</td></tr></table>

Table 1: Overall statistics of S2abEL. It consists of 52,257 data points for cell types, 9,564 for attributed source matching, and 8,429 for entity linking, with ground truth.

## 3.3 Dataset and Annotation Statistics

Dataset Statistics — Table 1 provides a summary of the statistics for S2abEL. ASM and EL annotations are only available for cells labeled positively in CTC. Metrics only are not linked to entities due to the lack of a controlled metric ontology in PwC. It is worth noting that S2abEL contains 3,610 outKB mentions versus 4,819 inKB mentions, presenting a significantly different challenge from prior datasets that mostly handle inKB mentions. More details are in Appendix A.

Post-hoc IAA Evaluation — We conducted a posthoc evaluation to verify the quality of annotations, where author B, who is a researcher with a Ph.D. in Computer Science, independently annotated five random tables. The Cohen’s Kappa scores show a substantial level of agreement (McHugh, 2012) between author B and the annotations (100% for CTC, 85.5% for ASM, and 60.6% for EL). These results demonstrate the quality and reliability of the annotations in S2abEL. A more detailed analysis on why EL agreement is relatively low can be found at Appendix D.

## 4 Method

In this section, we describe our approach for representing table cells, papers, and KB entities, as well as our model design for performing each of the sub-tasks defined in Section 3.1.

Cell Representation — For each table cell in a document, we collect information from both document text and the surrounding table. Top-ranked sentences were retrieved using BM25 (Robertson and Zaragoza, 2009) as context sentences, which often include explanations and descriptions of the table cell. The surrounding table captures the row and column context of the cell, which can offer valuable hints, such as the fact that mentions in the same row and column usually refer to the same type of entities. More details about cell representation features are in Table 9.

Paper Representation — For each referenced paper, we extract its index in the reference section, the last name of the first author, year, title, and abstract. Index, author name, and year are helpful for identifying inline citations (which frequently take the form of the index in brackets or the author and year in parens). Additionally, the title and abstract provide a summary of a paper which may contain information on new concepts it proposes.

KB Entity Representation — To represent each entity in the target KB, we use its abbreviation, full name, and description from the KB, if available. The abbreviation and full name of an entity are crucial for capturing exact mentions in the text, while the description provides additional context for the entity (Logeswaran et al., 2019).

Cell Type Classification — We concatenate features of cell representation (separated by special tokens) and input the resulting sequence to the pretrained language model SciBERT (Beltagy et al., 2019). For each token in the input sequence, we add its word embedding vector with an additional trainable embedding vector from a separate embedding layer to differentiate whether a token is in the cell, from context sentences, etc. (Subsequent mentions of SciBERT in this paper refer to this modified version). We pass the average of the output token embeddings at the last layer to a linear output layer and optimize for Cross Entropy loss. However, because the majority of cells in scientific tables pertain to experimental statistics, the distribution of cell types is highly imbalanced (as shown in Appendix A). To address this issue, we oversample the minority class data by randomly shuffling the context text extracted from the paper for those cells at a sentence level.

Attributed Source Matching — To enable contextualization between cell context and a potential source, we combine the representations of each table cell and potential attributed source in the document as the input to a SciBERT followed by a linear output layer. We optimize for the Binary Cross Entropy loss, where all non-attributed sources in the document are used as negative examples for a cell. The model output measures the likelihood that a source should be attributed to given a table cell.

Candidate Entity Retrieval — We design a method that combines candidates retrieved by two strategies: (i) dense retrieval (DR) (Karpukhin et al., 2020) that leverages embeddings to represent latent semantics of table cells and entities, and (ii) attributed source retrieval (ASR) which uses the attributed source information to retrieve candidate entities.

For DR, we fine-tune a bi-encoder architecture (Reimers and Gurevych, 2019) with two separate SciBERTs to optimize a triplet objective function. The model is only trained on cells whose gold referent entity exists in the KB. Top-ranked most similar entities based on the BM25F algorithm (Robertson and Zaragoza, 2009)<sup>6</sup> in Elasticsearch are used as negative examples. For each table cell $t _ { i } .$ , the top-k nearest entities ${ \mathcal { O } } _ { d r } ^ { i }$ in the embedding space with ranks are returned as candidates.

For ASR, we use the trained ASM model to obtain a list of papers ranked by their probabilities of being the attributed source estimated by the model. The candidate entity sequence $\mathcal { O } _ { a s r } ^ { i }$ is constructed by fetching entities associated with each potentially attributed paper in ranked order using the Paper-RelatesTo-Entity relations in PwC. Only entities of the same cell type as identified in CTC are retained. Note that including entities associated with lowerranked papers mitigates the errors propagated from the ASM model and the problem of imperfect entity and relation coverage that is common in real-world KBs.

We finally interleave ${ \mathcal { O } } _ { d r } ^ { i }$ and $\mathcal { O } _ { a s r } ^ { i }$ until we reach a pre-defined entity set size $K$

Entity Disambiguation with outKB Identification — Given a table cell and its entity candidates, we fine-tune a cross-encoder architecture (Reimers and Gurevych, 2019) with a SciBERT that takes as input the fused cell representation and entity representation, followed by a linear output layer. We optimize for BCE loss using the same negative examples used in CER training. The trained model is used to estimate the probability that a table cell matches an entity. If the top-ranked entity for a cell has a matching likelihood lower than 0.5, then the cell is considered to be outKB.

## 5 Evaluations

As no existing baselines exist for the end-to-end table EL task with outKB mention identification, we compare our methods against appropriate recent work by evaluating their performance on sub-tasks of our dataset (Section 5.1). Additionally, we report the performance of the end-to-end system to provide baseline results for future work (Section 5.2). Finally, to understand the connection and impact of each sub-task on the final EL performance, we conducted a component-wise ablation study (Section 5.3). This study provides valuable insights into the difficulties and bottlenecks in model performance.

The experiments are designed to evaluate the performance of methods in a cross-domain setting (following the setup in Kardas et al., 2020), where training, validation, and test data come from different disjoint topics. This ensures that the methods are not overfitting to the particular characteristics of a topic and can generalize well to unseen data from different topics.

## 5.1 Evaluating Sub-tasks

## 5.1.1 Cell Type Classification

We compare our method against AxCell’s cell type classification component (Kardas et al., 2020), which uses a ULMFiT architecture (Howard and Ruder, 2018) with LSTM layers pre-trained on arXiv papers. It takes as input the contents of table cells with a set of hand-crafted features to provide the context of cells in the paper. We use their publicly available implementation<sup>7</sup> with a slight modification to the output layer to suit our 5-class classification.

Table 2 shows that our method outperforms Ax-Cell somewhat in terms of F1 scores. Although we do not claim our method on this particular sub-task is substantially better, we provide baseline results using state-of-the-art transformer models.

## 5.1.2 Candidate Entity Retrieval

Since the goal of CER is to generate a small list of potential entities for a table cell, we evaluate the performance of the CER method using recall@K.

Figure 2 shows the results of evaluating dense retrieval (DR), attributed source retrieval (ASR), and a combination of both methods, with different candidate size limits K. We observe that seeding the candidate set with entities associated with attributed papers significantly outperforms DR, while interleaving candidates from ASR and DR produces the most promising results. These results demonstrate the effectiveness of utilizing information on attributed sources to generate high-quality candidates. It is worth noting that when K is sufficiently large, ASR considers all sources as attributed sources for a given cell, thus returning entities that are associated with any source. However, if the gold entity is not related to any cited source in the paper, it will still be missing from the candidate set. Increasing K further will not recover this missing entity, as indicated by the saturation observed in Figure 2.

<table><tr><td rowspan="3">Class</td><td colspan="6">Validation</td><td colspan="6">Test</td></tr><tr><td colspan="2">AxCell</td><td colspan="2"></td><td colspan="2">Ours</td><td colspan="2"> $\mathbf { A x C e l l }$ </td><td colspan="2"></td><td colspan="2">Ours</td></tr><tr><td>P</td><td>R</td><td> $F _ { 1 }$ </td><td>P</td><td>R</td><td> $F _ { 1 }$ </td><td>P</td><td>R</td><td> $F _ { 1 }$ </td><td>P</td><td>R</td><td> $F _ { 1 }$ </td></tr><tr><td>other</td><td>97.4</td><td>98.6</td><td>98.0</td><td>98.4</td><td>98.4</td><td>98.4</td><td>97.3</td><td>98.4</td><td>97.8</td><td>98.0</td><td>98.4</td><td>98.1</td></tr><tr><td>dataset</td><td>83.1</td><td>81.7</td><td>82.2</td><td>83.0</td><td>88.7</td><td>85.6</td><td>90.0</td><td>84.0</td><td>86.2</td><td>90.1</td><td>82.3</td><td>85.4</td></tr><tr><td>method</td><td>96.7</td><td>96.4</td><td>96.6</td><td>97.2</td><td>96.8</td><td>97.0</td><td>95.3</td><td>95.5</td><td>95.4</td><td>93.4</td><td>96.9</td><td>95.1</td></tr><tr><td>metric</td><td>71.3</td><td>72.8</td><td>71.9</td><td>88.6</td><td>79.7</td><td>83.7</td><td>86.6</td><td>86.1</td><td>85.0</td><td>88.0</td><td>87.4</td><td>86.8</td></tr><tr><td>dataset&amp;metric</td><td>97.1</td><td>41.9</td><td>58.0</td><td>88.8</td><td>77.6</td><td>82.1</td><td>82.6</td><td>63.4</td><td>68.1</td><td>75.3</td><td>64.8</td><td>61.9</td></tr><tr><td>Micro  $F _ { 1 }$ </td><td colspan="2">95.8</td><td colspan="2"></td><td colspan="2">96.8</td><td colspan="2">96.0</td><td colspan="2"></td><td colspan="2">96.2</td></tr></table>

Table 2: Results of cell type classification on our method and AxCell, with image classification papers fixed as the validation set and papers from each remaining category as the test set in turn.

![](images/a4a3744cf64fe2ecbeffdf68868aa7fd9fa2bafd472f11860a9eefbc362b55c9.jpg)  
Figure 2: Evaluation of different candidate entity retrieval methods. The method in the parenthesis indicates whether a fine-tuned SciBERT or BM25F is used.

Error Analysis — We examined the outputs of ASR and identified two main challenges. First, we observed that in 22.8% of the error cases when $K \ = \ 1 0 0$ , authors did not cite papers for referred concepts. These cases typically involve well-known entities such as LSTM (Hochreiter and Schmidhuber, 1997). In the remaining error cases, the authors did cite papers; however, the gold entity was not retrieved due to incomplete Paper-RelatesTo-Entity relations in the target KB or because the authors cited the wrong paper.

We additionally investigated the error cases from DR and found that a considerable fraction was caused by the use of generic words to refer to a specific entity. For instance, the validation set of a specific dataset entity was referred to as "val" in the table, the method proposed in the paper was referred to as "ours", and a subset of a dataset that represents data belonging to one of the classification categories was referred to as "window". Resolving the ambiguity of such references requires the model to have an understanding of the unique meaning of those words in the context.

When using the combined candidate sets, missing gold entities were only observed when both DR and ASR failed, leading to superior performance compared to using either method alone.

## 5.1.3 Entity Disambiguation with inKB Mentions

The state-of-the-art method closest to our table EL task is TURL (Deng et al., 2020), designed for general-domain tables with inKB cells. It is a structure-aware Transformer encoder pre-trained on the general-purpose WikiTables corpus (Bhagavatula et al., 2015), which produces contextualized embeddings for table cells, rows, and columns that are suitable for a range of downstream applications, including table EL. We used TURL’s public code<sup>8</sup> and fine-tuned it on the inKB cells of our dataset and compared it with our method using the same entity candidate set of size 50.

Table 3 shows that our model achieves a substantial improvement in accuracy over TURL on nine out of ten paper folds. The examples in Table 6 (appendix) demonstrate that our model is more effective at recognizing the referent entity when the cell mention is ambiguous and looks similar to other entities in the KB. This is because TURL as a generic table embedding method focuses on just cell content and position while our approach combines cell content with the full document. Our analysis further reveals that TURL made incorrect predictions for all cells whose mentions were shorter than four characters (likely an abbreviation or a pointer to a reference paper). Meanwhile, our method correctly linked 39% of these cells.

<table><tr><td>Test fold</td><td>Support</td><td>TURL</td><td>Ours</td></tr><tr><td>Question ans.</td><td>381</td><td>15.0</td><td>36.1</td></tr><tr><td>Öbject det.</td><td>2040</td><td>34.1</td><td>41.0</td></tr><tr><td>Speech rec.</td><td>175</td><td>34.3</td><td>54.3</td></tr><tr><td>Image gen.</td><td>168</td><td>7.7</td><td>35.1</td></tr><tr><td>Machine trans.</td><td>234</td><td>15.4</td><td>39.3</td></tr><tr><td>Text class.</td><td>246</td><td>52.4</td><td>68.7</td></tr><tr><td>Pose estim.</td><td>108</td><td>36.1</td><td>36.1</td></tr><tr><td>Semantic seg.</td><td>641</td><td>44.8</td><td>50.9</td></tr><tr><td>NLI</td><td>328</td><td>30.8</td><td>58.8</td></tr><tr><td>Misc.</td><td>81</td><td>14.8</td><td>33.3</td></tr><tr><td>Micro avg</td><td></td><td>32.5</td><td>44.8</td></tr></table>

Table 3: Accuracy for end-to-end entity linking for cells that refer to an inKB entity with 10-fold-cross-domain evaluation using our approach and TURL. Our method is specialized for tables in scientific papers and outperforms the more general-purpose TURL method.

<table><tr><td>Test fold</td><td>O/I ratio</td><td>OutKB  $F _ { 1 }$ </td><td>InKB hit@1</td><td>Overall acc.</td></tr><tr><td>Machine trans.</td><td>0.60</td><td>62.2</td><td>23.7</td><td>50.3</td></tr><tr><td>Image gen.</td><td>0.48</td><td>55.1</td><td>20.0</td><td>44.4</td></tr><tr><td>Misc.</td><td>2.74</td><td>85.1</td><td>19.5</td><td>74.9</td></tr><tr><td>Speech rec.</td><td>1.46</td><td>73.7</td><td>33.3</td><td>66.5</td></tr><tr><td>Question ans.</td><td>2.33</td><td>84.2</td><td>14.0</td><td>69.3</td></tr><tr><td>NLI</td><td>1.11</td><td>80.6</td><td>47.4</td><td>68.3</td></tr><tr><td>Text class.</td><td>1.24</td><td>77.1</td><td>35.4</td><td>66.8</td></tr><tr><td>Object det.</td><td>0.23</td><td>49.6</td><td>35.2</td><td>45.7</td></tr><tr><td>Semantic seg.</td><td>0.42</td><td>73.4</td><td>39.7</td><td>55.0</td></tr><tr><td>Pose estim.</td><td>1.06</td><td>72.2</td><td>35.2</td><td>59.9</td></tr><tr><td>Micro avg</td><td>0.75</td><td>71.4</td><td>33.3</td><td>57.6</td></tr><tr><td>+ gold CTC</td><td>0.75</td><td>72.4</td><td>33.4</td><td>58.2</td></tr><tr><td>+ gold Can.</td><td>0.75</td><td>71.5</td><td>33.4</td><td>57.6</td></tr><tr><td>+ gold both</td><td>0.75</td><td>72.5</td><td>33.4</td><td>58.2</td></tr></table>

Table 4: End-to-end EL results with 10-fold-crossdomain evaluation of our method on learned DR + ASR candidate sets of size 50 with the inKB threshold set to 0.5. Although our model achieved reasonable overall accuracy, it is still far from perfect, leaving ample room for future improvements in the end-to-end table EL task.

## 5.2 End-to-end Evaluation

We now evaluate the end-to-end performance of our approach on the EL task with outKB identification. In addition to re-ranking candidate entities, the method needs to determine when cell mentions refer to entities that do not exist in the target KB. We report $F _ { 1 }$ scores for outKB entities as the prediction is binary (precision and recall are reported in Appendix Table 8). For inKB mentions, we report the hit rate at top-1. Additionally, we evaluate overall performance using accuracy.<sup>9</sup> For each topic of papers, we report the ratio of outKB mentions to inKB mentions. The top block of Table 4 shows the end-to-end EL performance of our method. Our analysis shows a positive Pearson correlation (Cohen et al., 2009) of 0.87 between O/I ratio and overall accuracy, indicating our method tends to higher accuracy on datasets with more outKB mentions. Figure 3 shows the performance at various inKB thresholds.

![](images/9e0eedbe963d351bb46ff45c15694eb8b7fe1f54b76841eff6b611d172cf6d8a.jpg)  
Figure 3: Entity linking results with varying inKB thresholds. Note that the inKB hit rate is low (44.8%) even when all mentions are predicted with an entity (i.e., threshold is 0).

Error Analysis We sampled 100 examples of incorrect predictions for both outKB and inKB mentions and analyzed their causes of errors in Table 7 (Appendix E). Our analysis reveals that a majority of incorrect inKB predictions are due to the use of generic words. For outKB mentions, the model tends to get confused when they are similar to existing entities in the target KB.

## 5.3 Component-wise Ablation Study

To investigate how much of the error in our end-toend three-step system was due to errors introduced in the first two stages (specifically, wrong cell type classifications from CTC or missing correct candidates from CER), we tried measuring system performance with these errors removed. Specifically, we tried replacing the CTC output with the gold cell labels, or adding the gold entity to the output CER candidate set, or both.

The results in the bottom block of Table 4 show that there is no significant difference in performance with gold inputs. This could be because CTC and CER are easier tasks compared to ED, and if the model fails those tasks, it is likely to still struggle to identify the correct referent entity, even if that is present in the candidate set or the correct cell type is given.

## 6 Conclusion

In this paper, we present S2abEL, a high-quality human-annotated dataset for Entity Linking in machine learning results tables, which is, to the best of our knowledge, the first dataset for table EL in the scientific domain. We propose a model that serves as a strong baseline for the end-to-end table EL task with identification of outKB mentions, as well as for each of the sub-tasks in the dataset. We show that extracting context from paper text gives a significant improvement compared to methods that use only tables. Identifying attributed source papers for a concept achieves higher recall@k compared with dense retrieval for candidate entity generation. However, the best baselines still fall far below human performance, showing potential for future improvement.

## Limitations

In this section, we discuss some limitations of our work. First, our dataset only includes tables from English-language papers in the machine learning domain, linked to the Papers with Code KB, which limits its generalizability to other domains, languages, and KBs. Second, we acknowledge that the creation of S2abEL required significant manual effort from domain experts, making it a resourceintensive process that may not be easily scalable. Third, our approach of using attributed papers to aid in identifying referent entities relies on the target KB containing relations that associate relevant papers and entities together. Fourth, we do not compare against large GPT-series models and leave this as future work. Finally, while our experiments set one initial baseline for model performance on our task, substantially more exploration of different methods may improve performance on our task substantially.

## Acknowledgments

This work was supported in part by NSF Grant 2033558.

## References

Sören Auer, Christian Bizer, Georgi Kobilarov, Jens Lehmann, Richard Cyganiak, and Zachary Ives. 2007. Dbpedia: A nucleus for a web of open data. In Proceedings of the 6th International The Semantic Web and 2nd Asian Conference on Asian Semantic Web Conference, ISWC’07/ASWC’07, page 722–735, Berlin, Heidelberg. Springer-Verlag.

Iz Beltagy, Kyle Lo, and Arman Cohan. 2019. SciB-ERT: A pretrained language model for scientific text. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3615– 3620, Hong Kong, China. Association for Computational Linguistics.

Chandra Sekhar Bhagavatula, Thanapon Noraset, and Doug Downey. 2015. Tabel: Entity linking in web tables. In The Semantic Web-ISWC 2015: 14th International Semantic Web Conference, Bethlehem, PA, USA, October 11-15, 2015, Proceedings, Part I, pages 425–441. Springer.

Roi Blanco, Giuseppe Ottaviano, and Edgar Meij. 2015. Fast and space-efficient entity linking for queries. In Proceedings ofthe Eighth ACM International Conference on Web Search and Data Mining, WSDM ’15, page 179–188, New York, NY, USA. Association for Computing Machinery.

Arman Cohan, Waleed Ammar, Madeleine van Zuylen, and Field Cady. 2019. Structural scaffolds for citation intent classification in scientific publications. In Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3586–3596, Minneapolis, Minnesota. Association for Computational Linguistics.

Israel Cohen, Yiteng Huang, Jingdong Chen, Jacob Benesty, Jacob Benesty, Jingdong Chen, Yiteng Huang, and Israel Cohen. 2009. Pearson correlation coefficient. Noise reduction in speech processing, pages 1–4.

Silviu Cucerzan. 2007. Large-scale named entity disambiguation based on Wikipedia data. In Proceedings ofthe 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL), pages 708–716, Prague, Czech Republic. Association for Computational Linguistics.

Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2021. Autoregressive entity retrieval. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Xiang Deng, Huan Sun, Alyssa Lees, You Wu, and Cong Yu. 2020. Turl: table understanding through representation learning. Proceedings of the VLDB Endowment, 14(3):307–319.

Mohnish Dubey, Debayan Banerjee, Debanjan Chaudhuri, and Jens Lehmann. 2018. Earl: joint entity and relation linking for question answering over knowledge graphs. In The Semantic Web–ISWC 2018: 17th International Semantic Web Conference, Monterey, CA, USA, October 8–12, 2018, Proceedings, Part I 17, pages 108–126. Springer.

Sunyang Fu, David Chen, Huan He, Sijia Liu, Sungrim Moon, Kevin J. Peterson, Feichen Shen, Liwei Wang, Yanshan Wang, Andrew Wen, Yiqing Zhao, Sunghwan Sohn, and Hongfang Liu. 2020. Clinical concept extraction: A methodology review. Journal ofBiomedical Informatics, 109:103526.

Yingjie Gu, Xiaoye Qu, Zhefeng Wang, Baoxing Huai, Nicholas Jing Yuan, and Xiaolin Gui. 2021. Read, retrospect, select: An mrc framework to short text entity linking. Proceedings ofthe AAAI Conference on Artificial Intelligence, 35(14):12920–12928.

Andrew Head, Kyle Lo, Dongyeop Kang, Raymond Fok, Sam Skjonsberg, Daniel S Weld, and Marti A Hearst. 2021. Augmenting scientific papers with justin-time, position-sensitive definitions of terms and symbols. In Proceedings ofthe 2021 CHI Conference on Human Factors in Computing Systems, pages 1– 18.

Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation, 9(8):1735– 1780.

Johannes Hoffart, Mohamed Amir Yosef, Ilaria Bordino, Hagen Fürstenau, Manfred Pinkal, Marc Spaniol, Bilyana Taneva, Stefan Thater, and Gerhard Weikum. 2011. Robust disambiguation of named entities in text. In Proceedings ofthe 2011 Conference on Empirical Methods in Natural Language Processing, pages 782–792, Edinburgh, Scotland, UK. Association for Computational Linguistics.

Tom Hope, Doug Downey, Oren Etzioni, Daniel S Weld, and Eric Horvitz. 2022. A computational inflection for scientific discovery. arXiv preprint arXiv:2205.02007.

Yufang Hou, Charles Jochim, Martin Gleize, Francesca Bonin, and Debasis Ganguly. 2019. Identification of tasks, datasets, evaluation metrics, and numeric scores for scientific leaderboards construction. In Proceedings ofthe 57th Annual Meeting ofthe Association for Computational Linguistics, pages 5203– 5213, Florence, Italy. Association for Computational Linguistics.

Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification. In Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 328–339, Melbourne, Australia. Association for Computational Linguistics.

Hiroshi Iida, Dung Thai, Varun Manjunatha, and Mohit Iyyer. 2021. TABBIE: Pretrained representations of tabular data. In Proceedings ofthe 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3446–3456, Online. Association for Computational Linguistics.

Sarthak Jain, Madeleine van Zuylen, Hannaneh Hajishirzi, and Iz Beltagy. 2020. SciREX: A challenge dataset for document-level information extraction. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 7506– 7516, Online. Association for Computational Linguistics.

David Jurgens, Srijan Kumar, Raine Hoover, Dan Mc-Farland, and Dan Jurafsky. 2018. Measuring the evolution of a scientific field through citation frames. Transactions of the Association for Computational Linguistics, 6:391–406.

Marcin Kardas, Piotr Czapla, Pontus Stenetorp, Sebastian Ruder, Sebastian Riedel, Ross Taylor, and Robert Stojnic. 2020. AxCell: Automatic extraction of results from machine learning papers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8580– 8594, Online. Association for Computational Linguistics.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, Online. Association for Computational Linguistics.

Rodney Michael Kinney, Chloe Anastasiades, Russell Authur, Iz Beltagy, Jonathan Bragg, Alexandra Buraczynski, Isabel Cachola, Stefan Candra, Yoganand Chandrasekhar, Arman Cohan, Miles Crawford, Doug Downey, Jason Dunkelberger, Oren Etzioni, Robert Evans, Sergey Feldman, Joseph Gorney, David W. Graham, F.Q. Hu, Regan Huff, Daniel King, Sebastian Kohlmeier, Bailey Kuehl, Michael Langan, Daniel Lin, Haokun Liu, Kyle Lo, Jaron Lochner, Kelsey MacMillan, Tyler Murray, Christopher Newell, Smita Rao, Shaurya Rohatgi, Paul L Sayre, Zejiang Shen, Amanpreet Singh, Luca Soldaini, Shivashankar Subramanian, A. Tanaka, Alex D Wade, Linda M. Wagner, Lucy Lu Wang, Christopher Wilhelm, Caroline Wu, Jiangjiang Yang, Angele Zamarron, Madeleine van Zuylen, and Daniel S. Weld. 2023. The semantic scholar open data platform. ArXiv, abs/2301.10140.

Belinda Z. Li, Sewon Min, Srinivasan Iyer, Yashar Mehdad, and Wen-tau Yih. 2020. Efficient one-pass end-to-end entity linking for questions. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6433–6441, Online. Association for Computational Linguistics.

Lajanugen Logeswaran, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, Jacob Devlin, and Honglak Lee. 2019. Zero-shot entity linking by reading entity descriptions. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3449–3460, Florence, Italy. Association for Computational Linguistics.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.

Mary L McHugh. 2012. Interrater reliability: the kappa statistic. Biochemia medica, 22(3):276–282.

Varish Mulwad, Tim Finin, Vijay S Kumar, Jenny Weisenberg Williams, Sharad Dixit, Anupam Joshi, et al. 2023. A practical entity linking system for tables in scientific literature. In 3rd Workshop on Scientific Document Understanding at AAAI-2023.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2022. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(1).

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: Bm25 and beyond. Found. Trends Inf. Retr., 3(4):333–389.

Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Eric Michael Smith, Y-Lan Boureau, and Jason Weston. 2021. Recipes for building an open-domain chatbot. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 300–325, Online. Association for Computational Linguistics.

Pedro Ruas and Francisco M. Couto. 2022. Nilinker: Attention-based approach to nil entity linking. Journal ofBiomedical Informatics, 132:104137.

Sunil Sahu, Ashish Anand, Krishnadev Oruganty, and Mahanandeeshwar Gattu. 2016. Relation extraction from clinical texts using domain invariant convolutional neural network. In Proceedings of the 15th

Workshop on Biomedical Natural Language Processing, pages 206–215, Berlin, Germany. Association for Computational Linguistics.

Kurt Shuster, Mojtaba Komeili, Leonard Adolphs, Stephen Roller, Arthur Szlam, and Jason Weston. 2022. Language models that seek for knowledge: Modular search & generation for dialogue and prompt completion.

Hongyin Tang, Xingwu Sun, Beihong Jin, and Fuzheng Zhang. 2021a. A bidirectional multi-paragraph reading model for zero-shot entity linking. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 35, pages 13889–13897.

Nan Tang, Ju Fan, Fangyi Li, Jianhong Tu, Xiaoyong Du, Guoliang Li, Sam Madden, and Mourad Ouzzani. 2020. Rpt: relational pre-trained transformer is almost all you need towards democratizing data preparation. arXiv preprint arXiv:2012.02469.

Nan Tang, Ju Fan, Fangyi Li, Jianhong Tu, Xiaoyong Du, Guoliang Li, Sam Madden, and Mourad Ouzzani. 2021b. Rpt: Relational pre-trained transformer is almost all you need towards democratizing data preparation. Proc. VLDB Endow., 14(8):1254–1261.

Denny Vrandeciˇ c and Markus Krötzsch. 2014.´ Wikidata: A free collaborative knowledgebase. Commun. ACM, 57(10):78–85.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings ofthe 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353–355, Brussels, Belgium. Association for Computational Linguistics.

Ikuya Yamada, Koki Washio, Hiroyuki Shindo, and Yuji Matsumoto. 2019. Global entity disambiguation with pretrained contextualized embeddings of words and entities. arXiv preprint arXiv:1909.00426.

Ikuya Yamada, Koki Washio, Hiroyuki Shindo, and Yuji Matsumoto. 2022. Global entity disambiguation with BERT. In Proceedings ofthe 2022 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 3264–3271, Seattle, United States. Association for Computational Linguistics.

Wenhao Yu, Zongze Li, Qingkai Zeng, and Meng Jiang. 2019. Tablepedia: Automating pdf table reading in an experimental evidence exploration and analytic system. In The World Wide Web Conference, WWW ’19, page 3615–3619, New York, NY, USA. Association for Computing Machinery.

Wenhao Yu, Wei Peng, Yu Shu, Qingkai Zeng, and Meng Jiang. 2020. Experimental evidence extraction system in data science with hybrid table features and ensemble learning. In Proceedings ofThe Web Conference 2020, WWW ’20, page 951–961, New York, NY, USA. Association for Computing Machinery.

<table><tr><td rowspan="2">Fold</td><td colspan="3">CTC</td><td colspan="3">ASM</td><td colspan="3">EL</td></tr><tr><td># paper # table</td><td></td><td># cell</td><td></td><td></td><td># paper # table # cell</td><td># paper # table # cell</td><td></td><td></td></tr><tr><td>Question ans.</td><td>58</td><td>139</td><td>6,422</td><td>57</td><td>128</td><td>1,320</td><td>57</td><td>127</td><td>1,282</td></tr><tr><td>Object det.</td><td>54</td><td>159</td><td>17,020</td><td>52</td><td>141</td><td>2,686</td><td>51</td><td>135</td><td>2,527</td></tr><tr><td>Image class.</td><td>27</td><td>94</td><td>2,681</td><td>27</td><td>82</td><td>608</td><td>27</td><td>81</td><td>597</td></tr><tr><td>Speech rec.</td><td>22</td><td>88</td><td>3,516</td><td>21</td><td>76</td><td>649</td><td>21</td><td>74</td><td>612</td></tr><tr><td>Image gen.</td><td>25</td><td>37</td><td>1,184</td><td>23</td><td>34</td><td>290</td><td>23</td><td>34</td><td>288</td></tr><tr><td>Machine trans.</td><td>28</td><td>48</td><td>2,199</td><td>25</td><td>42</td><td>412</td><td>25</td><td>41</td><td>378</td></tr><tr><td>Text class.</td><td>21</td><td>75</td><td>4,085</td><td>20</td><td>55</td><td>688</td><td>19</td><td>51</td><td>600</td></tr><tr><td>NLI</td><td>32</td><td>83</td><td>3,385</td><td>30</td><td>68</td><td>787</td><td>30</td><td>66</td><td>697</td></tr><tr><td>Pose estim.</td><td>13</td><td>47</td><td>4,447</td><td>11</td><td>31</td><td>550</td><td>7</td><td>18</td><td>222</td></tr><tr><td>Semantic seg,</td><td>32</td><td>82</td><td>5,733</td><td>30</td><td>75</td><td>927</td><td>30</td><td>75</td><td>919</td></tr><tr><td>Misc.</td><td>15</td><td>34</td><td>1,585</td><td>13</td><td>30</td><td>308</td><td>13</td><td>30</td><td>307</td></tr></table>

Table 5: Detailed statistics of S2abEL.

<table><tr><td rowspan=1 colspan=1>Cell Content Column header</td><td rowspan=1 colspan=1>TURL result</td><td rowspan=1 colspan=1>Our result</td><td rowspan=1 colspan=1>Gold entity</td></tr><tr><td rowspan=1 colspan=4>InKB</td></tr><tr><td rowspan=1 colspan=1>&quot;[33]&quot;             &quot;&quot;</td><td rowspan=1 colspan=1>Cityscapes, PoseTrack,LAMBADA</td><td rowspan=1 colspan=1>PointNet, GAN, CRF</td><td rowspan=1 colspan=1>PointNet</td></tr><tr><td rowspan=1 colspan=1>&quot;Text GCN&quot;      &quot;Model&quot;</td><td rowspan=1 colspan=1>Global Conv. Net.,End-to-End Mem. Net.</td><td rowspan=1 colspan=1>Graph Conv. Net.,Global Conv. Net.</td><td rowspan=1 colspan=1>Graph Conv. Net.</td></tr></table>

Table 6: Incorrect examples for end-to-end EL from TURL. The table includes the cell content and the column header in the first two columns, the top-3 ranked results from TURL and our approach in the third and fourth columns, respectively, and the gold entity in the last column.

Shuo Zhang, Edgar Meij, Krisztian Balog, and Ridho Reinanda. 2020. Novel entity discovery from web tables. In Proceedings ofThe Web Conference 2020, WWW ’20, page 1298–1308, New York, NY, USA. Association for Computing Machinery.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.

## A Detailed Dataset Statistics

S2abEL consists of 11 folds, each corresponding to a topic. Table 5 provides detailed statistics on the number of papers, tables, and cells for each sub-task and topic. The class distribution for CTC is as follows: other (74%), dataset (8%), method (14%), metric (3%), and dataset&metric (0.4%). For ASM, 1,532 (16.6%) cells have missing attributed paper, 1,095 (11.9%) cells attribute to the paper itself, 6,598 (71.5%) cells attribute to an entry in the reference section of the paper. For EL, 3,610 (42.8%) cells refer to outKB entities and (57.2%) cells refer to inKB entities.

## B Training Details

We trained all our models for two epochs with a batch size of 32, using the AdamW optimizer (Loshchilov and Hutter, 2019) with linear decay warm-up. The initial learning rate was 2e-5 and the warm-up ratio was 10%. All models were trained using a single 48Gb NVIDIA A6000 GPU. For the triplet loss function in DR, we used Euclidean as the distance function with a margin of 1. For the Candidate Entity Retrieval and Entity Disambiguation tasks, we used negative examples of size 50 at training time. Additionally for the ED task, we set candidate set size limitation as 50 when making predictions.

## C Annotation Interface and Guidelines

Our annotation interface with annotation guidelines is at https://github.com/allenai/ s2abel/blob/main/common\_utils/ Annotation%20Interface.pdf. Note that there might be cells that contain a subentity mentions consisting of an entity mention and a non-entity mention string, e.g., "Bert-large", "Bert with 6 layers frozen". For these cells, we asked the annotators to focus on the primary entity and our current model considers these mentions as mentions of the main entity. Thus those two mentions are labeled as method, and linked to https: paperswithcode.com/method/bert. We also specifically asked the annotators to mark cells that contain mentions of more than one primary entity or are confusing to understand, which are excluded from the dataset. We leave the tasks of linking subentities explicitly and cells to multiple entities for future work.

<table><tr><td rowspan=1 colspan=1>Cause        Percent |</td><td rowspan=1 colspan=1>Example cell</td><td rowspan=1 colspan=1>Our result</td><td rowspan=1 colspan=1>Gold entity</td></tr><tr><td rowspan=1 colspan=4>InKB</td></tr><tr><td rowspan=2 colspan=1>Variants         21%</td><td rowspan=1 colspan=1>&quot;bottle&quot;</td><td rowspan=1 colspan=1>PASCAL VOC,PASCAL VOC 2007</td><td rowspan=1 colspan=1>PASCAL VOC 2007</td></tr><tr><td rowspan=1 colspan=1>&quot;BigGAN&quot;</td><td rowspan=1 colspan=1>BigGan-deep, BigGan</td><td rowspan=1 colspan=1>BigGan</td></tr><tr><td rowspan=1 colspan=1>Top below threshold  16%</td><td rowspan=1 colspan=1>&quot;Car&quot;</td><td rowspan=1 colspan=1>outKB, KITTI</td><td rowspan=1 colspan=1>KITTI</td></tr><tr><td rowspan=1 colspan=1>Abbreviations      19%</td><td rowspan=1 colspan=1>&quot;FR-EN&quot;</td><td rowspan=1 colspan=1>Arcade Learning Environment</td><td rowspan=1 colspan=1>WMT 2014</td></tr><tr><td rowspan=1 colspan=1>Generic words      39%</td><td rowspan=1 colspan=1>&quot;Ours&quot;</td><td rowspan=1 colspan=1>Neural Turing Machine</td><td rowspan=1 colspan=1>OHEM</td></tr><tr><td rowspan=1 colspan=4>OutKB</td></tr><tr><td rowspan=1 colspan=1>Similar names      56%</td><td rowspan=1 colspan=1>&quot;DPN&quot;</td><td rowspan=1 colspan=1>Dual Path Network</td><td rowspan=1 colspan=1>Deep Parsing Network</td></tr><tr><td rowspan=1 colspan=1>Generic words      22%</td><td rowspan=1 colspan=1>&quot;12&quot;</td><td rowspan=1 colspan=1>HUB5 English</td><td rowspan=1 colspan=1>bAbi</td></tr></table>

Table 7: Representative examples of erroneous end-to-end EL cases. The table includes the cause and the percentage for that cause in the first two columns, an example of cell content for that cause and our incorrect prediction in the third and fourth columns, and the gold entity in the last column.

<table><tr><td>Test fold</td><td>Precision Recall</td></tr><tr><td>Machine trans.</td><td>46.4 94.4</td></tr><tr><td>Image gen.</td><td>38.7 95.7</td></tr><tr><td>Misc.</td><td>77.0 95.1</td></tr><tr><td>Speech rec.</td><td>62.8 89.3</td></tr><tr><td>Question ans.</td><td>76.9 93.1</td></tr><tr><td>NLI</td><td>74.9 87.2</td></tr><tr><td>Text class.</td><td>66.2 92.2</td></tr><tr><td>Object det.</td><td>34.0 91.1</td></tr><tr><td>Semantic seg.</td><td>61.4 91.2</td></tr><tr><td>Pose estim.</td><td>63.8 83.3</td></tr><tr><td>Micro avg</td><td>58.6 91.4</td></tr><tr><td>+ gold CTC</td><td>59.5 92.7</td></tr><tr><td></td><td>58.7 91.4</td></tr><tr><td>+ gold Can.</td><td></td></tr><tr><td>+ gold both</td><td>59.5 92.7</td></tr></table>

Table 8: Additional end-to-end Entity linking results for outKB cells.

<table><tr><td rowspan=1 colspan=1>Feature</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1>cell content</td><td rowspan=1 colspan=1>cell&#x27;s raw text</td></tr><tr><td rowspan=1 colspan=1>region</td><td rowspan=1 colspan=1>cell&#x27;s relative location with reference to the top-leftnumeric cell in the table, i.e., top-left, top-right,bottom-left, and bottom-right</td></tr><tr><td rowspan=1 colspan=1>context sentences</td><td rowspan=1 colspan=1>top-ranked sentences in the full document(including table captions, section headers, etc.)regarding the cell content based on BM25</td></tr><tr><td rowspan=1 colspan=1>row context</td><td rowspan=1 colspan=1>concatenated cell&#x27;s row separated by special tokens</td></tr><tr><td rowspan=1 colspan=1>column context</td><td rowspan=1 colspan=1>concatenated cell&#x27;s column separated by special tokens</td></tr><tr><td rowspan=1 colspan=1>position</td><td rowspan=1 colspan=1>cell&#x27;s 2D position in the table in terms of distancefrom the top left corner</td></tr><tr><td rowspan=1 colspan=1>reverse position</td><td rowspan=1 colspan=1>cell&#x27;s 2D position in the table in terms of distancefrom the bottom right corner.</td></tr><tr><td rowspan=1 colspan=1>has reference</td><td rowspan=1 colspan=1>whether cell has a reference</td></tr></table>

Table 9: Features for cell representation.

## D Annotation Error Analysis

Our investigation showed that the main cause of disagreement in the EL phase was that there were cells whose matching entities were confusing to the annotators, due to their insufficient background in the specific academic area and/or the paper not clearly indicating which entity it was and/or whether should be considered a variant of an existing entity or a different entity entirely.

## E Error Case Study

Table 6 presents examples where TURL made incorrect EL predictions while our approach made correct predictions. Table 7 summarizes the main causes of incorrect predictions made by our approach for both inKB and outKB mentions.