# Improving Biomedical Abstractive Summarisation with Knowledge Aggregation from Citation Papers

Chen Tang<sup>1</sup>, Shun Wang<sup>2</sup>, Tomas Goldsack<sup>2</sup> and Chenghua Lin<sup>2,3</sup>∗ <sup>1</sup>Department of Computer Science, The University of Surrey, UK <sup>2</sup>Department of Computer Science, The University of Sheffield, UK <sup>3</sup>Department of Computer Science, The University of Manchester, UK chen.tang@surrey.ac.uk, chenghua.lin@manchester.ac.uk {swang209, tgoldsack1}@sheffield.ac.uk

## Abstract

Abstracts derived from biomedical literature possess distinct domain-specific characteristics, including specialised writing styles and biomedical terminologies, which necessitate a deep understanding of the related literature. As a result, existing language models struggle to generate technical summaries that are on par with those produced by biomedical experts, given the absence of domain-specific background knowledge. This paper aims to enhance the performance of language models in biomedical abstractive summarisation by aggregating knowledge from external papers cited within the source article. We propose a novel attention-based citation aggregation model that integrates domain-specific knowledge from citation papers, allowing neural networks to generate summaries by leveraging both the paper content and relevant knowledge from citation papers. Furthermore, we construct and release a large-scale biomedical summarisation dataset that serves as a foundation for our research. Extensive experiments demonstrate that our model outperforms state-of-the-art approaches and achieves substantial improvements in abstractive biomedical text summarisation.

## 1 Introduction

Biomedical text summarisation plays a pivotal role in facilitating the comprehension of the vast and constantly expanding body of biomedical literature (Xie et al., 2022), which poses a significant challenge for clinicians and domain experts who strive to remain well-informed in their respective fields. To address this challenge, the generation of high-quality summaries from the extensive corpus of biomedical literature holds immense potential in supporting research and advancements within the biomedical domain (DeYoung et al., 2021).

One of the key challenges in biomedical natural language generation (NLG) lies in effectively handling domain-specific terminologies that are prevalent in biomedical texts. Consequently, a plethora of research studies have been conducted with a primary focus on enhancing language quality by better integrating domain-specific knowledge in the biomedicine domain (Sotudeh Gharebagh et al., 2020; Tangsali et al., 2022; An et al., 2021; Tang et al., 2023b) However, most prior works have predominantly attempted to incorporate knowledge by leveraging additional annotations within the paper content. These annotations include frequent items (Givchi et al., 2022), named entities (Schulze and Neves, 2016; Peng et al., 2021), entity relations (Shang et al., 2011), as well as external knowledge systems such as biomedical ontologies (Chandu et al., 2017) and external terminology searching tools (Gigioli et al., 2018). Surprisingly, the inclusion of external knowledge derived from citation papers has been rarely explored in previous biomedical studies. Existing corpora for biomedical text summarisation are typically constructed in a manner that models solely rely on the source article when generating a summary. However, as shown in Figure 1, there exists strong connections among papers in the citation network with shared research backgrounds, terminologies, and abstract styles, which will be a useful source of knowledge for improving biomedical abstractive summarisation but not captured in existing datasets.

![](images/20f52243c4eea31185a5f9871025f2d632564b84570808c9595ae2dd08cba538.jpg)  
Figure 1: The overview of our proposed citation knowledge aggregation framework. In this framework, language models are trained to incorporate features from both the content of the main paper and the abstracts of its cited papers. The rationale behind this approach is that the cited papers often share relevant research backgrounds, terminologies and writing styles, which can be used as a good template for the summary generation.

To address this gap in the existing biomedical summarisation dataset, we construct a novel biomedical summarisation dataset utilising an opensource biomedical literature corpus provided by the Allen Institute<sup>1</sup>. During the dataset construction process, we applied rigorous filtering criteria to eliminate low-quality samples. Specifically, we discarded samples with an insufficient number of citations (less than three distinct citations), as well as unqualified papers whose unique identifiers (UIDs) or citation UIDs were inaccessible within the corpus. Additionally, we designed heuristic rules to select and transform the unstructured raw data corpus into a structured dataset in JsonL format. The final dataset comprises over 10,000 instances, with each instance having an average of 16 citations. To the best of our knowledge, this is the largest biomedical literature dataset<sup>2</sup> specifically tailored for citation paper-enhanced biomedical text summarisation. Furthermore, we provide the corresponding methods for collecting the citation network, including cited papers and their associations.

Facilitated by our biomedical summarisation dataset, we further propose a novel approach to biomedical document summarisation whereby we enhance neural models with external domainspecific knowledge in the form of the abstracts of cited papers. Accordingly, we introduce an attention-based network (Vaswani et al., 2017) that dynamically aggregates features extracted from the citation abstracts with the encoded content features of the main paper. This aggregation is achieved by applying attention mechanisms to the associated abstracts of all cited papers, which provides the subsequent summary decoding process with additional features derived from abstracts of the citation papers. Within this framework, the base language model can effectively leverage both the features of the main paper and the additional domain-specific knowledge obtained from cited papers. Consequently, this integration leads to enhanced performance in text summarisation. Extensive experiments demonstrate that our model outperforms state-of-the-art baselines in abstractive biomedical text summarisation. We also conducted an in-depth quantitative analysis to verify the performance gain obtained by our attention-based citation knowledge enhancement framework<sup>3</sup>. Our contributions are summarised as follows:

• We construct a large-scale biomedical literature dataset, which can be used for enhancing biomedical text summarisation with the extracted external knowledge from cited papers.

• We propose a novel framework that can effectively leverage citation papers to enhance the performance of large-scale language models on abstractive summarisation of biomedical literature.

• We conduct extensive experiments to evaluate the effectiveness of our proposed framework, including comparisons with SOTA models and an in-depth analysis of the performance gain achieved by aggregating different quantities of citations.

## 2 Related Work

In recent years, a variety of large-scale pre-trained models (PLMs), such as BART (Lewis et al., 2019); T5 (Raffel et al., 2020); GPT-2 (Radford et al., 2019), have demonstrated remarkable performance improvements across various tasks (Loakman et al., 2023; Zhang et al., 2023; Zhao et al., 2023; Tang et al., 2022b) in the Natural Language Generation (NLG) Domain. These PLMs have also been widely applied to biomedical text summarisation. These models, e.g. BioBERT (Lee et al., 2020) and BioBART (Yuan et al., 2022), have achieved remarkable performance by training on extensive biomedical literature corpora, such as Pubmed<sup>4</sup> and MIMIC-III<sup>5</sup>. However, certain high-level knowledge, e.g., the understanding of medical terminologies, cannot be adequately captured solely through the implicit modeling of word probabilities. To address this limitation, the improvement of biomedical background knowledge understanding is able to necessitate the integration of additional knowledge systems, such as conceptual ontologies. These ontologies explicitly model representations of domain-specific knowledge learned by neural networks. Recent studies have proposed incorporating biomedical knowledge, including terminologies (Tang et al., 2023b)) and concepts (Chandu et al., 2017), to enhance the performance of these language models and bridge the gap between language understanding and specialized biomedical knowledge. Indeed, several notable works have focused on enhancing summarisation through citations in the open domain, such as An et al. (2021) and Yasunaga et al. (2019). However, it is important to highlight that the progress of language models in the biomedical domain has been hindered by the limited availability of datasets and resources. This scarcity has impeded the further advancement and improvement of pre-trained language models (PLMs) specifically tailored for biomedical applications. In this study, we require a dataset that contains retrievable citation papers, making traditional raw data corpora such as Pubmed and MIMIC-III inadequate. To date, the sole public dataset we could find is the Text Analysis Conference (TAC) 2014 Biomedical Summarization track (Cohan et al., 2014). However, this dataset is limited in size comprising merely 313 instances, and is somewhat outdated. Therefore, we construct a novel dataset for investigating biomedical citation-enhanced summarisation.

## 3 Dataset Construction

## 3.1 Construction Process

In order to create a dataset containing biomedical literature and its associated citations, we process a semi-structured raw corpus <sup>6</sup> released by Allen Institute. We refer to this dataset as BioCiteDB throughout the paper. The construction process of the dataset is outlined in algorithm 1, where C represents the raw corpus, and $D$ represents the processed dataset. To ensure the quality and relevance of the data, the selected papers have to meet the following requirements: (1) The papers must include the "Introduction" section, as it is considered the most crucial part for generating abstracts; (2) The papers must have at least three distinct citations to ensure the quality of curated data; (3) The essential elements of the papers, including UID (Pubmed id), Title, Abstract, Sections, and Citations, must be accessible within the raw corpus. As a result of this construction process, the dataset D comprises structured data in JsonL<sup>7</sup> format, with each sample representing an individual paper.

Algorithm 1: Construction of BioCiteDB   
Input: Samples $c _ { i } \in C ;$ Citation limit R   
Output: Json objects $d _ { i } \in D$   
1 Initialise $D  { \dot { \emptyset } }$   
2 foreach $\underline { { c _ { i } } }$ in $C$ do   
3 Initialise object $d _ { i }$ with $c _ { i }$   
4 retrieve files $f _ { j }$ to a queue $q$   
5 Initialise object $p _ { i }$   
6 foreach $f _ { \dot { \lambda } }$ in $\mathcal { A }$ do   
7 if $f _ { j }$ missing elements then   
8 break   
9 end   
10 extract distinct citations $r _ { n } \in f _ { i }$   
11 if $\left. r _ { n } \right. > = R$ then   
12 $r _ { n }$ .uid p<sub>i</sub>.citations   
13 extend $d _ { i }$ with $p _ { i }$   
14 break   
15 end   
16 end   
17 $D$ append $d _ { i }$   
18 end   
19 foreach $\underline { { d _ { i } } }$ in $\underline { { \boldsymbol { D } } }$ do   
20 foreach $\underline { { r _ { n } } }$ in $\frac { d _ { i } } { \ d t }$ .citations do   
21 if $\underline { { r _ { n } \notin D } }$ then   
22 exclude $r _ { n }$   
23 end   
24 end   
25 end

<table><tr><td>Datasets</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td># Samples</td><td>9144</td><td>1143</td><td>1143</td></tr><tr><td># Papers</td><td>18194</td><td>4762</td><td>4618</td></tr><tr><td>Avg. # Distinct Citations of Doc</td><td>6.28</td><td>6.23</td><td>6.26</td></tr><tr><td>Avg. # Total Citations of Doc</td><td>16.56</td><td>16.85</td><td>16.96</td></tr><tr><td>Avg. # Chunks in Doc</td><td>37.33</td><td>37.33</td><td>37.13</td></tr><tr><td>Avg. # Sentences in Doc</td><td>529.94</td><td>526.8</td><td>529.11</td></tr><tr><td>Avg. # Words in Doc</td><td>9858.09</td><td>9901.96</td><td>9907.31</td></tr><tr><td>Avg. # Sentences in Summary</td><td>13.96</td><td>13.98</td><td>13.69</td></tr><tr><td>Avg. # Words in Summary</td><td>220.25</td><td>222.04</td><td>217.28</td></tr></table>

Table 1: Data statistics of Biomed Ref dataset. Doc is the abbr. of Document, and Intro is the abbr. of introduction. Chunks are split by section and subsections in a paper.

## 3.2 Data Statistics

The statistical analysis of our processed dataset is presented in Table 1. Additionally, the distribution of citations per paper is visualized in Figure 2 (a), (b), and (c), while the proportions of data size are depicted in Figure 2 (d) of the same figure. The results obtained from both the statistical analysis and visual representations in Table 1 and Figure 2 both validate the data quality of the constructed dataset, thus indicating the effectiveness of our data construction process and the consistency of the dataset splits. This validation supports the notion that training and inference tasks conducted on this dataset can be regarded as fair and reliable.

![](images/a63149ad7a6834848e37012ed722770a48b55cb2d74b24ce053f4117ce73fb1e.jpg)  
(a)

![](images/9e312fcf683ed98b6d756a74765ea912b412cc15097f2859c824d01c59f16da8.jpg)  
(b)

![](images/f103634bffd540526d39fe7a2c28bdea4db8b1ef6e7e137117d95858676d7d6e.jpg)  
(c)

![](images/f8939f075af19db2d6f1f965efa81689d854020baa6ccc10ddb57cf133ec9f5d.jpg)  
(d)  
Figure 2: A visulisation of the distribution of citations per paper and the data size associated with each split. To avoid clutter and maintain clarity, we have excluded papers with over 50 citations from the visualization, as they constitute a relatively small proportion of the dataset and fall into the long tail category. Some papers in our corpus are only used as citations other than the papers in data splits, so we categorise them as “others”.

Algorithm 2: Extracting Citation Graph G.   
Input: $\overline { { d _ { i } \in D ; h o p _ { m a x } ; N } }$   
Output: The set of related papers P   
1 Initialise current hop $h o p _ { n } = 0 ;$ a double-ended   
queue DQ  (d<sub>i</sub>.uid, hop<sub>n</sub>); a queue recording   
visited nodes V Q;   
2 while s<sup>i</sup> in S do   
3 pop uid and hop<sub>n</sub> from DQ   
4 if hop<sub>n</sub> > hop<sub>max</sub> then   
5 return P   
6 end   
7 $P \gets ( u i d , h o p _ { n } )$   
8 if $| P | > N$ then   
9 return P   
10 end   
11 get d<sub>j</sub> by uid   
12 VQ uid   
13 foreach ${ \underline { { r _ { n } } } } \in d _ { j }$ .citations do   
14 if ${ \underline { { r _ { n } } } } .$ .uid / vq and $\overline { { r _ { n } } } \in D$ then   
15 P r<sub>n</sub>.uid and $\overline { { V Q \gets } } r _ { n } . u i d$   
16 end   
17 end   
18 end

## 3.3 Extract Citation Graph

Scientific papers are intricately connected through citation relationships, forming a network of interconnected nodes. This citation graph provides valuable insights into the relatedness of papers. In order to retrieve relevant papers within this citation graph, we propose an algorithm outlined in algorithm 2. $h o p _ { m a x }$ defines the maximum number of hops between papers that the algorithm can traverse, and N specifies the maximum number of retrieved papers at each hop. As output, P represents papers as nodes, while citation relationships are represented as edges in the network. Due to the high computational cost of processing long documents for summarisation, we set $h o p _ { m a x }$ to 1 and $n e i g h b o r _ { m a x }$ to 12, taking into account the limitations of our available computing resources. However, it is worth noting that the attention-based citation aggregation module can be extended to incorporate Graph Attention Networks (Zhou et al., 2020), which have the capability to integrate multilayer citation graphs (Zhang et al., 2023).

## 4 Methodology

As illustrated in Figure 3, our proposed framework is designed to enhance the performance of the base language model by leveraging the collective knowledge from a set of citation papers. For our experiments, we select BART (Lewis et al., 2019), a widely-used summarization model that has demonstrated promising results in the biomedical domain (Goldsack et al., 2022, 2023), as the base model. In this study, we adopt a strategy where we concatenate the abstracts of the citation papers with the input document to form the model’s input. This approach is motivated by the goal of enabling the model to capture and emulate the writing style present in relevant papers. By incorporating this additional information, we aim to improve the model’s ability to generate high-quality summaries that align with the conventions and patterns observed in the domain-specific literature.

![](images/a107b4627be7f40850edabca05bd242f9c7b6c3438549446c6429cd9a0ff5256.jpg)  
Figure 3: The illustration of our proposed framework. doc is the abbr. of the input document referring to the paper content. abs is the abbr. of the abstracts and abs<sub>c</sub> denotes the abstracts of citation papers.

## 4.1 Task Definition

The task is formulated as follows: Given a paper document $d _ { i } \in D$ as the input, where D represents the paper corpus, and $d _ { i }$ denotes the i-th paper. In addition, the citations papers $\begin{array} { r l } { D ^ { c } } & { { } = } \end{array}$ $\{ d _ { 1 } ^ { c } , d _ { 2 } ^ { c } , . . . , d _ { k } ^ { c } \}$ are also provided as the input. The abstracts of $d _ { k } ^ { c } \in D ^ { c }$ are denoted $a b s _ { k } ^ { c }$ . Either d<sub>i</sub> or $d _ { j } ^ { c }$ consists of a sequence of words represented as $X = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { t } \}$ where $x _ { t }$ denotes t-th word in X. The goal is to generate a summary $Y = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { t } \}$ by modeling the conditional probability distribution $P ( Y | X \in d _ { i } , X \in D ^ { c } )$

## 4.2 Knowledge Aggregation from Citations

Input At the initial stage, both the input document $d _ { i }$ and its retrieved N citation abstracts $a b s ^ { c }$ are concatenated and encoded by language models. Byte-Pair Encoding (Radford et al., 2019) is implemented in the transformation from text into fixed word embeddings:

$$
E _ { d o c } = \mathbf { L } \mathbf { M } _ { e m b } ( [ T o k ^ { C L S } , x _ { t } \in d _ { i } ] )\tag{1}
$$

$$
E _ { a b s _ { j } ^ { c } } = \mathbf { L } \mathbf { M } _ { e m b } ( [ T o k ^ { A B S } , x _ { t } \in a b s _ { j } ^ { c } ] )\tag{2}
$$

$$
E _ { Q _ { j } } = \mathrm { c o n c a t } ( E _ { d o c } , E _ { a b s _ { j } ^ { c } } )\tag{3}
$$

where $L M _ { e m b }$ represents the module responsible for tokenising and converting words into subword embeddings. $T o k ^ { C L S }$ is a special token that signifies the global context tag in the input text. $T o k ^ { A B S }$ is a special token used to indicate the separation between the input document and the cited abstracts. $E _ { Q _ { \mathcal { I } } }$ denotes the embeddings generated for the j-th $( \dot { j } \in [ 1 , N ] )$ document abstract pair.

Encoding In order to capture the relevance of each cited abstract, we employ an attention mechanism to measure the importance of $d _ { i }$ with respect to abs<sup>c</sup>. The attention score is denoted as $a t t n _ { j } ^ { i }$ and the process of aggregating knowledge is illustrated as follows:

$$
E _ { Q } = \mathrm { c o n c a t } ( [ E _ { Q _ { 1 } } , . . . , E _ { Q _ { N } } ] )\tag{4}
$$

$$
Q = \mathbf { L M } _ { e n c } ( E _ { Q } ) , Q \in \mathbb { R } ^ { N \times L \times M }\tag{5}
$$

where $E _ { Q }$ denotes the matrix of embeddings for all composed $Q _ { j }$ , and it is encoded by the language model encoder to generate the encoded features $Q .$

$$
Q ^ { C L S } = \mathrm { F i r s t \_ P o o l } ( Q ) , Q ^ { C L S } \in \mathbb { R } ^ { N \times M }\tag{6}
$$

$$
A t t n \_ l o g i t s = Q ^ { C L S } W ^ { Q } , A t t n \in \mathbb { R } ^ { N \times 1 }\tag{7}
$$

$$
A t t n = \mathrm { s o f t m a x } ( A t t n ) , A t t n \in \mathbb { R } ^ { N \times 1 }
$$

$$
F = A ^ { T } Q , F \in \mathbb { R } ^ { L \times M }\tag{8}
$$

(9)

In the above equations, First\_Pool collects features that represent the global context of the input $d _ { i }$ and $a b s _ { j } ^ { c }$ pairs. As the hidden states of the neural encoder, $Q$ incorporates features from both the documents and the abstracts. Therefore, the representations of the first position in $Q$ (represented as $Q ^ { C L S } )$ correspond to the global context token $T o k ^ { C L S }$ . The attention logits matrix is obtained by applying a trainable parameter $W ^ { Q } \in \mathbb { R } ^ { M \times 1 }$ to the features of $Q ^ { C L S }$ . After applying the sof tmax function for normalization, Attn represents the importance of the input features and is used to reweight the original encoded features $Q ,$ resulting in the final features $F .$

## 4.3 Summary Generation

In line with other abstractive summarization systems, we employ an auto-regressive decoder to generate summary tokens $y _ { t }$ in a sequential manner. The process is described as follows:

$$
H _ { t } = \operatorname { D e c o d e r } ( y _ { < t } , F )\tag{10}
$$

$$
P ( y _ { t } | y _ { < t } , X ) = \operatorname { s o f t m a x } ( H _ { t } W ^ { D } )\tag{11}
$$

$$
y _ { t } \stackrel { \mathrm { s a m p l i n g } } { \longleftarrow } P ( y _ { t } | y _ { < t } , F )\tag{12}
$$

where t represents the current time step. X corresponds to the input, consisting of the words from $d _ { i }$ and $a b s _ { 1 } ^ { c } , . . . , a b s _ { j } ^ { c }$ , provided to the neural model. $H _ { t }$ refers to the hidden state of the decoder module at time step t. This state is computed by the language models using the infused features $F ,$ which encapsulate the information from the input document and its cited abstracts, along with the previously predicted tokens $y _ { < t } . \ W ^ { D }$ denotes a trainable parameter, and $P ( y _ { t } | y _ { < t } , F )$ represents the probability distribution over the vocabulary, which includes special tokens. Employing a sampling strategy, such as argmax, we obtain the predicted token $y _ { t }$

## 4.4 Training and Inference

Finally, as shown in Figure 3, the neural model is trained to fit on the citation-enhanced training set by the following objective function:

$$
\mathcal { L } = - \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \log P ( y _ { t } | y _ { < t } , X )\tag{13}
$$

where $\mathcal { L }$ is the cross-entropy loss employed to train the model in modeling the conditional probabilities over the token sequence $P ( y _ { t } | y _ { < t } , F )$ . By minimizing ${ \mathcal { L } } ,$ the language model learns to predict the referenced abstract corresponding to the input document.

## 5 Experiment

## 5.1 Experimental Setup

Baselines We include a range of competitive PLM models as our baselines. We also provide the results of two rule-based systems, namely LEAD-3 and ORACLE, which serve as benchmarks representing the upper and lower bounds of model performance. LEAD-3 extracts the first 3 sentences from the input as the summary, which can be considered as the lower bound of the performance. ORACLE select sentences from the input document and compose a summary with the highest score<sup>8</sup>, which is the upper bound of extractive summarisation systems. The PLM models serve as baselines for abstractive biomedical summarisation. The Long-Document Transformer (LED) is a Transformer-based models which are able to process long sequences due to their selfattention operation (Beltagy et al., 2020). PE-GASUS (Zhang et al., 2020) is pre-training large Transformer-based encoder-decoder models on massive text corpora with a new self-supervised objective, which is tailored for abstractive text summarisation. BART (Lewis et al., 2019) is a widely used PLM model based on a denoising autoencoder that has proved effective for long text generation tasks. Pubmed-X refers to several PLM-based baselines that have been pre-trained on a largescale biomedical literature corpus Pubmed (Cohan et al., 2018) (the size of the dataset is 215k), where X denotes the name of a PLM. Additionally, we include ChatGPT for comparison. However, as it is close-source and very expensive for training, we were unable to use ChatGPT as the base language model to fine-tune on our dataset. Instead, we compare the outputs of ChatGPT in a zero-shot setting.

Evaluation Metrics In the domain of text summarisation (Sun et al., 2021; Tang et al., 2022a; Xie et al., 2022), ROUGE (Lin, 2004) is the most used metric for the evaluation of generated summaries. For evaluating the quality of the generated summaries, we implement the ROUGE metric with the python package of rouge\_score. Specifically, we report the unigram and bigram overlaps (ROUGE-1 and ROUGE-2, respectively) between the generated summaries and the reference (golden) summaries. Additionally, we include the longest common subsequence (ROUGE-L) metric to evaluate the fluency of the generated summaries. For each ROUGE metric, we provide fine-grained measurements of precision, recall, and F-values, offering a comprehensive assessment of the summarisation performance.

In addition to the ROUGE metric, we conduct an extensive automatic evaluation utilising a broader range of evaluation metrics. Specifically, we employ BERTScore (Zhang et al., 2019) (BeS) and BartScore (Yuan et al., 2021) (BaS) to assess the quality of the generated outputs. We also introduce some readability metrics, e.g. Flesch-Kincaid (FLK) and Coleman-Liau Index (CLI), to evaluate the readability of the generated text. This comprehensive evaluation allows for a more robust assessment of the summarisation performance across multiple dimensions.

<table><tr><td rowspan="2">Models</td><td rowspan="2">PPL↓</td><td colspan="3">ROUGE-1↑</td><td colspan="3">ROUGE-2↑</td><td colspan="3">ROUGE-L↑</td></tr><tr><td>Precision</td><td>Recall</td><td>F-value</td><td>Precision</td><td>Recall</td><td>F-value</td><td>Precision</td><td>Recall</td><td>F-value</td></tr><tr><td>LEAD-3</td><td></td><td>0.5512</td><td>0.1838</td><td>0.2645</td><td>0.2039</td><td>0.0647</td><td>0.0941</td><td>0.4979</td><td>0.1646</td><td>0.2374</td></tr><tr><td>ORACLE</td><td></td><td>0.5669</td><td>0.4121</td><td>0.4676</td><td>0.2478</td><td>0.1764</td><td>0.2015</td><td>0.5195</td><td>0.3762</td><td>0.4276</td></tr><tr><td>ChatGPT</td><td></td><td>0.4965</td><td>0.3899</td><td>0.4242</td><td>0.1720</td><td>0.1358</td><td>0.1471</td><td>0.4551</td><td>0.3564</td><td>0.3880</td></tr><tr><td>LED</td><td>11.36</td><td>0.3250</td><td>0.3129</td><td>0.3150</td><td>0.0940</td><td>0.0909</td><td>0.0912</td><td>0.2934</td><td>0.2822</td><td>0.2844</td></tr><tr><td>PEGASUS</td><td>12.54</td><td>0.3063</td><td>0.3036</td><td>0.3003</td><td>0.0837</td><td>0.0833</td><td>0.0821</td><td>0.2715</td><td>0.2681</td><td>0.2657</td></tr><tr><td>BART</td><td>10.89</td><td>0.3581</td><td>0.2963</td><td>0.3171</td><td>0.0862</td><td>0.0709</td><td>0.0760</td><td>0.3276</td><td>0.2719</td><td>0.2907</td></tr><tr><td>Pubmed-LED</td><td>11.01</td><td>0.3462</td><td>0.3394</td><td>0.3395</td><td>0.0877</td><td>0.0860</td><td>0.0859</td><td>0.3109</td><td>0.3044</td><td>0.3047</td></tr><tr><td>Pubmed-PEGASUS</td><td>18.27</td><td>0.3806</td><td>0.2463</td><td>0.2926</td><td>0.0980</td><td>0.0635</td><td>0.0752</td><td>0.3438</td><td>0.2188</td><td>0.2618</td></tr><tr><td>Pubmed-BART</td><td>10.80</td><td>0.3789</td><td>0.3242</td><td>0.3426</td><td>0.1027</td><td>0.0875</td><td>0.0926</td><td>0.3464</td><td>0.2963</td><td>0.3134</td></tr><tr><td>Pubmed-BART</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>- w one citation</td><td>11.51</td><td>0.3816</td><td>0.3288</td><td>0.3461</td><td>0.1070</td><td>0.0926</td><td>0.0971†</td><td>0.3473</td><td>0.2998</td><td>0.3154</td></tr><tr><td>- w citation agg.</td><td>10.54</td><td>0.3758</td><td>0.3427†</td><td>0.3522†</td><td>0.1039</td><td>0.0946†</td><td>0.0973†</td><td>0.3450</td><td>0.3143†</td><td>0.3233†</td></tr></table>

Table 2: Automatic evaluation based on ROUGE scores. LEAD-3, ORACLE, and ChatGPT were excluded from the performance comparisons as they were not trained on the datasets. However, we use them as reference models to provide insights into the potential performance achievable on our datasets. For each metric, the best overall score is highlighted in bold, and the baseline score is underlined.  /  indicates the higher/lower the better, respectively. - w one citation to denote the input configuration where the document is composed with one randomly selected citation abstract. - w citation agg. denotes our proposed citation abstract aggregation framework. † denotes that the citation-enhanced model results are statistically significant with respect to the base model (Pubmed-BART) by way of Mann-Whitney U test.

## 5.2 Implementation Details

All of the pre-trained models used are restored from the publicly available checkpoints on Hugging Face<sup>9</sup>. The checkpoints we selected include: LED<sup>10</sup>, PEGASUS<sup>11</sup>, BART<sup>12</sup>, Pubmed-LED<sup>13</sup>, Pubmed-PEGASUS<sup>14</sup>, and Pubmed-BART<sup>15</sup>.

To make the comparison fair, all input text is chunked according to the minimal input size limitation of selected language models. In our experiments, it is BART (1024 tokens). Models are trained for up to 10 epochs on a Tesla A40 machine, which has 40 GB GPU memory, and the best checkpoints are kept based on the perplexity of generated responses during validation for the generation on the testset. The batch size is set to 16, and the learning rate is $1 e ^ { - 4 }$ , with the Adam optimizer selected for training. For more details, please refer to the Appendix A.1.

<table><tr><td rowspan="2">Model</td><td colspan="2">Referenced</td><td colspan="2">Unreferenced</td></tr><tr><td>BeS↑</td><td>BaS↑</td><td>FLK↓</td><td>CLI↓</td></tr><tr><td>Golden</td><td>100</td><td>-0.27</td><td>16.49</td><td>15.69</td></tr><tr><td>ChatGPT</td><td>85.56</td><td>-3.11</td><td>16.19</td><td>16.31</td></tr><tr><td>Pubmed-LED</td><td>82.85</td><td>-3.37</td><td>16.33</td><td>15.31</td></tr><tr><td>Pubmed-PEGASUS</td><td>81.97</td><td>-3.42</td><td>15.78</td><td>14.52</td></tr><tr><td>Pubmed-BART</td><td>83.34</td><td>-3.36</td><td>14.45</td><td>13.32</td></tr><tr><td>Pubmed-BART</td><td></td><td></td><td></td><td></td></tr><tr><td>- w one citation</td><td>83.34</td><td>-3.36</td><td>14.78</td><td>13.51</td></tr><tr><td>- w citation agg.</td><td>83.60</td><td>-3.35</td><td>13.82</td><td>13.20</td></tr></table>

Table 3: Automatic evaluation on more metrics. BeS and BaS denote the F1 values of BERTScore and BartScore, respectively. FLK and CLI denote the readability scores of Flesch-Kincaid and Coleman-Liau Index, respectively.

## 5.3 Automatic Evaluation

The results of all experiments are presented in Table 2. It can be observed that our proposed framework (-w citation agg.) significantly outperforms all baseline models across all ROUGE scores (F1 scores), indicating substantial enhancements in the summarisation capability of biomedical papers. To be more specific, the incorporation of citation knowledge has contributed to a substantial improvement in recall, with ROUGE-1 exhibiting a 5.7% increase and ROUGE-2 demonstrating an 8.1% increase. This suggests that the integration of citation knowledge has facilitated the utilisation of more similar expressions extracted from the reference abstracts.

In addition, within our framework, the language model achieves substantially lower perplexity (PPL) and ROUGE-L scores, signifying an improvement in the language quality and reduced confusion during summary generation. We hypothesise that the decrease in PPL and ROUGE-L indicates that the language model has learned writing styles and relevant biomedical terminologies by referring to the abstracts of cited papers.

Regarding the ablation study, -w one citation yields a slight improvement compared to the baseline model Pubmed-BART but exhibits a higher perplexity. This observation suggests that the direct inclusion of random citation content may introduce certain noise. In contrast, our attention-based mechanism enables the neural networks to dynamically select and aggregate important information from multiple citations, effectively addressing confusion issues associated with additional inputs.

In Table 3, we present the results of additional evaluation metrics. BertScore and BartScore, as machine learning-based metrics, measure the semantic similarity between the generated summaries and the reference abstracts. Flesch-Kincaid and Coleman-Liau metrics assess text readability on a vocabulary level. Across all these metrics, -w citation agg. outperforms all baseline models, showcasing the advantages of introducing citation knowledge with our framework. Further analysis within the "Pubmed-BART" model reveals that using a single citation results in a slight decrease in BERTScore and BartScore, along with a slightly higher Flesch-Kincaid score (14.78) and Coleman-Liau Index score (13.51). However, employing citation aggregation leads to improvements across all metrics, with BERTScore (83.60), BartScore (- 3.35), Flesch-Kincaid score (13.82), and Coleman-Liau Index score (13.20). This analysis confirms our initial hypothesis, that directly introducing a random citation may introduce noise that hampers model performance, while our aggregation model comprehensively considers all citation papers, effectively reducing the random noise introduced by a single citation.

## 5.4 Human Evaluation

In order to obtain a more comprehensive evaluation of the generated summaries, we also incorporate human evaluation. This evaluation focuses on four key aspects: fluency, readability, relevance, and informativeness. Fluency assessment aims to measure the overall quality of the language used in the summaries. Readability evaluation determines the extent to which the summaries are easily understandable by readers. Relevance assessment examines whether the content of the summaries is pertinent and aligned with the content of the input document. Informativeness measurement evaluates the extent to which the generated summaries provide sufficient and meaningful information derived from the given input. By incorporating human evaluation, we can assess subjective aspects of summary quality that automated metrics may not fully capture.

<table><tr><td rowspan="2">Score (1 to 5)</td><td colspan="4">Human Evaluation</td></tr><tr><td>Flu</td><td>Rea</td><td>Rel</td><td>Inf</td></tr><tr><td>Golden ChatGPT</td><td>4.86* 3.80</td><td>4.70* 3.77</td><td>5.0* 4.33*</td><td>5.0* 4.33*</td></tr><tr><td>Pubmed-LED Pubmed-PEGASUS Pubmed-BART</td><td>2.70 2.33 2.73</td><td>2.67 2.33 2.73</td><td>3.83* 3.53* 3.77*</td><td>3.70 3.27 3.57</td></tr><tr><td>Pubmed-BART - w one citation</td><td>2.87</td><td>2.77</td><td>3.93* 4.0*</td><td>3.63 3.83*</td></tr></table>

Table 4: Results of Human Evaluation. Flu, Rea, Rel and Inf denote Fluency, Readability, Relevance, and Informativeness, respectively. The best scores are in bold, and the second best are underlined. ChatGPT and Golden are not included in comparison. We calculate Fleiss’ Kappa κ for each metric. The majority of results demonstrate a moderate level of agreement (κ (0.4, 0.6]), and results with a higher level of agreement are marked with ∗.

Considering the difficulty of evaluating generated summaries, which requires a thorough understanding of the content in both the source papers and the summaries, it is imperative that human evaluators possess a strong background in academic writing and biomedical knowledge. We invite 3 qualified evaluators by snowball sampling to rate 30 randomly sampled instances from the testset. In order to minimise biases and increase interannotator agreement, the evaluators were provided with the same annotation guide (see Appendix A.3). The results of the human evaluation are presented in Table 4. It can be observed that both the - w one citation and - w citation agg. models exhibit superior performance compared to other baseline models, thereby affirming the effectiveness of our proposed framework.

![](images/15f4cdc35758dcdb1832e4160ab868ed0863ef0e3e020fc51c515619aa75a734.jpg)  
(a)

![](images/523aa9f6304e0ef0a1545fe91066d1a51e5ba66ae48b1c665bab0e1dcf360599.jpg)

![](images/ed16009be2bd9954a22968fbc19ef1ec3bcd31636e067c3a9eef6868f060aab4.jpg)  
(c)

![](images/0cb352909e1ad79bdf7017546a3c263efc5a4bebd1f0cc5aefb7a972ac01bb8a.jpg)  
(d)  
Figure 4: (a), (b) and (c) illustrate the bar charts depicting the performance enhancement achieved by the -w citation agg. method over the PubmedBART model. The improvement is calculated as the difference between the scores obtained by -w citation agg. and PubmedBART. In these bar charts, we report the ROUGE F1 scores. Additionally, Figure 1(d) exhibits the smoothed curve of the ROUGE scores data, obtained using Gaussian kernel smoothing technique. Due to the limit of GPU memory, we limit the input of the citation aggregation network to 12 citation papers.

To delve further into the evaluation, the metrics of Relevance and Informativeness underscore the improved capability to extract relevant information from the input content and generate comprehensive abstracts. Additionally, the fluency and readability metrics assess the language quality, indicating that the language model generates abstracts that are more coherent and natural. However, it is important to note that the tested Pretrained Language Models (PLMs) exhibited a notable disparity in language quality when compared to the performance of ChatGPT. This discrepancy can be attributed to the substantial difference in model size, with Chat-GPT having 130 billion parameters, whereas the tested PLMs have less than 5 billion parameters.

## 5.5 In-depth Analysis

To further investigate the impact of the citation knowledge aggregation module, we conduct an evaluation to assess the improvement in the generated abstracts. This evaluation involves comparing the performance of our proposed framework, denoted as -w citation agg., against the base model Pubmed-BART using ROUGE scores. The results, presented in Figure 4 as (a), (b), and (c), illustrate the increase in ROUGE scores (F value) for different numbers of citations. The inclusion of citations is shown to have a positive effect on the abstract generation process. The Gaussian kernel smoothed increasing curve, depicted in Figure 4 (d), indicates a clear trend: as more citation abstracts are introduced, the language model exhibits greater improvements. The results highlight the potential of leveraging citation information to enhance the quality of generated abstracts.

## 6 Conclusion

In conclusion, we proposed a novel attentionbased citation aggregation model that incorporates domain-specific knowledge from citation papers. By integrating this additional information, our model enables neural networks to generate summaries that benefit from both the paper content and the associated knowledge extracted from citation papers. Furthermore, we introduced a specialized biomedical summarisation dataset, which served as a valuable resource for evaluating and advancing our research. The effectiveness of our approach was demonstrated through extensive experiments, where our model consistently outperformed stateof-the-art methods in biomedical text summarisation. The results highlight the significant improvements achieved by leveraging knowledge from citation papers and the potential for our model to enhance the understanding of biomedical literature through natural language generation techniques.

## Acknowledgements

Chen Tang is supported by the China Scholarship Council (CSC) for his doctoral study (File No.202006120039). We also gratefully acknowledge the anonymous reviewers for their insightful comments.

## Limitations

In the field of text summarisation, two main approaches are commonly employed: extractive summarisation and abstractive summarisation. While extractive summarisation composes summaries by directly selecting sentences from the input content, abstractive summarisation generates summaries that are not bound to the input content, providing greater flexibility but posing challenges in management and control. In this work, due to resource and time constraints, we focused on implementing an abstractive summarisation model and did not further conduct experiments to develop an extractive summarisation counterpart using our proposed algorithm. However, it is worth noting that our proposed approach has shown promising results, emphasizing the importance of leveraging citation papers to enhance the performance of language models in generating high-quality biomedical summaries. Theoretically, the aggregation of knowledge from citation papers can also be beneficial for extractive summarization approaches.

## Ethics Statement

Our new dataset is derived from an existing publicly available corpus released by the Allen Institute, which is a comprehensive biomedical literature corpus licensed under the Apache License 2.0. We have diligently adhered to the terms and conditions of the license and followed all provided instructions. Furthermore, we have familiarized ourselves with and acknowledged the ACM Code of Ethics and Professional Conduct<sup>16</sup>. We approach our professional responsibilities with utmost seriousness, ensuring that our study upholds ethical principles in every aspect.

## References

Chenxin An, Ming Zhong, Yiran Chen, Danqing Wang, Xipeng Qiu, and Xuanjing Huang. 2021. Enhancing scientific papers summarization with citation graph. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 12498–12506.

Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150.

Khyathi Chandu, Aakanksha Naik, Aditya Chandrasekar, Zi Yang, Niloy Gupta, and Eric Nyberg. 2017. Tackling biomedical text summarization: OAQA at BioASQ 5B. In BioNLP 2017, pages 58– 66, Vancouver, Canada,. Association for Computational Linguistics.

Arman Cohan, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, and Nazli Goharian. 2018. A discourse-aware attention model for abstractive summarization of long documents. In Proceedings of the 2018 Conference of the North

American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 615–621, New Orleans, Louisiana. Association for Computational Linguistics.

Arman Cohan, Luca Soldaini, and Nazli Goharian. 2014. Towards citation-based summarization of biomedical literature. In Proceedings of the Text Analysis Conference (TAC’14).

Jay DeYoung, Iz Beltagy, Madeleine van Zuylen, Bailey Kuehl, and Lucy Lu Wang. 2021. MSˆ2: Multidocument summarization of medical studies. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 7494– 7513, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Paul Gigioli, Nikhita Sagar, Anand Rao, and Joseph Voyles. 2018. Domain-aware abstractive text summarization for medical documents. In 2018 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pages 2338–2343. IEEE.

Azadeh Givchi, Reza Ramezani, and Ahmad Baraani-Dastjerdi. 2022. Graph-based abstractive biomedical text summarization. Journal ofBiomedical Informatics, 132:104099.

Tomas Goldsack, Zheheng Luo, Qianqian Xie, Carolina Scarton, Matthew Shardlow, Sophia Ananiadou, and Chenghua Lin. 2023. Overview of the biolaysumm 2023 shared task on lay summarization of biomedical research articles. In The 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks, pages 468–477.

Tomas Goldsack, Zhihao Zhang, Chenghua Lin, and Carolina Scarton. 2022. Making science simple: Corpora for the lay summarisation of scientific literature. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10589–10604, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Henglin Huang, Chen Tang, Tyler Loakman, Frank Guerin, and Chenghua Lin. 2022. Improving Chinese story generation via awareness of syntactic dependencies and semantics. In Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 178–185, Online only. Association for Computational Linguistics.

Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. 2020. Biobert: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics, 36(4):1234–1240.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy,

Ves Stoyanov, and Luke Zettlemoyer. 2019. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Tyler Loakman, Chen Tang, and Chenghua Lin. 2023. TwistList: Resources and baselines for tongue twister generation. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 579–589, Toronto, Canada. Association for Computational Linguistics.

Keqin Peng, Chuantao Yin, Wenge Rong, Chenghua Lin, Deyu Zhou, and Zhang Xiong. 2021. Named entity aware transfer learning for biomedical factoid question answering. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 19(4):2365– 2376.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(140):1–67.

Frederik Schulze and Mariana Neves. 2016. Entitysupported summarization of biomedical abstracts. In Proceedings of the Fifth Workshop on Building and Evaluating Resourcesfor Biomedical Text Mining (BioTxtM2016), pages 40–49, Osaka, Japan. The COLING 2016 Organizing Committee.

Yue Shang, Yanpeng Li, Hongfei Lin, and Zhihao Yang. 2011. Enhancing biomedical text summarization using semantic relation extraction. PLoS one, 6(8):e23862.

Sajad Sotudeh Gharebagh, Nazli Goharian, and Ross Filice. 2020. Attend to medical ontologies: Content selection for clinical abstractive summarization. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 1899– 1905, Online. Association for Computational Linguistics.

Edward Sun, Yufang Hou, Dakuo Wang, Yunfeng Zhang, and Nancy X. R. Wang. 2021. D2S: Document-to-slide generation via query-based text summarization. In Proceedings ofthe 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1405–1418, Online. Association for Computational Linguistics.

Chen Tang, Frank Guerin, Yucheng Li, and Chenghua Lin. 2022a. Recent advances in neural text generation: A task-agnostic survey. arXiv preprint arXiv:2203.03047.

Chen Tang, Chenghua Lin, Henglin Huang, Frank Guerin, and Zhihao Zhang. 2022b. EtriCA: Eventtriggered context-aware story generation augmented by cross attention. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 5504–5518, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Chen Tang, Hongbo Zhang, Tyler Loakman, Chenghua Lin, and Frank Guerin. 2023a. Enhancing dialogue generation via dynamic graph knowledge aggregation. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4604–4616, Toronto, Canada. Association for Computational Linguistics.

Chen Tang, Hongbo Zhang, Tyler Loakman, Chenghua Lin, and Frank Guerin. 2023b. Terminology-aware medical dialogue generation. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE.

Chen Tang, Zhihao Zhang, Tyler Loakman, Chenghua Lin, and Frank Guerin. 2022c. NGEP: A graph-based event planning framework for story generation. In Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 186–193, Online only. Association for Computational Linguistics.

Rahul Tangsali, Aditya Jagdish Vyawahare, Aditya Vyankatesh Mandke, Onkar Rupesh Litake, and Dipali Dattatray Kadam. 2022. Abstractive approaches to multidocument summarization of medical literature reviews. In Proceedings of the Third Workshop on Scholarly Document Processing, pages 199–203, Gyeongju, Republic of Korea. Association for Computational Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Qianqian Xie, Jennifer Amy Bishop, Prayag Tiwari, and Sophia Ananiadou. 2022. Pre-trained language models with domain knowledge for biomedical extractive summarization. Knowledge-Based Systems, 252:109460.

Bohao Yang, Chen Tang, and Chenghua Lin. 2023a. Improving medical dialogue generation with abstract meaning representations. arXiv preprint arXiv:2309.10608.

Bohao Yang, Chen Tang, Kun Zhao, Chenghao Xiao, and Chenghua Lin. 2023b. Effective distillation

of table-based reasoning ability from llms. arXiv preprint arXiv:2309.13182.

Michihiro Yasunaga, Jungo Kasai, Rui Zhang, Alexander R Fabbri, Irene Li, Dan Friedman, and Dragomir R Radev. 2019. Scisummnet: A large annotated corpus and content-impact models for scientific paper summarization with citation networks. In Proceedings ofthe AAAI conference on artificial intelligence, volume 33, pages 7386–7393.

Hongyi Yuan, Zheng Yuan, Ruyi Gan, Jiaxing Zhang, Yutao Xie, and Sheng Yu. 2022. Biobart: pretraining and evaluation of a biomedical generative language model. arXiv preprint arXiv:2204.03905.

Weizhe Yuan, Graham Neubig, and Pengfei Liu. 2021. Bartscore: Evaluating generated text as text generation. Advances in Neural Information Processing Systems, 34:27263–27277.

Hongbo Zhang, Chen Tang, Tyler Loakmana, Chenghua Lina, and Stefan Goetze. 2023. Cadge: Contextaware dialogue generation enhanced with graphstructured knowledge aggregation. arXiv preprint arXiv:2305.06294.

Jingqing Zhang, Yao Zhao, Mohammad Saleh, and Peter Liu. 2020. Pegasus: Pre-training with extracted gap-sentences for abstractive summarization. In International Conference on Machine Learning, pages 11328–11339. PMLR.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675.

Kun Zhao, Bohao Yang, Chenghua Lin, Wenge Rong, Aline Villavicencio, and Xiaohui Cui. 2023. Evaluating open-domain dialogues in latent space with next sentence prediction and mutual information. arXiv preprint arXiv:2305.16967.

Jie Zhou, Ganqu Cui, Shengding Hu, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, and Maosong Sun. 2020. Graph neural networks: A review of methods and applications. AI open, 1:57–81.

## A Appendices

## A.1 Implementation Details

ChatGPT Prompts The performance of Chat-GPT is highly reliable on the quality of input prompts. We manually design and test prompts of abstract summarisation, and select the best cases as the experimental results.

Others The Gaussian kernel smoothing used in Figure 4 is implemented with the gaussian\_filter1d function from the python package of scipy.ndimage. The ROUGE score evaluation is implemented with the python package rouge\_score. The readability scores such as Flesch-Kincaid (FLK) and Coleman-Liau Index (CLI), are implemented with the python package py-readability-metrics. BertScore is bert\_score, and BartScore is from the GitHub repository of https://github.com/ neulab/BARTScore.

## A.2 Automatic Evaluation

Table 6 shows the full results of BertScore and BartScore.

## A.3 Human Evaluation

In addition to automatic evaluation metrics, we conducted a comprehensive human evaluation to assess the quality of biomedical summarization generated by the different models. The human evaluation aimed to capture important aspects of summarization, including fluency, readability, and relevance.

For the human evaluation, we recruited a group of expert annotators with a strong background in biomedical research. The annotators were provided with a set of summaries generated by each model and were asked to rate them on a Likert scale ranging from 1 to 5. The Likert scale allowed annotators to provide a subjective assessment of the summaries based on their expertise and judgment. The four aspects evaluated in the human evaluation were as follows:

Fluency: Annotators assessed the language quality and coherence of the summaries. They considered factors such as grammar, sentence structure, and overall fluency of the generated text. Higher ratings on the Likert scale indicated better fluency.

Readability: Evaluators focused on the readability and comprehensibility of the summaries. They assessed whether the generated summaries were clear, concise, and understandable to a non-expert audience. Higher ratings indicated better readability.

Relevance: An important criterion was the relevance of the summaries to the original input documents. Annotators evaluated whether the summaries captured the main ideas, key findings, and important concepts present in the source documents. Higher ratings indicated greater relevance. Informativeness: Evaluate the extent to which the generated summaries provide sufficient and meaningful information derived from the given input. Assess the comprehensiveness and completeness of the summary. Consider the inclusion of important details and relevant facts.

<table><tr><td rowspan=1 colspan=1>Index</td><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>Template</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=2>Generate Summaries  Write a summary according to given paper content, which is part of a medicalscientific paper (A). The length of generated summary is expected to be largerthan 130 words and less than  $\{ M A X _ { A } B S _ { L } E N \}$ words. \n\n The papercontent of (A) is: {DOC} \n|n Output:</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Generate the HumanEvaluation Guideline</td><td rowspan=1 colspan=1>Write a detailed humam evaluation guideline based on the following content:{X}</td></tr></table>

Table 5: The examples of ChatGPT prompt templates. The variable in the brackets {} will be replaced by the actual text during the utilisation.

<table><tr><td>Model</td><td>BerS-P↑</td><td>BertS-R↑</td><td>BertS-F1↑</td><td>BartS↑</td></tr><tr><td>Golden</td><td>100</td><td>100</td><td>100</td><td>-0.26</td></tr><tr><td>ChatGPT</td><td>86.46</td><td>84.71</td><td>85.56</td><td>-3.11</td></tr><tr><td>Pubmed-LED</td><td>82.91</td><td>82.83</td><td>82.85</td><td>-3.37</td></tr><tr><td>Pubmed-PEGASUS</td><td>82.55</td><td>81.43</td><td>81.97</td><td>-3.42</td></tr><tr><td>Pubmed-BART</td><td>84.26</td><td>82.49</td><td>83.35</td><td>-3.38</td></tr><tr><td>Pubmed-BART</td><td></td><td></td><td></td><td></td></tr><tr><td>- w one citation</td><td>84.20</td><td>82.54</td><td>83.34</td><td>-3.36</td></tr><tr><td>- w citation agg.</td><td>84.33</td><td>82.92</td><td>83.60</td><td>-3.35</td></tr></table>

Table 6: BeS and BaS denote of BERTScore and BartScore, respectively. P, R, F1 represents the precision, recall and F1 values, respectively.

By utilising a Likert scale with a range of 1 to 5, we were able to capture nuanced evaluations from the annotators. This human evaluation provided valuable insights into the overall performance of the models from the perspectives of fluency, readability, and relevance, allowing us to gain a deeper understanding of their summarization capabilities in the biomedical domain.

## A.4 Future Works

In this paper, we propose a novel framework designed to enhance the performance of biomedical text summarisation using pre-trained language models. Recent years have witnessed the emergence of increasingly potent open-source language models, exemplified by Llama 2<sup>17</sup> and Baichuan<sup>18</sup>. However, the practical implementation of our approach on these immensely large-scale models has been constrained by high computational demands. Consequently, we anticipate the need for more advanced GPU hardware or optimised models, such as distilled models (Yang et al., 2023b), to render the training of these models feasible. Presently, there are two primary ways for advancing our research in citation network-enriched text summarisation:

and we have to wait for more advanced GPU devices or more optimised models (e.g. distilled models) to make training those models to be practical. Currently, there are two main direction to further improve our citaion networks enhanced text summarisation: (1) The development of a more efficient neural network that can effectively incorporate the graph-based features derived from citations (Tang et al., 2023a; Yang et al., 2023a). (2) The identification and extraction of key information from both the input document and its associated citations to enhance language understanding (Huang et al., 2022; Tang et al., 2022c). We defer the exploration of these directions to future research endeavors.