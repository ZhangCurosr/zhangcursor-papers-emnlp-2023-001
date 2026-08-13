# Dual-Channel Span for Aspect Sentiment Triplet Extraction

Pan Li<sup>1</sup> Ping Li<sup>1</sup>∗ Kai Zhang<sup>2</sup>

<sup>1</sup> School of Computer Science, Southwest Petroleum University, Chengdu, China <sup>2</sup> School of Computer Science and Technology, East China Normal University, Shanghai, China <sup>1</sup> {lp970717, dping.li}@gmail.com <sup>2</sup> zk1980@hotmail.com

## Abstract

Aspect Sentiment Triplet Extraction (ASTE) is one of the compound tasks of fine-grained aspect-based sentiment analysis (ABSA), aiming at extracting the triplets of aspect terms, corresponding opinion terms and the associated sentiment orientation. Recent efforts in exploiting span-level semantic interaction have shown superior performance on ASTE task. However, span-based approaches could suffer from excessive noise due to the large number of spans that have to be considered. To ease this burden, we propose a dual-channel span generation method to coherently constrain the search space of span candidates. Specifically, we leverage the syntactic relations among aspect/opinion terms and their part-of-speech characteristics to generate useful span candidates, which empirically reduces span enumeration by nearly a half. Besides, the interaction between syntactic and part-of-speech views brings relevant linguistic information to learned span representations. Extensive experiments on two public datasets demonstrate both the effectiveness of our design and the superiority on ASTE task <sup>1</sup>.

## 1 Introduction

Aspect Sentiment Triplet Extraction (ASTE) is a compound task in fine-grained Aspect-Based Sentiment Analysis (ABSA) (Pontiki et al., 2014). It is composed of three fundamental subtasks: Aspect Term Extraction (ATE) (Yin et al., 2016; Ma et al., 2019; Chen and Qian, 2020; Li et al., 2020), Opinion Term Extraction (OTE) (Yang and Cardie, 2012, 2013; Wan et al., 2020) and Aspect Sentiment Classification (ASC) (Wang et al., 2016; Tang et al., 2016; Xue and Li, 2018; Tang et al., 2020; Li et al., 2021). In particular, ASTE aims to extract the sentiment triplet of aspect terms, corresponding opinion terms and their associated sentiment polarity in a given sentence. For example, in the sentence "My vegetable risotto was burnt, and infused totally in a burnt flavor", there are two sentiment triplets, namely, (“vegetable risotto”, “burnt”, Negative”) and (“flavor”, “burnt”, Negative”), where “vegetable risotto” and “flavor” are aspect terms, “burnt” is the opinion term corresponding to the aspect of interest, and “Negative” is the sentiment polarity of these two triplets.

![](images/141ea8d0c4fe0a09c4c18e059c9882be297f6fe1d8f815a62399c243da905bb1.jpg)  
Figure 1: A sentence with dependency tree and part-ofspeech in ASTE task.

When the idea of ASTE was first proposed, a two-stage pipeline method (Peng et al., 2020) was developed for this task. However, staged processing scheme often lead to error propagation between subtasks. More than that, opinion terms are generally associated with the aspect target, staged pipeline method breaks this interaction. To address those issue, some end-to-end approaches (Wu et al., 2020a; Xu et al., 2021; Chen et al., 2021b, 2022b) are devised, which attempt to simultaneously extract aspect-opinion pairs and perform sentiment classification by introducing novel tagging schemes. In particular, most of existing end-toend models (Wu et al., 2020a; Chen et al., 2021b, 2022b) build the interaction between aspect and its corresponding opinion at token-level, i.e., word-toword interactions. Despite of its efficacy, it is hard to guarantee the consistency of predicted sentiment polarity between multiple word-to-word pairs when many aspects/opinions are expressed using multiple words. On account of this, recent work (Xu et al., 2021; Chen et al., 2022d) adopt span-level interactions in the sentiment triplet structure. Compared with the token-level pairing, span-level interaction is proved to bring significant gains to the model.

However, one prominent problem with spanbased methods is that they usually enumerate all spans in a sentence, which will bring about high computational cost and many noises. Specifically, the number of enumerated spans for a sentence of length n is $O ( n ^ { 2 } )$ , while the number of possible interactions between all opinion and aspect candidate spans is $O ( n ^ { 4 } )$ at the later span-pairing stage, implying a lot of invalid aspect/opinion spans and span pairs. Moreover, most of the existing spanbased methods model direct interactions between two spans. The high-order interactions are generally overlooked.

To address those issues, we explore the linguistic phenomena in the spans. Our observations are two-fold: First, multiple words composed of the span of an aspect/opinion target are generally syntactically dependent, and multiple dependency relations can transmit higher-order interactions between spans. For example, in Figure 1, the aspect term “vegetable risotto” has an intra-span syntactic dependency “compound” and an inter-span dependency “nsubj” with “burnt”. On the other hand, the span “flavor” is indirectly related to the first “burnt” (i.e., the one associated with “vegetable risotto”) within 2 hops in the syntactic tree. This indirect relation may suggest the relevance of the sentiment polarity of the two span pairs, namely, (“vegetable risotto”, “burnt”) and (“flavor”, “burnt”). In effect, the ground truth of sentiment polarity of these two aspect-opinion pairs are the same, as shown in Figure 1.

Second, we also observe that there are some frequent patterns in aspect and opinion spans in terms of part-of-speech. For instance, in many cases aspect terms are noun or noun phrase which we refer to as NN and (NN NN), respectively. Moreover, it is fairly common that opinion terms are adjective (denoted by JJ). As shown in Figure 1, the aspect “vegetable risotto” has the part-ofspeech structure (NN  NN), and opinion term "burnt” is JJ. Therefore, it is possible to extract the aspect/opinion spans according to the lexical characteristics of the words so as to avoid enumerating all word combination.

Motivated by the two observations, we propose a dual-channel span generation approach for aspectlevel sentiment triplet extraction, which we term as Dual-Span. Dual-Span utilizes two relational graph attention networks (RGAT) to separately learn high-order syntactic dependency between words/spans and linguistic features in constructed part-of-speech relations among words. Then a gating mechanism is adopted to fuse the syntactic and lexical information of span candidates, which helps to enhance the feature representation of spans. On the other hand, instead of enumerating all possible spans, the span candidates are extracted from two channels, i.e., the syntactic dependency relations and part-of-speech based relations, thus largely reducing the noisy information in favor of valid span pairing.

Our main contributions are as follows:

• We devise a dual-channel span generation method for aspect sentiment triplet extraction, which produces a span candidate set much smaller than the greedily enumerated one by leveraging the syntactic dependency and partof-speech correlation among tokens/spans in a dual-channel manner.

• We construct the intra-span and inter-span relations based on the part-of-speech correlation of spans/words, on top of which the high-order linguistic interactions is able to be captured by relational graph neural networks.

• We combine the syntactic information learned from dependency tree with the part-of-speech information learned from constructed lexical relations to enrich span representation. We conduct extensive experiments on benchmark datasets to evaluate the efficacy and efficiency of the proposed method. The experimental results show that our model Dual-Span outperforms all state-of-the-art methods on the ASTE task.

## 2 Related Work

Aspect-based sentiment analysis (ABSA) (Pontiki et al., 2014; Schouten and Frasincar, 2016; Xue and Li, 2018; Chen et al., 2022a; Trusca and Frasincar, 2023) is fine-grained sentiment analysis. The early work of ABSA was to identify its three sentiment elements (i.e., aspect, opinion, sentiment polarity) as basic tasks: ATE (e.g., (Yin et al., 2016; Ma et al., 2019; Chen and Qian, 2020; Li et al., 2020), OTE (Yang and Cardie, 2012, 2013; Wan et al., 2020)) and ASC (e.g., (Wang et al., 2016; Tang et al., 2016; Xue and Li, 2018; Du et al., 2019;

![](images/8883b32ef59b258d324387816c2f5816eda2a2bc1c95c89340d8a5f629f52e7c.jpg)  
Figure 2: The overall architecture of our Dual-Span.

Li et al., 2021; Brauwers and Frasincar, 2023)). Subsequently, some studies began to consider multiple sentiment element composite tasks in order to better understand fine-grained sentiment analysis: aspect term polarity co-extraction (APCE) (Li and Lu, 2017; He et al., 2019; Li et al., 2019), Aspect-Opinion Pair Extraction (AOPE) (Zhao et al., 2020; Wu et al., 2020a; Gao et al., 2021; Chakraborty et al., 2022) and Aspect Category Sentiment Analysis (ACSA) (Schmitt et al., 2018; Hu et al., 2019; Cai et al., 2020; Liu et al., 2021).

Some recent works started to consider the integrity among the three sentiment elements and thus proposed the ASTE task. A diversity of techniques were proposed for it: two-stage pipeline (Peng et al., 2020), multi-task unified framework (Li et al., 2019; Zhang et al., 2020; Yan et al., 2021), multi-round machine reading comprehension method (Mao et al., 2021; Chen et al., 2021a; Liu et al., 2022) and end-to-end method (Wu et al., 2020a; Xu et al., 2020; Chen et al., 2021b, 2022c; Xu et al., 2021; Chen et al., 2022d). The span-level based approaches adopt end-to-end implementation. For instance, Span-ASTE (Xu et al., 2021) enumerates aspect and viewpoint spans and directly exploits their interaction to solve ASTE tasks, while SBN (Chen et al., 2022d) proposed a span-level bidirectional network that enumerates all possible spans as input, and completes the ASTE task by designing two decoders and adopting inference strategies. Despite that, it still remains an open challenge to improve the search efficiency and feature representation for the span of sentiment triplets.

## 3 Proposed Framework

In this section, the overall architecture of our proposed model Dual-Span is shown in Figure 2, which consists of four main components: sentence encoding, feature enhancing module, dual-channel span generation and triplet module.

## 3.1 Task Definition

For a sentence $X = \{ w _ { 1 } , w _ { 2 } , . . . , w _ { n } \}$ of length $n ,$ the ASTE task is to extract the set of aspect sentiment triplets $\begin{array} { r c l } { \mathcal { T } } & { = } & { \{ ( a , o , s ) _ { m } \} _ { m = 1 } ^ { | \mathcal { T } | } } \end{array}$ from the given sentence X, where a, o and $s \in$ P OS, NEU, NEG represent the aspect term, opinion term and sentiment polarity, respectively. is the number of sentiment triplets contained sentence X.

## 3.2 Sentence Encoding

To obtain contextual representations for each word, we explore two sentence encoding methods, namely, BiLSTM and BERT.

BiLSTM We first use the GloVe (Pennington et al., 2014) embedding to get the embedding matrix $E \in \mathbb { R } ^ { | V | * d _ { w } }$ of the corpus, where V  represents the vocabulary size, and $d _ { s }$ represents the embedding dimension. For the embedding tokens $E _ { x } ~ = ~ \{ e _ { 1 } , e _ { 2 } , . . . , e _ { n } \}$ in the sentence, we use BiLSTM to get its hidden representation $H \_ =$ $\{ h _ { 1 } , h _ { 2 } , \ldots , h _ { n } \}$ , where $h \in \mathbb { R } ^ { 2 d _ { n } }$ is obtained by splicing the hidden state $\vec { h } \in R ^ { d _ { n } }$ generated by forward LSTM and the hidden state $\stackrel {  } { h } \in { \cal R } ^ { d _ { n } }$ generated by backward LSTM:

$$
h = [ \stackrel { \right. } { h } ; \stackrel { \left. } { h } ]\tag{1}
$$

BERT An alternative approach is to utilize BERT (Devlin et al., 2019) as the sentence encoder to generate contextualized word representations. Given a sentence $X = \{ w _ { 1 } , w _ { 2 } , \ldots , w _ { n } \}$ with n words, the hidden representation sequence $H = \{ h _ { 1 } , h _ { 2 } , \ldots , h _ { n } \}$ is the output of the encod ing layer of BERT at the last transformer block.

## 3.3 Feature Enhancing Module

As aforementioned, spans (or intra-span words) involve syntactical dependency and part-of-speech correlation, therefore incorporating those information into feature representations can be beneficial for span pairing and sentiment prediction. To capture the high order dependency relations, here we devise a graph neural network based method to encode the syntactic dependency and part-of-speech relations of intra- and inter-spans in high orders. In particular, we construct the part-of-speech relational graph (corresponding to a multi-relation matrix as shown in Figure 3 (b)). Then we apply two relational graph attention networks to learn the high order interactions between words on syntactic dependency tree of the sentence in question and constructed part-of-speech graph, respectively.

## 3.3.1 Part-of-speech Graph Construction

The goal of part-of-speech graph construction is to characterize the word formation patterns of aspect and opinion terms so as to better identify the possible spans. Specifically, we adopt the following three rules to construct the part-of-speech graph $G ^ { P o s } = ( V , R ^ { P o s } )$ of a given sentence X. First, following previous work (Chakraborty et al., 2022), assuming that aspect terms are usually nouns and opinion terms are usually adjectives, we can define part-of-speech relations based on part-of-speech tags NN or $J J$ . In particular, we consider the relations between words in a given window that contains words tagged with NN or $J J$ . Therefore, a relational edge $\bar { R } _ { i , j } ^ { P o s }$ of $G ^ { P o s }$ is defined for two words i and j as the combination of part-of-speech tags of the two words, whose representation vector is $r _ { i , j } ^ { p } \in \mathbb { R } ^ { d _ { p } }$ , where $d _ { p }$ is the dimension of part-of-speech combination embedding. Besides, we consider the special syntactic relation nsubj, since opinion terms are usually directly used to modify aspect terms, leading to better extraction of aspect-opinion pairs. Finally, for each word’s partof-speech, we add a self-loop relational edge to itself, as the diagonal elements shown in Figure 3.

![](images/db068358adb4435fa190fff12e3d9e8076d5d47401d009721f2683b1837f350f.jpg)  
Figure 3: An example sentence with dependency tree and part-of-speech adjacency matrices in ASTE task.

On the other hand, the syntactic dependency graph $G ^ { S y n } = \left( V , R ^ { S y n } \right)$ is constructed according to the dependency parsing tree, where edges are represented by syntactic relation types. Moreover, we define the self-dependency for each word. So for a given sentence of length n, the syntactic relation between words w<sub>i</sub> and $w _ { j }$ is denoted as $R _ { i , j } ^ { S y n }$ whose corresponding vectorization representation is denoted as the vector $r _ { i , j } ^ { s } \in \mathbb { R } ^ { d _ { s } }$ , where $d _ { s }$ is the dimension of syntactic relation embeddings.

## 3.3.2 High-order Feature Learning with Relational Graph Attention Network

Next, we use relational graph attention networks (RGAT) to capture the multiple types of linguistic features and high-order interaction between spans/words on syntactic dependency graph and part-of-speech graph, respectively. Moreover, we use two graph attentional network based modules, namely, SynGAT and PosGAT to learn syntactic dependency graphs and part-of-speech graphs, respectively, which will distinguish between various syntactic relationships and part-of-speech relationships when calculating the attention weight between nodes. In particular, following previous work (Bai et al., 2021), we denote two specific relations on each edge by $r _ { i , j } ^ { s }$ and $r _ { i , j } ^ { p }$ , respectively. Specifically, for the i th node, the update process is as follows:

$$
h _ { i } ^ { s y n } ( l ) = \| _ { z = 1 } ^ { Z } \sigma \left( \sum _ { j \in N ( i ) } \hat { \alpha } _ { i , j } ^ { l z } \left( W _ { s 1 } ^ { l z } h _ { j } ^ { s y n } ( l - 1 ) + W _ { s 2 } ^ { l } r _ { i , j } ^ { s y n } \right) \right)\tag{2}
$$

$$
h _ { i } ^ { p o s } ( l ) = \| _ { z = 1 } ^ { Z } \sigma \left( \sum _ { j \in \mathcal { N } ( i ) } \hat { \beta } _ { i , j } ^ { l z } \left( W _ { p 1 } ^ { l z } h _ { j } ^ { p o s } ( l - 1 ) + W _ { p 2 } ^ { l } r _ { i , j } ^ { p o s } \right) \right)\tag{3}
$$

where $W _ { s 2 } ^ { l } \in \mathbb { R } ^ { \frac { d } { z } \times d }$ and $W _ { p 2 } ^ { l } \in \mathbb { R } ^ { \frac { d } { z } \times d }$ are parameter matrices. z denotes the number of attention heads, and σ is the sigmoid activation. ${ \mathcal { N } } _ { i }$ is the set of immediate neighbors of node i. $\hat { \alpha } _ { i , j } ^ { ( l z ) } , \hat { \beta } _ { i , j } ^ { ( l z ) }$ are the normalized attention coefficients for the $z { \mathrm { - t h } }$ head at the l-th layer.

To fuse syntactic dependency and part-of-speech relation features, we introduce a gating mechanism (Cho et al., 2014) to merge the two views as follows:

$$
g = \sigma \left( W _ { g } \left[ h ^ { s y n } : h ^ { p o s } \right] + b _ { g } \right)\tag{4}
$$

$$
h = g \circ h ^ { s y n } + ( 1 - g ) \circ h ^ { p o s }\tag{5}
$$

where is element-wise product operation. $\left\lceil h ^ { s y n } : h ^ { p o s } \right\rceil$ is the concatenation of $h ^ { s y n }$ and $h ^ { p o s }$ and $W _ { g }$ and $b _ { g }$ are model parameters. This way, g is learned to optimize the feature fusion.

## 3.4 Dual-Channel Span Generation

In this section, we propose a dual-channel span generation module, which consists of two parts: dual-channel span generation and span classification.

## 3.4.1 Dual-Channel Span Generation

Syntactic Span Generation Given a sentence X whose syntactic dependency graph is $\begin{array} { r l } { G ^ { S y n } } & { { } = } \end{array}$ $( V , R ^ { S y n } )$ , if there is a dependency edge $e _ { i j }$ between words $w _ { i }$ and $w _ { j }$ , then all words positioned between them are considered to be a span $\mathbf { s } _ { i , j } ^ { s y n }$ . In particular, self-dependent edges represents spans of length $L _ { s } = 1$ . We define the representation of $\mathbf { s } _ { i , j } ^ { s y n }$ as follows

$$
{ \bf s } _ { i , j } ^ { s y n } = [ h _ { i } : h _ { j } : f _ { w i d t h } ( i , j ) ] , \mathrm { i f } e _ { i , j } = 1\tag{6}
$$

where $f _ { w i d t h } ( i , j )$ denotes trainable embedding of span length $( \mathrm { i . e . , } j - i + 1 ) . e _ { i , j } = 1$ suggests that there is an edge between $w _ { i }$ and $w _ { j }$

Part-of-speech Span Generation For a given sentence $X ~ = ~ \{ w _ { 1 } , w _ { 2 } , \ldots , w _ { n } \}$ , if the part-ofspeech tag of word $w _ { o }$ is NN or $J J ,$ , the words in a predefined window will be exhaustively enumerated and then the enumeration is further combined with central word $w _ { o }$ to form spans. The part-ofspeech induced span $\mathbf { s } _ { k , l } ^ { p o s }$ can be represented as:

$$
\begin{array} { c } { \mathbf { s } _ { k , l } ^ { p o s } = [ h _ { k } : h _ { l } : f _ { w i d t h } ( k , l ) ] , } \\ { \mathrm { i f } p o s _ { o } = N N \mathrm { o r } J J , \mathrm { a n d } o \in [ k , l ] } \end{array}\tag{7}
$$

where $f _ { w i d t h } ( k , l )$ refers to the trainable embedding of span length.

Finally, we merge the two types of span candidates: $\bar { S = } \bar { \mathbf { s } _ { i , j } ^ { s y n } } \cup \bar { \mathbf { s } _ { k , l } ^ { p o s } }$

Compared to exhaustive enumeration on the whole sentence in previous span-based approaches, whose time complexity of enumerated spans is $O ( n ^ { 2 } )$ , for a sentence of length n. However, in our syntactic span generation, the parsing tree containing 2n edge dependencies (Qi et al., 2020) (including self-dependent edges), so the number of generated spans is $O ( 2 n )$ . On the other hand, the statistics shows that in the benchmark datasets, there are about 2.5 part-of-speech NN and JJ in each sentence on average. Therefore, in the part-of-speech span generation procedure, the number of span candidates is $O ( 2 . 5 S _ { w i n d o w } ( S _ { w i n d o w } - 1 ) ) \leq n ,$ where $S _ { w i n d o w }$ is the window size to restrict span length and generally set to be a small value (e.g., $S _ { w i n d o w } = 3$ in our experiments). That is, the time complexity of our method to generate the span is $O ( n )$ , which significantly reduce the span candidate size.

## 3.4.2 Span Classification

After obtaining the span candidates S, we further narrow down the pool of possible spans by leveraging two auxiliary tasks, namely, ATE and OTE tasks. Specifically, all span candidates in S will be classified into one of the three categories: Aspect, Opinion, Invalid by a span classifier. Next, nz spans are singled out with higher prediction scores $\Phi _ { a s p e c t }$ or $\Phi _ { o p i n i o n } .$ , where z is the threshold hyper-parameter and $\Phi _ { a s p e c t }$ and $\Phi _ { o p i n i o n }$ are obtained by

$$
\Phi _ { a s p e c t } ( \mathbf { s } _ { i , j } ) = \mathrm { s o f t m a x } \left( \mathrm { F F N N } _ { t = a s p e c t } \left( \mathbf { s } _ { i , j } \right) \right)\tag{8}
$$

$$
\Phi _ { o p i n i o n } ( \mathbf { s } _ { i , j } ) = \mathrm { s o f t m a x } \left( \mathrm { F F N N } _ { t = o p i n i o n } \left( \mathbf { s } _ { i , j } \right) \right)\tag{9}
$$

where FFNN denotes a feed-forward neural network with non-linear activation.

## 3.5 Triplet Module

Based on the shrinked candidate pool of aspect and opinion terms, the aspect candidate $\mathbf { s } _ { a , b } ^ { a } \in S ^ { a }$ and opinion candidate $\mathbf { s } _ { c , d } ^ { o } \in \ S ^ { o }$ are paired and represented as

$$
{ \bf g } _ { s _ { a , b } ^ { a } , { \bf s } _ { c , d } ^ { o } } ^ { { } } = \left[ { \bf s } _ { a , b } ^ { a } : { \bf s } _ { c , d } ^ { o } : r _ { a b , c d } ^ { s } : f _ { d i s t a n c e } ( a , b , c , d ) \right] .\tag{10}
$$

where $f _ { d i s t a n c e } ( a , b , c , d )$ denotes trainable embeddings of span length. $r _ { a b , c d } ^ { s }$ is a trainable embedding vector which is the average pooling of the dependency vectors between words ab and cd. Additionally, since opinions are more likely to modify the aspects that match them, we consider the dependency relationship $r _ { a b , c d } ^ { s } \in$ $R ^ { S y n }$ between them. Then, sentiment classification is performed for the obtained span pairs, where the sentiment types are defined as $r \in \mathbf { \Sigma }$ R = Positive, Negative, Neutral, Invalid Formally, the triplet prediction is written as

$$
P \left( r \mid \mathbf { s } _ { a , b } ^ { a } , \mathbf { s } _ { c , d } ^ { o } \right) = \mathrm { s o f t m a x } \left( \mathrm { F F N N } _ { r } \left( \mathbf { g } _ { \mathbf { s } _ { a , b } ^ { a } , \mathbf { s } _ { c , d } ^ { o } } \right) \right)\tag{11}
$$

## 3.6 Training objective

The loss function for training is defined as the sum of the negative log-likelihoods from the span-pair classification in the span-classification and triplets modules:

$$
\begin{array} { r l } & { \mathcal { L } = - \displaystyle \sum _ { \mathbf { s } _ { i , j } \in S } \log P \left( \hat { t } _ { i , j } \mid \mathbf { s } _ { i , j } \right) } \\ & { \quad \quad - \displaystyle \sum _ { \mathbf { s } _ { a , b } ^ { t } \in S ^ { a } , \mathbf { s } _ { c , d } ^ { o } \in S ^ { o } } \log P \left( \hat { r } \mid \mathbf { s } _ { a , b } ^ { a } , \mathbf { s } _ { c , d } ^ { o } \right) } \end{array}\tag{12}
$$

where $\hat { t } _ { i , j }$ and $\hat { r } _ { i , j }$ are the ground truth labels for span $\mathbf { s } _ { i , j }$ and span-pair $( \mathbf { s } _ { a , b } ^ { a } , \mathbf { s } _ { c , d } ^ { o } )$ , respectively. S, $S ^ { a }$ and $S ^ { o }$ are the final span representation, pruned aspect and opinion candidate pools in dual-channel span generation, respectively.

## 4 Experiments

## 4.1 Datasets

To verify the effectiveness of our proposed model, we conduct experiments on four public datasets, i.e., Lap14, Res14, Res 15 and Res16, which come from the sentiment evaluation benchmarks SemEval 2014 (Pontiki et al., 2014), SemEval 2015 (Pontiki et al., 2015) and SemEval 2016 (Pontiki et al., 2016), respectively. Moreover, the four datasets have two versions: ASTE-Data-v1(D1 for short) (Peng et al., 2020) and ASTE-Data-v2(D2 for short) (Xu et al., 2020). Statistics for public datasets are shown in the appendix A.1.

## 4.2 Experimental Setting

We initialize word embedding with two different encoders: BiLSTM-based and BERT-based encoders. The hidden dimension of BiLSTM-based encoder is set to 300 with dropout rate 0.5. To alleviate overfitting, the input embedding dropout rate is 0.7. For the proposed model, we use the AdamW optimizer (Loshchilov and Hutter, 2017) with a learning rate of 1e-3 in the training. In the implementation of BERT-based encoding, the model parameters are optimized using Adamw with a maximum learning rate of 5e-5 and weight decay of 1e-2. We run the model for 20 training epochs. For other parameter groups, the same parameter settings are used for both embedding initialization schemes. The maximum span length $L _ { s }$ is fixed to 8, the span pruning threshold z is set to 0.5, and the part-of-speech window $S _ { w i n d o w }$ is 3. We choose the best model parameters based on the F1 score on the validation set and report the average of the results for 5 different random seeds.

## 4.3 Baselines

We compare our model to the following state-ofthe-art methods:

• Pipeline: including CMLA+ (Wang et al., 2017), RINANTE+ (Dai and Song, 2019), Li-unified-R (Li et al., 2019), Peng-twostage (Peng et al., 2020) and IMN+IOG (Wu et al., 2020b).

• End-to-end: OTE-MTL (Zhang et al., 2020), JET (Xu et al., 2020), GTS-CNN, GTS-BiLSTM, GTS-BERT (Wu et al., 2020a), $\mathrm { S ^ { 3 } E ^ { 2 } }$ (Chen et al., 2021b), BART-ABSA (Yan et al., 2021), MTDTN (Zhao et al., 2022), EMC-GCN (Chen et al., 2022c). These approaches are end-to-end models that include a unified grid tagging scheme and a positionaware tagging scheme.

• MRC: Dual-MRC (Mao et al., 2021), BMRC (Chen et al., 2021a), COM-MRC (Zhai et al., 2022). All these method are based on the framework of machine reading comprehension.

• Span-based: Span-ASTE (Xu et al., 2021), SBN (Chen et al., 2022d). Span-based models consider all possible spans in a sentence and match aspect terms with opinion terms in an end-to-end manner.

## 4.4 Main Results

We conduct experiments on the two versions of four benchmark datasets, i.e., D1 and D2, whose results are shown in Table 1 and Table 2, respectively. As can be seen from the two tables, under the comprehensive performance indicator F1, the proposed Dual-span consistently outperforms all baselines both for BiLTSM encoder and BERT encoder. Moreover, our model achieves the superior performance in precision and/or recall in most cases. On the other side, the experimental results suggest that non-pipeline methods (i.e., End-to-end, MRC-based, Span-based) are better than pipeline methods, which should be attributed to the fact that the pipeline methods do not consider the correlation between sentiment elements, thus leading to error propagation between stages. It is noteworthy that among tagging based end-to-end methods, some methods that employ syntactic structure of the sentence such as S3E2, MTDTN and EMC-GCN generally outperform the methods that only learn tagging information (e.g., OTE-MTL, GTS and JET), suggesting that the syntactic features of sentences are meaningful for triplet representation. In particular, our end-to-end Dual-Span model outperforms all end-to-end based methods and spanbased methods Span-ASTE, SBN, which can be attributed to the fact that our method not only utilizes the syntactic relationship and other linguistic features of sentences for span representation learning, but reduce the noise for span generation and pairing, which can facilitate valid span pairing. Specifically, the F1 score of Dua-Span on datasets D1 and D2 outperforms over other state-of-the-art models by about 2% on average.

<table><tr><td rowspan="2"></td><td rowspan="2">Category</td><td rowspan="2">Model</td><td colspan="3">Lap14</td><td colspan="3">Res14</td><td colspan="3">Res15</td><td colspan="3">Res16</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td rowspan="6">BILSTMM</td><td rowspan="2">Pipeline</td><td>Li-unified-R(2019) Peng-two-stage(2019)</td><td>42.25 48.62</td><td>42.78 45.52</td><td>42.47 47.02</td><td>41.44 58.89</td><td>68.79 60.41</td><td>51.68 59.64</td><td>43.34 51.07</td><td>50.73 46.04</td><td>46.69 48.71</td><td>38.19 59.25</td><td>53.47 58.09</td><td>44.51 58.67</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3"></td><td>IMN+IOG(2020)</td><td>49.21</td><td>46.23</td><td>47.68</td><td>59.57</td><td>63.88</td><td>61.65</td><td>55.24</td><td>52.33</td><td>53.75</td><td></td><td></td><td></td></tr><tr><td>GTS-CNN(2020)</td><td>55.93</td><td>47.52</td><td>51.38</td><td>70.79</td><td>61.71</td><td>65.94</td><td>60.09</td><td>53.57</td><td>56.64</td><td>62.63</td><td>66.98</td><td>64.73</td></tr><tr><td>GTS-BiLSTM(2020)</td><td>59.42</td><td>45.13</td><td>51.30</td><td>67.28</td><td>61.91</td><td>64.49</td><td>63.26</td><td>50.71</td><td>56.29</td><td>66.07</td><td>65.05</td><td>65.56</td></tr><tr><td rowspan="2">Span-based</td><td>S3E2(2021)</td><td>59.43</td><td>46.23</td><td>52.01</td><td>69.08</td><td>64.55</td><td>66.74</td><td>61.06</td><td>56.44</td><td>58.66</td><td></td><td>71.08 63.13</td><td></td><td>66.87</td></tr><tr><td></td><td>Span-ASTE(2021)</td><td>59.85</td><td>45.67</td><td>51.80</td><td>72.52 62.43</td><td>67.08</td><td>64.29</td><td></td><td>52.12</td><td>57.56</td><td>67.25</td><td>61.75</td><td>64.37</td></tr><tr><td rowspan="5">BERT</td><td rowspan="2">Ours</td><td>Dual-Span</td><td>60.14</td><td>47.65</td><td>53.53</td><td>74.82</td><td>61.97</td><td>68.19</td><td>64.71</td><td>57.37</td><td>60.76</td><td>73.47</td><td>62.46</td><td>67.49</td></tr><tr><td>MTDTN(2022)</td><td>61.98</td><td>54.71</td><td>58.12</td><td>70.00</td><td>71.78</td><td>70.88</td><td>59.03</td><td>62.68</td><td>60.80</td><td>69.04</td><td>69.98</td><td>69.51</td></tr><tr><td rowspan="2">End-to-end</td><td>EMC-GCN(2022)</td><td>61.46</td><td>55.56</td><td>58.32</td><td>71.85</td><td>72.12</td><td>71.98</td><td>59.89</td><td>61.05</td><td>60.38</td><td>65.08</td><td>71.66</td><td>68.18</td></tr><tr><td>BMRC(2021)</td><td></td><td></td><td>57.83</td><td></td><td></td><td>70.01</td><td></td><td></td><td>58.74</td><td></td><td></td><td>67.49</td></tr><tr><td rowspan="2">MRC-based Ours</td><td>COM-MRC(2022)</td><td>64.73</td><td>56.09</td><td>60.09</td><td>76.45</td><td>69.67</td><td>72.89</td><td>68.50</td><td>59.74</td><td>63.65</td><td>72.80</td><td>70.85</td><td>71.79</td></tr><tr><td>Dual-Span</td><td>64.50</td><td>58.59</td><td>61.36</td><td>77.55</td><td>73.52</td><td>75.47</td><td>67.66</td><td>66.14</td><td>66.85</td><td>72.44</td><td>73.47</td><td>72.94</td></tr></table>

Table 1: Experimental results on dataset D1, including two versions of BiLSTM and BERT. All baseline results are from the original paper. Best results are in bold and the second best are underlined.
<table><tr><td rowspan="2">Category</td><td rowspan="2">Model</td><td colspan="3">Lap14</td><td colspan="3">Res14</td><td colspan="3">Res15</td><td colspan="3">Res16</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td rowspan="4">Pipeline</td><td>CMLA+(2017)</td><td>30.09</td><td>36.92</td><td>33.16</td><td>39.18</td><td>47.13</td><td>42.79</td><td>34.56</td><td>39.84</td><td>37.01</td><td>41.34</td><td>42.10</td><td>41.72</td></tr><tr><td>RINANTE+(2019)</td><td>21.71</td><td>18.66</td><td>20.07</td><td>31.42</td><td>39.38</td><td>34.95</td><td>29.88</td><td>30.06</td><td>29.97</td><td>25.68</td><td>22.30</td><td>23.87</td></tr><tr><td>Li-unified-R(2019)</td><td>40.56</td><td>44.28</td><td>42.34</td><td>41.04</td><td>67.35</td><td>51.00</td><td>44.72</td><td>51.39</td><td>47.82</td><td>37.33</td><td>54.51</td><td>44.31</td></tr><tr><td>Peng-two-stage(2019)</td><td>37.38</td><td>50.38</td><td>42.87</td><td>43.24</td><td>63.66</td><td>51.46</td><td>48.07</td><td>57.51</td><td>52.32</td><td>46.96</td><td>64.24</td><td>54.21</td></tr><tr><td rowspan="5">End-to-end</td><td>OTE-MTL (2020)</td><td>49.53</td><td>39.22</td><td>43.42</td><td>62.00</td><td>55.97</td><td>58.71</td><td>56.37</td><td>40.94</td><td>47.13</td><td>62.88</td><td>52.10</td><td>56.96</td></tr><tr><td>JET-BERTM=6(2020)</td><td>55.39</td><td>47.33</td><td>51.04</td><td>70.56</td><td>55.94</td><td>62.40</td><td>64.45</td><td>51.96</td><td>57.53</td><td>70.42</td><td>58.37</td><td>63.83</td></tr><tr><td>GTS-BERT(2020)</td><td>57.52</td><td>51.92</td><td>54.58</td><td>70.92</td><td>69.49</td><td>70.20</td><td>59.29</td><td>58.07</td><td>58.67</td><td>68.58</td><td>66.86</td><td>67.58</td></tr><tr><td>BART-ABSA(2021)</td><td>61.41</td><td>56.19</td><td>58.69</td><td>65.52</td><td>64.99</td><td>65.25</td><td>59.14</td><td>59.38</td><td>59.26</td><td>66.60</td><td>68.68</td><td>67.62</td></tr><tr><td>EMC-GCN(2022)</td><td>61.46</td><td>55.56</td><td>58.32</td><td>71.85</td><td>72.12</td><td>71.98</td><td>59.89</td><td>61.05</td><td>60.38</td><td>65.08</td><td>71.66</td><td>68.18</td></tr><tr><td rowspan="3">MRC-based</td><td>Dual-MRC(2021)</td><td>57.39</td><td>53.88</td><td>55.58</td><td>71.55</td><td>69.14</td><td>70.32</td><td>63.78</td><td>51.87</td><td>57.21</td><td>68.60</td><td>66.24</td><td>67.40</td></tr><tr><td>BMRC(2021)</td><td>70.55</td><td>48.98</td><td>57.82</td><td>75.61</td><td>61.77</td><td>67.99</td><td>68.51</td><td>53.40</td><td>60.02</td><td>71.20</td><td>61.08</td><td>65.75</td></tr><tr><td>COM-MRC(2022)</td><td>62.35</td><td>58.16</td><td>60.17</td><td>75.46</td><td>68.91</td><td>72.01</td><td>68.35</td><td>61.24</td><td>64.53</td><td>71.55</td><td>71.59</td><td>71.57</td></tr><tr><td rowspan="2">Span-based</td><td>Span-ASTE-BERT(2021)</td><td>63.44</td><td>55.84</td><td>59.38</td><td>72.89</td><td>70.89</td><td>71.85</td><td>62.18</td><td>64.45</td><td>63.27</td><td>69.40</td><td>71.17</td><td>70.26</td></tr><tr><td>SBN(2022)</td><td>65.68</td><td>59.88</td><td>62.65</td><td>76.36</td><td>72.43</td><td>74.34</td><td>69.93</td><td>60.41</td><td>64.82</td><td>71.59</td><td>72.57</td><td>72.08</td></tr><tr><td>Ours(BERT)</td><td>Dual-Span</td><td>67.14</td><td>62.13</td><td>64.49</td><td>77.01</td><td>74.00</td><td>75.47</td><td>67.97</td><td>66.34</td><td>67.13</td><td>73.56</td><td>73.48</td><td>73.49</td></tr></table>

Table 2: Experimental results on dataset D2, all baselines are from the original text. Best results are in bold and the second best are underlined.

<table><tr><td>Model</td><td>Lap14</td><td>Res14</td><td>Res15</td><td>Res16</td></tr><tr><td>w/o Dual-RGAT</td><td>59.95</td><td>69.77</td><td>63.4</td><td>70.12</td></tr><tr><td>w/o SynGAT</td><td>61.62</td><td>70.12</td><td>63.02</td><td>70.13</td></tr><tr><td>w/o PosGAT</td><td>64.23</td><td>71.32</td><td>62.42</td><td>70.89</td></tr><tr><td>Transformer</td><td>63.16</td><td>72.14</td><td>63.32</td><td>72.49</td></tr><tr><td>Dual-GAT</td><td>61.71</td><td>72.67</td><td>64.23</td><td>72.13</td></tr><tr><td>w/o SynSpan</td><td>62.77</td><td>73.27</td><td>63.86</td><td>70.89</td></tr><tr><td>w/o PosSpan</td><td>64.39</td><td>73.46</td><td>65.51</td><td>71.34</td></tr><tr><td>Dual-Span</td><td>64.49</td><td>75.47</td><td>67.13</td><td>73.49</td></tr></table>

Table 3: Experimental results of ablation study.

<table><tr><td rowspan="2">D2</td><td rowspan="2">Model</td><td colspan="3">ATE</td><td colspan="3">OTE</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td rowspan="3">Lap14</td><td>GTS-BERT</td><td>76.63</td><td>82.68</td><td>79.53</td><td>76.11</td><td>78.44</td><td>77.25</td></tr><tr><td>Span-ASTE</td><td>81.48</td><td>86.39</td><td>83.86</td><td>83.00</td><td>82.28</td><td>82.63</td></tr><tr><td>Dual-Span</td><td>80.67</td><td>87.92</td><td>84.14</td><td>78.96</td><td>84.07</td><td>81.44</td></tr><tr><td rowspan="3">Res14</td><td>GTS-BERT</td><td>78.12</td><td>85.64</td><td>81.69</td><td>81.12</td><td>88.24</td><td>84.53</td></tr><tr><td>Span-ASTE</td><td>83.56</td><td>87.59</td><td>85.50</td><td>82.93</td><td>89.67</td><td>86.16</td></tr><tr><td>Dual-Span</td><td>83.19</td><td>89.95</td><td>86.44</td><td>83.38</td><td>88.99</td><td>86.10</td></tr><tr><td rowspan="3">Res15</td><td>GTS-BERT</td><td>75.13</td><td>81.57</td><td>78.21</td><td>74.96</td><td>82.52</td><td>78.49</td></tr><tr><td>Span-ASTE</td><td>78.97</td><td>84.68</td><td>81.72</td><td>77.36</td><td>84.86</td><td>80.93</td></tr><tr><td>Dual-Span</td><td>81.49</td><td>84.16</td><td>82.80</td><td>78.17</td><td>87.25</td><td>82.46</td></tr><tr><td rowspan="3">Res16</td><td>GTS-BERT</td><td>75.06</td><td>89.42</td><td>81.61</td><td>78.99</td><td>88.71</td><td>83.57</td></tr><tr><td>Span-ASTE</td><td>79.78</td><td>88.50</td><td>83.89</td><td>82.59</td><td>90.91</td><td>86.54</td></tr><tr><td>Dual-Span</td><td>78.45</td><td>90.24</td><td>83.93</td><td>82.24</td><td>88.26</td><td>85.14</td></tr></table>

Table 4: Experimental results of ATE and OTE tasks on dataset D2.

## 4.5 Model Analysis

## 4.5.1 Ablation Study

To further explore the effectiveness of different modules in Dual-Span, we conduct ablation experiments on the D2 dataset. Table 3 shows the experimental results in terms of F1 scores in D2. W/o SynGAT and w/o PosGAT denote the removal of syntactic graph attention network (SynGAT) and part-of-speech graph attention network (PosGAT), respectively, while W/o Dual-RGAT denotes the removal of both SynGAT and PosGAT. We also compare our approach with unitary graph attention networks Dual-GAT that performs attention convolution on syntactic dependency graphs and part-of-speech graphs respectively without distinguishing edge types. By comparing w/o SynGAT, w/o PosGAT, w/o Dual-RGAT and Dual-Span, we observe that both the dependency relationship and part-of-speech features of the sentence are informative to the representation of spans. In particular, the syntactic structural feature and part-of-speech information can be complementary. This is manifested by the outperformance of Dual-RGAT over Transformer and Dual-GAT.When removing the syntactic span generation module (corresponding to w/o SynSpan) or part-of-speech span generation module (corresponding to w/o PosSpan), the performance is also degraded. This observation illustrates that span candidate size can be effectively reduced. Overall, each module of our Dual-span contributes to the overall performance of the ASTE task.

## 4.5.2 Effectiveness of Dual-Span in Span Generation

We use two subtasks, namely, ATE and OTE of ABSA, to explore the effectiveness of dual-channel span Generation strategy. We evaluate our model on the D2 dataset with F1 metric and the results are shown in the table 4. On ATE task, Dual-Span is consistently superior to Span-ASTE and GTS, indicating that syntactic and part-of-speech correlation based candidate reduction and representation are effective for aspect term identification. However, on the OTE task, our model is slightly inferior to Span-ASTE on most of the benchmark datasets, which is caused by lower P values. We expect the reason behind lies in the part-of-speech based span generation. In effect, we only consider the spans involving words that are tagged with JJor NN. However, opinion terms can be tagged with V BN, which we do not include. We leave the expansion of more valid part-of-speech spans in the future work.

<table><tr><td>Model</td><td>Lap14</td><td>Res14</td><td>Res15</td><td>Res16</td></tr><tr><td>Span-ASTE</td><td>0.8579</td><td>1.1131</td><td>0.5368</td><td>0.6597</td></tr><tr><td>Dual-Span</td><td>0.4443</td><td>0.5587</td><td>0.2472</td><td>0.3169</td></tr></table>

Table 5: Experimental results of time consumption (second) to generate spans on the D2 dataset.

In order to verify that our proposed dual-channel span generation strategy can noticeably reduce the computational cost of span enumeration, we test the time consumption of Dual-span and Span-ASTE on span enumeration under the same runtime environment. From the results shown in Table 5, we can see that the proposed dual-channel span generation strategy cuts time cost in half.

## 5 Conclusion

In this work, we present a Dual-Span model for improving the performance on ASTE task. Based on the observations of syntactic relations and partof-speech features among spans, we design a dualchannel span generation method to refine the span candidate set so as to mitigate the negative impacts of invalid spans. Moreover, we employ relational graph neural networks to capture the high-order interactions between possible spans from both views: syntactic dependency relation and part-of-speech relation. Our experimental results demonstrate that the proposed method brings meaningful gains to ASTE as well as ATE task, compared to all baselines. We also note that for OTE task, our method is generally inferior to the vanilla span-based method that enumerate all possible spans. The reason may lie in the limited part-of-speech relations, which will be considered in the future work.

## Acknowledgements

This work is supported by National Nature Science Foundation (NO.62276099) and SWPU Innovation Foundation (NO.642).

## References

Xuefeng Bai, Pengbo Liu, and Yue Zhang. 2021. Investigating typed syntactic dependencies for targeted sentiment classification using graph attention neural network. IEEE ACM Trans. Audio Speech Lang. Process., 29:503–514.

Gianni Brauwers and Flavius Frasincar. 2023. A survey on aspect-based sentiment classification. volume 55, pages 65:1–65:37.

Hongjie Cai, Yaofeng Tu, Xiangsheng Zhou, Jianfei Yu, and Rui Xia. 2020. Aspect-category based sentiment analysis with hierarchical graph convolutional network. In COLING, pages 833–843. International Committee on Computational Linguistics.

Mohna Chakraborty, Adithya Kulkarni, and Qi Li. 2022. Open-domain aspect-opinion co-mining with doublelayer span extraction. In KDD, pages 66–75. ACM.

Chenhua Chen, Zhiyang Teng, Zhongqing Wang, and Yue Zhang. 2022a. Discrete opinion tree induction for aspect-based sentiment analysis. In ACL (1), pages 2051–2064. Association for Computational Linguistics.

Hao Chen, Zepeng Zhai, Fangxiang Feng, Ruifan Li, and Xiaojie Wang. 2022b. Enhanced multi-channel graph convolutional network for aspect sentiment triplet extraction. In ACL, pages 2974–2985, Dublin, Ireland. Association for Computational Linguistics.

Hao Chen, Zepeng Zhai, Fangxiang Feng, Ruifan Li, and Xiaojie Wang. 2022c. Enhanced multi-channel graph convolutional network for aspect sentiment triplet extraction. In ACL (1), pages 2974–2985. Association for Computational Linguistics.

Shaowei Chen, Yu Wang, Jie Liu, and Yuelin Wang. 2021a. Bidirectional machine reading comprehension for aspect sentiment triplet extraction. In AAAI, pages 12666–12674. AAAI Press.

Yuqi Chen, Keming Chen, Xian Sun, and Zequn Zhang. 2022d. A span-level bidirectional network for aspect sentiment triplet extraction. In EMNLP, pages 4300– 4309. Association for Computational Linguistics.

Zhexue Chen, Hong Huang, Bang Liu, Xuanhua Shi, and Hai Jin. 2021b. Semantic and syntactic enhanced aspect sentiment triplet extraction. In ACL/IJCNLP (Findings), volume ACL/IJCNLP 2021 of Findings of ACL, pages 1474–1483. Association for Computational Linguistics.

Zhuang Chen and Tieyun Qian. 2020. Enhancing aspect term extraction with soft prototypes. In EMNLP, pages 2107–2117. Association for Computational Linguistics.

Kyunghyun Cho, Bart van Merrienboer, Çaglar Gülçehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In EMNLP, pages 1724–1734. ACL.

Hongliang Dai and Yangqiu Song. 2019. Neural aspect and opinion term extraction with mined rules as weak supervision. In ACL (1), pages 5268–5277. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT (1), pages 4171–4186. Association for Computational Linguistics.

Chunning Du, Haifeng Sun, Jingyu Wang, Qi Qi, Jianxin Liao, Tong Xu, and Ming Liu. 2019. Capsule network with interactive attention for aspect-level sentiment classification. In EMNLP, pages 5488– 5497. Association for Computational Linguistics.

Lei Gao, Yulong Wang, Tongcun Liu, Jingyu Wang, Lei Zhang, and Jianxin Liao. 2021. Question-driven span labeling model for aspect-opinion pair extraction. In AAAI, pages 12875–12883. AAAI Press.

Ruidan He, Wee Sun Lee, Hwee Tou Ng, and Daniel Dahlmeier. 2019. An interactive multi-task learning network for end-to-end aspect-based sentiment analysis. In ACL (1), pages 504–515. Association for Computational Linguistics.

Mengting Hu, Shiwan Zhao, Li Zhang, Keke Cai, Zhong Su, Renhong Cheng, and Xiaowei Shen. 2019. CAN: constrained attention networks for multi-aspect sentiment analysis. In EMNLP, pages 4600–4609. Association for Computational Linguistics.

Hao Li and Wei Lu. 2017. Learning latent sentiment scopes for entity-level sentiment analysis. In AAAI, pages 3482–3489. AAAI Press.

Kun Li, Chengbo Chen, Xiaojun Quan, Qing Ling, and Yan Song. 2020. Conditional augmentation for aspect term extraction via masked sequence-tosequence generation. In ACL, pages 7056–7066. Association for Computational Linguistics.

Qimai Li, Zhichao Han, and Xiao-Ming Wu. 2018. Deeper insights into graph convolutional networks for semi-supervised learning. In Proceedings ofthe Thirty-Second AAAI Conference on Artificial Intelligence, pages 3538–3545.

Ruifan Li, Hao Chen, Fangxiang Feng, Zhanyu Ma, Xiaojie Wang, and Eduard H. Hovy. 2021. Dual graph convolutional networks for aspect-based sentiment analysis. In ACL, pages 6319–6329. Association for Computational Linguistics.

Xin Li, Lidong Bing, Piji Li, and Wai Lam. 2019. A unified model for opinion target extraction and target sentiment prediction. In AAAI, pages 6714–6721. AAAI Press.

Jian Liu, Zhiyang Teng, Leyang Cui, Hanmeng Liu, and Yue Zhang. 2021. Solving aspect category sentiment analysis as a text generation task. In EMNLP (1), pages 4406–4416. Association for Computational Linguistics.

Shu Liu, Kaiwen Li, and Zuhe Li. 2022. A robustly optimized BMRC for aspect sentiment triplet extraction. In NAACL, pages 272–278. Association for Computational Linguistics.

Ilya Loshchilov and Frank Hutter. 2017. Fixing weight decay regularization in adam. CoRR, abs/1711.05101.

Dehong Ma, Sujian Li, Fangzhao Wu, Xing Xie, and Houfeng Wang. 2019. Exploring sequence-tosequence learning in aspect term extraction. In ACL, pages 3538–3547. Association for Computational Linguistics.

Yue Mao, Yi Shen, Chao Yu, and Longjun Cai. 2021. A joint training dual-mrc framework for aspect based sentiment analysis. In AAAI, pages 13543–13551. AAAI Press.

Haiyun Peng, Lu Xu, Lidong Bing, Fei Huang, Wei Lu, and Luo Si. 2020. Knowing what, how and why: A near complete solution for aspect-based sentiment analysis. In AAAI, pages 8600–8607. AAAI Press.

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. Glove: Global vectors for word representation. In EMNLP, pages 1532–1543. ACL.

Maria Pontiki, Dimitris Galanis, Haris Papageorgiou, Ion Androutsopoulos, Suresh Manandhar, Mohammad Al-Smadi, Mahmoud Al-Ayyoub, Yanyan Zhao, Bing Qin, Orphée De Clercq, Véronique Hoste, Marianna Apidianaki, Xavier Tannier, Natalia V. Loukachevitch, Evgeniy V. Kotelnikov, Núria Bel, Salud María Jiménez Zafra, and Gülsen Eryigit. 2016. Semeval-2016 task 5: Aspect based sentiment analysis. In SemEval@NAACL-HLT, pages 19–30. The Association for Computer Linguistics.

Maria Pontiki, Dimitris Galanis, Haris Papageorgiou, Suresh Manandhar, and Ion Androutsopoulos. 2015. Semeval-2015 task 12: Aspect based sentiment analysis. In SemEval@NAACL-HLT, pages 486–495. The Association for Computer Linguistics.

Maria Pontiki, Dimitris Galanis, John Pavlopoulos, Harris Papageorgiou, Ion Androutsopoulos, and Suresh Manandhar. 2014. Semeval-2014 task 4: Aspect based sentiment analysis. In SemEval@COLING, pages 27–35. The Association for Computer Linguistics.

Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton, and Christopher D. Manning. 2020. Stanza: A python natural language processing toolkit for many human languages. In ACL (demo), pages 101–108. Association for Computational Linguistics.

Martin Schmitt, Simon Steinheber, Konrad Schreiber, and Benjamin Roth. 2018. Joint aspect and polarity classification for aspect-based sentiment analysis with end-to-end neural networks. In EMNLP, pages 1109–1114. Association for Computational Linguistics.

Kim Schouten and Flavius Frasincar. 2016. Survey on aspect-level sentiment analysis. volume 28, pages 813–830.

Duyu Tang, Bing Qin, and Ting Liu. 2016. Aspect level sentiment classification with deep memory network. In EMNLP, pages 214–224. The Association for Computational Linguistics.

Hao Tang, Donghong Ji, Chenliang Li, and Qiji Zhou. 2020. Dependency graph enhanced dual-transformer structure for aspect-based sentiment classification. In ACL, pages 6578–6588. Association for Computational Linguistics.

Maria Mihaela Trusca and Flavius Frasincar. 2023. Survey on aspect detection for aspect-based sentiment analysis. volume 56, pages 3797–3846.

Hai Wan, Yufei Yang, Jianfeng Du, Yanan Liu, Kunxun Qi, and Jeff Z. Pan. 2020. Target-aspect-sentiment joint detection for aspect-based sentiment analysis. In AAAI, pages 9122–9129. AAAI Press.

Wenya Wang, Sinno Jialin Pan, Daniel Dahlmeier, and Xiaokui Xiao. 2017. Coupled multi-layer attentions for co-extraction of aspect and opinion terms. In AAAI, pages 3316–3322. AAAI Press.

Yequan Wang, Minlie Huang, Xiaoyan Zhu, and Li Zhao. 2016. Attention-based LSTM for aspectlevel sentiment classification. In EMNLP, pages 606– 615. The Association for Computational Linguistics.

Zhen Wu, Chengcan Ying, Fei Zhao, Zhifang Fan, Xinyu Dai, and Rui Xia. 2020a. Grid tagging scheme for aspect-oriented fine-grained opinion extraction. In Findings-EMNLP, pages 2576–2585, Online. Association for Computational Linguistics.

Zhen Wu, Fei Zhao, Xin-Yu Dai, Shujian Huang, and Jiajun Chen. 2020b. Latent opinions transfer network for target-oriented opinion words extraction. In AAAI, pages 9298–9305. AAAI Press.

Lu Xu, Yew Ken Chia, and Lidong Bing. 2021. Learning span-level interactions for aspect sentiment triplet extraction. In ACL, pages 4755–4766. Association for Computational Linguistics.

Lu Xu, Hao Li, Wei Lu, and Lidong Bing. 2020. Position-aware tagging for aspect sentiment triplet extraction. In EMNLP, pages 2339–2349. Association for Computational Linguistics.

Wei Xue and Tao Li. 2018. Aspect based sentiment analysis with gated convolutional networks. In ACL, pages 2514–2523. Association for Computational Linguistics.

Hang Yan, Junqi Dai, Tuo Ji, Xipeng Qiu, and Zheng Zhang. 2021. A unified generative framework for aspect-based sentiment analysis. In ACL/IJCNLP (1), pages 2416–2429. Association for Computational Linguistics.

Bishan Yang and Claire Cardie. 2012. Extracting opinion expressions with semi-markov conditional random fields. In EMNLP, pages 1335–1345. ACL.

Bishan Yang and Claire Cardie. 2013. Joint inference for fine-grained opinion extraction. In ACL, pages 1640–1649. The Association for Computer Linguistics.

Yichun Yin, Furu Wei, Li Dong, Kaimeng Xu, Ming Zhang, and Ming Zhou. 2016. Unsupervised word and dependency path embeddings for aspect term extraction. In IJCAI, pages 2979–2985. IJCAI/AAAI Press.

Zepeng Zhai, Hao Chen, Fangxiang Feng, Ruifan Li, and Xiaojie Wang. 2022. COM-MRC: A contextmasked machine reading comprehension framework for aspect sentiment triplet extraction. In EMNLP, pages 3230–3241. Association for Computational Linguistics.

Chen Zhang, Qiuchi Li, Dawei Song, and Benyou Wang. 2020. A multi-task learning framework for opinion triplet extraction. In EMNLP (Findings), volume EMNLP 2020 of Findings ofACL, pages 819–828. Association for Computational Linguistics.

He Zhao, Longtao Huang, Rong Zhang, Quan Lu, and Hui Xue. 2020. Spanmlt: A span-based multi-task learning framework for pair-wise aspect and opinion terms extraction. In ACL, pages 3239–3248. Association for Computational Linguistics.

Yichun Zhao, Kui Meng, Gongshen Liu, Jintao Du, and Huijia Zhu. 2022. A multi-task dual-tree network for aspect sentiment triplet extraction. In COLING, pages 7065–7074. International Committee on Computational Linguistics.

## A Appendix

## A.1 Dataset statistics

We counted the data of the two versions separately <sup>2</sup> <sup>3</sup>. As shown in the Table 6, in addition to the number of sentences and triples in the dataset, the number of words and multi-word spans is also counted. In addition, we perform part-of-speech statistics on the D1 and D2 data sets. The results are shown in Table 7. The number of parts of speech with JJ or NN accounts for a relatively high proportion of the overall $A \& O ,$ , and the distribution of the rest of the parts of speech is scattered and unrepresentative. As there is a trade-off between prediction accuracy and time consumption, we only consider spans involving words tagged with JJ or NN in Section 4.5.2.

## A.2 Hyperparameter analysis

Figure 4 shows the sensitivity analysis of the hyperparameters $S _ { w i n d o w } , L _ { s } ,$ , on the D2 dataset. From the figure, we can observe that the effect is the best when the part-of-speech window $S _ { w i n d o w }$ is 3. In fact, when the part-of-speech window is set to $S _ { w i n d o w } = 3 .$ , it can basically cover all aspect terms and opinion terms whose parts-of-speech are NN and JJ. When $S _ { w i n d o w }$ exceeds 3, more noise and complexity may be introduced. When the hyperparameter $L _ { s } { = } 8$ , the performance is the best.

## A.3 Impact of SynGAT and PosGAT Layers

To explore the impact of the number of layers of SynGAT and PosGAT in Dual-Span, we evaluate the number of layers of SynGAT and PosGAT on the D2 dataset, where multiple layers indicate that node information can be propagated to higher-order neighbors. As shown in Figure 6, our model achieves the best performance when both SynGAT and PosGAT are two layers. Specifically, on the syntactic dependency tree of a sentence, 2-hops are helpful for the interaction between aspect and opinion terms, while on the part-of-speech graph, 2-hop relations involving NN or JJ are conducive for capturing valid spans. Note that, the performance declines as the Dual-RGAT goes deeper, which may be due to the oversmoothing (Li et al., 2018) of graph neural networks.

![](images/1ec711ee7591ed8233fb8fc966a5a80139c313a1769ce4a0083b0386025dfa65.jpg)

![](images/cab47505472759f900abd69b2d1b799037e180cf67a4aa6d12344e4b4c58b2cf.jpg)  
Figure 4: Sensitivity analysis of the hyperparameters $S _ { w i n d o w }$ and $L _ { s }$ on dataset D2.

![](images/cbde69effdf908d77ce742eab9023bc10470b57c0365cc6ed2d432c6d742764c.jpg)  
Figure 5: Performance v.s the number of layers in graph networks.

## A.4 Visualization of Linguistic Correlations

To explore how the syntactic dependency and partof-speech correlation between words contribute to valid span generation, we visualize the attention scores of the syntactic relations and part-of-speech adjacency matrix in two RGAT modules, where rows are queries and columns are keys. As shown in the figure 6, the sampled text “Good creative rolls !” (i.e., the fourth example in the section A.5) contains the triplet: (“rolls”, “good creative”, “positive”). Since our model employs syntactic dependency relation to learn representations, and exploits the part-of-speech information around indicative words (e.g., the words with “NN" tag) as well, the inter-span and intra-span relations can be successfully captured by graph attention networks. So the aspect term “rolls” pays attention to opinion terms “good" and “ creative" (the last row of the left plot in Figure 6), while the two words with “JJ" tag, i.e., “good" and “ creative" shows more strong correlation than with “rolls" in part-of-speech graph (corresponding to the right plot of Figure 6), demonstrating that they are more likely to fall into the same category. As a result, our model gives the correct triplet, in contrast to Span-ASTE whose prediction is (“creative rolls”, “good”, “positive”).

<table><tr><td rowspan="2"></td><td rowspan="2">Dataset</td><td colspan="4">Lap14</td><td colspan="4">Res14</td><td colspan="4">Res15</td><td colspan="4">Res16</td></tr><tr><td>#S</td><td>#T</td><td>#SW</td><td>#MW</td><td>#S</td><td>#T</td><td>#SW</td><td>#MW</td><td>#S</td><td>#T</td><td>#SW</td><td>#MW</td><td>#S</td><td>#T</td><td>#SW</td><td>#MW</td></tr><tr><td></td><td>Train</td><td>899</td><td>1452</td><td>815</td><td>637</td><td>1259</td><td>2356</td><td>1614</td><td>742</td><td>603</td><td>1038</td><td>696</td><td>342</td><td>863</td><td>1421</td><td>931</td><td>490</td></tr><tr><td>D1</td><td>dev</td><td>225</td><td>383</td><td>213</td><td>170</td><td>315</td><td>580</td><td>386</td><td>194</td><td>151</td><td>239</td><td>165</td><td>74</td><td>216</td><td>348</td><td>232</td><td>116</td></tr><tr><td></td><td>test</td><td>332</td><td>547</td><td>291</td><td>256</td><td>493</td><td>1008</td><td>674</td><td>334</td><td>325</td><td>493</td><td>303</td><td>190</td><td>328</td><td>525</td><td>351</td><td>174</td></tr><tr><td>D2</td><td>Train</td><td>906</td><td>1460</td><td>824</td><td>636</td><td>1266</td><td>2338</td><td>1586</td><td>752</td><td>605</td><td>1013</td><td>678</td><td>335</td><td>857</td><td>1394</td><td>918</td><td>476</td></tr><tr><td></td><td>dev</td><td>219</td><td>346</td><td>190</td><td>156</td><td>310</td><td>577</td><td>388</td><td>189</td><td>148</td><td>249</td><td>165</td><td>84</td><td>210</td><td>339</td><td>216</td><td>123</td></tr><tr><td></td><td>test</td><td>328</td><td>543</td><td>291</td><td>252</td><td>492</td><td>994</td><td>657</td><td>337</td><td>322</td><td>485</td><td>297</td><td>188</td><td>326</td><td>514</td><td>344</td><td>170</td></tr></table>

Table 6: Statistics for the two versions of the dataset. #S and #T represent the number of sentences and the number of triplets, respectively. #SW indicates that the aspect and opinion terms in the triplets are both single-word spans, while #MW indicates that at least one of the aspect or opinion terms are multi-word spans.

<table><tr><td rowspan="2">Dataset</td><td colspan="3">Lap14</td><td colspan="3">Res14</td><td colspan="3">Res15</td><td colspan="3">Res16</td></tr><tr><td>A&amp;O</td><td>JJ&amp;NN</td><td>Rat</td><td>A&amp;O</td><td>JJ&amp;NN</td><td>Rat</td><td>A&amp;O</td><td>JJ&amp;NN</td><td>Rat</td><td>A&amp;O</td><td>JJ&amp;NN</td><td>Rat</td></tr><tr><td>D1</td><td>4447</td><td>2949</td><td>0.66</td><td>7350</td><td>5944</td><td>0.81</td><td>3282</td><td>2714</td><td>0.83</td><td>4262</td><td>3523</td><td>0.83</td></tr><tr><td>D2</td><td>4698</td><td>3113</td><td>0.66</td><td>7818</td><td>6346</td><td>0.81</td><td>3494</td><td>2882</td><td>0.82</td><td>4494</td><td>3707</td><td>0.82</td></tr></table>

Table 7: Aspect and opinion term part-of-speech statistics on public datasets. Among them, A&O, JJ&NN represent the number of aspect and opinion terms and the number of parts of speech are JJ and NN, respectively, and Rat represents their ratio.

<table><tr><td>Review</td><td>Ground-truth</td><td>Span-ASTE</td><td>Dual-Span</td></tr><tr><td>The baterry is very longer.</td><td>(baterry, longer, P)</td><td>(baterry, longer, N)</td><td>(baterry, longer, P)</td></tr><tr><td>And windows 7 works like a charm.</td><td>(windows 7, charm, P)</td><td>0</td><td>(windows 7, charm, P)</td></tr><tr><td>I wanted it for it &#x27;s mobility and man,this little bad boy is very nice.</td><td>(mobility, nice, P)</td><td>(mobility, wanted, P), (mobility, nice, P)</td><td>(mobility, nice, P)</td></tr><tr><td>Good creative rolls !</td><td>(rolls, good creative, P)</td><td>(creative rolls, good, P)</td><td>(rolls, good creative, P)</td></tr><tr><td>The wine list was extensive-though the staff did not seem knowledgeable about wine pairings .</td><td>(wine list, extensive, P), (staff, not seem knowledgeable, N)</td><td>(wine list, extensive, P)</td><td>(wine list, extensive, P), (staff, not seem knowledgeable, N)</td></tr><tr><td>for 7 years they have put out the most tasty, most delicious food and kept it that way .</td><td>(food, tasty, P), (food, delicious, P)</td><td>(food, delicious, P)</td><td>(food, tasty, P), (food, delicious, P)</td></tr></table>

Table 8: Case study on dataset D2.

<table><tr><td>Review</td><td>Ground-truth(part of speech)</td><td>Dual Span(part of speech)</td></tr><tr><td>The OS is fast and fluid, everything</td><td>fast(JJ), organized(VBN),</td><td>fast(JJ),</td></tr><tr><td>is organized and it &#x27;s just beautiful. This place has ruined me for neighborhood sushi.</td><td>fluid(NN), beautiful(JJ) ruined(VBN)</td><td>fluid(NN), beautiful(JJ) ∅</td></tr><tr><td>I think the pizza is so</td><td>overrated(JJ),</td><td></td></tr><tr><td>overrated and was under cooked. Decor needs to be upgraded but the food is amazing!</td><td>under cooked(IN, VBN) upgraded(VBN), amazing(JJ)</td><td>overrated(JJ) amazing(JJ)</td></tr></table>

Table 9: A case study of prediction errors in OTE tasks on the D2 dataset.

<table><tr><td>Review</td><td>A&amp;O(part of speech)</td><td>Ground-truth</td><td>Dual-Span</td></tr><tr><td>The OS is fast and fluid, everything is organized and it &#x27;s just beautiful.</td><td>os(NNP), fast(JJ), fluid(NN), organized(VBN), beautiful(JJ)</td><td>(os, fast, P), (os, organized, P), (os, fluid, P), (os, beautiful, P)</td><td>(os, fast, P), (os, fluid, P), (os, beautiful, P)</td></tr><tr><td>This place has ruined me for neighborhood sushi.</td><td>sushi(NN), ruined(VBN)</td><td>(sushi, ruined, P)</td><td>0</td></tr><tr><td>I think the pizza is so overrated and was under cooked.</td><td>pizza(NN), overrated(JJ), under cooked(IN, VBN)</td><td>(pizza, overrated, N), (pizza, under cooked, N)</td><td>(pizza, overrated, N)</td></tr><tr><td>Decor needs to be upgraded but the food is amazing!</td><td>&#x27;decor(NN), food(NN), upgraded(VBN), amazing(JJ)</td><td>(decor, upgraded, N), (food, amazing, P)</td><td>(food, amazing, P)</td></tr><tr><td>The manager was rude and handled the situation extremely poorly.</td><td>manager(NN), rude(VBN)</td><td>(manager, rude, N)</td><td>(manager, rude, N), (manager, poorly, N), (situation, poorly, N)</td></tr></table>

Table 10: A case study of prediction errors on the D2 dataset.

![](images/54d50dc3b6fc62db253cc705bd0a89f13c108abf8d6e3380ca002cd128a5e954.jpg)

(a) Review: “Good creative rolls!”  
![](images/a816630c549d04b96518e5e4f36ee54098bf7222f09edafc2044399ab83436af.jpg)  
(b) Review: “for 7 years they have put out the most tasty, most delicious food and kept it that way...”  
Figure 6: Visualization of adjacency tensors of syntactic (left) and part-of-speech combination features (right).

## A.5 Case Study

We use several examples from the test set of dataset D2 to analyze and validate our method, as shown in the Table 8. For the first example, our method Dual-Span may perform better in predicting sentiment consistency than Span-ASTE. On the 2nd,

5th, and 6th examples, it can be shown that our method makes full use of syntactic and semantic information to improve the accuracy of effective span capture. It can be shown in the fourth example that our method reduces the span boundary error by using part-of-speech structural features.

## A.6 Error analysis

In order to explore the reason behind the slight inferiority of our model on the OTE task, compared to Span-ASTE on most benchmark datasets, we conduct error case study analysis on the datasets of the D2 version. As shown in Table 9, since we do not use the tag “VBN” in constructing partof-speech graph, our method fails to extract the opinion words with the part of speech "V BN" in the four examples, e.g. “organized (VBN)” and “upgraded(VBN)”. Additionally, to identify the limitations of our work and potential areas for improvement in the future, we perform error sample analysis on the D2 dataset. As shown in Table 10, for opinions whose part of speech is not JJ, our method is more likely to give wrong prediction results. Moreover, there are some non-aspect or opinion words whose part-of-speech are NN or JJ, which also mislead the model to make wrong span identification.