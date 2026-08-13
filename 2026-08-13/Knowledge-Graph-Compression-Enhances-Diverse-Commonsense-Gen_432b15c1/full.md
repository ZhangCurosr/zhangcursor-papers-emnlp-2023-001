# Knowledge Graph Compression Enhances Diverse Commonsense Generation

EunJeong Hwang<sup>1,2</sup>, Veronika Thost<sup>3</sup>, Vered Shwartz<sup>1,2</sup>, Tengfei Ma<sup>4</sup>

<sup>1</sup> University of British Columbia <sup>2</sup> Vector Institute for AI

<sup>3</sup> MIT-IBM Watson AI Lab, IBM Research

<sup>4</sup> Stony Brook University {ejhwang,vshwartz}@cs.ubc.ca veronika.thost@ibm.com tengfei.ma@stonybrook.edu

## Abstract

Generating commonsense explanations requires reasoning about commonsense knowledge beyond what is explicitly mentioned in the context. Existing models use commonsense knowledge graphs such as ConceptNet to extract a subgraph of relevant knowledge pertaining to concepts in the input. However, due to the large coverage and, consequently, vast scale of ConceptNet, the extracted subgraphs may contain loosely related, redundant and irrelevant information, which can introduce noise into the model. We propose to address this by applying a differentiable graph compression algorithm that focuses on more salient and relevant knowledge for the task. The compressed subgraphs yield considerably more diverse outputs when incorporated into models for the tasks of generating commonsense and abductive explanations. Moreover, our model achieves better quality-diversity tradeoff than a large language model with 100 times the number of parameters. Our generic approach can be applied to additional NLP tasks that can benefit from incorporating external knowledge.<sup>1</sup>

## 1 Introduction

Commonsense knowledge graphs (CSKGs) have been used to improve the performance of downstream applications such as question answering (Yasunaga et al., 2021) and dialogue (Tu et al., 2022), as well as for enhancing neural models for commonsense reasoning tasks (Lin et al., 2019; Yu et al., 2022). Typically, these methods extract keywords from the input and construct a subgraph around them using the KG knowledge, which is then incorporated into the model.

Recent popular CSKGs such as ConceptNet (Speer et al., 2017) and ATOMIC (Sap et al., 2019) represent nodes in natural language, which allows flexibility but also adds redundancy and noise (Wu et al., 2023). Moreover, the retrieved subgraphs around a task’s concepts potentially include information that is not relevant to the context. For example, in Figure 1, the goal is to generate a reason why the input sentence (“A shark interviews a fish”) defies commonsense. The concepts tank and business are semantically irrelevant to either the input or the reference output sentences. Including irrelevant information introduces noise that can deteriorate the model’s performance. Recent work has addressed this by pruning noisy paths based on low edge confidence scores in knowledge base embeddings (Lin et al., 2019) or by using language models (LMs) (Yasunaga et al., 2021). Yet, the relevance of paths is not determined in relation to the given task.

![](images/a151b4aad4498fe43e08dfa795f23d958ccdbcbbeae4e787f2e491d6463da664.jpg)  
Figure 1: An example from ComVE (Wang et al., 2020). The subgraph obtained for the input sentence includes unimportant information (in red) that can lead to noisy outputs.

In this paper, we propose to use differentiable graph compression that enables the model to learn how to select the crucial concepts that are actually related to the task. Our method contains two main components: using self-attention scores to select relevant concept nodes in the retrieved subgraph, and employing optimal transport loss to ensure the chosen concepts preserve the most crucial information of the original graph. In this way, the irrelevant or redundant concepts can be automatically eliminated in the subgraph.

We demonstrate the usefulness of our method on two commonsense generation tasks: commonsense explanation generation and abductive commonsense reasoning. Our method outperforms a range of baselines that use KGs in terms of both diversity and quality of the generations. We further conduct a comprehensive analysis, exploring a different setup, such as the scenario of incorporating new knowledge into the subgraph. Different from the baselines, our method enables the model to maintain performance, even in the presence of potentially increased noisy data. Finally, we show that our approach demonstrates better quality-diversity tradeoff than the large language model vicuna-13b, which has 100 times more parameters.

## 2 Background

KG-Enhanced Neural Methods. KGs have been used to enhance models for question answering (Lin et al., 2019; Feng et al., 2020; Yasunaga et al., 2021), relation classification (Wang et al., 2021), textual entailment (Kapanipathi et al., 2020), and more. Typically, such methods extract a subgraph of knowledge related to keywords in the input, which is then either embedded or represented in natural language before being incorporated into the model. For example, both Wang et al. (2023) and Wang, Fang, et al. (2023) used CSKGs to enhance a commonsense inference and a QA model by including the abstraction of concepts in the input (e.g. vacation  relaxing event). However, some knowledge may be irrelevant in the context of the particular question.

To reduce such noise, prior methods have proposed to score and prune the paths. Lin et al. (2019) used TransE (Wang et al., 2014) to score each edge in the path, while Yasunaga et al. (2021) scores nodes based on the likelihood of a pre-trained LM to generate it after the input. In both methods, the scores are not trained to represent a node’s importance in relation to the task.

Generating Commonsense Explanations. This paper focuses on the task of generating commonsense explanations, in particular focusing on the following datasets. In ComVE (Wang et al., 2020) the goal is to generate explanations for why a given sentence, such as “A shark interviews a fish”, does not make sense. α-NLG (Bhagavatula et al., 2020) presents models with a past observation, such as “Mike spends a lot of his time on the internet” and a future observation such as “Now other people love the internet because of Mike’s website”. The goal is to generate a plausible explanation for what might have happened in-between, such as “Mike created a website that helps people search”. In a related line of work, researchers collected or generated commonsense explanations for existing tasks (e.g., Camburu et al., 2018; Rajani et al., 2019; Brahman et al., 2021).

Diverse Sentence Generation. One of the desired aspects of generating commonsense explanations is the diversity of the outputs. Popular LM decoding methods such as top-k (Fan et al., 2018), top-p (Holtzman et al., 2020), and truncated sampling (Hewitt et al., 2022) generate diverse outputs by pruning the probability distribution over the vocabulary for the next token and then sampling a token from the pruned distribution. An alternative approach is to use a mixture of experts (MoE) to produce diverse outputs (Shen et al., 2019; Cho et al., 2019). Our approach extends MoKGE Yu et al. (2022), a model for commonsense explanation generation. MoKGE uses a combination of KGs to diversify the outputs of a MoE model. However, the knowledge that MoKGE retrieves from the KG is not filtered, hence may contain loosely related, redundant and irrelevant information, which can negatively impact the model’s performance in generating high-quality diverse outputs. In our approach, we employ knowledge graph compression to prioritize more important information.

## 3 Method

Our goal is to generate diverse sentences, $\left\{ y _ { 1 } , y _ { 2 } , . . . , y _ { k } \right\}$ that explain a given instance x (see Sec 2 for the specific task descriptions). The objective is to maximize the probability of generating each $y _ { i } \colon P ( y _ { i } | x )$ , as well as to diversify them. Previous KG-enhanced approaches usually add an external graph $\mathcal { G } _ { x }$ to make the generation also conditioned on the graph: $P ( y _ { i } | x , \mathcal { G } _ { x } )$ . However, as we discussed in Sec 1, $\mathcal { G } _ { x }$ often contains redundancy or noise. For example, given a target concept A, there is a semantically similar concept (e.g. a synonym) $A ^ { \prime }$ and a noisy concept B in the graph $\mathcal { G } _ { x } )$ Obviously, $A ^ { \prime }$ will negatively impact the diversity of generations because the model may select both

![](images/8cfede9e78cbe54be68101516870517137f3650b29664ce9a9cd23ec5ead5ba3.jpg)  
Figure 2: Overview of our approach. We retrieve a subgraph from ConceptNet for the given input sentence, compress it, and use MoE to generate diverse sentences for containing concepts from the compressed graph.

A and A′ for generation and the semantics of the generations are similar; concept B will hurt the generation quality since it is irrelevant to the context. So, a natural idea to solve the problem is to eliminate these concepts by compressing the graph.

Our method extends MoKGE (Yu et al., 2022) by compressing the retrieved external knowledge graph. The framework is illustrated in Figure 2 and described in detail subsequently. In a nutshell, it aims to identify the concepts within the KG that provide the most relevant knowledge for a particular instance. We first extract a subgraph from the KG based on the given input sentence, and encode it into a vector representation (Sec 3.1). Then, we learn a compressed graph that maintains only the most relevant concepts for the given instance (Sec 3.2). We train the model with the corresponding losses (Sec 3.3) and finally apply MoE to generate diverse outputs (Sec 3.4).

## 3.1 KG Subgraph Extraction and Encoding

The subgraph extraction and encoding follows MoKGE (Yu et al., 2022).

Subgraph Extraction. We first associate each input sentence with the set of concepts from the KG that match its tokens. For example, given the sentence $q = \sp { \ast } \mathbf { A }$ shark interviews a fish” (the “query”), we extract the concepts $C _ { q } =$ fish, shark, interview from ConceptNet.<sup>2</sup> Second, we fix a radius h and extract a subgraph $\mathcal { G } _ { q }$ with node set $V _ { q } \supseteq C _ { q }$ from the KG such that it contains all KG nodes and edges that are up to h = 2 hops around the concepts in $C _ { q }$ (e.g. shark

swim  fish).

Graph Encoding. To obtain embeddings for the concept nodes, we apply an off-the-shelf graph encoder over the extracted subgraph (Wu et al., 2021). In our implementation, we follow Yu et al. (2022) and use the relational graph convolutional network (R-GCN; Schlichtkrull et al., 2018). R-GCN computes node representations by iteratively aggregating neighboring node representations and thereby taking the relation types into account. In this way, the final embeddings capture the structural patterns of the subgraph.

## 3.2 Differentiable Graph Compression

As we discussed before, the extracted subgraphs often contain redundancy and noise, and we aim to compress the graph and remove the irrelevant information. This introduces two challenges: (1) how to make the graph compression differentiable so that it can be trained in the context of downstream tasks; and (2) how to maintain the most important and relevant information in the compressed graph.

Self-Attention for Concept Scoring. Since we want to select concepts for the generation step (Sec 3.4), we can’t apply differentiable pooling methods (Ying et al., 2018; Ma and Chen, 2020) and instead choose to construct a semantically meaningful subgraph containing the relevant nodes and edges. To do so, we apply self-attention and hence essentially use the features computed in the previous step as main criterion to determine the concepts’ importance. Specifically, we compute self-attention scores $Z \in \mathbb { R } ^ { C \times 1 }$ as proposed by Lee et al. (2019) using graph convolution (Kipf

and Welling, 2017):

$$
Z = \sigma ( \tilde { D } ^ { - \frac { 1 } { 2 } } \tilde { A } \tilde { D } ^ { - \frac { 1 } { 2 } } X \Theta _ { a t t } )
$$

where σ is the non-linear activation function tanh; $C : = | V _ { q } |$ is the number of concept nodes in the subgraph; $\tilde { A } \in \mathbb { R } ^ { C \times C }$ is the adjacency matrix extended by self-connections; $\tilde { D }$ is the degree matrix of ${ \tilde { A } } ,$ , which is used for normalization; $\bar { \boldsymbol { X } } \in \mathbb { R } ^ { C \times F }$ is the matrix of concept embeddings obtained in the previous step, with embedding dimension $F ;$ and $\mathbf { \bar { \Theta } } _ { \mathbf { \Lambda } } \mathbf { \Theta } _ { \Theta _ { a t t } } \in \mathbb { R } ^ { F \times 1 }$ is the parameter matrix for the self-attention scores. Given the concept scores $Z ,$ we consider a pre-set assignment ratio $s \in ( 0 , 1 ]$ and form the compressed graph, $\mathcal { G } ^ { \prime }$ , by selecting $s \%$ of concept nodes. We denote S as the number of concept nodes selected. In the example in Figure 2, the compressed (third) graph contains 80% of the nodes in the original subgraph.

Optimal Transport for Regularization. The self-attention based concept selection make the graph compressed in an differentiable way, however the attention parameters can only be trained from downstream generation tasks which cannot gurantee the compression quality as well as generalizability. Consider the case with concept A and its synonym $A ^ { \prime }$ in the retrieved graph $\mathcal { G } _ { q } ,$ , if $A$ is selected by the attention scores, it is highly possible A′ also has a high score to be selected, so the redundancy cannot be removed.

For this reason, we additionally apply optimal transport (OT; Peyré and Cuturi, 2019), a method commonly used for measuring the distance between two probability measures. Here, we regard a graph as a discrete distribution, similarly to Ma and Chen (2020), and minimize the OT distance between the original graph and its compressed version. To this end, we define an optimal transport loss between graphs. Given a m-node graph and a n-node graph, we assume they have discrete distributions $\begin{array} { r } { \mu = \sum _ { i = 1 } ^ { m } a _ { i } \sigma _ { x _ { i } } } \end{array}$ and $\begin{array} { r } { \nu = \sum _ { j = 1 } ^ { n } b _ { j } \sigma _ { x _ { j } } , } \end{array}$ where $x _ { i }$ and $x _ { j }$ indicate the nodes, σ is a delta function, $\boldsymbol { a } = ( a _ { 1 } , . . . , a _ { m } )$ and $b = ( b _ { 1 } , . . . , b _ { n } )$ are weights of nodes (generally uniform). If we define a cost matrix M whose element $M _ { i j }$ indicates the transport cost from node $x _ { i }$ to node $x _ { j }$ , then the optimal transport distance is:

$$
W ( \mu , \nu ) = \operatorname* { m i n } _ { T } < T , M >\tag{1}
$$

$T \in \mathbf { R } ^ { m * n }$ is called a transportation plan, whose element $T _ { i j }$ denotes the transportation probability from $x _ { i }$ to $x _ { j } .$ , and it meets the requirements that $T 1 _ { n } = a .$ , and $T ^ { T } 1 _ { m } = b$

Once the optimal transport distance is minimized, the compressed graph is expected to keep as much information of the original graph. Thus redundant concepts will be largely removed, since involving them in the compressed graph will lead to less information kept. Take a simple example, given an original graph with nodes $\{ A , A ^ { \prime } , C \}$ , the subgraph with node $\{ A , C \}$ should be more informative than the one with $\{ A , A ^ { \prime } \}$ , and its optimal transport distance between the original graph should be smaller.

Since solving an OT problem is computationally expensive, we add an entropy regularization term $\begin{array} { r } { E ( T ) = \sum _ { i j } T _ { i j } ( \log T _ { i j } - 1 ) } \end{array}$ , to allow for solving it approximately using Sinkhorn’s algorithm (Cuturi, 2013) in practice, following prior work. With a hyperparameter $\gamma > 0$ , the entropy-regularized loss becomes:

$$
W _ { \gamma } ( \mu , \nu ) = \operatorname* { m i n } _ { T } < T , M > - \gamma E ( T )\tag{2}
$$

## 3.3 Loss Functions for Training

Following Yu et al. (2022), we train BART-base (Lewis et al., 2020) in a seq2seq architecture on the commonsense explanation generation task, with a generation loss, and apply a KG concept loss in addition. We also include an optimal transport loss.

Generation Loss. For sentence generation, we maximize the conditional probability of the target sequence y given the input sequence x concatenated with the selected KG concepts $c _ { 1 } , c _ { 2 } , . . . c _ { S }$ We utilize the standard auto-regressive crossentropy loss as follows:

$$
\mathcal { L } _ { \mathrm { g } } = - \sum _ { t = 1 } ^ { | y | } \log P ( y _ { t } | x , c _ { 1 } , c _ { 2 } , . . . , c _ { S } , y _ { < t } )
$$

where t is the timestep of the actual output. In the generation step, the model auto-regressively generates the output y with input x and S selected concepts.

KG Concept Loss. The effectiveness of the concept selection can be measured in terms of which of the chosen concepts appear in the output sentence a (the reference answer). More specifically, we consider a regular binary cross entropy loss with targets $y _ { c } = I ( c \in V _ { q } \cap C _ { a } )$ for each $c \in V _ { q }$ . Here, $I ( \cdot )$ represents the indicator function. and $C _ { a }$ is the set of concepts that are present in the output. To obtain a probability for each of the $S$ concepts in the compressed graph, we apply an MLP. The resulting loss is as follows:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c } } = - \left( \sum _ { c \in V _ { q } \cap C _ { a } } y _ { c } \log P ( c ) + \sum _ { c \in V _ { q } - C _ { a } } ( 1 - y _ { c } ) \log 1 - P ( c ) \right) } \end{array}
$$

Optimal Transport Loss. To make the optimal transport distance differentiable, we solve Eq. 2 using the Sinkhorn’s algorithm (Cuturi, 2013):

Starting with any positive vector $v ^ { 0 }$ , we iteratively update u and v as follows:

$$
u ^ { i + 1 } = a \oslash K v ^ { i } ; v ^ { i + 1 } = b \oslash K ^ { T } u ^ { i + 1 }\tag{3}
$$

where $\oslash$ is the element-wise division and $K$ is an intermediate variable derived from the cost matrix $M \colon K = \exp ( - M / \gamma )$

After k steps, we arrive at the k-step result $P ^ { k } =$ $\mathrm { d i a g } ( u ^ { k } ) K \bar { \mathrm { d i a g } } ( v ^ { k } )$ as an approximated optimal transportation plan, hence the optimal transport loss is approximated by

$$
\mathcal { L } _ { \mathrm { t } } = W _ { \gamma } ^ { k } ( G , G _ { c } ) = < P ^ { k } , M > - \gamma E ( P ^ { k } )
$$

Altogether, our model is trained with three loss functions:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { g } } + \alpha \mathcal { L } _ { \mathrm { c } } + \beta \mathcal { L } _ { \mathrm { t } }\tag{4}
$$

where α and $\beta$ are hyperparameters that control the relative importance of the individual loss functions. In our experimental setup, we set both α and $\beta$ to a value of 0.3.

## 3.4 Diverse Generation based on MoE

To encourage more diverse outputs, we follow previous work (Shen et al., 2019; Cho et al., 2019; Yu et al., 2022) and use mixture of experts (MoE).

We use K experts, where each expert is responsible for generating a unique set of KG concepts. The model is trained using hard-EM algorithm (Dempster et al., 1977). Since it is similar to (Yu et al., 2022)), we put the details in Appendix E. In Figure 2, the nodes in the 4th graph highlighted in green, red, and blue colors indicate the $K = 3$ respective experts assigned to handle different concepts. The utilization of our compressed graph version helps the model better prioritize the crucial concepts during output generation, as we demonstrate in our experiments.

## 4 Experimental Setup

## 4.1 Datasets

ComVE (Wang et al., 2020) was part of the SemEval 2020 commonsense validation task. Given a nonsensical sentence, the task is to generate explanations for why it doesn’t make sense. The dataset contains 10k training examples and roughly 1000 examples each for test and validation. Each example comes with 3 reference output sentences. The other dataset, α-NLG (Bhagavatula et al., 2020), addresses the abductive commonsense reasoning task. Given a past observation and a future observation, the goal is to generate plausible explanations for what might have happened in-between. The dataset consists of 50k training examples, 1,779 validation and 3,560 test examples. Each example in the dataset includes up to 5 reference outputs.

## 4.2 Baselines

MoE-based Methods. MoE-embed (Cho et al., 2019) and MoE-prompt (Shen et al., 2019) produce diverse sentences by sampling different mixture components. While MoE-embed employs independent latent variables when generating diverse outputs, MoE-prompt shares the latent variable between the experts. MoKGE (Yu et al., 2022) is the approach that we extend by adding graph compression. It generates outputs by incorporating KG concepts on top of MoE-based methods.

Other Methods to Improve Diversity. To show that our method yields a sophisticated concept selection beyond regular filtering, we compare it to a simple synonym filtering on top of MoKGE, applied during the inference step, that yields a set of unique KG concepts for generating outputs. This baseline prevents the model from selecting similar concepts when generating the outputs. Second, we consider the common pruning approach, which removes irrelevant paths from the potentially noisy subgraph, following KagNet (Lin et al., 2019). To measure the quality of the path, the path is decomposed into a set of triples. Each triple is scored based on the scoring function of the knowledge graph embedding technique, TransE (Bordes et al., 2013) and the score for each path is the product of its triple scores. The threshold for pruning is a hyperparameter and set to 0.15 following Lin et al. (2019).

Large Language Model (LLM). Lastly, we compare to Vicuna-13b (Chiang et al., 2023). This large LM was built upon LLaMA-13b (Touvron et al., 2023), a transformer-based LM trained on trillions of tokens exclusively sourced from publicly available data. Vicuna-13b performs on par with ChatGPT (Chiang et al., 2023). We employ Vicuna-13b in a 2-shot setup (see Appendix A for the prompts).

<table><tr><td>ComVE</td><td>self-bleu-3 (↓)</td><td>self-bleu-4 (↓)</td><td>distinct-2 (个)</td><td>entropy-4 (介)</td><td>bleu-4 (介个)</td><td>rouge-1 (个)</td></tr><tr><td>MoE, embed</td><td> $3 3 . 6 4 _ { 0 . 2 }$ </td><td> $2 8 . 2 1 _ { 0 . 1 }$ </td><td> $4 6 . 5 7 _ { 0 . 2 }$ </td><td> $9 . 6 1 _ { 0 . 1 }$ </td><td> $1 8 . 6 6 _ { 0 . 5 }$ </td><td> $4 3 . 7 2 _ { 0 . 2 }$ </td></tr><tr><td>MoKGE, embed</td><td> $3 5 . 3 6 _ { 1 . 1 }$ </td><td> $2 9 . 7 1 _ { 1 . 2 }$ </td><td> $4 7 . 5 1 _ { 0 . 4 }$ </td><td> $9 . 6 3 _ { 0 . 1 }$ </td><td> $\mathbf { 1 9 . 1 3 } _ { 0 . 1 }$ </td><td> $4 3 . 7 _ { 0 . 1 }$ </td></tr><tr><td> $+ \mathrm { S A G } + \mathrm { O T } \left( \mathrm { o u r s } \right)$ </td><td> $3 2 . 1 9 _ { 0 . 6 }$ </td><td> $2 6 . 2 8 _ { 0 . 6 }$ </td><td> $\mathbf { 4 9 . 0 5 } _ { 0 . 1 }$ </td><td> $\mathbf { 9 . 6 9 } _ { 0 . 0 }$ </td><td> $1 9 . 0 8 _ { 0 . 2 }$ </td><td> $4 3 . 6 5 _ { 0 . 3 }$ </td></tr><tr><td>MoE, prompt</td><td> $3 3 . 4 2 _ { 0 . 3 }$ </td><td> $2 8 . 4 _ { 0 . 3 }$ </td><td> $4 6 . 9 3 _ { 0 . 2 }$ </td><td> $9 . 6 \phantom { } _ { 0 . 2 }$ </td><td> $1 8 . 9 1 _ { 0 . 4 }$ </td><td> $4 3 . 7 1 _ { 0 . 5 }$ </td></tr><tr><td>MoKĠE, prompt</td><td> $3 0 . 9 3 _ { 0 . 9 }$ </td><td> $2 5 . 3 _ { 1 . 1 }$ </td><td> $4 8 . 4 4 _ { 0 . 2 }$ </td><td> $9 . 6 7 _ { 0 . 2 }$ </td><td> $1 9 . 0 1 _ { 0 . 1 }$ </td><td> $4 3 . 8 3 _ { 0 . 3 }$ </td></tr><tr><td>+ filtering</td><td> $3 4 . 0 1 _ { 0 . 5 }$ </td><td> $2 8 . 9 2 _ { 0 . 5 }$ </td><td> $4 7 . 4 9 _ { 0 . 9 }$ </td><td> $9 . 6 4 _ { 0 . 1 }$ </td><td> $1 9 . 0 2 _ { 0 . 4 }$ </td><td> $4 3 . 4 8 _ { 0 . 6 }$ </td></tr><tr><td>+ pruning</td><td> $3 3 . 4 3 _ { 2 . 0 }$ </td><td> $2 8 . 2 7 _ { 2 . 2 }$ </td><td> $4 8 . 2 6 _ { 0 . 7 }$ </td><td> $9 . 6 4 _ { 0 . 0 }$ </td><td> $1 8 . 6 7 _ { 0 . 2 }$ </td><td> $4 3 . 1 0 _ { 0 . 3 }$ </td></tr><tr><td> $+ \operatorname { S A G } \left( \mathrm { o u r s } \right)$ </td><td> $2 8 . 4 6 _ { 0 . 8 }$ </td><td> $2 2 . 8 1 _ { 1 . 2 }$ </td><td> $4 8 . 3 3 _ { 0 . 6 }$ </td><td> $9 . 6 6 _ { 0 . 0 }$ </td><td> $1 9 . 0 0 _ { 0 . 6 }$ </td><td> $4 3 . 8 0 _ { 0 . 5 }$ </td></tr><tr><td> $+ \mathrm { S A G } + \mathrm { O T } \left( \mathrm { o u r s } \right)$ </td><td> $2 7 . 3 2 _ { 0 . 3 }$ </td><td> $2 1 . 9 4 _ { 0 . 4 }$ </td><td> ${ \bf 4 8 . 9 4 } _ { 0 . 1 }$ </td><td> $\mathbf { 9 . 6 9 } _ { 0 . 0 }$ </td><td> $\mathbf { 1 9 . 3 1 } _ { 0 . 3 }$ </td><td> $4 4 . 1 6 _ { 0 . 1 }$ </td></tr><tr><td>α-NLG</td><td>self-bleu-3 (↓)</td><td> $\mathrm { s e l f - b l e u } { - 4 } \left( \downarrow \right)$ </td><td>distinct-2 (介)</td><td> $\mathrm { e n t r o p y { - } 4 } \left( \Uparrow \right)$ </td><td> $\mathsf { b l e u } \mathrm { - } 4 \left( \Uparrow \right)$ </td><td> $\mathrm { r o u g e { - } l } \left( \Uparrow \right)$ </td></tr><tr><td>MoE, embed</td><td> $2 9 . 0 2 _ { 1 . 0 }$ </td><td> $2 4 . 1 9 _ { 1 . 0 }$ </td><td> $3 6 . 2 2 _ { 0 . 3 }$ </td><td> $1 0 . 8 4 _ { 0 . 0 }$ </td><td> ${ \bf 1 4 . 3 1 } _ { 0 . 2 }$ </td><td> $3 8 . 9 1 _ { 0 . 2 }$ </td></tr><tr><td> $\mathbf { M o K G E } , \mathbf { e m b e d }$ </td><td> $2 9 . 1 7 _ { 1 . 5 }$ </td><td> $2 4 . 0 4 _ { 1 . 6 }$ </td><td> $3 8 . 1 5 _ { 0 . 3 }$ </td><td> $1 0 . 9 _ { 0 . 1 }$ </td><td> $1 3 . 7 4 _ { 0 . 2 }$ </td><td> $3 8 . 0 6 _ { 0 . 2 }$ </td></tr><tr><td> $+ \mathrm { S A G } + \mathrm { O T } \left( \mathrm { o u r s } \right)$ </td><td> $2 4 . 9 8 _ { 0 . 2 }$ </td><td> $\mathbf { 1 9 . 8 3 } _ { 0 . 2 }$ </td><td> $3 8 . 9 3 _ { 0 . 3 }$ </td><td> $\mathbf { 1 0 . 9 3 } _ { 0 . 0 }$ </td><td> $1 3 . 0 6 _ { 0 . 3 }$ </td><td> $3 7 . 7 7 _ { 0 . 3 }$ </td></tr><tr><td>MoE, prompt</td><td> $2 8 . 0 5 _ { 2 . 0 }$ </td><td> $2 3 . 1 8 _ { 1 . 9 }$ </td><td> $3 6 . 7 1 _ { 0 . 1 }$ </td><td> $1 0 . 8 5 _ { 0 . 0 }$ </td><td> $1 4 . 2 6 _ { 0 . 3 }$ </td><td> $3 8 . 7 8 _ { 0 . 4 }$ </td></tr><tr><td>MoKGE, prompt</td><td> $2 7 . 4 0 _ { 2 . 0 }$ </td><td> $2 2 . 4 3 _ { 2 . 4 }$ </td><td> $3 8 . 0 1 _ { 0 . 6 }$ </td><td> $1 0 . 8 8 _ { 0 . 2 }$ </td><td> $1 4 . 1 7 _ { 0 . 2 }$ </td><td> $3 8 . 8 2 _ { 0 . 7 }$ </td></tr><tr><td>+ filtering</td><td> $3 1 . 3 8 _ { 2 . 9 } $ </td><td> $2 6 . 3 6 _ { 2 . 8 }$ </td><td> $3 7 . 9 5 _ { 0 . 6 }$ </td><td> $1 0 . 7 8 _ { 0 . 6 }$ </td><td> $1 3 . 8 9 _ { 0 . 2 }$ </td><td> $3 8 . 0 7 _ { 0 . 1 }$ </td></tr><tr><td>+ pruning</td><td> $3 1 . 8 4 _ { 2 . 3 }$ </td><td> $2 6 . 7 2 _ { 2 . 4 }$ </td><td> $3 8 . 2 1 _ { 0 . 2 }$ </td><td> $1 0 . 7 8 _ { 0 . 0 }$ </td><td> $1 3 . 7 3 _ { 0 . 1 }$ </td><td> $3 8 . 0 1 _ { 0 . 2 }$ </td></tr><tr><td>+ SAG (ours)</td><td> $2 8 . 4 9 _ { 0 . 8 }$ </td><td> $2 3 . 5 9 _ { 0 . 5 }$ </td><td> $3 8 . 0 5 _ { 0 . 4 }$ </td><td> $1 0 . 8 6 _ { 0 . 0 }$ </td><td> $1 3 . 4 1 _ { 0 . 5 }$ </td><td> $3 8 . 0 0 _ { 0 . 3 }$ </td></tr><tr><td>+ SAG+OT (ours)</td><td> $2 3 . 9 9 _ { 0 . 7 }$ </td><td> $\mathbf { 1 8 . 8 0 } _ { 0 . 6 }$ </td><td> $\mathbf { 3 9 . 0 2 } _ { 0 . 7 }$ </td><td> $\mathbf { 1 0 . 8 8 } _ { 0 . 0 }$ </td><td> $1 4 . 2 1 _ { 0 . 5 }$ </td><td> $3 8 . 9 3 _ { 0 . 2 }$ </td></tr></table>

Table 1: Diversity and quality evaluation on ComVE and α-NLG datasets. All experiments are run three times with different random seeds, and the standard deviations are reported in subscript.

tribution within the generated sentence.

## 4.3 Metrics

Following the same evaluation setting in previous works, we assess the performance of the generated sentences in terms of both diversity and quality.

Quality. We use standard metrics for automatic evaluation of generative tasks: BLEU (Papineni et al., 2002) and ROUGE (Lin, 2004), which are based on n-gram overlap between the generated sentences and human-written reference outputs. They assess the highest accuracy by comparing the best generated sentences to the target sentences.

## 5 Results and Discussion

Pairwise Diversity. Self-BLEU (Zhu et al., 2018) is used to evaluate how each sentence is similar to the other generated sentences based on n-gram overlap. Self-BLEU-3/4 are diversity scores based on 3/4-gram overlap. The metrics compute the average of sentence-level self-BLEU scores between all pairwise combinations of generated outputs. Hence, lower self-BLEU scores indicate a greater variety between the sentences in the set generated for each input sample.

Corpus Diversity. Distinct-k (Li et al., 2016) is calculated by counting the number of unique kgrams in the generated sentence and dividing it by the total number of generated tokens, to prevent preference towards longer sentences. Additionally, we report entropy-k (Zhang et al., 2018), which evaluates the evenness of the empirical n-gram dis-

Comparison to Baselines, Table 1. We observe similar trends for the two datasets and across the two model series, based on embedding and prompts. Overall, the differences are strongest for self-BLEU and Distinct-2, two aspects that are particularly important for diverse generation. This suggests that our model is able to reason about different possible contexts. On both datasets, our method, MoKGE+SAG+OT, outperforms the mixtures of experts by large margins in terms of diversity and, at the same time, achieves comparable or better performance in terms of quality. Note that, on ComVE, the quality differences between the best and our, second-best model are within standard deviation.

The effectiveness of our approach is especially evident from the comparison to the filtering and pruning baselines. Recall that these approaches similarly aim at better exploiting the KG by improving diversity and removing noise, respectively. However, we observe a considerable decrease in diversity and nearly always also slightly in quality. This shows that simple solutions, unrelated to the task at hand, are seemingly not able to identify the most relevant knowledge. More specifically, for the filtering baseline, we observed that the model is unable to learn what concepts to choose for unseen data. As a result, its ability to generalize to unseen data is limited, resulting in lower diversity scores on the test data. Altogether, this demonstrates that our approach, based on the compressed graph, is effective in suppressing redundant information present in the KG and promoting other knowledge that is more relevant in the given context.

We additionally confirm that our optimal transport loss helps the model to keep the KG subgraph more coherently; see especially the α-NLG results.

Generation Examples, Figure 4. Observe that MoKGE tends to generate sentences with simpler structure and fewer concepts, whereas our model employs a broader range of KG concepts. This makes the generations effectively more similar to the human-written ones, where each of the three sentences addresses a different context. We show more examples in Appendix B.

Testing Robustness with Potentially more Redundancy and Noise, Table 2. We created a more challenging scenario by extending the extracted subgraphs with additional, related knowledge potentially including more of both relevant and redundant information. This was done by applying COMET (Bosselut et al., 2019), a transformer that was trained to generate KG triples (i.e., entity-relation-entity tuples) based on given entityrelation pairs. The original MoKGE model seems to struggle in this scenario: its performance decreases without exception in terms of all metrics. In contrast, our approach, applied on top of MoKGE, is successful in both retaining the performance of MoKGE alone and even the improvements of MoKGE+SAG+OT.

Comparison with Large Language Model, Table 3 & Figure 4. We compare our method to Vicuna-13b. Most interestingly, our proposal outperforms the LLM on Distinct-2 and Entropy-4. Note that even MoKGE alone is slightly better than the LLM in these aspects, yet our method is effective in extending the gap by better exploiting the external knowledge. Figure 4 gives example outputs and shows that the LLM is still prone to generating sentences with similar structure (e.g. “I wore a wig to ...”), as it can be seen with α-NLG. Furthermore, while the generated sentence “I wore a wig to a party and felt great” explains the first observation “I always wondered why I loved wearing wigs”, it fails to explain the second observation “I got beat up and reminded of why I shouldn’t”. In the ComVE task, the generated sentences are diverse in terms of both sentence structure and word usage, but the model sometimes generates sentences that are less logical, such as “Writing in a paper with an eraser is not permanent”. In contrast, our approach enables MoKGE to generate a wider range of sentences that incorporate relevant concepts and enhance the interpretability of the generation process.

![](images/1ee6ff4eaf765277a44c365fe6cc96d59acc943cfe0901ea2f075078419b8751.jpg)  
Figure 3: Self BLEU-3, Distinct-2, and ROUGE-l per assignment ratio on α-NLG dataset. MoKGE-prompt with Self Attention and Optimal Transport is used for the experiment.

## 6 Analysis

Compression Ratios, Figure 3. This hyperparameter determines the amount of concept nodes to be kept in the compressed subgraph. Maintaining 65% of the nodes in the subgraph yields the optimal performance in terms of both diversity and quality on both datasets (see Appendix C for ComVE dataset). Interestingly, we do not observe a great negative impact on performance, even up to a ratio of 30%. This shows that ConceptNet apparently contains much information that is not necessarily beneficial for diverse generations in the context of a particular task and hence justifies our proposal.

Unique Concepts in the Output, Appendix D. The comparison of MoKGE and MoKGE+SAG+OT shows that MoKGE tends to generate more sentences containing 0 or 1 concepts only. This indicates that the lower diversity scores of MoKGE may be due to the selection of irrelevant concepts during output generation, showing the model’s inability to effectively utilize them. The table shows that our method increases the numbers of KG knowledge actually used by the model and thus its success in injecting external bias into LMs.

<table><tr><td>ComVE</td><td>self-bleu-3 (↓)</td><td>self-bleu-4 (↓)</td><td>distinct-2 (个)</td><td>entropy-4 (介)</td><td></td><td>bleu-4 (个) rouge-1 (介)</td></tr><tr><td>MoKGE</td><td> $3 0 . 9 3 _ { 0 . 9 }$ </td><td> $2 5 . 3 _ { 1 . 1 }$ </td><td> $4 8 . 4 4 _ { 0 . 2 }$ </td><td> $9 . 6 7 _ { 0 . 2 }$ </td><td> $1 9 . 0 1 _ { 0 . 1 }$ </td><td> $4 3 . 8 3 _ { 0 . 3 }$ </td></tr><tr><td>+COMET</td><td> $3 2 . 7 3 _ { 1 . 5 }$ </td><td> $2 7 . 4 5 _ { 1 . 8 }$ </td><td> $4 8 . 3 2 _ { 0 . 2 }$ </td><td> $9 . 6 4 _ { 0 . 0 }$ </td><td> $1 8 . 6 8 _ { 0 . 3 }$ </td><td> $4 3 . 5 1 _ { 0 . 4 }$ </td></tr><tr><td> $+ \mathrm { C O M E T + S A G + O T }$ </td><td> $2 7 . 2 3 _ { 1 . 2 }$ </td><td> $2 1 . 6 8 _ { 1 . 5 }$ </td><td> $4 8 . 6 5 _ { 0 . 6 }$ </td><td> $\mathbf { 9 . 6 8 } _ { 0 . 0 }$ </td><td> $1 9 . 3 8 _ { 0 . 4 }$ </td><td> $4 3 . 9 9 _ { 0 . 4 }$ </td></tr><tr><td>α-NLG</td><td>self-bleu-3 (↓)</td><td>self-bleu-4 (↓)</td><td>distinct-2 (介)</td><td>entropy-4 (介)</td><td></td><td>bleu-4 (个) rouge-1 (个)</td></tr><tr><td>MoKGE</td><td> $2 7 . 4 0 _ { 2 . 0 }$ </td><td> $2 2 . 4 3 _ { 2 . 4 }$ </td><td> $3 8 . 0 1 _ { 0 . 6 }$ </td><td> $\mathbf { 1 0 . 8 8 } _ { 0 . 2 }$ </td><td> $\mathbf { 1 4 . 1 7 } _ { 0 . 2 }$ </td><td> $3 8 . 8 2 _ { 0 . 7 }$ </td></tr><tr><td>+COMET</td><td> $3 1 . 4 1 _ { 2 . 4 }$ </td><td> $2 6 . 3 2 _ { 2 . 4 }$ </td><td> $3 7 . 9 9 _ { 0 . 2 }$ </td><td> $1 0 . 7 7 _ { 0 . 1 }$ </td><td> $1 3 . 8 7 _ { 0 . 3 }$ </td><td> $3 7 . 9 6 _ { 0 . 1 }$ </td></tr><tr><td> $+ \mathrm { C O M E T + S A G + O T }$ </td><td> $2 5 . 4 8 _ { 1 . 0 }$ </td><td> $2 1 . 1 4 _ { 1 . 3 }$ </td><td> $3 8 . 3 6 _ { 0 . 3 }$ </td><td> $1 0 . 8 4 _ { 0 . 0 }$ </td><td> $1 4 . 0 7 _ { 0 . 4 }$ </td><td> $3 8 . 6 5 _ { 0 . 4 }$ </td></tr></table>

Table 2: Performance with COMET and COMET with Self Attention and Optimal Transport. MoKGE-prompt is used for experiments.
<table><tr><td>ComVE</td><td>self-bleu-3 (↓)self-bleu-4 (↓)</td><td></td><td></td><td>distinct-2 (介) entropy-4 (介) bleu-4 (介) rouge-1 (介)</td><td></td><td></td></tr><tr><td>Vicuna-13b  $\mathbf { M o K G E + S A G + O T }$ </td><td> $\mathbf { 1 8 . 1 0 } _ { 0 . 0 }$   $2 7 . 3 2 _ { 0 . 3 }$ </td><td> $1 2 . 7 4 _ { 0 . 0 }$   $2 1 . 9 4 _ { 0 . 4 }$ </td><td> $4 8 . 4 0 _ { 0 . 0 }$   ${ \bf 4 8 . 9 4 } _ { 0 . 1 }$ </td><td> $9 . 6 5 _ { 0 . 0 }$   $\mathbf { 9 . 6 9 } _ { 0 . 0 }$ </td><td> $1 7 . 6 5 _ { 0 . 0 }$   $\mathbf { 1 9 . 3 1 } _ { 0 . 3 }$ </td><td> $4 3 . 9 7 _ { 0 . 0 }$   $4 4 . 1 6 _ { 0 . 1 }$ </td></tr><tr><td>α-NLG</td><td></td><td></td><td></td><td>self-bleu-3 (↓) self-bleu-4 (↓) distinct-2 (介) entropy-4 (介) bleu-4 (介) rouge-1(介)</td><td></td><td></td></tr><tr><td>Vicuna-13b</td><td> $3 3 . 2 3 _ { 0 . 0 }$ </td><td> $2 7 . 3 9 _ { 0 . 0 }$ </td><td> $3 7 . 9 7 _ { 0 . 0 }$ </td><td></td><td></td><td></td></tr><tr><td> $\mathbf { M o K G E + S A G + O T }$ </td><td> $2 3 . 9 9 _ { 0 . 7 }$ </td><td> $\mathbf { 1 8 . 8 0 } _ { 0 . 6 }$ </td><td> $\mathbf { 3 9 . 0 2 } _ { 0 . 7 }$ </td><td> $1 0 . 3 8 _ { 0 . 0 }$   $\mathbf { 1 0 . 8 8 } _ { 0 . 0 }$ </td><td> $\mathbf { 1 7 . 3 0 } _ { 0 . 0 }$   $1 4 . 2 1 _ { 0 . 5 }$ </td><td> $\mathbf { 4 0 . 5 8 } _ { 0 . 0 }$   $3 8 . 9 3 _ { 0 . 2 }$ </td></tr></table>

Table 3: Comparison between Vicuna-13b with 2-shot setup and MoKGE with SAG Pooling. MoKGE-prompt is used for experiments. Vicuna-13b was ran 1 time.  
![](images/9397fa637296347289a0e229e9f0104f9a5ed21d6c33a7028fe92c1e2d2531a4.jpg)  
Figure 4: Examples of model generated sentences using MoKGE, Vicuna-13b, and MoKGE with Self Attention + Optimal Transport.

Human Evaluation, Table 4. We conducted human evaluation on the outputs produced by our model MoKGE+SAG+OT and the baseline MoKGE for the α-NLG task. We randomly selected 30 generations from each model. The annotation was performed by 3 researchers in the lab. We instructed the annotators to score the diversity and correctness (quality) of each generation on a scale of 0 to 3. Table 4 shows a consistent performance improvement across both diversity and quality when comparing our model to the baseline.

<table><tr><td>Model</td><td>diversity</td><td>quality</td></tr><tr><td>MoKGE</td><td>1.88</td><td>1.93</td></tr><tr><td>MoKGE+SAG+OT</td><td>2.10</td><td>2.08</td></tr></table>

Table 4: Human evaluation performance on 30 randomly selected α-NLG samples.

## 7 Conclusion

We present a differentiable graph compression algorithm that enables the model to focus on crucial information. Through experiments on two commonsense explanation generation tasks, we show that our approach not only improves the diversity but also maintains the quality of outputs. Moreover, our graph compression helps the model regain performance when new and potentially noisy information is added to graphs. Our work opens up future research in effectively selecting and incorporating symbolic knowledge into NLP models.

## Limitations

Use of Single Word Concept. Since ConceptNet contains mostly single words, we limit additional KG concepts to single words only. However, it can easily be extended into phrases and we leave it to future work to investigate how to effectively utilize longer phrases.

Use of Relations. When additional KG concepts are added to the model, we focus more on the concept nodes, not the edges. However, relation edges may provide additional insight. We leave the exploration of this for future work.

## Ethics Statement

Data The datasets used in our work, SemEval-2020 Task 4 Commonsense Validation and Explantation (ComVE; Wang et al., 2020) and Abductive Commonsense Reasoning (α-NLG; Bhagavatula et al., 2020), are publicly available. The two datasets aim to produce commonsense explanations and do not include any offensive, hateful, or sexual content. The commonsense knowledge graph,

ConceptNet, was collected through crowdsourcing and may also introduce bias to our model (Mehrabi et al., 2021). However, we only use single word nodes from ConceptNet, which limits the impact of such bias.

Models The generative models presented in the paper are trained on a large-scale publicly available web corpus and may also bring some bias when generating sentences.

## Acknowledgements

This work was funded, in part, by the Vector Institute for AI, Canada CIFAR AI Chairs program, an NSERC discovery grant, and a research gift from AI2.

## References

Chandra Bhagavatula, Ronan Le Bras, Chaitanya Malaviya, Keisuke Sakaguchi, Ari Holtzman, Hannah Rashkin, Doug Downey, Wen-tau Yih, and Yejin Choi. 2020. Abductive commonsense reasoning. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multirelational data. In Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc.

Antoine Bosselut, Hannah Rashkin, Maarten Sap, Chaitanya Malaviya, Asli Celikyilmaz, and Yejin Choi. 2019. COMET: Commonsense transformers for automatic knowledge graph construction. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4762–4779, Florence, Italy. Association for Computational Linguistics.

Faeze Brahman, Vered Shwartz, Rachel Rudinger, and Yejin Choi. 2021. Learning to rationalize for nonmonotonic reasoning with distant supervision. Proceedings of the AAAI Conference on Artificial Intelligence, 35:12592–12601.

Oana-Maria Camburu, Tim Rocktäschel, Thomas Lukasiewicz, and Phil Blunsom. 2018. e-snli: Natural language inference with natural language explanations. Advances in Neural Information Processing Systems, 31.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An opensource chatbot impressing gpt-4 with 90% chatgpt quality.

Jaemin Cho, Minjoon Seo, and Hannaneh Hajishirzi. 2019. Mixture content selection for diverse sequence generation.

Marco Cuturi. 2013. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 26.

A. P. Dempster, N. M. Laird, and D. B. Rubin. 1977. Maximum likelihood from incomplete data via the EM algorithm. Journal ofthe Royal Statistical Society: Series B, 39:1–38.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889–898, Melbourne, Australia. Association for Computational Linguistics.

Yanlin Feng, Xinyue Chen, Bill Yuchen Lin, Peifeng Wang, Jun Yan, and Xiang Ren. 2020. Scalable multihop relational reasoning for knowledge-aware question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1295–1309, Online. Association for Computational Linguistics.

John Hewitt, Christopher Manning, and Percy Liang. 2022. Truncation sampling as language model desmoothing. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 3414– 3427, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The curious case of neural text degeneration. In International Conference on Learning Representations.

Pavan Kapanipathi, Veronika Thost, Siva Sankalp Patel, Spencer Whitehead, Ibrahim Abdelaziz, Avinash Balakrishnan, Maria Chang, Kshitij P. Fadnis, R. Chulaka Gunasekara, Bassem Makni, Nicholas Mattei, Kartik Talamadupula, and Achille Fokoue. 2020. Infusing knowledge into the textual entailment task using graph convolutional networks. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications ofArtificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 8074–8081. AAAI Press.

Thomas N. Kipf and Max Welling. 2017. Semisupervised classification with graph convolutional networks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net.

Junhyun Lee, Inyeop Lee, and Jaewoo Kang. 2019. Self-attention graph pooling. In Proceedings of the 36th International Conference on Machine Learning,

ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pages 3734–3743. PMLR.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7871–7880, Online. Association for Computational Linguistics.

Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. 2016. A diversity-promoting objective function for neural conversation models. In Proceedings of the 2016 Conference of the North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, pages 110–119, San Diego, California. Association for Computational Linguistics.

Bill Yuchen Lin, Xinyue Chen, Jamin Chen, and Xiang Ren. 2019. KagNet: Knowledge-aware graph networks for commonsense reasoning. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2829–2839, Hong Kong, China. Association for Computational Linguistics.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Tengfei Ma and Jie Chen. 2020. Unsupervised learning of graph hierarchical abstractions with differentiable coarsening and optimal transport.

Ninareh Mehrabi, Pei Zhou, Fred Morstatter, Jay Pujara, Xiang Ren, and Aram Galstyan. 2021. Lawyers are dishonest? quantifying representational harms in commonsense knowledge resources. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 5016–5033, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Gabriel Peyré and Marco Cuturi. 2019. Computational optimal transport. Foundations and Trends in Machine Learning, 11 (5-6):355–602.

Nazneen Fatema Rajani, Bryan McCann, Caiming Xiong, and Richard Socher. 2019. Explain yourself! leveraging language models for commonsense

reasoning. In Proceedings ofthe 57th Annual Meeting of the Association for Computational Linguistics, pages 4932–4942, Florence, Italy. Association for Computational Linguistics.

Maarten Sap, Ronan Le Bras, Emily Allaway, Chandra Bhagavatula, Nicholas Lourie, Hannah Rashkin, Brendan Roof, Noah A Smith, and Yejin Choi. 2019. Atomic: An atlas of machine commonsense for ifthen reasoning. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 3027–3035.

Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne Van Den Berg, Ivan Titov, and Max Welling. 2018. Modeling relational data with graph convolutional networks. In The Semantic Web: 15th International Conference, ESWC 2018, Heraklion, Crete, Greece, June 3–7, 2018, Proceedings 15, pages 593– 607. Springer.

Tianxiao Shen, Myle Ott, Michael Auli, and Marc’Aurelio Ranzato. 2019. Mixture models for diverse machine translation: Tricks of the trade. International Conference on Machine Learning.

Robyn Speer, Joshua Chin, and Catherine Havasi. 2017. Conceptnet 5.5: An open multilingual graph of general knowledge. In Proceedings ofthe AAAI conference on artificial intelligence, volume 31.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models.

Quan Tu, Yanran Li, Jianwei Cui, Bin Wang, Ji-Rong Wen, and Rui Yan. 2022. MISC: A mixed strategyaware model integrating COMET for emotional support conversation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 308–319, Dublin, Ireland. Association for Computational Linguistics.

Cunxiang Wang, Shuailong Liang, Yili Jin, Yilong Wang, Xiaodan Zhu, and Yue Zhang. 2020. SemEval-2020 task 4: Commonsense validation and explanation. In Proceedings of the Fourteenth Workshop on Semantic Evaluation, pages 307–321, Barcelona (online). International Committee for Computational Linguistics.

Ruize Wang, Duyu Tang, Nan Duan, Zhongyu Wei, Xuanjing Huang, Jianshu Ji, Guihong Cao, Daxin Jiang, and Ming Zhou. 2021. K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 1405–1418, Online. Association for Computational Linguistics.

Weiqi Wang\*, Tianqing Fang\*, Wenxuan Ding, Baixuan Xu, Xin Liu, Yangqiu Song, and Antoine Bosselut.

2023. Car: Conceptualization-augmented reasoner for zero-shot commonsense question answering. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023.

Weiqi Wang, Tianqing Fang, Baixuan Xu, Chun Yi Louis Bo, Yangqiu Song, and Lei Chen. 2023. Cat: A contextualized conceptualization and instantiation framework for commonsense reasoning. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics, ACL 2023.

Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen. 2014. Knowledge graph embedding by translating on hyperplanes. In Proceedings of the AAAI conference on artificial intelligence, volume 28.

Siwei Wu, Xiangqing Shen, and Rui Xia. 2023. Commonsense knowledge graph completion via contrastive pretraining and node clustering.

Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and Philip S. Yu. 2021. A comprehensive survey on graph neural networks. IEEE Transactions on Neural Networks and Learning Systems, 32(1):4–24.

Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. 2021. QA-GNN: Reasoning with language models and knowledge graphs for question answering. In Proceedings of the 2021 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 535–546, Online. Association for Computational Linguistics.

Zhitao Ying, Jiaxuan You, Christopher Morris, Xiang Ren, Will Hamilton, and Jure Leskovec. 2018. Hierarchical graph representation learning with differentiable pooling. Advances in neural information processing systems, 31.

Wenhao Yu, Chenguang Zhu, Lianhui Qin, Zhihan Zhang, Tong Zhao, and Meng Jiang. 2022. Diversifying content generation for commonsense reasoning with mixture of knowledge graph experts. In Findings of the Association for Computational Linguistics: ACL 2022, pages 1896–1906, Dublin, Ireland. Association for Computational Linguistics.

Yizhe Zhang, Michel Galley, Jianfeng Gao, Zhe Gan, Xiujun Li, Chris Brockett, and Bill Dolan. 2018. Generating informative and diverse conversational responses via adversarial information maximization. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc.

Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, and Yong Yu. 2018. Texygen: A benchmarking platform for text generation models.

## A Prompt used with Vicuna-13b

We present the prompts that we used for Vicuna-13b for ComVE (Figure 5) and α-NLG (Figure 6).

# few-shot examples   
< input sentence >   
Give three reasons explaining why the above   
sentence does not make sense:   
1. < reference1 >   
2. < reference2 >   
3. < reference3 >   
# target example   
< input sentence >   
Give three reasons explaining why the above   
sentence does not make sense:

Figure 5: Vicuna prompt for the ComVE dataset.  
```markdown
# few-shot examples
Write three sentences that likely happened in
between the past event: < past event > and the
future event: < future event >:
1. < reference1 >
2. < reference2 >
3. < reference3 >
# target example
Write three sentences that likely happened in
between the past event: < past event > and the
future event: < future event >:
```

Figure 6: Vicuna prompt for the α-NLG dataset.
<table><tr><td rowspan="2">Data</td><td rowspan="2">Model</td><td colspan="4"># of KG Concepts</td></tr><tr><td>0</td><td>1</td><td>2</td><td>3&lt;=</td></tr><tr><td>ComVE</td><td>MoKGE +SAG+OT</td><td>5.9 +0.1</td><td>23.2 -3.1</td><td>28.9 +1.5</td><td>42.1 +1.0</td></tr><tr><td>α-NLG</td><td>MoKGE +SAG+OT</td><td>16.8 -2.0</td><td>31.9 -1.1</td><td>29.0 +1.7</td><td>22.3 +1.4</td></tr></table>

Table 5: Comparison of models with MoKGE and MoKGE with Self Attention and Optimal Transport on the number of unique concepts in generated outputs. All KG concepts are lemmatized.

## B Additional Generation Examples

We show additional sentences generated by MoKGE and MoKGE+SAG+OT for ComVE (Figure 7) and α-NLG (Figure 8).

## C Assignment Ratio for ComVE

We show the performance on ComVE with varying node assignment ratios in Figure 9.

## D Concept Inclusiveness

We analyze how well the model incorporates KG concepts in output generation in Table 5.

## E Mixture of Experts

Given input sentence q and target sentence y, MoE employs a multinomial latent variable $\delta \in$ $\{ 1 , 2 , . . . , K \}$ and decomposes the marginal likelihood as:

$$
P ( y | x , g _ { x } ) = \sum _ { \delta = 1 } ^ { K } P ( \delta | x , \mathcal { G } _ { x } ^ { \prime } ) P ( y | \delta , x , \mathcal { G } _ { x } ^ { \prime } )
$$

Each δ represents an expert, which is responsible for explaining $( x , \mathcal { G } _ { x } ^ { \prime } , y )$ observation.

With the above decomposition, the model minimizes the loss function (Eq.(4))

$$
\begin{array} { r } { \nabla \log P ( y | x , \mathcal { G } _ { x } ^ { \prime } ) = \sum _ { \delta = 1 } ^ { K } P ( \delta | x , y , \mathcal { G } _ { x } ^ { \prime } ) \cdot \nabla \log P ( y , \delta | x , \mathcal { G } _ { x } ^ { \prime } ) } \end{array}
$$

and is trained using hard-EM algorithm (Dempster et al., 1977) as follows:

• E-step: choose expert $\delta ^ { \mathrm { b e s t } }$ with minimal loss.

$$
\delta ^ { \mathrm { b e s t } } = \underset { \delta } { \operatorname { a r g m i n } } - \log P ( \boldsymbol { y } , \delta | \boldsymbol { x } , \mathcal { G } _ { \boldsymbol { x } } ^ { \prime } )
$$

• M-step: update the parameters of the chosen expert $\delta ^ { \mathrm { b e s t } }$ from E-step.

## F Hyper-parameters

We used BART-base (Lewis et al., 2020), which is built on the Transformer architecture with a 6 layer encoder-decoder. For model training, we used Adam optimizer with a batch size of 60 and a learning rate of 3e-5. For the ComVE dataset, the warmup steps are set to 5000. For the α-NLG dataset, the weight decay is set to 0.01. For optimal transport, γ is set to 1.0. As to the weights in the discrete distributions, $\left\{ { a } _ { i } \right\}$ are set evenly as $1 / m$ and $\{ b _ { j } \}$ are all set as $1 / n$ , where m and n are number of nodes in the graphs.

![](images/ef8e09acb1b7ee1326984a6791643366520313b2c10b77da6120fbfb4c1db8d3.jpg)  
ComVE  
Figure 7: Additional examples of model generated sentences with MoKGE and MoKGE+SAG+OT on ComVE dataset.

## Kelly decided she wanted to paint some plaster. [likely explanation between two events] Kelly had to redo it.

## Human:

1. However, she realized she did it wrong.

2. Kelly applied the plaster wrong.

3. She finished her project, but then the dog ran over it and ruined it.

## MoKGE:

concepts: ['turn', 'paint']

1. Kelly painted it badly and it turned out badly.

2. Kelly painted it badly. She did not realize how bad it looked.

3. She did not do it correctly and it turned out badly.

## MoKGE+SAG+OT:

concepts: ['good', 'white', 'plaster', 'wall', 'paint']

1. She painted it badly and it didn't look good.

2. Kelly painted the walls with white paint. It looked bad.

3. Kelly did not like the look of the plaster.

## Roy loved everything about rice. [likely explanation between two events] Now Roy produces millions of pounds of rice and has sales worldwide!.

## Human:

1. Roy begin growing his own rice.

2. Roy decided to become a rice farmer.

3. Roy decided to grow rice.

4. Roy started growing his own rice and selling it.

5. Roy wanted to start producing rice.

## MoKGE:

concepts: ['company', 'rice', 'farm']

1. Roy decided to start his own rice company with his passion.

2. Roy decided to start his own rice farm with his rice beans.

3. He decided to start his own rice company with his passion.

## MoKGE+SAG+OT:

concepts: ['company', 'money', 'production', 'rice', 'farm']

1. Roy's passion got him into the field of rice production.

2. Roy started his own rice farm. Roy learned everything.

3. Roy decided to start his own rice company with his own money.

α-NLG

Figure 8: Additional examples of model generated sentences with MoKGE and MoKGE+SAG+OT on α-NLG dataset.

![](images/5a01cc7700c7ba443f5008fb9f10d33d4641f397026eee3f583f114158815b67.jpg)  
Figure 9: Self BLEU-3, Distinct-2, and ROUGE-l per assignment ratio on ComVE dataset.