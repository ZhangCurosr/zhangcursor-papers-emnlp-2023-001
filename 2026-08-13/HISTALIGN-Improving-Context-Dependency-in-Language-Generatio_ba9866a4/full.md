# HISTALIGN: Improving Context Dependency in Language Generation by Aligning with History

David Wan Shiyue Zhang Mohit Bansal

UNC Chapel Hill

{davidwan, shiyue, mbansal}@cs.unc.edu

## Abstract

Language models (LMs) can generate hallucinations and incoherent outputs, which highlights their weak context dependency. Cache-LMs, which augment LMs with a memory of recent history, can increase context dependency and have shown remarkable performance in diverse language generation tasks. However, we find that even with training, the performance gain stemming from the cache component of current cache-LMs is suboptimal due to the misalignment between the current hidden states and those stored in the memory. In this work, we present HISTALIGN, a new training approach to ensure good cache alignment such that the model receives useful signals from the history. We first prove our concept on a simple and synthetic task where the memory is essential for correct predictions, and we show that the cache component of HISTALIGN is better aligned and improves overall performance. Next, we evaluate HISTALIGN on diverse downstream language generation tasks, including prompt continuation, abstractive summarization, and data-to-text. We demonstrate that HISTALIGN improves text coherence and faithfulness in open-ended and conditional generation settings, respectively. HISTALIGN is also generalizable across different model families, showcasing its strength in improving context dependency of LMs in diverse scenarios.<sup>1</sup>

## 1 Introduction

Language modeling (LM), or language generation, requires decent context dependency. For both openended and conditional generation tasks, we want the model generation to be consistent with its previous generation or the input context. However, incoherence and hallucination problems are pervasive in current model generations (Holtzman et al.,

![](images/bdb33eb75d411dd0b0d46332f127730d980bea725e9e20b3fa5b360bf0b08990.jpg)  
Figure 1: An illustration of HISTALIGN and baseline cache-LM. The input example is from Chang and Mc-Callum (2022). Our HISTALIGN is able to assign high probabilities to both king and woman, and thus is able to tune down the weight of the hallucinated token queen from the softmax probability. Current cache language models (baseline) give high probabilities to irrelevant tokens in the cache and thus are at risk of producing hallucinated or incoherent tokens.

2020; Cao et al., 2018; Maynez et al., 2020), which suggests the weak context dependency of LMs.

Cache language model (Grave et al., 2017b, Cache-LM) is a simple yet effective method to improve context dependency by equipping LM with an additional memory of recent history (local context) and enabling it to directly “copy” from the history. Such models showed considerable improvement in language modeling and downstream generation tasks (Merity et al., 2017; See et al., 2017). However, since the introduction of Transformers (Vaswani et al., 2017), local memory has been less used due to the powerful self-attention mechanism, and more works have been focusing on leveraging long-term or external memory (Khandelwal et al., 2020; Yogatama et al., 2021). Nonetheless, Zhong et al. (2022) showed that using local memory on top of a Transformer is still beneficial.

In this paper, we focus on applying local cache to Transformer-based LMs and show that better alignment of the cache component leads to stronger gains. First, we show that cache-LM theoretically breaks the softmax bottleneck (Yang et al., 2018) that limits the capacity of any parametric LM to model highly context-dependent natural language. Then, we find that, in current cache-LMs, the signals provided by the memory component are minor, even when using the cache component during training (Zhong et al., 2022). We hypothesize that the main bottleneck comes from the misalignment of the current hidden states and those in the memory, because of which more relevant memories are not given higher weights than less relevant ones. We demonstrate this problem through a synthetic task: Ambiguous Template (Chang and McCallum, 2022), an example of which is shown in Figure 1. When asking the model to predict the next word given the context “After debating whether to bow to the woman or the king, the jester decided to bow to the \_\_ ,” current cache-LM does not give the highest probabilities to the desired words king and woman. Instead, we find that irrelevant words, such as to and jester have high cache probabilities. When combining these probabilities with the original softmax, the desired words cannot be ranked as top tokens. We find that this problem exists in pretrained LMs of various sizes, fine-tuned models, as well as models with cache augmented.

Next, we address this misalignment issue by proposing a new fine-tuning scheme, HISTALIGN, in which we augment the LM training objective with a contrastive loss to encourage the model to align the current hidden states with those in the history. As shown in Figure 1, our cache component gives higher probabilities for king and woman than other less relevant words in the cache. Unlike the typical contrastive loss that treats all negative examples equally, we propose to learn a ranking of negative tokens, i.e., more semantically similar tokens are ranked higher. As shown in Figure 2, when we align the space for the token housing, we want words such as accommodations to be closer than less relevant words like children. Hence, the cache can also be useful even when the exact target word is not present in the history. We demonstrate the stronger cache performance of HISTALIGN through the synthetic ambiguous template task and showcase its strength in improving coherence for open-ended prompt continuation and faithfulness for abstractive summarization and data-to-text.

To summarize, our contributions are as follows:

• We discuss why cache-LM with local memory can improve context dependency through a softmax bottleneck lens.

• We show the misalignment problem present in current cache language models and their training strategy.

• We propose a new training method, HISTAL-IGN, based on order-informed contrastive learning, which alleviates the misalignment problem and makes better use of memories.

• We demonstrate that HISTALIGN improves the coherence of open-ended generation as well as the faithfulness of conditional generation, and it works across different model families and adds little computational overhead.

## 2 Related Work

Cache-LM and Pointer Network. Adding a cache component to a language model (LM) was first introduced for speech recognition (Kuhn and De Mori, 1990). Grave et al. (2017c) extended this idea to RNN-based neural LM, which they call neural cache-LM. Cache-LM predicts the next token by combining the RNN model’s outputs with the similarities between the cache and the current hidden state. The cache saves tuples of hidden state and next token prediction, i.e., $( h _ { i } , x _ { i + 1 } )$ , from recent history (see Section 3.2). Essentially, the cache component enables the model to copy tokens from the history. Similar to cache-LM, a pointer network (Vinyals et al., 2015; Merity et al., 2017) also combines generating and copying of tokens but uses $h _ { i }$ as a representation of $x _ { i }$ (instead of $x _ { i + 1 } )$ This means that a pointer network requires learning additional transformations between the current representation and those in the past and a gating component for interpolation (Merity et al., 2017; See et al., 2017).<sup>2</sup> In contrast, cache-LM doesn’t need extra parameters to be learned and can be applied directly at testing time. It is more efficient to be used for larger cache sizes (i.e., extending cache-LM to long-term and external memory), and has been shown to perform better than pointer-network (Grave et al., 2017b; Zhong et al., 2022).

While cache-LM can be directly applied at test time, a recent work (Zhong et al., 2022) showed that it leads to more improvement when using cache during training time as well. Nonetheless, such proposed learning objectives for cache-LMs usually only provide distant supervision to the cache component. In contrast, we introduce direct supervision to the cache, which aligns the current representation with its history.

LM with Local or External Memory. Cache-LM and pointer network were originally proposed to only use hidden states from the local context, i.e., previous tokens in the input context. Though this technique has been proven to be helpful for language modeling and other language generation tasks (Gulcehre et al., 2016; Grave et al., 2017c; Merity et al., 2017; See et al., 2017), it has been less used after the Transformer architecture became popular, because the self-attention mechanism can attend to any token in the input context. Therefore, many works (Grave et al., 2017a; Khandelwal et al., 2020; Yogatama et al., 2021; Zhong et al., 2022; Min et al., 2022) proposed to use long-term or external memory beyond local context by applying retrieval techniques. Though our work can be extended to the external cache setting, we focus only on incorporating local memory, and we show that local memory is still helpful on top of Transformer because it breaks the softmax bottleneck (Yang et al., 2018) of parametric language models. A concurrent work (Chang et al., 2023) also demonstrates how a pointer network breaks softmax bottleneck by examples and empirical results, while we discuss this in a more mathematical way in Section 4.1.

Context Dependency in Language Generation. Existing language generation models demonstrate weak context dependency. For open-ended generation tasks, Holtzman et al. (2020) pointed out that strong LMs can produce very incoherent text following an input prompt. This incoherence issue has also been long observed in the story generation literature (Rashkin et al., 2020; Alabdulkarim et al., 2021). For conditional generation tasks, for example, summarization, Cao et al. (2018); Maynez et al. (2020) showed that around 30% and 70% model-generated summaries contain hallucinations for two popularly used summarization datasets, respectively. Similar unfaithfulness problems have also been seen in data-to-text generation (Chen et al., 2020a), machine translation (Weng et al., 2020), etc. Though many approaches have been introduced to alleviate incoherence (Li et al., 2022a) or unfaithfulness (Cao and Wang, 2021; Wan and Bansal, 2022), in this work, we explore a simple yet general cache-LM method to increase context dependency for diverse tasks. The concurrent work (Chang et al., 2023) uses pointer network type of architectures to improve next-word distribution and summarization factuality. They modify the softmax head by using additional contextdependent embeddings. In contrast, we simply apply the original cache-LM architecture and improve it with a novel training objective.

## 3 Preliminaries

## 3.1 Language Modeling

We focus on autoregressive language modeling (LM). Here, for simplicity, we assume that the LM is decoder-only, i.e., the context of the current step is the generated tokens of previous steps. We show that the same approach can easily be generalized to encoder-decoder models in Section 4.3. Given the context $c _ { t } = x _ { 1 } , . . . , x _ { t - 1 }$ , the probability of next token $x _ { t } = w$ is predicted by a softmax head:

$$
P _ { l m } ( w | c _ { t } ) \propto \exp ( h _ { t } ^ { \top } e _ { w } )\tag{1}
$$

where $e _ { w }$ is the output embedding of token w and $h _ { t }$ is the output context vector (hidden state) from the model at the t-th step. The model is trained by minimizing the cross-entropy loss: $l _ { x e } =$ $- \sum t$ log $P _ { l m } ( x _ { t } | c _ { t } )$

## 3.2 Cache Language Models

Cache language models augment a memory component to language models. Following Grave et al. (2017c), we consider cache to be a list of tuples of context vector and target token, $( h _ { i } , x _ { i } )$ . Assume we only consider the history of the local context, then the local memory of t-th step is written as:

$$
\mathcal { M } _ { \mathrm { l o c a l } } = \{ ( h _ { i } , x _ { i } ) \} _ { 1 \leq i \leq t - 1 }\tag{2}
$$

Then, the next-token prediction aggregates the logits from the softmax head and the similarities be-

tween $h _ { t }$ and those saved in the memory:

$$
\sum _ { \begin{array} { l } { P _ { c l m } ( w | c _ { t } ) \propto \exp ( h _ { t } ^ { \top } e _ { w } ) + } \\ { \sum _ { \begin{array} { c } { 1 } \end{array} } \mathbb { 1 } _ { \{ x _ { i } = w \} } \exp ( \sin ( h _ { t } , h _ { i } ) ) } \\ { ( h _ { i } , x _ { i } ) \in \mathcal { M } _ { \mathrm { l o c a l } } } \end{array} }\tag{3}
$$

where sim $\left( \cdot , \cdot \right)$ can be an arbitrary similarity function. Here, we follow Zhong et al. (2022) and use the scaled dot product: si $\begin{array} { r } { \mathfrak { n } ( \bar { h } _ { 1 } , h _ { 2 } ) = \frac { h _ { 1 } \cdot h _ { 2 } } { \sqrt { d } } } \end{array}$ , where d is the hidden dimension size.

While Grave et al. (2017c) only incorporated cache during evaluation, TRIME (Zhong et al., 2022) showed that it brings more benefits when also incorporated during training, i.e., minimizing $\begin{array} { r } { l _ { t r i m e } = - \sum _ { t } \log P _ { c l m } ( x _ { t } | c _ { t } ) } \end{array}$ . Here, we also use cache in both training and evaluation, but we improve the training objective by introducing direct supervision on the cache (see Section 4.2).

## 4 Our Methodology

## 4.1 Breaking Softmax Bottleneck

We first want to connect using local memory with the softmax bottleneck problem (Yang et al., 2018) and show that Transformer’s self-attention cannot break this bottleneck, while the local cache can.

Parametric autoregressive language models (Section 3.1), including Transformer-based LMs, use a softmax function operating on context vectors (or hidden states) H $\doteq \mathbb { R } ^ { N \times d }$ and output embedding matrix $\mathbf { E } \in \mathbb { R } ^ { V \times d } . ~ N$ is the number of contexts, assuming every token in the training set has a different context, then N is the number of tokens in the training set. V is the vocabulary size, and d is the hidden dimension size. Then, the next token probabilities form a log-probability matrix $\mathbf { A } \in \bar { \mathbb { R } } ^ { N \times V }$ $( A _ { t w } = \log P ( w | h _ { t } ) )$ . Ideally, since every context is unique, the rank of A should be as large as $V$ (assuming $V < N )$ . However, as A is roughly equivalent to $\mathbf { H } \mathbf { E } ^ { \top }$ , its rank is strictly upper bounded by hidden size d (please refer to Yang et al. (2018) for the formal proof). This low-rank problem greatly limits the LM’s capacity to model highly contextdependent natural language. This can be seen in Figure 1, where queen achieves higher probability than woman. The reason for LM’s difficulty in such bimodal distribution, as explained in Chang and McCallum (2022), is that the four words king, woman, man, queen tend to form a parallelogram in the embedding space, and if the model’s hidden state wishes to be close to the output embeddings of king and woman, it will also be close to those of man and queen.

To break this bottleneck, one simple solution is to increase $d ,$ as we see larger models usually have better performance. Another solution proposed by Yang et al. (2018) and extended by Kanai et al. (2018); Yang et al. (2019); Chang and Mc-Callum (2022) is to use multiple softmax heads – mixture of softmax (MoS), e.g., $P ( w | h _ { t } )$ α $\exp ( h _ { t } ^ { ( 1 ) \top } e _ { w } ) + \exp ( h _ { t } ^ { ( 2 ) \top } e _ { w } ) + \exp ( h _ { t } ^ { ( 3 ) \top } e _ { w } ) ]$ Each $h _ { t } ^ { ( k ) }$ is a different context vector. However, adding softmax heads is fairly computationally expensive. Comparing MoS to Eq. 3, we can see that adding exp $( h _ { t } ^ { \top } e _ { w } )$ and $\exp ( \sin ( h _ { t } , h _ { i } ) )$ resembles MoS without adding extra softmax heads. Another way to understand this connection is that when using local memory, A is roughly equivalent to $\mathbf { H } \mathbf { E } ^ { \top } + \mathbf { H } \mathbf { H } _ { c } ^ { \top }$ , where H<sub>c</sub> are the hidden states in the local context.<sup>3</sup> Assuming $\mathbf { E } _ { c } = \mathbf { E } + \mathbf { H } _ { c } , \mathbf { A }$ becomes $\mathbf { H E } _ { c }$ . Different from $\mathbf { E } , \mathbf { E } _ { c }$ is no longer a static output embedding matrix of size $V \times d$ but a context-dependent embedding tensor of size $N \times V \times d .$ Hence, the rank of A is no longer upper bounded by d. Note that this connection also holds for using long-term or external memories.

## 4.2 HISTALIGN

Cache-LM combines the original softmax probabilities with the cache probabilities by aggregating the similarity scores between the current hidden state and those in the cache. To use the cache module effectively, the similarity function sim( , ) plays an important role in Eq. 3. If the similarities between the current hidden state and less relevant memories are higher than more relevant ones, it would steer the model away from selecting the most useful information from the cache. By assigning a high probability to the correct local memories, e.g., those corresponding to king and woman in the example of Figure 1, we can ensure that when the probabilities are combined, they will be scored higher than irrelevant and hallucinated tokens. However, we find that even when directly maximizing log $P _ { c l m }$ (Zhong et al., 2022), there is no guarantee that the current representations are well aligned with relevant information stored in the memory, as shown by the baseline probabilities in Figure 1 (see Section 6.1 for more details).

Hence, to deal with this misalignment, we propose a new contrastive objective that encourages higher similarities between the hidden states of similar target tokens. During training, given the current hidden state $h _ { t }$ and the corresponding next token $x _ { t }$ , we construct a positive set $\mathcal { P } _ { t }$ from caches by selecting memories with the same target token:

![](images/77b4e194edbae4439d84cd2d49f33074e85bf7db0e3125e44364263e90d6fa0f.jpg)  
Figure 2: Illustration of our HISTALIGN training approach. We first get local cache by combining the hidden states in local context with their target tokens, and then rank them according to embedding similarity. The ranked memories are then used to train with the margin loss. This ensures that negative yet similar words (e.g. accommodations) will be closer in the vector space than irrelevant words (e.g. children).

$$
\mathcal { P } _ { t } = \{ ( h _ { i } , x _ { i } ) \} _ { x _ { i } = x _ { t } , 1 \leq i \leq t - 1 }\tag{4}
$$

All other memories are taken as negative examples. An example is shown in step 2 of Figure 2. For predicting the token housing, we have two previous mentions of the word housing, and the other words, including flat, children, accommodations, etc., are considered as negative.

In the typical contrastive loss, such as InfoNCE (van den Oord et al., 2019), all negative examples are treated equally. However, we hope to learn an ordering of the negative examples – more similar examples are ranked higher than less similar ones. In the example in Figure 2, accommodations is more similar to housing than children. This ensures that even when predicting words that do not have previous mentions in the local cache, our model can still output a reasonable alternative.

To achieve this, we construct a ranking of memories by computing the cosine similarities between the embedding of the current target word and the embeddings of words in the cache, i.e., cosim $( e _ { t } , e _ { i } )$ . After sorting tokens from the most similar w.r.t. semantic similarity to the least, we use the following max-margin loss (Liu et al., 2022c):

$$
l _ { c o n t . } = \sum _ { t } \sum _ { i \in \mathcal { P } _ { t } } \sum _ { j > i , j \notin \mathcal { P } _ { t } } \operatorname* { m a x } _ { \mathbf { \Phi } } \left( 0 , \sin ( h _ { t } , h _ { j } ) \right.\tag{5}
$$

where $\lambda _ { i , j } = ( j - i ) \lambda _ { : }$ and λ is the margin tuned based on validation loss.

The final objective of HISTALIGN is a combination of the original LM cross-entropy loss $l _ { x e }$ and this ranking-based contrastive loss:

$$
l _ { h i s t a l i g n } = l _ { x e } + \alpha l _ { c o n t . }\tag{6}
$$

where α is a tunable weight of the contrastive loss.   
Note that during the inference time, we use Eq. 3.

## 4.3 Extension to Encoder-Decoder Models

HISTALIGN can be easily adapted to encoderdecoder models. For conditional generation tasks, the target text is usually short, hence, coherence is not a big issue. What is more crucial is whether the target generation stays true to the input context, e.g., the input document for summarization or the input table for data-to-text. Therefore, we define the local cache to be the input tokens and their corresponding encoder hidden states, as opposed to the output tokens and decoder hidden states for decoder-only models. We then calculate the similarity between the current decoder hidden state with those encoder hidden states stored in the cache.

## 5 Experimental Setup

Here, we describe the tasks and the experimental setups. Please refer to Appendix A for more details.

## 5.1 Tasks and Datasets

Ambiguous Template is a useful synthetic dataset collated by Chang and McCallum (2022), in which each example is generated using templates with diagonal words<sup>4</sup> from semantic analogy relations in the Google (English) analogy dataset (Mikolov et al., 2013). This is a simple yet effective setting to examine whether the model can copy the correct tokens from history and not hallucinate semantically similar tokens, e.g., queen and man of the example in Figure 1. Since the target words can always be found in the context, we can also evaluate the performance only with the cache component.

Open-Ended Generation evaluates the language modeling capability by asking the model to generate a continuation given a prompt (Holtzman et al., 2020; Su et al., 2022; Li et al., 2022b). We use WritingPrompts (Fan et al., 2018), and treat the first 50 tokens as the prompt and allow the model to generate up to 256 tokens using the canonical nucleus sampling $( p = 0 . 9 5 )$ (Holtzman et al., 2020).

Abstractive Summarization is the task of providing an abridged version of the input document. One crucial problem is ‘hallucination’, where the generated summaries contain facts or entities that are wrong or not present in the document (Cao et al., 2018; Maynez et al., 2020). We evaluate on two widely-used English News summarization datasets, XSum (Narayan et al., 2018) and CNN/DM (Hermann et al., 2015).

Data-to-Text is the task of describing structured data, where faithfulness is extremely important, as humans do not tolerate any hallucinations in cases such as describing medical reports or financial statistics (Thomson and Reiter, 2020). We evaluate on LogicNLG (Chen et al., 2020a).

## 5.2 Systems

We use GPT2-small and GPT2-large (Radford et al., 2019) for ambiguous template and prompt continuation, and we use BART-large (Lewis et al., 2020) for both summarization and data-to-text. For all tasks, we choose to finetune pre-trained LMs. The first baseline we compare to is fine-tuning with the original cross-entropy loss $( l _ { x e }$ in Section 3.1), which is named by the original model name in our result tables. Then, we also compare to the most recent cache-LM learning objective, TRIME (Zhong et al., 2022) $( l _ { t r i m e }$ in Section 3.2).

## 5.3 Evaluations

Ambiguous Template. As a proof-of-concept experiment, we evaluate under both a full setting, using the combined probability in Eq. 3, as well as a cache-only setting, only using the cache similarity scores to predict the next token. We evaluate the performance via the accuracy of having the two diagonal words within the top-k predictions (Acc@k), where $k = \{ 2 , 5 , 1 0 , 2 5 \}$ . Ideally, we want to see 100% accuracy with $k = 2 .$ , which indicates that the two diagonal words are the top 2 choices. Note that when only using the cache, a k value of 50 would achieve perfect accuracy, as it would include the entire local history. In addition, we want to empirically verify that cache LM with local memory can break the softmax bottleneck. To this end, we calculate the rank of log-probability matrix $\mathbf { A } \in \mathbb { R } ^ { N \times V }$ (Section 4.1) using 500 examples (concretely, N = 4750 and V = 50257 for GPT-2 based models) under the full setting.

Open-Ended Generation. We mainly evaluate the coherence of model-generated continuations. Following Su et al. (2022), coherence is approximated by the cosine similarity of the SimCSE (Gao et al., 2021) sentence embeddings of the prompt and the continuation. In addition, following previous works, we report n-gram diversity (Meister et al., 2022) and MAUVE (Pillutla et al., 2021) scores for a more general evaluation. We hope HISTALIGN not to harm diversity and MAUVE. We also run human evaluation on Amazon MTurk to ask workers to compare the continuations generated by TRIME and HISTALIGN. More details can be found in Appendix B.1.

Abstractive Summarization. We mainly evaluate the faithfulness of generated summaries by three widely-used automatic metrics: FactCC (Kryscinski et al., 2020) and DAE (Goyal and Durrett, 2021), which are entailment-based metric; and Entity Precision (Nan et al., 2021, $\mathrm { P _ { E N T } } )$ , which calculates the percentage of entities in the summary that are present in the document. We also report ROUGE-L (Lin, 2004) for general content selection evaluation. Similarly, we conduct human evaluation, where we ask crowd workers to judge whether each summary (of 100 randomly selected examples) is faithful and informative. Please refer to Appendix B.2 for more details.

<table><tr><td></td><td colspan="4">Full</td><td colspan="4">Cache-Only</td><td>Full</td></tr><tr><td>Model</td><td>Acc@2</td><td>Acc@5</td><td>Acc@10</td><td>Acc@25</td><td>Acc@2</td><td>Acc@5</td><td>Acc@10</td><td>Acc@25</td><td>Rank</td></tr><tr><td>GPT2-Small</td><td>50.00</td><td>62.20</td><td>68.80</td><td>76.96</td><td>0.00</td><td>35.57</td><td>50.56</td><td>75.71</td><td>762</td></tr><tr><td>TRIME</td><td>46.43</td><td>56.41</td><td>76.47</td><td>97.96</td><td>0.00</td><td>39.89</td><td>66.51</td><td>100.00</td><td>836</td></tr><tr><td>HISTALIGN</td><td>63.47</td><td>72.26</td><td>89.71</td><td>100.00</td><td>58.62</td><td>70.62</td><td>79.59</td><td>94.31</td><td>854</td></tr><tr><td>GPT2-Large</td><td>75.43</td><td>84.76</td><td>87.93</td><td>91.26</td><td>0.05</td><td>37.83</td><td>73.88</td><td>100.00</td><td>1280</td></tr><tr><td>TRIME</td><td>77.40</td><td>91.10</td><td>94.56</td><td>97.17</td><td>0.11</td><td>22.32</td><td>84.89</td><td>100.00</td><td>1377</td></tr><tr><td>HISTALIGN</td><td>82.22</td><td>92.84</td><td>96.57</td><td>98.34</td><td>82.15</td><td>92.51</td><td>99.94</td><td>100.00</td><td>1377</td></tr></table>

Table 1: Results on Ambiguous Template. HISTALIGN achieves the best performance in both full and cache-only settings. We also empirically show that TRIME and HISTALIGN break the softmax bottleneck.
<table><tr><td>Model</td><td>Acc@2</td><td>Acc@5</td><td>Acc@10</td><td>Acc@25</td></tr><tr><td>LLaMA2-7B</td><td>0</td><td>0</td><td>0</td><td>100</td></tr><tr><td>TRIME</td><td>0</td><td>0</td><td>0</td><td>100</td></tr><tr><td>HISTALIGN</td><td>100</td><td>100</td><td>100</td><td>100</td></tr></table>

Table 2: Cache-Only results on Ambiguous Template with LLaMA2-7B model.

Data-to-Text. We mainly evaluate the faithfulness of model generations by NLI-Acc and SP-Acc (Chen et al., 2020a) and two more recent metrics – TAPEX-Acc and TAPAS-Acc (Liu et al., 2022a). NLI-Acc is an entailment-based metric pre-trained on TabFact dataset (Chen et al., 2020b) using TaBERT (Yin et al., 2020), and SP-Acc first parses the sentence into a logical program and evaluates the execution accuracy. TAPEX-Acc and TAPAS-Acc are entailment-based metrics trained with TAPEX (Liu et al., 2022b) and TAPAS (Eisenschlos et al., 2020), respectively. Same as previous works (Chen et al., 2020a), we report BLEU (Papineni et al., 2002) for a surface-level evaluation.

## 6 Results

We verify the strength of HISTALIGN at aligning the cache component and thus improve the nexttoken prediction on ambiguous template in Section 6.1, coherence in open-ended prompt continuation in Section 6.2, and faithfulness in abstractive summarization and data-to-text in Section 6.3 and Section 6.4, respectively.

## 6.1 Importance of Cache on Ambiguous Template

We show the results of the Ambiguous Template in Table 1. First, it can be seen that the original GPT2 model has pretty bad performance in the cache-only setting, especially considering Acc@2. This is expected because the original model is fine-tuned using the cross-entropy loss without the cache component involved, and thus applying cache at test time may not be helpful. Second, though TRIME (Zhong et al., 2022) generally outperforms the original model in the full setting, its cache-only Acc@2 and Acc@5 are similar to the original model. Considering that all target words are present in the history, this result indicates that despite the fact that TRIME uses cache during training, its cache component is still misaligned and has limited contributions to the final performance.

In contrast, HISTALIGN achieves high Acc@2 with only the cache module, substantially outperforming the original model and TRIME on both model sizes, which demonstrates the effectiveness of our contrastive loss for aligning memories better. As a result, HISTALIGN outperforms both baselines across all k in the full setting. And the improvement holds for both model sizes, though with smaller gaps for the large model. This observation is consistent with our discussion in Section 4.1 that a larger model with a larger hidden dimension suffers less from the softmax bottleneck, while local memory can help break this bottleneck of any parametric LM. This is also empirically verified by the rank of the log-probability matrix reported in Table 1, where we see that the rank of the original model is upper-bounded by its hidden dimension (768 for GPT2-small and 1280 for GPT2-large), and having a local cache breaks this bottleneck. Finally, we present two qualitative examples in Table 9. See detailed discussions in Appendix C.

Experiment on recent LLM. We also fine-tune LLaMA2 7B model (Touvron et al., 2023). Interestingly, we find that LLaMA2 achieves 0% accuracy for Acc@{2,5,10} when evaluated zero-shot. After fine-tuning, the model achieves 100% accuracy without any cache. This is expected, as the task is a simple synthetic task, and the model, compared to GPT2-large, is 10x larger, and the hidden size is 3.2x larger (1280 4096). Thus, as mentioned in Section 4.1, the model alleviates the softmax bottleneck due to its larger hidden size.

<table><tr><td>Model</td><td>diversity</td><td>MAUVE</td><td>coherence</td></tr><tr><td>GPT2-small</td><td> $8 8 . 1 3 _ { \pm 0 . 1 2 }$ </td><td> $8 6 . 6 2 _ { \pm 1 . 1 0 }$ </td><td> $5 3 . 7 7 _ { \pm 0 . 2 9 }$ </td></tr><tr><td>TRIME HISTALIGN</td><td> $8 8 . 5 3 { \scriptstyle \pm 0 . 1 4 }$   ${ \bf 9 0 . 0 7 } _ { \pm 0 . 1 9 }$ </td><td> $8 6 . 7 6 { \scriptstyle \pm 0 . 5 8 }$   $\mathbf { 8 7 . 4 6 _ { \pm 0 . 8 0 } }$ </td><td> $5 7 . 5 8 { \scriptstyle \pm 1 . 0 5 }$   ${ \bf 6 1 . 3 0 _ { \pm 0 . 1 5 } }$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>GPT2-large TRIME</td><td> $8 8 . 8 2 _ { \pm 0 . 0 7 }$ </td><td> $8 6 . 1 8 _ { \pm 0 . 9 4 }$ </td><td> $5 2 . 3 9 _ { \pm 0 . 1 0 }$ </td></tr><tr><td></td><td> $\mathbf { 9 0 . 7 0 { \scriptstyle \pm 0 . 0 8 } }$ </td><td> $\mathbf { 8 7 . 2 7 _ { \pm 0 . 8 5 } }$ </td><td> $5 3 . 1 1 _ { \pm 0 . 1 9 }$ </td></tr><tr><td>HISTALIGN</td><td> $8 9 . 4 1 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $8 6 . 8 3 { \scriptstyle \pm 1 . 0 2 }$ </td><td> ${ \bar { 5 } } 3 . { \bar { 5 } } 1 _ { \pm 0 . 0 5 }$ </td></tr></table>

Table 3: Automatic evaluation results of open-ended generation. Numbers are 3-run averages the 95% confidence intervals.

However, we still observe the two problems with LLaMA2. First, the problem of softmax bottleneck still exists, as the rank of its output log-probability matrix A is still upper-bounded by its hidden size of 4096, as we find that its empirical rank is 3332. This means that it is still theoretically less expressive than highly context-dependent natural language. Second, TRIME is still not able to make good use of the cache, i.e., misalignment still exists. As shown in the Table 2, TRIME achieves 0% accuracy for Acc@{2,5,10} under the cache-only setting, which shows that the issue of misalignment is even more apparent for larger language models: Since the token logits perform well enough, the model does not learn to use the cache anymore. Nevertheless, as shown in the table, our training objective can enforce the use of the local cache and achieve 100% accuracy, which is consistent with our findings from smaller models.

The presence of these two issues showcases that there is still room for improvement on LM’s context dependency, as HISTALIGN outperforms TRIME in making good use of cache.

## 6.2 Coherence in Open-Ended Generation

The results of prompt continuation can be found in Table 3. Across both sizes of the model, we observe an improvement in coherence with TRIME and a larger improvement with HISTALIGN. The effect of HISTALIGN is especially prominent for the smaller model, where coherence increases by 7.5 points compared to the original model, and 3.7 points over TRIME. This validates our hypothesis that HISTALIGN can improve the coherence of LMs. When looking at MAUVE, HISTALIGN improves by 0.8 points and 0.7 points over GPT2 and TRIME respectively when using small models. On the large model, while TRIME achieves the best performance, HISTALIGN still improves over the original model by 0.7 points. A similar trend can be observed for diversity. Holistically, HISTAL-IGN improves coherence while maintaining similar diversity and MAUVE.

<table><tr><td colspan="3">Fluency</td><td colspan="3">Coherence</td></tr><tr><td>Win↑</td><td>Tie</td><td>Lose↓</td><td>Win↑</td><td>Tie</td><td>Lose↓</td></tr><tr><td>46.33</td><td>36.33</td><td>17.33</td><td>48.33</td><td>32.00</td><td>19.66</td></tr></table>

Table 4: Human evaluation results of open-ended generation. We conduct a pairwise comparison between HISTALIGN with TRIME $\mathrm { ( ^ { 6 6 } W i n ^ { \prime } }$ means humans prefer our HISTALIGN over TRIME) and show the percentage of passages that are judged as coherent and fluent. HISTALIGN is statistically significantly better $( p < 0 . 0 5 )$ than TRIME on fluency and coherence.

Besides automatic evaluations, we also conduct a human evaluation, the results of which are shown in Table 4. On both fluency and coherence, human raters prefer the continuations by HISTALIGN more than that by TRIME. This confirms the observation from the automatic evaluations that HISTALIGN does improve especially on coherence.

## 6.3 Faithfulness in Abstractive Summarization

The summarization results are shown in Table 5. TRIME improves faithfulness over the baseline on XSum, but the improvement is not clear on CNN/DM. In contrast, our HISTALIGN method greatly improves over the baseline, especially on DAE and $\mathrm { P _ { e n t } }$ , which are specifically targeted towards hallucinations. Concretely, we improve FactCC by 0.91 points, DAE by 4.78 points, and $\mathrm { P _ { e n t } }$ by 3 points on the XSum dataset. HISTALIGN improves the metrics on CNN/DM as well though to a smaller degree. This shows that allowing the model to pay specific attention to previous contexts in the input is helpful in reducing hallucinations.

We note that the ROUGE-L score for HISTAL-IGN is lower than the original model. This ROUGEfaithfulness tradeoff has been observed by many previous works (Chen et al., 2021; Kryscinski et al., 2020; Wan and Bansal, 2022; Wan et al., 2023), where the reference summary inherently contains hallucinations and thus does not overlap highly with the more faithful generated summaries.

To confirm this, we conduct a human evaluation. The results are shown in Table 6. HISTALIGN achieves the best faithfulness score, which is statistically significantly better than BART. This confirms our observation from automatic metric results in Table 5. Though there is a small drop in informativeness, the difference between the three methods has no statistical significance.<sup>5</sup> This shows that the drop in automated metrics such as ROUGE-L does not necessarily mean a decrease in informativeness.

<table><tr><td></td><td colspan="4">XSum</td><td colspan="4">CNN/DM</td></tr><tr><td>Model</td><td>Rouge-L</td><td>FactCC</td><td>DAE↓</td><td> $\mathrm { P _ { e n t } }$ </td><td>Rouge-L</td><td>FactCC</td><td>DAE↓</td><td> $\mathrm { P _ { e n t } }$ </td></tr><tr><td>BART</td><td>36.41</td><td>22.16</td><td>67.96</td><td>72.72</td><td>30.63</td><td>72.63</td><td>6.98</td><td>93.53</td></tr><tr><td>TRIME</td><td>36.50</td><td>22.94</td><td>66.34</td><td>74.25</td><td>30.60</td><td>72.65</td><td>7.08</td><td>93.39</td></tr><tr><td>HISTALIGN</td><td>35.45</td><td>23.07</td><td>63.18</td><td>75.71</td><td>29.96</td><td>74.93</td><td>5.73</td><td>93.80</td></tr></table>

Table 5: Performance on abstractive summarization tasks. HISTALIGN consistently improves faithfulness over the two baseline methods on both datasets.
<table><tr><td>Model</td><td>Faithfulness</td><td>Informativeness</td></tr><tr><td>BART</td><td>19.33</td><td>63.67</td></tr><tr><td>TRIME</td><td>20.00</td><td>66.33</td></tr><tr><td>HISTALIGN</td><td>26.33*</td><td>65.33</td></tr></table>

Table 6: Human evaluation results on XSum. \* indicates that it is statistically significantly better (p < 0.05) than BART. Krippendorff’s αs are 0.52 and 0.34 for faithfulness and informativeness, respectively.

## 6.4 Faithfulness in Data-to-Text Generation

The results on LogicNLG are shown in Table 7. Similar to abstractive summarization, HISTALIGN can improve faithfulness on LogicNLG. Out of the four faithfulness metrics, HISTALIGN achieves the highest NLI-Acc, TAPEX-Acc, and TAPAS-Acc: HISTALIGN achieves 0.6 and 0.8 point improvements on TAPEX-Acc over BART and TRIME respectively, and a 1.74 point improvement on TAPAS-Acc over the BART model. In the meantime, HISTALIGN obtains the best BLEU scores.

## 7 Discussion and Conclusion

In this work, we improve the context dependency of LMs by introducing a novel cache-LM training objective, HISTALIGN, which improves the existing cache-LM objective by adding an order-informed contrastive loss for the cache component. On a synthetic dataset, we show that HISTALIGN is effective at retrieving the desired memories from the cache and breaking the softmax bottleneck. Furthermore, we demonstrate the effectiveness of HISTALIGN at improving the coherence of open-ended generation and improving faithfulness of abstractive summarization and data-to-text generation.

We want to emphasize a couple of salient points with the recent trend of pushing for larger and more powerful models. Firstly, attention mechanisms alone cannot break the softmax bottleneck, as shown in Table 2. Secondly, while increasing the model size can mitigate this bottleneck, the problem will persist unless we reach a size that truly encapsulates the complexity of human language. Cache-LM is a light alternative for breaking softmax bottleneck theoretically and improving context dependency empirically.

<table><tr><td>Model</td><td>BLEU-(1/2/3)</td><td>NA</td><td>SA</td><td>TX</td><td>TS</td></tr><tr><td>BART</td><td>56.27/37.07/25.63</td><td>85.46</td><td>53.45</td><td>63.97</td><td>63.74</td></tr><tr><td>TRIME</td><td>56.12/36.84/25.29</td><td>84.55</td><td>52.85</td><td>63.74</td><td>65.11</td></tr><tr><td>HISTALIGN</td><td>56.65/37.56/26.25</td><td>85.67</td><td>53.12</td><td>64.58</td><td>65.48</td></tr></table>

Table 7: Performance on LogicNLG (data-to-text generation) evaluated by BLEU scores, NLI-Acc (NA), SP-Acc (SA), TAPEX-Acc (TA), and TAPAS-Acc (TS). HISTALIGN improves over two baselines on BLEU and three faithfulness metrics: NA, TX, and TS.

## Acknowledgments

We thank the reviewers and Haw-Shiuan Chang for helping with providing the Ambiguous Template data. This work was supported by NSF-CAREER Award 1846185, NSF-AI Engage Institute DRL-2112635, DARPA Machine Commonsense (MCS) Grant N66001-19-2-4031, and a Bloomberg Data Science Ph.D. Fellowship. The views contained in this article are those of the authors and not of the funding agency.

## Limitations

While we focus on the local memory to show that current LMs still benefit from better local context dependency, our method is also compatible with external memories, which can potentially further improve the performance of HISTALIGN in future work. We evaluate HISTALIGN using GPT2 and BART that at most consist of 774M parameters, which is smaller than the latest large LMs that can have billions of parameters. On the Ambiguous Template task, we do show that this problem exists for recent LLMs with LLaMA2 7B models and our method improves the cache alignment, but we hope that in the future we can explore scaling up the approach on large LMs to various tasks. We believe that our method is still helpful for larger models. But as larger models suffer less from softmax bottleneck (Section 4.1), how much it can help is an interesting problem to study in the future. Another current limitation of this work is that due to the additional hyper-parameters (the λ of the margin and the weight α of the contrastive loss), it becomes less straightforward to incorporate our HISTALIGN objective into pre-training compared to TRIME. The training objective also considers that each token has a fixed margin (and thus assumes that each token is equally different), which can be improved by dynamically adjusting the margins. Although fine-tuning is cheaper and we show effective gains using HISTALIGN in fine-tuning, how to use HISTALIGN to pre-train LMs is also an interesting future work direction.

## Ethical Considerations

As the OpenAI team pointed out, GPT-2 does not distinguish fact from fiction, so it can not support use cases that require the generated text to be true. In addition, GPT-2 reflects the biases inherent to the data they were trained on, so it can not be deployed unless the deployers first carry out a study of biases relevant to the intended use case. Though our HISTALIGN improves the coherence of GPT-2 generations, the above statement still holds. Similarly, despite that HISTALIGN improved the faithfulness of BART-large generations for abstractive summarization and data-to-text generation, such systems cannot be directly deployed and used in factualitysensitive scenarios without further checks in place.

## References

Amal Alabdulkarim, Siyan Li, and Xiangyu Peng. 2021. Automatic story generation: Challenges and attempts. In Proceedings of the Third Workshop on Narrative Understanding, pages 72–83, Virtual. Association for Computational Linguistics.

Shuyang Cao and Lu Wang. 2021. CLIFF: Contrastive learning for improving faithfulness and factuality in abstractive summarization. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6633–6649, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Ziqiang Cao, Furu Wei, Wenjie Li, and Sujian Li. 2018. Faithful to the original: Fact-aware neural abstractive summarization. In Proceedings of the Thirty-

Second AAAI Conference on Artificial Intelligence and Thirtieth Innovative Applications of Artificial Intelligence Conference and Eighth AAAI Symposium on Educational Advances in Artificial Intelligence, AAAI’18/IAAI’18/EAAI’18. AAAI Press.

Haw-Shiuan Chang and Andrew McCallum. 2022. Softmax bottleneck makes language models unable to represent multi-mode word distributions. In Proceedings ofthe 60th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 8048–8073, Dublin, Ireland. Association for Computational Linguistics.

Haw-Shiuan Chang, Zonghai Yao, Alolika Gon, Hong Yu, and Andrew McCallum. 2023. Revisiting the architectures like pointer networks to efficiently improve the next word distribution, summarization factuality, and beyond. In Findings ofthe Association for Computational Linguistics: ACL 2023 (Findings of ACL).

Sihao Chen, Fan Zhang, Kazoo Sone, and Dan Roth. 2021. Improving faithfulness in abstractive summarization with contrast candidate generation and selection. In Proceedings of the 2021 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 5935–5941, Online. Association for Computational Linguistics.

Wenhu Chen, Jianshu Chen, Yu Su, Zhiyu Chen, and William Yang Wang. 2020a. Logical natural language generation from open-domain tables. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 7929– 7942, Online. Association for Computational Linguistics.

Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. 2020b. Tabfact: A large-scale dataset for table-based fact verification. In International Conference on Learning Representations.

Bradley Efron and Robert J. Tibshirani. 1993. An Introduction to the Bootstrap. Number 57 in Monographs on Statistics and Applied Probability. Chapman & Hall/CRC, Boca Raton, Florida, USA.

Julian Eisenschlos, Syrine Krichene, and Thomas Müller. 2020. Understanding tables with intermediate pre-training. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 281–296, Online. Association for Computational Linguistics.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889–898, Melbourne, Australia. Association for Computational Linguistics.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Tanya Goyal and Greg Durrett. 2021. Annotating and modeling fine-grained factuality in summarization. In Proceedings ofthe 2021 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1449–1462, Online. Association for Computational Linguistics.

Edouard Grave, Moustapha M Cisse, and Armand Joulin. 2017a. Unbounded cache model for online language modeling with open vocabulary. Advances in neural information processing systems, 30.

Edouard Grave, Armand Joulin, and Nicolas Usunier. 2017b. Improving neural language models with a continuous cache. In International Conference on Learning Representations.

Edouard Grave, Armand Joulin, and Nicolas Usunier. 2017c. Improving neural language models with a continuous cache. In International Conference on Learning Representations.

Caglar Gulcehre, Sungjin Ahn, Ramesh Nallapati, Bowen Zhou, and Yoshua Bengio. 2016. Pointing the unknown words. In Proceedings of the 54th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 140–149, Berlin, Germany. Association for Computational Linguistics.

Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. 2015. Teaching machines to read and comprehend. In Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The curious case of neural text degeneration. In International Conference on Learning Representations.

Sekitoshi Kanai, Yasuhiro Fujiwara, Yuki Yamanaka, and Shuichi Adachi. 2018. Sigsoftmax: Reanalysis of the softmax bottleneck. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc.

Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. 2020. Generalization through memorization: Nearest neighbor language models. In International Conference on Learning Representations.

Wojciech Kryscinski, Bryan McCann, Caiming Xiong, and Richard Socher. 2020. Evaluating the factual consistency of abstractive text summarization. In

Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9332–9346, Online. Association for Computational Linguistics.

Roland Kuhn and Renato De Mori. 1990. A cachebased natural language model for speech recognition. IEEE transactions on pattern analysis and machine intelligence, 12(6):570–583.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 7871–7880, Online. Association for Computational Linguistics.

Quentin Lhoest, Albert Villanova del Moral, Yacine Jernite, Abhishek Thakur, Patrick von Platen, Suraj Patil, Julien Chaumond, Mariama Drame, Julien Plu, Lewis Tunstall, Joe Davison, Mario Šaško, Gunjan Chhablani, Bhavitvya Malik, Simon Brandeis, Teven Le Scao, Victor Sanh, Canwen Xu, Nicolas Patry, Angelina McMillan-Major, Philipp Schmid, Sylvain Gugger, Clément Delangue, Théo Matussière, Lysandre Debut, Stas Bekman, Pierric Cistac, Thibault Goehringer, Victor Mustar, François Lagunas, Alexander Rush, and Thomas Wolf. 2021. Datasets: A community library for natural language processing. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 175–184, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. 2022a. Contrastive decoding: Open-ended text generation as optimization.

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. 2022b. Contrastive decoding: Open-ended text generation as optimization.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Ao Liu, Haoyu Dong, Naoaki Okazaki, Shi Han, and Dongmei Zhang. 2022a. PLOG: Table-to-logic pretraining for logical table-to-text generation. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 5531– 5546, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Qian Liu, Bei Chen, Jiaqi Guo, Morteza Ziyadi, Zeqi Lin, Weizhu Chen, and Jian-Guang Lou. 2022b. TAPEX: Table pre-training via learning a neural SQL executor. In International Conference on Learning Representations.

Yixin Liu, Pengfei Liu, Dragomir Radev, and Graham Neubig. 2022c. BRIO: Bringing order to abstractive summarization. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2890–2903, Dublin, Ireland. Association for Computational Linguistics.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.

Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020. On faithfulness and factuality in abstractive summarization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1906–1919, Online. Association for Computational Linguistics.

Clara Meister, Tiago Pimentel, Gian Wiher, and Ryan Cotterell. 2022. Locally typical sampling. Transactions of the Association for Computational Linguistics, abs/2202.00666.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2017. Pointer sentinel mixture models. In International Conference on Learning Representations.

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc.

Sewon Min, Weijia Shi, Mike Lewis, Xilun Chen, Wen tau Yih, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Nonparametric masked language modeling.

Feng Nan, Ramesh Nallapati, Zhiguo Wang, Cicero Nogueira dos Santos, Henghui Zhu, Dejiao Zhang, Kathleen McKeown, and Bing Xiang. 2021. Entitylevel factual consistency of abstractive text summarization. In Proceedings of the 16th Conference of the European Chapter ofthe Associationfor Computational Linguistics: Main Volume, pages 2727–2733, Online. Association for Computational Linguistics.

Shashi Narayan, Shay B. Cohen, and Mirella Lapata. 2018. Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1797–1807, Brussels, Belgium. Association for Computational Linguistics.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaid Harchaoui. 2021. Mauve: Measuring the gap between neural text and human text using divergence frontiers. In Advances in Neural Information Processing Systems, volume 34, pages 4816–4828. Curran Associates, Inc.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. OpenAI blog.

Hannah Rashkin, Asli Celikyilmaz, Yejin Choi, and Jianfeng Gao. 2020. PlotMachines: Outlineconditioned generation with dynamic plot state tracking. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4274–4295, Online. Association for Computational Linguistics.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Abigail See, Peter J Liu, and Christopher D Manning. 2017. Get to the point: Summarization with pointergenerator networks. In Proceedings of the 55th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1073– 1083.

Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier. 2022. A contrastive framework for neural text generation. In Advances in Neural Information Processing Systems.

Craig Thomson and Ehud Reiter. 2020. A gold standard methodology for evaluating accuracy in data-to-text systems. In Proceedings of the 13th International Conference on Natural Language Generation, pages 158–168, Dublin, Ireland. Association for Computational Linguistics.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten,

Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and finetuned chat models.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2019. Representation learning with contrastive predictive coding.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. 2015. Pointer networks. Advances in neural information processing systems, 28.

David Wan and Mohit Bansal. 2022. FactPEGASUS: Factuality-aware pre-training and fine-tuning for abstractive summarization. In Proceedings of the 2022 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 1010–1028, Seattle, United States. Association for Computational Linguistics.

David Wan, Mengwen Liu, Kathleen McKeown, Markus Dreyer, and Mohit Bansal. 2023. Faithfulness-aware decoding strategies for abstractive summarization. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 2864–2880, Dubrovnik, Croatia. Association for Computational Linguistics.

Rongxiang Weng, Heng Yu, Xiangpeng Wei, and Weihua Luo. 2020. Towards enhancing faithfulness for neural machine translation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2675–2684, Online. Association for Computational Linguistics.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, and William W. Cohen. 2018. Breaking the softmax bottleneck: A high-rank RNN language model. In International Conference on Learning Representations.

Zhilin Yang, Thang Luong, Russ R Salakhutdinov, and Quoc V Le. 2019. Mixtape: Breaking the softmax bottleneck efficiently. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc.

Pengcheng Yin, Graham Neubig, Wen-tau Yih, and Sebastian Riedel. 2020. TaBERT: Pretraining for joint understanding of textual and tabular data. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8413–8426, Online. Association for Computational Linguistics.

Dani Yogatama, Cyprien de Masson d’Autume, and Lingpeng Kong. 2021. Adaptive semiparametric language models. Transactions of the Association for Computational Linguistics, 9:362–373.

Zexuan Zhong, Tao Lei, and Danqi Chen. 2022. Training language models with memory augmentation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5657–5673, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

## A Experimental Setup Details

Unless specified, we use Huggingface’s Transformers library (Wolf et al., 2020) to train the models. We use the trainer’s default setting, including AdamW optimizer (Loshchilov and Hutter, 2019) and a linear rate scheduler. We use mixed precision and deepspeed. We use RTX A6000 GPUs with 48GB memory and A100 GPUs with 80GB memory.

For hyperparameter tuning, we try learning rate of {1e-5,3e-5,5e-5} and λ between {0.001,0.0001,0.00001}, and contrastive weight {0.5,1.0} for all tasks. For HISTALIGN, we use λ of 0.001 and the contrastive weight α = 1, unless otherwise specified.

## A.1 Ambiguous Template

The dataset consists of 122k, 250k, and 122k examples for train, dev, and test sets, respectively. The test set has no overlap of diagonal words with the training set. Following Chang and McCallum (2022), we freeze output vocab to prevent overfitting, and get loss only from the last token (the target token). We select the model based on validation loss. For GPT2-large-based models, training the original model took around 30 minutes, TRIME and HISTALIGN took around an hour with 4 RTX A6000. On GPT2-small-based models, training took 10 minutes, 15 minutes, and 15 minutes, for orig, TRIME, and HISTALIGN, respectively.

<table><tr><td>Task</td><td>Dataset</td><td>Learning rate</td><td>Steps/Epochs</td><td>Warmup ratio</td><td>Batch size</td><td>Max input tokens</td><td>Max output tokens</td></tr><tr><td colspan="2">Ambiguous Template</td><td>1e-5</td><td>5 ep.</td><td>0.0</td><td>256</td><td></td><td>1024</td></tr><tr><td colspan="2">Open-ended Generation WritingPrompt</td><td>5e-5</td><td>3 ep.</td><td>0.2</td><td>512</td><td></td><td>1024</td></tr><tr><td rowspan="2">Summarization</td><td>XSum</td><td>5e-5</td><td>15,000 steps</td><td>0.0</td><td>128</td><td>512</td><td>64</td></tr><tr><td>CNN/DM</td><td>3e-5</td><td>20,000 steps</td><td>0.0</td><td>128</td><td>512</td><td>128</td></tr><tr><td>Data-to-text</td><td>LogicNLG</td><td>5e-5</td><td>10 eps</td><td>0.0</td><td>64</td><td>500</td><td>200</td></tr></table>

Table 8: Hyper-parameters for all tasks.
<table><tr><td></td><td>Context</td><td>Houston and Pennsylvania are my favorites, and I especially love</td><td>The brother and the granddaughter are my favorites, and I especially love the</td></tr><tr><td>Full</td><td>GPT2-large TRIME HISTALIGN</td><td>Pennsylvania, Harris, Pittsburgh Pennsylvania, Philadelphia, Pittsburgh Pennsylvania, Houston, Harris</td><td>brother, niece, granddaughter brother, niece, granddaughter brother, granddaughter, niece</td></tr><tr><td>Cache- only</td><td>GPT2-large TRIME HISTALIGN</td><td>and, I, Houston I, and, Houston Pennsylvania, Houston, and</td><td>the, granddaughter, I the, granddaughter, favorites granddaughter, brother, the</td></tr></table>

Table 9: Qualitative examples from Ambiguous Template. For both the full and cache-only settings, HISTALIGN retrieves the correct two tokens from the context as the top predictions.

## A.2 Prompt Continuation

WritingPrompts<sup>6</sup> (Fan et al., 2018) contain 273k, 16k, and 15k examples in the train, dev, and test sets. We use the full train and dev sets, while we sample 5000 examples from the test set for final evaluation to save time. We first train the models using the different objectives on the training set. We split the text into blocks of 512 tokens. For generation, we decode with nucleus sampling with p = 0.95 and three random seeds={0,1,42}, and average the scores. Training the original small model takes around 1.5 hours, TRIME takes around 2 hours, and HISTALIGN takes around 3 hours. Training GPT-2 large, TRIME and HISTALIGN takes around 6 hours, 7 hours, and 11.5 hours on 2 A100s, respectively.

## A.3 Summarization

XSum is a news summarization dataset consisting of BBC articles and contains 204k/11k/11k examples in the train/dev/test set. CNN/DM consists of Dailymail and CNN articles and the dataset consists of 287k/13k/11k examples in the train/dev/test set. We use the official packages for the faithfulness metrics.<sup>7</sup> We calculate $\mathrm { P _ { e n t } }$ by using spacy to extract entities and only consider [PERSON, FAC, GPE, ORG, NORP, LOC, EVENT] as the allowed entity types. We use Huggingface’s Dataset library (Lhoest et al., 2021) for loading the XSum (Narayan et al., 2018) and CNN/DM (Hermann et al., 2015) datasets. And we use Huggingface’s Metrics library for calculating ROUGE scores. Training the original model, TRIME, and HISTAL-IGN all took around 5 hours for XSum and training orig, TRIME and HISTALIGN all took around 4 hours for CNN/DM on 4 A6000s.

## A.4 Data-to-text

We follow Liu et al. (2022a) for pre-processing dataset, such as adding numerical pre-computation to the tables. We use a contrastive weight α = 0.5. LogicNLG (Chen et al., 2020a) consists of 28k training, 4k validation, and 4k test examples. We use original evaluation scripts for the faithfulness metrics, and the BLEU calculation script provided by the original dataset.<sup>8</sup> Training the original model and TRIME took 2 hours, and HISTALIGN took around 5 hours on 2 A100.

## B Human Evaluation Details

For both human evaluations, we use Amazon Mechanical Turk to do the annotation. We have the same set of requirements: The workers need to be from the United States, have more than 10,000 number of HITS approved, and an approval rate greater than 98%.

## B.1 Open-ended Generation

We use Amazon Mechanical Turk to annotate whether human prefers the continuation by TRIME or by HISTALIGN. We do not include the original model, since TRIME shows better performance on the automatic metrics. We select examples where the difference between their characters is less than 200 characters to ensure that the length is similar (since shorter texts will naturally be more coherent). We collect 3 annotations per example for 100 randomly selected examples, yielding 300 annotations. We take the percentage of passages that are judged as coherent and/or fluent.

We pay 0.5 USD per HIT, and the average time it takes is around 2.5 minutes, which yields an hourly rate of  \$12 per hour. An example of the annotation page is shown in Figure 3.

## B.2 Summarization

We follow the same setup as Wan and Bansal (2022), and also use a qualification test where we rate the faithfulness of the selected generated summaries. Only workers with the correct annotation can perform the actual task.

We select the most important sentences and replace the less relevant sentences with an ellipsis to reduce the overload for the workers. We select ten most relevant sentences from the document by cosine similarity of the sentence embedding using SentenceTransformer<sup>9</sup> (Reimers and Gurevych, 2019) for each summary and then combine and show all the selected relevant sentences from each summary.

Each task consists of three unique workers, where we take the mean as the scores for this document. The final score is the mean factuality score across all documents. The average time for each task is around 2.5 minutes and we pay 0.5 USD per task, hence an hourly rate of  \$12 per hour. An example of the annotation page is shown in Figure 4.

## C Qualitative Results on Ambiguous Template

We present two qualitative examples in Table 9. We see that both the original model and TRIME have difficulty in outputting the two correct words as the top two choices. This is also reflected by the cacheonly results, where irrelevant words, such as and, I, the get high probabilities. In fact, the cache similarities of the original model are similar to those of TRIME, again indicating that there is no guarantee of well-aligned memories, despite training with the cache. HISTALIGN nevertheless returns the two target words as the top two choices for both the full and cache-only settings, showing that the model benefits from the well-aligned memories through our contrastive objective.

## D Sample Outputs

We show sample outputs for prompt continuations in Figure 5, summarization in Figure 6 and Figure 7, and data-to-text in Figure 8.

![](images/12adf5a24fb982e6cea637d1d33d5c44c7894c5c1503e3e6a286e928222e1137.jpg)

[WP] A crime fighting duo having a petty argument in the middle of fighting the villain. You are the villain. I watched in disbelief as the two vigilantes turned on eachother. "I'm getting the credit for this one!" Yelled

![](images/f6036437557c9dc1c4a16b7763d5bae49adaa28497656add8842d1ea4a0a44a0.jpg)  
Figure 3: Human annotation page for evaluating coherence and fluency for prompt continuation.

Please evaluate whether the three summaries are coherent, consistent with the information in the article, and are informative

The article has been separated into sentences and displayed line-by-line. We have shortened the text for your convenience by removing the least related sentences, indicated by ellipsis (...).

Note that the five summaries shown are different and not in any particular order

Please evaluate the summary in the following three categories:

Irrespective of the given document, please select coherent if the summary is fluent. The summary has better structure and flow, and is easier to follow. The facts are presented in a more logical order.

## Consistency/Factuality

Please avoid using general knowledge, and only consider in the context of the provided document.

Select not consistent if facts in the summary are not supported by the document, such as cases like these:

• The summary contradicts information in the document. The summary might say "A fire broke out in Seattle", but an document says it broke out in Portland. Or the summary might say "the Republicans won the election", but the document indicates the Democrats won instead.

• The summary adds (hallucinates) a fact that is not mentioned anywhere in the document. For example, the summary might say thet "A fire broke out at 2am", but the document doesn't mention the time when the fire broke out.

## Informativeness

Please select informative if the summary expresses the main points of the document. Summary should contain relevant and important information and few unimportant details.

If you select the summary to be not consistent with the document, please only consider the consistent information when evaluating this category

## Document:

First-team coach Andy Smith, goalkeeping coach Marco Tabuas and fitness coach Maykel Moreira have now also left Vale Park, the League One club has confirmed. The three all arrived when Ribeiro was appointed in the summer

"They helped assist the club to their best home start to a season ever at Vale Park and progression to the FA Cup third round.'

Vale face a home game with Chesterfield on Friday before a trip to Oldham on Monday, prior to playing Championship side Huddersfield Town in the FA Cup on 7 January, They have lost their last three league games to slip to 17th in the table - just six points above the relegation zone.   
Ribeiro's former assistant, Michael Brown, has been placed in temporary charge of the Burslem-based club.

<table><tr><td colspan="3">Port Vale manager Bruno Ribeiro has left the club by mutual consent after less than two months in charge.</td></tr><tr><td>Coherence:</td><td>Consistency:</td><td>Informativeness:</td></tr><tr><td> coherent</td><td>consistent</td><td>Oinformative</td></tr><tr><td>not coherent</td><td> not consistent</td><td>not informative</td></tr></table>

## Summary:

Port Vale head coach Bruno Ribeiro has left the club by mutual consent after less than two months in charge Coherence: Consistency: Informativeness: coherent consistent informative not coherent Onot consistent not informative

Summary:   
Port Vale have sacked manager Bruno Ribeiro after just three months in charge.   
Coherence: Consistency: Informativeness: coherent consistent informative not coherent not consistent not informative

Figure 4: Human annotation page for evaluating faithfulness and informativeness for abstractive summarization.

<table><tr><td>Prompt: Every time you die, you are transported to the time of the first event that ultimately caused your death. One day, you are hit by a car and sent 12 years into the past. It&#x27;s been a strange life; constantly setting GPT2-Large: up and killing other people. But it&#x27;s done, and now it ’s time to settle down in this new world. I’d never actually</td></tr><tr><td>thought about my past life before, I just knew that I went back and lived, but this time there &#x27;s a complication. I’ve been to many places in my life, so why is there such a big problem? It ’s the weirdest thing, I could go anywhere in time and never get the feeling, but this time it’s weird. The cars are just like the last time, the ones that hit me, there’s just a hole in the middle of them. I&#x27;m in a small, abandoned office, and the walls aren&#x27;t quite made out of cement. I think that’s a nice touch ..</td></tr><tr><td>TRImE: up and killing other people. But it&#x27;s done, and now it&#x27;s time to put the final touches on this latest creation. I&#x27;ll be frank, I don&#x27;t really know what I&#x27;m supposed to do. A couple months ago, I was on my way to my parents, and there was an accident on the side of the road. It was some kind of accident; the car was hit by a drunk driver, and it fell on my head. Luckily, I was alive. I was given a few months to live, by the hospital. I can live with that, at least until I can figure out what happened. Anyway, I&#x27;m now waiting for the day when I meet my next death, and I have to be prepared for the worst. I can try to avoid death ...</td></tr><tr><td>HisTALIGN: up and stopping things, trying to prevent every mistake that might happen. If it weren&#x27;t for the fact that I was always so careful with everything, I might be a normal person who just goes about their business. A normal person like you wouldn&#x27;t have the ability to go back in time to prevent yourself from ever having died in the first place. Every mistake that might have happened could have killed me then and there, and it wasn&#x27;t an easy road to travel down. I was lucky enough to have only</td></tr><tr><td>been born on July 4th, 1996. I&#x27;ve been trying to do the right thing all these years. I&#x27;ve avoided death on more than one ... Prompt: Crown Minotaur Expedition The first week of the expedition was slow going. The terrain was rocky and unforgiving, as if the Mother Nature had forgotten to smooth out this patch of land. The Minotaurs had the most trouble due to the</td></tr><tr><td>GPT2-Large: steep slopes of the mountain pass. When a Minotaurel came up the mountain pass, she would pass straight through, just passing through the pass. It was just one of many impassable ridges, that the Miatarel would face to traverse the pass. The only reason it was never impassable was due to the massive, hairy beasts, which covered half of the pass, and caused many people to leave the mountains. This, plus the numerous Minotaurel who passed through the pass would create an increasingly high attrition rate on the expedition. Still, the Miatarel had learned of the pass through the passing .</td></tr><tr><td>TRimE: steep slopes of the mountain pass. When a mountain pass was at its highest, the Minotaurs were no longer able to pass into the valley below. It was just too steep. Luckily there was plenty of rock that they could use as a ramp and they would be able to crawl up with their arms, legs, and wings intact. The trip began on the third day after lunch. The trail was still rough, as many of the boulders were nearly a foot high. By now they could barely lift the weight of their bodies, much less their bags full of</td></tr><tr><td>HisTALIGN: steep slopes of the mountain pass. When a mountain pass was at its highest, the Minotaurs were no longer able to pass into the valley below. It was just too steep. Luckily, they were able to set up camps and set out in the first few days. This was only going to be a matter of time until they realized where they were going. This place was far away from anything they knew of. The only light was the lanterns on their back, and the lanterns were only good for a few minutes. As for how they got</td></tr></table>

Figure 5: Sample outputs for open-ended prompt continuation.

<table><tr><td>Document: David Lipton, second in command at the IMF, outlined some of these risks in a speech to the National Association for Business Economics in Washington on Tuesday. &quot;The IMF&#x27;s latest reading of the global economy shows once again a weakening baseline,&quot; he said. &quot;We are clearly at a delicate juncture.&quot; The comments come after weaker-than-expected trade figures from China showing that exports plunged by a quarter from a year ago. The IMF has already said it is likely it will downgrade its current forecast of 3.4% for global growth when it next releases its economic predictions in April. The dismal picture is one that has on-going ramifications for businesses and industries that bet on China&#x27;s growth story. Read more from</td></tr><tr><td>Karishma: Why a story about bulk shipping matters BART: The International Monetary Fund (IMF) has warned that the global economy is at a &quot;critical juncture&quot;. TRIME: The International Monetary Fund (IMF) has warned that the global economy is in a &quot;dangerous situation&quot;.</td></tr><tr><td>Document: Coventry University&#x27;s Scarborough campus has been built on the town&#x27;s former Weaponness Park and Ride site. About 200 students have begun courses at the site, though it is expected to eventually be home to more than 2,000 students. The building, which includes engineering and science labs, a mock law court and a library, is part of a £50m sports and education facility. Professor Craig Gaskell said: &quot;Launching our new state-of-the-art building is a huge milestone for us and demonstrates our commitment to Scarborough and the Yorkshire coast area.&quot; A new University Technical College has been built nearby and Scarborough Athletic FC&#x27;s new 2,000-seater stadium is also under construction on the site. Coventry University also has a campus near London&#x27;s Liverpool Street Station and recently announced it will open a campus in Dagenham in September 2017.</td></tr><tr><td>BART:A university has officially opened its first campus in North Yorkshire. TRIME: A new university campus has been officially opened in North Yorkshire. HisTALIGN: A university campus on the Yorkshire coast has opened to the public.</td></tr></table>

Figure 6: Sample outputs for XSum summarization.

![](images/eb58fe476e73fa2159592f77e04385ca3767ccf77c8afe155108333f7fea8e57.jpg)  
Figure 7: Sample outputs for CNN/DM summarization.

<table><tr><td></td><td>home team score</td><td>away team</td><td>away team score</td></tr><tr><td rowspan="4">Table for &quot;1928 vfl season&quot;:</td><td>11.12 (78)</td><td>st kilda</td><td>21.11 (137)</td></tr><tr><td>8.9 (57)</td><td>geelong</td><td>8.8 (56)</td></tr><tr><td>11.15 (81)</td><td>richmond</td><td>10.13 (73)</td></tr><tr><td>22.17 (149) 18.18 (126)</td><td>hawthorn</td><td>11.13 (79)</td></tr><tr><td></td><td>11.17 (83)</td><td>fitzroy essendon</td><td>11.13 (79) 18.11 (119)</td></tr><tr><td colspan="4">Reference: St Kilda had the highest Score as an Away Team in the 1928 Vfl Season</td></tr><tr><td colspan="4">BART: Hawthorn had the lowest Away Team Score of any team in the 1928 Vfl Season TRIME: Geelong had the lowest Score of 8.8 (56) while Hawthorn had the highest Score of 11.13 (79) HisTALIGN: St Kilda was the Away Team with the highest Score in the 1928 Vfl Season</td></tr><tr><td colspan="4"></td></tr><tr><td colspan="4"></td></tr><tr><td></td><td>name john hearne</td><td>matches 29</td><td></td></tr><tr><td>Table for &quot;1893 english cricket season&quot;:</td><td>tom richardson</td><td>23</td><td></td></tr><tr><td></td><td></td><td>28</td><td></td></tr><tr><td></td><td>johny briggs</td><td></td><td></td></tr><tr><td></td><td>arthur mold</td><td>28</td><td></td></tr><tr><td></td><td>bill lockwood</td><td></td><td>27</td></tr><tr><td colspan="4">Reference: John Hearne, played in more Match than any other Player, with 20 9</td></tr><tr><td colspan="4">TRIME: Bill Lockwood and Arthur Mold both played 27 Match in the 1893 English Cricket Season HiSTALIGN: John Hearne had the most Match with 29</td></tr></table>

Figure 8: Sample outputs for data-to-text generation.