# Tagging-Assisted Generation Model with Encoder and Decoder Supervision for Aspect Sentiment Triplet Extraction

Xianlong Luo<sup>1,2</sup> Meng Yang <sup>1,2</sup>∗ Yihao Wang<sup>1,2</sup>

<sup>1</sup>School of Computer Science and Engineering, Sun Yat-Sen University

<sup>2</sup>Key Laboratory of Machine Intelligence and Advanced Computing (SYSU),

Ministry of Education, China

luoxlong@mail2.sysu.edu.cn, yangm6@mail.sysu.edu.cn, wangyh357@mail2.sysu.edu.cn

## Abstract

ASTE (Aspect Sentiment Triplet Extraction) has gained increasing attention. Recent advancements in the ASTE task have been primarily driven by Natural Language Generationbased (NLG) approaches. However, most NLG methods overlook the supervision of the encoder-decoder hidden representations and fail to fully utilize the semantic information provided by the labels to enhance supervision. These limitations can hinder the extraction of implicit aspects and opinions. To address these challenges, we propose a tagging-assisted generation model with encoder and decoder super vision (TAGS), which enhances the supervision of the encoder and decoder through multiple perspective tagging assistance and label semantic representations. Specifically, TAGS enhances the generation task by integrating an additional sequence tagging task, which improves the encoder’s capability to distinguish the words of triplets. Moreover, it utilizes sequence tagging probabilities to guide the decoder, improving the generated content’s quality. Furthermore, TAGS employs a selfdecoding process for labels to acquire the semantic representations of the labels and aligns the decoder’s hidden states with these semantic representations, thereby achieving enhanced semantic supervision for the decoder’s hidden states. Extensive experiments on various public benchmarks demonstrate that TAGS achieves state-of-the-art performance.

## 1 Introduction

Aspect Sentiment Triplet Extraction (ASTE) aims to extract sentiment triplets from a sentence, i.e., Aspect: the aspect term represents an explicit mention of a discussed target, Opinion: the mentioned comment terms/phrases, Sentiment: sentiment polarity of the aspect, holding significant potential in downstream research and applications. Unlike sentence sentiment classification, ASTE emphasizes the explanation for sentiments, explicitly highlighting the causes of sentiments and the entities to which they are attached. This task involves addressing challenges such as the diversity of emotional expressions and the complexity of linguistic contexts. For instance, in the sentence "Food wise, it’s ok but a bit pricey for what you get considering the restaurant isn’t a fancy place," three sentiment triplets can be extracted: (food, ok, neutral), (food, pricey, neutral), and (restaurant, isn’t a fancy place, neutral).

Existing Methods The current mainstream approaches for ASTE can be classified into two categories: sequence tagging-based approaches and sequence generation-based approaches. ASTE employed a sequence tagging method initially introduced by Peng et al. (2020). However, the sequence tagging-based approaches in ASTE fail to capture the semantic information conveyed by the labels, which can result in semantic mismatches in the predicted results (Zhang et al., 2021b). By leveraging the rich label semantic information and mitigating the potential error propagation in pipeline methods (Paolini et al., 2021; Yu et al., 2023), generation methods achieve better performance in ASTE.

Generation-based approaches still face two significant challenges. Firstly, the supervision of hidden representations within encoder-decoder architectures has been overlooked, leading to potential issues such as the degeneration of neural language models and difficulty in identifying distinctive information (Su et al., 2022). In the context of the ASTE task, this oversight can fail to extract implicit aspects and opinions (Cai et al., 2021; Peper and Wang, 2022). Secondly, during training, the semantic information of the labels has yet to be fully utilized. Traditional supervision utilizes labels in the form of one-hot probability vectors without fully leveraging the semantic information of the labels at the hidden state level.

TAGS To address the challenges mentioned above, we propose a novel tagging-assisted generation model called TAGS, which enhances the supervision of both the encoder and decoder through multiple-perspective tagging assistance and label semantic representations. TAGS consists of two modules: "Empowering Generation through Sequence Tagging" (EGST) and "Label-Driven Semantic Alignment" (LDSA).

In EGST, we utilize a sequence tagging task to enhance the generation task through three aspects: Multitask Learning, Guided Generation, and Result Optimization. Multitask learning: we enhance the supervision in the encoder of the generation model by introducing a sequence tagging task. This additional task empowers the encoder to distinguish between triplet and irrelevant words effectively, thereby benefiting the generation task. Guided Generation: We incorporate the sequence tagging outputs into the decoder’s attention mechanism. This encourages the model to focus more on the keywords identified by the sequence tagging task. Result Optimization: Finally, during inference, we utilize the sequence tagging results to optimize the generation results, thereby improving the quality of the results.

In LDSA, we further enhance the supervision for the decoder’s hidden states in the generation model by utilizing the semantic information conveyed by labels. Firstly, we convert label triplets into a natural context, referred to as a label sentence, and input the label sentence into the TAGS model to obtain a more accurate hidden state, which also serves as a semantic label representation. Subsequently, we dynamically align the hidden states of the decoder to the label’s semantic representation according to the comparison results between the tokens corresponding to the semantic representation and the ground truth tokens. By this alignment, the model can better capture the semantic information conveyed by the labels, making the generation more in line with the intended label semantics.

Extensive experimental results validate the effectiveness of the TAGS model. In summary, our contributions to this work are threefold:

1. We propose a novel ASTE generation model, which utilizes sequence tagging to assist the generation via enhancing the supervision of the encoder’s hidden state and incorporating sequence tagging probabilities and results to improve the generation process.

2. We obtain the semantic representation of labels at the decoder level and achieve semantic alignment of the decoder’s hidden state to the labels in the generation model.

3. The experimental results show that our proposed framework significantly outperforms recent SOTA methods.

## 2 Problem statement

The input of the ASTE task is a sentence ${ \textbf { X } } =$ $\{ x _ { 1 } , x _ { 2 } , . . . , x _ { n } \}$ , where each x<sub>i</sub> represents a word and n is the maximum length of the sentence. The goal of the ASTE task is to generate a set of sentiment triplets $\mathbf { T } ~ = ~ \{ ( \mathbf { a } , \mathbf { o } , \mathbf { s } ) _ { k } \} _ { k = 1 } ^ { | T | }$ , where T means the number of triplets in T. Each triplet consists of an aspect term (a), an opinion term (o), and the corresponding sentiment polarity (s) $( s \in \{ P O S , N E U , N E G \} )$

Our proposed TAGS is an encoder-decoder model designed for the generation task, in which the input is a natural sentence and the generation target, i.e., the label sentence, is constructed by concatenating triplets from the set T as follows: $Y = " a _ { 1 } , o _ { 1 } , s _ { 1 } ; a _ { 2 } , o _ { 2 } , s _ { 2 } ; . . . ; a _ { k } , o _ { k } , s _ { k } "$ , where $a _ { i } , o _ { i } ,$ , and $s _ { i }$ correspond to the i-th triplet $( a , o , s ) _ { i }$

## 3 Methodology

Fig. 1 shows our proposed TAGS method. TAGS comprises two modules, an Empowering Generation through Sequential Tagging module (EGST) and a Label-Driven Semantic Alignment (LDSA) module. EGST leverages sequence tagging task to enhance the generation model in three aspects: Multitask Learning, Guided Generation, and Result Optimization. LDSA utilizes a label self-decoding process to obtain the semantic representation of labels and aligns the decoder’s hidden states to the semantic representation during training, thereby achieving enhanced semantic supervision for the decoder’s hidden states.

## 3.1 Empowering Generation through Sequence Tagging

TAGS leverages sequence tagging to enhance the generation task from multiple perspectives, shown in the right part of Fig. 1. Firstly, TAGS employs a sequence tagging task as an additional task to enhance the supervision of the encoder, thereby improving its ability to differentiate between triplet and irrelevant words. By sharing parameters between the sequence tagging model and the generation model, the enhanced discriminative power obtained from the sequence tagging task can also benefit the triplets extraction process in the generation model. Next, the sequence tagging task probabilities are integrated into the generation model, compelling it to prioritize the words identified as crucial by the sequence tagging results. This integration ensures that the generation model produces content closely aligned with those words. Lastly, TAGS utilizes the sequence tagging task results during inference to optimize the generated results. By considering the results from both methods, TAGS achieves a more comprehensive information fusion, enhancing overall model performance.

![](images/b6e1069acf0c3f30b4362e510e0ef9427f8bc0af3628d8ec5de09fdaf8d18851.jpg)  
Figure 1: Overview of our TAGS framework, which consists of two parts: Empowering Generation through Sequence Tagging (right) and Label-Driven Semantic Alignment (left). Sequence tagging enhances the generation task in three aspects. Encoder Multitask Learning: $\mathcal { L } _ { \mathrm { t a g g i n g } }$ for sequence tagging task. Guided Generation: the decoder’s Tag Attention incorporates probabilities $( p _ { i } )$ from the sequence tagging task as additional weights for its cross-attention mechanism. (The figure illustrates the generation of the q-th word.) Result Optimization: "Inference" optimizes the generation results using tagging results. Semantic representation of the label $( \hat { h } _ { D e } ^ { Y } )$ is obtained by inputting the label sentence $( Y )$ into the model. $\mathcal { L } _ { \mathrm { a l i g n m e n t } }$ is computed based on the cosine similarity results $( \hat { h } _ { D e } ^ { X }$ and $\hat { h } _ { D e } ^ { Y } )$ and the alignment labels L.

## 3.1.1 Sequence Tagging Task

We perform multitask learning by simultaneously training a generation task and a sequence tagging task.

Tagging Scheme In our designed sequence tagging scheme, each word will be classified into one of 7 categories. The "N" category represents nonkeywords, while the remaining 6 categories represent aspects and opinions, each combined with three sentiment types (positive, negative, and neutral). Read Appendix A.1 for detailed descriptions.

Tagging Task A tagging sample is denoted as (X, Z), where Z is the tagging label $\{ z _ { 1 } , z _ { 2 } , z _ { 3 } , . . . , z _ { n } \}$ . The encoder encodes X to obtain hidden states $H _ { \mathrm { E n } } ^ { X }$

$$
H _ { \mathrm { E n } } ^ { X } = E n ( [ x _ { 1 } , x _ { 2 } , \ldots , x _ { n } ] ) = [ h _ { 1 } ^ { X } , h _ { 2 } ^ { X } , \ldots , h _ { n } ^ { X } ]\tag{1}
$$

where En is Encoder, $H _ { \mathrm { E n } } ^ { X } \in \mathbb { R } ^ { n \times d }$ , d denotes the hidden dimension. $H _ { \mathrm { E n } } ^ { X }$ is also the encoder hidden state for the generation task. Pass $H _ { \mathrm { E n } } ^ { X }$ through a fully connected layer to obtain the tag probabilities $p _ { i } \colon$

$$
p _ { i } = \mathrm { s o f t m a x } ( W _ { 1 } h _ { i } ^ { X } + b _ { 1 } )\tag{2}
$$

where $W _ { 1 } \in \mathbb { R } ^ { 7 \times d } , b _ { 1 } \in \mathbb { R } ^ { 7 }$ , and $p _ { i } \in \mathbb { R } ^ { 7 }$ represents the probability distribution of the i-th word across 7 tags. We calculate the sequence tagging loss using cross-entropy loss:

$$
{ \mathcal { L } } _ { \mathrm { t a g g i n g } } = \sum _ { i = 1 } ^ { n } \mathbf { C E } ( p _ { i } , z _ { i } )\tag{3}
$$

3.1.2 Generation guided by sequence tagging Tag Attention To leverage the guidance information the sequence tagging task provides, we compute the probability $\widetilde { p _ { i } }$ of the i-th word being a keyword. $p _ { i } [ 0 ]$ edenotes the probability of the i-th word belonging to the non-keyword category $( " \mathrm { N " } )$ Consequently, $\widetilde { p } _ { i } = 1 - p _ { i } [ 0 ]$ indicates the probeability of the i-th word belonging to the keyword category. We incorporate $\widetilde { p _ { i } }$ into the cross-attention emechanism of the decoder in the generation model as follows:

$$
\widetilde { a } _ { t i } = \frac { e x p ( ( 1 + \widetilde { p } _ { i } ) \cdot a _ { t i } ) } { \sum _ { j = 1 } ^ { n } e x p ( ( 1 + \widetilde { p } _ { j } ) \cdot a _ { t j } ) }\tag{4}
$$

where $a _ { t i }$ represents the attention score at the tth row and i-th column in the cross-attention score matrix before applying softmax. $\widetilde { a } _ { t i }$ denotes the fienal adjusted attention score after applying softmax. $( 1 + \widetilde p _ { i } )$ ensures a balanced contribution from both ethe sequence tagging task and inherent generation task to the attention distribution. Compared to the formulation without adding 1 to $\widetilde { p _ { i } }$ (Appendix $\mathbf { A . } 3 )$ ethis formulation effectively enhances the generation process while mitigating the potential impact of tagging errors on overall generation quality $( \mathsf { A p - }$ pendix B.2).

Generation Task A sample is denoted as $( \mathbf { X } , \mathbf { Y } )$ , where Y represents the label sentence $\{ y _ { 1 } , y _ { 2 } , y _ { 3 } , . . . , y _ { m } \}$ , with m being the maximum length of Y. The loss function for the generation task with model parameters θ is defined as follows:

$$
\mathcal { L } _ { \mathrm { g e n e r a t i o n } } = - \sum _ { t = 1 } ^ { m } \log p _ { \theta } ( y _ { t } | X , y _ { < t } )\tag{5}
$$

## 3.1.3 Inference

During inference, we leverage the sequence tagging results to optimize the generated outputs. The main operation involves comparing the generated triplets from the generation task with the triplets from the sequence tagging task. If the generated aspect is a subset of the sequence tagging aspect set, or vice versa, and the generated opinion is a subset of the sequence tagging opinion set, or vice versa, the triplet is retained. Otherwise, the triplet is discarded. Read Appendix A.2 for details.

## 3.2 Label-Driven Semantic Alignment

The form of supervision, similar to Equation 5, lacks fine-grained supervision at the hidden state level and fails to fully utilize the semantic information embedded in the labels. In Label-Driven Semantic Alignment (shown in the left part of Fig. 1), we employ a label self-decoding process to obtain a more accurate decoder hidden state, which serves as a semantic representation of the label. During training, We align the decoder’s hidden state to the semantic representation, thereby enhancing the supervision of the decoder’s hidden state. This alignment ensures that the generated output closely matches the semantic content of the label.

Label Semantic Representation During training, we input the label sentence Y into the model to obtain the decoder’s hidden state:

$$
H _ { \mathrm { D e } } ^ { Y } = E n { \cdot } D e ( [ y _ { 1 } , y _ { 2 } , \ldots , y _ { m } ] ) = [ \hat { h } _ { 1 } ^ { Y } , \hat { h } _ { 2 } ^ { Y } , \ldots , \hat { h } _ { m } ^ { Y } ]\tag{6}
$$

where $E n \ – D e$ means encoder-decoder architecture. Since the label sentence contains only the words of the correct triplets, the model can effortlessly extract the correct triplets from it. In this case, the model’s input and output are both the label sentence, essentially forming a self-decoding process. Furthermore, due to the absence of irrelevant words in the input, $H _ { \mathrm { D e } } ^ { Y }$ is more accurate compared to $H _ { \mathrm { D e } } ^ { X }$ , where $H _ { \mathrm { D e } } ^ { \overline { { X } } ^ { \prime } } = D e ( H _ { \mathrm { E n } } ^ { X } ) =$ $[ \hat { h } _ { 1 } ^ { X } , \hat { h } _ { 2 } ^ { X } , \dots , \hat { h } _ { m } ^ { X } ]$ , as demonstrated in Experiment 4.3.3. Therefore, we regard $H _ { D e } ^ { Y }$ as an accurate semantic representation of the label that can provide substantial supervision at the decoder stage.

Alignment Labels The main objective of semantic alignment is to establish alignment between $H _ { \mathrm { D e } } ^ { X }$ and $H _ { \mathrm { D e } } ^ { Y }$ . One significant challenge arises from the fact that even though $H _ { \mathrm { D e } } ^ { Y }$ represents a more accurate hidden state, its corresponding output tokens $Y ^ { \prime }$ , as shown in Equation $^ { 7 , }$ may not always match the ground truth token sequence $Y$ during the early stages of training. Therefore, we compare $y _ { i } ^ { \prime }$ with $y _ { i } ,$ , and only when $y _ { i } ^ { \prime }$ is equal to $y _ { i }$ , it indicates that $\hat { h } i ^ { Y }$ is correct. We then allow $\hat { h } _ { i } ^ { X }$ to be close to $\hat { h } _ { i } ^ { Y }$ Otherwise, we move $\hat { h } _ { i } ^ { X }$ away from $\hat { h } _ { i } ^ { Y }$ . Use $L _ { i }$ to represent the comparison result between $y _ { i } ^ { \prime }$ and y<sub>i</sub>:

$$
Y ^ { \prime } = ( \mathrm { L m \_ h e a d } \ ( H _ { \mathrm { D e } } ^ { Y } ) ) . \mathrm { a r g m a x } ( )
$$

$$
L _ { i } = \mathrm { E q u a l } ( y _ { i } ^ { \prime } , y _ { i } )\tag{7}
$$

(8)

where Lm\_head represents a linear layer that takes the decoder’s hidden states as input and outputs a probability distribution over the vocabulary. The predicted tokens $Y ^ { \prime }$ are obtained by selecting the words with the highest probability using the argmax operation. The function "Equal" outputs 1 when the inputs are equal and 0 otherwise.

Alignment Task Alignment is achieved by adjusting the distance between $\hat { h } _ { i } ^ { X }$ and $\hat { h } _ { i } ^ { Y }$ accoding to $L _ { i }$ . Employ cosine similarity to quantify the distance:

$$
s _ { i } = \cos ( \hat { h } _ { i } ^ { X } , \hat { h } _ { i } ^ { Y } )\tag{9}
$$

$$
s _ { i } ^ { \prime } = \mathrm { R e L u } ( s _ { i } )\tag{10}
$$

where cos is the cosine similarity, and ReLu is used to limit the similarity values between 0 and 1 (Appendix B.3). We compute the alignment loss using binary cross-entropy to enforce the cosine similarity scores to align with the labels $L \colon$

$$
\mathcal { L } _ { \mathrm { a l i g n m e n t } } = \sum _ { i = 1 } ^ { m } { \mathrm { B C E l o s s } ( s _ { i } ^ { \prime } , L _ { i } ) }\tag{11}
$$

Final Loss. Therefore, the final loss is defined as follows:

$$
\mathcal { L } = \alpha _ { 1 } \mathcal { L } _ { \mathrm { g e n e r a t i o n } } + \alpha _ { 2 } \mathcal { L } _ { \mathrm { t a g g i n g } } + \alpha _ { 3 } \mathcal { L } _ { \mathrm { a l i g n m e n t } }\tag{12}
$$

where $\alpha _ { 1 } , \alpha _ { 2 }$ and $\alpha _ { 3 }$ are hyperparameters.

## 4 Experiments

## 4.1 Experiment Setup

ASTE Dataset We evaluate our TAGS on four popular ASTE datasets shown in Table 1: 14Res, 14Lap, 15Res, 16Res (Pontiki et al., 2014, 2015, 2016), which are modified for ASTE task by Fan et al. (2019); Peng et al. (2020); Xu et al. (2020a); Wu et al. (2020).

Baseline Models We categorize the comparison models into the following three types:

1.Sequence tagging-based models, such as OTE-MTL (Zhang et al., 2020), GTS (Wu et al., 2020), JET (Xu et al., 2020b), EMC-GCN (Chen et al., 2022a), SyMux (Fei et al., 2022), SCEDD (Zhang et al., 2022b), BDTF (Zhang et al., 2022a), SA-Transformer (Yuan et al., 2023), STAGE (Liang et al., 2023).

2.Generation-based models, such as GAS(Zhang et al., 2020), Paraphrase (Wu et al., 2020), BARTABSA (Yan et al., 2021), PASTE (Mukherjee et al., 2021), Seq2Path (Mao et al., 2022), DLO (Hu et al., 2022a), LEGO-ABSA (Gao et al., 2022), EHG (Lv et al., 2023) and Mvp (Gou et al., 2023).

3.Models based on other methods: reinforcement learning based model ASTE-RL (Jian et al., 2021), reading comprehension based model BMRC (Chen et al., 2021), and span-level models Span-ASTE (Xu et al., 2021) and SBN (Chen et al., 2022b).

Experiment Details We employ the T5-base model (Raffel et al., 2020) from the huggingface Transformer library as our pre-trained generative encoder-decoder model. During training, we set the learning rate to 3e-4 for T5 and 5e-3 for all the linear layers. The model is trained for 40 epochs on Nvidia 3090 GPUs, and the hyperparameters of Equation 12 are set as follows: $\alpha _ { 1 } = 1 0 , \alpha _ { 2 } = 1$ and $\alpha _ { 3 } = 1$ . The probability threshold in the inference stage is 0.999. All the reported results are the average of five runs with different random seeds.

Table 1: Statistics of datasets. S and T mean the total number of sentences and triplets. POS, NEU, and NEG represent the number of positive, neutral, and negative sentiment triplets, respectively.
<table><tr><td colspan="2">Dataset</td><td>S</td><td>T</td><td>POS</td><td>NEU</td><td>NEG</td></tr><tr><td rowspan="3">15Res</td><td>train</td><td>605</td><td>1013</td><td>783</td><td>25</td><td>205</td></tr><tr><td>dev</td><td>148</td><td>249</td><td>185</td><td>11</td><td>53</td></tr><tr><td>test</td><td>322</td><td>485</td><td>317</td><td>25</td><td>143</td></tr><tr><td rowspan="3">16es</td><td>train</td><td>857</td><td>1394</td><td>1015</td><td>50</td><td>329</td></tr><tr><td>dev</td><td>210</td><td>339</td><td>252</td><td>11</td><td>76</td></tr><tr><td>test</td><td>326</td><td>514</td><td>407</td><td>29</td><td>78</td></tr><tr><td rowspan="3">141aP</td><td>train</td><td>906</td><td>1460</td><td>817</td><td>126</td><td>517</td></tr><tr><td>dev</td><td>219</td><td>345</td><td>169</td><td>36</td><td>140</td></tr><tr><td>test</td><td>328</td><td>541</td><td>364</td><td>63</td><td>114</td></tr><tr><td rowspan="3">14Res</td><td>train</td><td>1266</td><td>2337</td><td>1015</td><td>50</td><td>329</td></tr><tr><td>dev</td><td>310</td><td>577</td><td>252</td><td>11</td><td>76</td></tr><tr><td>test</td><td>492</td><td>994</td><td>407</td><td>29</td><td>78</td></tr></table>

Evaluation Metrics Following previous works (Peng et al., 2020), we employ widely used evaluation metrics, namely $\mathrm { F _ { 1 } }$ scores $( \mathrm { F _ { 1 } } )$ , recall (R), and precision (P).

## 4.2 Main Results

The main results are reported in Table 2. In this task, F1 is the most important metric (Peng et al., 2020; Chen et al., 2022b; Gao et al., 2022; Gou et al., 2023). TAGS significantly outperforms the previous state-of-the-art method Mvp (Gou et al., 2023), specifically achieving a lead of up to 3.13% on the 16res dataset and 2.01% on the 15res dataset according to the $F _ { 1 }$ metric.

Based on the principles of sequence taggingbased methods, these approaches tend to be conservative, which means they only predict a triplet when they are highly confident. Consequently, the precision of these methods tends to be higher than the recall, as shown in both the OTE-MTL and JET methods in Table 2. In contrast, generation methods tend to over-predict the number of triplets due to their strong creativity. Consequently, the recall in the results of generation methods is generally higher than the precision.

By introducing a sequence tagging task, the

Table 2: Main results on 4 datasets of ASTE tasks. The best results are in bold, while the second best are underlined. denotes the replication results, while the other results are obtained from original papers.
<table><tr><td rowspan=2 colspan=2>Model</td><td rowspan=1 colspan=3>16res</td><td rowspan=1 colspan=2>15res</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>14lap</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>14res</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>P    R    F1</td><td rowspan=1 colspan=3>P    R    F1</td><td rowspan=1 colspan=1>P</td><td rowspan=1 colspan=2>R    F1</td><td rowspan=1 colspan=2>P    R</td><td rowspan=1 colspan=1>F1</td></tr><tr><td rowspan=1 colspan=2>BMRC</td><td rowspan=1 colspan=3>71.20 61.0865.75</td><td rowspan=1 colspan=1>68.51</td><td rowspan=1 colspan=1>53.40</td><td rowspan=1 colspan=1>60.02</td><td rowspan=1 colspan=1>70.55</td><td rowspan=1 colspan=1>48.98</td><td rowspan=1 colspan=1>57.82</td><td rowspan=1 colspan=2>75.61 61.77</td><td rowspan=1 colspan=1>67.99</td></tr><tr><td rowspan=1 colspan=2>ASTE-RL</td><td rowspan=1 colspan=1>67.21</td><td rowspan=1 colspan=2>69.6968.41</td><td rowspan=1 colspan=1>65.45</td><td rowspan=1 colspan=1>60.29</td><td rowspan=1 colspan=1>62.72</td><td rowspan=1 colspan=1>64.80</td><td rowspan=1 colspan=1>54.99</td><td rowspan=1 colspan=1>59.50</td><td rowspan=1 colspan=2>70.6068.65</td><td rowspan=1 colspan=1>69.71</td></tr><tr><td rowspan=1 colspan=2>Span-ASTE</td><td rowspan=1 colspan=1>69.45</td><td rowspan=1 colspan=2>71.1770.26</td><td rowspan=1 colspan=1>62.18</td><td rowspan=1 colspan=1>64.45</td><td rowspan=1 colspan=1>63.27</td><td rowspan=1 colspan=1>63.44</td><td rowspan=1 colspan=1>55.84</td><td rowspan=1 colspan=1>59.38</td><td rowspan=1 colspan=2>72.8970.89</td><td rowspan=1 colspan=1>71.85</td></tr><tr><td rowspan=1 colspan=2>SBN</td><td rowspan=1 colspan=1>71.59</td><td rowspan=1 colspan=2>72.57 72.08</td><td rowspan=1 colspan=1>69.93</td><td rowspan=1 colspan=2>60.41 64.82</td><td rowspan=1 colspan=1>65.68</td><td rowspan=1 colspan=1>59.88</td><td rowspan=1 colspan=1>62.65</td><td rowspan=1 colspan=2>76.36 72.43</td><td rowspan=1 colspan=1>74.34</td></tr><tr><td rowspan=1 colspan=2>OTE-MTL</td><td rowspan=1 colspan=1>62.88</td><td rowspan=1 colspan=1>52.10</td><td rowspan=1 colspan=1>56.96</td><td rowspan=1 colspan=1>56.37</td><td rowspan=1 colspan=1>40.94</td><td rowspan=1 colspan=1>47.13</td><td rowspan=1 colspan=1>49.53</td><td rowspan=1 colspan=1>39.22</td><td rowspan=1 colspan=1>43.42</td><td rowspan=1 colspan=2>62.0055.97</td><td rowspan=1 colspan=1>58.71</td></tr><tr><td rowspan=1 colspan=2>JET</td><td rowspan=1 colspan=1>70.42</td><td rowspan=1 colspan=1>58.37</td><td rowspan=1 colspan=1>63.83</td><td rowspan=1 colspan=1>64.45</td><td rowspan=1 colspan=1>51.96</td><td rowspan=1 colspan=1>57.53</td><td rowspan=1 colspan=1>55.39</td><td rowspan=1 colspan=1>47.33</td><td rowspan=1 colspan=1>51.04</td><td rowspan=1 colspan=2>70.5655.94</td><td rowspan=1 colspan=1>62.40</td></tr><tr><td rowspan=1 colspan=2>GTS</td><td rowspan=1 colspan=1>66.08</td><td rowspan=1 colspan=1>69.91</td><td rowspan=1 colspan=1>67.93</td><td rowspan=1 colspan=1>62.59</td><td rowspan=1 colspan=1>57.94</td><td rowspan=1 colspan=1>60.15</td><td rowspan=1 colspan=1>57.82</td><td rowspan=1 colspan=1>51.32</td><td rowspan=1 colspan=1>54.36</td><td rowspan=1 colspan=1>67.76</td><td rowspan=1 colspan=1>67.29</td><td rowspan=1 colspan=1>67.50</td></tr><tr><td rowspan=1 colspan=2>EMC-GAN</td><td rowspan=1 colspan=1>64.43</td><td rowspan=1 colspan=1>72.63</td><td rowspan=1 colspan=1>67.69</td><td rowspan=1 colspan=1>60.45</td><td rowspan=1 colspan=1>62.72</td><td rowspan=1 colspan=1>61.55</td><td rowspan=1 colspan=1>59.61</td><td rowspan=1 colspan=1>56.30</td><td rowspan=1 colspan=1>57.90</td><td rowspan=1 colspan=1>70.37</td><td rowspan=1 colspan=1>72.84</td><td rowspan=1 colspan=1>71.58</td></tr><tr><td rowspan=1 colspan=2>SyMux</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>72.76</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>63.13</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>60.11</td><td rowspan=1 colspan=2>1     1</td><td rowspan=1 colspan=1>74.84</td></tr><tr><td rowspan=1 colspan=2>SCEDD</td><td rowspan=1 colspan=1>66.11</td><td rowspan=1 colspan=1>71.37</td><td rowspan=1 colspan=1>68.64</td><td rowspan=1 colspan=1>59.41</td><td rowspan=1 colspan=1>62.73</td><td rowspan=1 colspan=1>61.03</td><td rowspan=1 colspan=1>61.84</td><td rowspan=1 colspan=1>60.08</td><td rowspan=1 colspan=1>60.95</td><td rowspan=1 colspan=2>70.2773.02</td><td rowspan=1 colspan=1>71.62</td></tr><tr><td rowspan=1 colspan=2>SA-Transformer</td><td rowspan=1 colspan=1>72.01</td><td rowspan=1 colspan=1>62.87</td><td rowspan=1 colspan=1>67.13</td><td rowspan=1 colspan=1>62.82</td><td rowspan=1 colspan=1>58.31</td><td rowspan=1 colspan=1>60.48</td><td rowspan=1 colspan=1>61.28</td><td rowspan=1 colspan=1>48.98</td><td rowspan=1 colspan=1>54.44</td><td rowspan=1 colspan=2>70.7665.85</td><td rowspan=1 colspan=1>68.22</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>BDTF</td><td rowspan=1 colspan=1>71.44</td><td rowspan=1 colspan=1>73.13</td><td rowspan=1 colspan=1>72.27</td><td rowspan=1 colspan=1>68.76</td><td rowspan=1 colspan=1>63.71</td><td rowspan=1 colspan=1>66.12</td><td rowspan=1 colspan=1>68.94</td><td rowspan=1 colspan=1>55.97</td><td rowspan=1 colspan=1>61.74</td><td rowspan=1 colspan=1>75.53</td><td rowspan=1 colspan=1>73.24</td><td rowspan=1 colspan=1>74.35</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>STAGE</td><td rowspan=1 colspan=1>77.67</td><td rowspan=1 colspan=2>68.4472.75</td><td rowspan=1 colspan=1>72.33</td><td rowspan=1 colspan=2>58.93 64.94</td><td rowspan=1 colspan=1>70.56</td><td rowspan=1 colspan=1>55.16</td><td rowspan=1 colspan=1>61.88</td><td rowspan=1 colspan=2>78.5169.30</td><td rowspan=1 colspan=1>73.61</td></tr><tr><td rowspan=1 colspan=2>GAS</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>70.10</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>62.10</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>60.78</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>72.16</td></tr><tr><td rowspan=1 colspan=2>Paraphrase</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>71.70</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>62.56</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>61.13</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>72.03</td></tr><tr><td rowspan=1 colspan=2>BARTASA</td><td rowspan=1 colspan=1>66.6</td><td rowspan=1 colspan=1>68.68</td><td rowspan=1 colspan=1>67.62</td><td rowspan=1 colspan=1>59.14</td><td rowspan=1 colspan=1>59.38</td><td rowspan=1 colspan=1>59.26</td><td rowspan=1 colspan=1>61.41</td><td rowspan=1 colspan=1>56.19</td><td rowspan=1 colspan=1>58.69</td><td rowspan=1 colspan=1>65.52</td><td rowspan=1 colspan=1>64.99</td><td rowspan=1 colspan=1>65.25</td></tr><tr><td rowspan=1 colspan=2>PASTE</td><td rowspan=1 colspan=1>66.1</td><td rowspan=1 colspan=1>69.8</td><td rowspan=1 colspan=1>67.9</td><td rowspan=1 colspan=1>61.7</td><td rowspan=1 colspan=1>60.8</td><td rowspan=1 colspan=1>61.3</td><td rowspan=1 colspan=1>61.2</td><td rowspan=1 colspan=1>53.6</td><td rowspan=1 colspan=1>57.1</td><td rowspan=1 colspan=1>66.7</td><td rowspan=1 colspan=1>66.5</td><td rowspan=1 colspan=1>66.6</td></tr><tr><td rowspan=1 colspan=2>DLO</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>72.23</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>63.52</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>61.33</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>72.02</td></tr><tr><td rowspan=1 colspan=2>Seq2path†</td><td rowspan=1 colspan=1>71.59</td><td rowspan=1 colspan=1>75.41</td><td rowspan=1 colspan=1>73.40</td><td rowspan=1 colspan=1>62.62</td><td rowspan=1 colspan=1>65.48</td><td rowspan=1 colspan=1>64.02</td><td rowspan=1 colspan=1>64.57</td><td rowspan=1 colspan=1>60.04</td><td rowspan=1 colspan=1>62.22</td><td rowspan=1 colspan=2>73.28 74.23</td><td rowspan=1 colspan=1>73.75</td></tr><tr><td rowspan=1 colspan=2>LEGO-ABSA</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>69.9</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=2>1    64.4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=2>1   62.2</td><td rowspan=1 colspan=2>1     1</td><td rowspan=1 colspan=1>73.7</td></tr><tr><td rowspan=1 colspan=2>EHG</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=2>1   72.35</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=2>1   63.58</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=2>1   61.53</td><td rowspan=1 colspan=2>1     1</td><td rowspan=1 colspan=1>71.82</td></tr><tr><td rowspan=1 colspan=2>MvP</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=2>1  73.48</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=2>1   65.89</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=2>1  63.33</td><td rowspan=1 colspan=3>1     1  74.05</td></tr><tr><td rowspan=1 colspan=2>TAGS</td><td rowspan=1 colspan=1>76.37</td><td rowspan=1 colspan=2>76.85 76.61</td><td rowspan=1 colspan=1>70.23</td><td rowspan=1 colspan=2>65.73 67.90</td><td rowspan=1 colspan=1>65.11</td><td rowspan=1 colspan=2>62.20 64.53</td><td rowspan=1 colspan=3>77.38 72.86 75.05</td></tr></table>

TAGS method alleviates the excessive creativity of the generation model by directing its focus toward keywords. This not only enhances the quality of the generated output but also objectively limits the number of excessively generated triplets. Leveraging the semantic alignment with labels, TAGS further enhances the quality of the generated triplets. Consequently, compared to conventional generation methods, our method can extract more correct triplets with fewer predicted triplets. This leads to higher precision, recall, and consequently, a higher F1 score. Furthermore, when compared to conventional sequence tagging methods, TAGS surpasses them due to the generation model’s ability to utilize semantic information from the labels and its inherent creativity. Thus, TAGS outperforms most previous methods in terms of F1 score, precision, and recall.

Table 3: Ablation study. The results reported are the average F1 scores.
<table><tr><td>Model</td><td>16Res</td><td>15Res</td><td>14Lap</td><td>14Res</td></tr><tr><td>Full Model</td><td>76.61</td><td>67.90</td><td>64.53</td><td>75.05</td></tr><tr><td>w/o Tagging traing</td><td>72.83</td><td>64.34</td><td>62.15</td><td>72.96</td></tr><tr><td>w/o Tag Attention</td><td>75.68</td><td>66.51</td><td>63.14</td><td>74.18</td></tr><tr><td>w/o Inference</td><td>75.49</td><td>66.66</td><td>64.19</td><td>74.34</td></tr><tr><td>w/o Alignment</td><td>75.37</td><td>65.82</td><td>63.20</td><td>73.44</td></tr></table>

## 4.3 Ablation

The results of the ablation experiments are presented in Table 3.

Effectiveness of the Sequence Tagging Task: The "w/o Tagging training" condition denotes the removal of the sequence tagging task, including multitask training, tag attention, and the specialized inference stage. It means that the model only relies on the Semantic Alignment component. Compared to the "Full Model", the performance under this condition decreased in all datasets: 16res (-3.78%), 15res (-3.56%), 14Lap (-2.38%), and 14Res (-2.09%), providing evidence for the effectiveness of the sequence tagging task. To further investigate the role of the sequence tagging task, we conducted Experiments 4.3.1.

Effectiveness of Tag Attention: The "w/o Tag attention" condition refers to the absence of tag attention while still retaining the training of the sequence tagging task, special inference stage, and the Semantic Alignment component. When compared to the "full model," there was an average performance decrease of 1.11% across all datasets, providing evidence for the effectiveness of Tag Attention. In Appendix B.2, we further analyze the impact of different utilization methods for sequence tagging probabilities on Tag Attention. This analysis enables us to gain a deeper understanding of how the utilization of sequence tagging probabilities influences the performance of Tag Attention.

Effectiveness of Inference: The "w/o Inference" condition refers to the absence of a special inference stage. In comparison to the "Full model," there was an average performance decrease of 0.85% across all datasets. This provides evidence for the effectiveness of the Inference stage. In Experiment 4.3.2, we further investigate the experimental results related to the threshold hyperparameter in the inference stage.

Effectiveness of the Semantic Alignment: The "w/o Alignment" condition refers to the removal of the Semantic Alignment component. Compared to the "Full Model", the performance under this condition decreased in all datasets: 16res (-1.24%), 15res (-2.08%), 14Lap (-1.33%), and 14Res (-1.61%). This demonstrates the effectiveness of the Semantic Alignment component in improving overall performance. To further investigate the impact of the loss function on the Semantic Alignment component, we conducted Experiment B.3.

## 4.3.1 Loss Hyperparameters

In this section, we investigate the impact of loss hyperparameters. First, we fix $\alpha _ { 2 }$ and vary $\alpha _ { 1 }$ and $\alpha _ { 3 }$ as shown in Fig. 2(a). As $\alpha _ { 1 }$ gradually increases, the performance initially improves and then decreases, achieving the best result at 10. Comparing the three curves in the graph, the curve corresponding to $\alpha _ { 3 } = 1$ achieves the best result. Next, we fix $\alpha _ { 1 } = 1 0$ and vary $\alpha _ { 2 }$ and $\alpha _ { 3 }$ as shown in Fig. 2(b). As the α<sub>2</sub> increases, the performance initially improves and then decreases, achieving the best result at 1. Furthermore, the curve corresponding to $\alpha _ { 3 } = 1$ achieves the best result. We select the loss ratios corresponding to the optimal performance as our final hyperparameter settings: $\alpha _ { 1 } = 1 0 \ : \mathrm { \Omega }$ α<sub>2</sub> = 1, $\alpha _ { 3 } = 1$ . This suggests that our method primarily focuses on the generative task, with the other two components serving as auxiliary factors.

Table 4: F1 results on the development dataset for different thresholds.
<table><tr><td>threshold</td><td>16res 15res</td><td>14lap</td><td>14res</td></tr><tr><td>0.9</td><td>77.14 73.91</td><td>62.14</td><td>65.88</td></tr><tr><td>0.99</td><td>77.62 74.27</td><td>62.61</td><td>66.00</td></tr><tr><td>0.999</td><td>77.73 74.35</td><td>62.78</td><td>66.08</td></tr><tr><td>0.9999</td><td>77.66</td><td>74.29 62.77</td><td>66.00</td></tr></table>

![](images/a2756f9dd0ff422626d302f64a7274c1cdb742f8329eb05391f20b4639bf9cba.jpg)  
(a) generation ratio

![](images/5e2cf3a54632d3bd58c83abb0e980eee16ff3b03a726bc1aa82e3c964a275e8f.jpg)  
(b) tagging ratio  
Figure 2: $F _ { 1 }$ result with different loss ratios.

## 4.3.2 Threshold Hyperparameter in Inference

We conducted experiments on the development set to determine the most suitable probability threshold hyperparameter. We experimented with four different values for the threshold hyperparameter. The results are shown in Table 4. As the threshold increases, the performance initially improves and then decreases, achieving the best result at 0.999. This threshold value is very close to 1. In the generated results, the probability of each word is also very close to 1, even for some incorrect words. Therefore, when we require a threshold to filter out potentially erroneous triplets, this threshold should also be very close to 1. Hence, 0.999 is a reasonable choice.

## 4.3.3 Correctness of Semantic Representation

To demonstrate that $H _ { \mathrm { D e } } ^ { Y }$ is more accurate, during self-decoding, we replace each label sentence with the original input sentence with a probability of $^ { r } \cdot$ This increases the influence of irrelevant words on semantic representation. We then train the TAGS model using the semantic representation obtained from this self-decoding process and the corresponding performance reflects the correctness of the semantic representation. We conducted experiments on the 16res dataset and the results are presented in Fig. 3. The results indicate that as r increases, performance decreases. This demonstrates that an increasing number of irrelevant words in the input lead to a decrease in the correctness of the semantic representation, resulting in a gradual decline in performance.

![](images/d8332d2be99bdcd4c75887ea50ece91e059e02bbe884bc5c948ac7305370c35d.jpg)  
Figure 3: $F _ { 1 }$ result with different r.

## 4.4 Results on Other ABSA Tasks

The proposed model provides a unified framework to effectively address the Aspect-Based Sentiment Analysis (ABSA) problem. To demonstrate the effectiveness of TAGS and its generalizability across different tasks, we conducted experiments on two ABSA tasks: AOPE and UABSA. We compared TAGS with the models in Appendix B.4.

AOPE focuses on extracting (aspect, opinion) pairs, similar to ASTE, but without sentiment analysis. This task requires accurate identification of keywords in the sequence tagging task, as well as the assistance of Tag Attention and Semantic Alignment components. The F1 results for the AOPE task are presented in Table 5. TAGS outperforms the previous model on all four datasets: 2.28% for 16Res, 1.50% for 14Lap, 0.83% for 15Res, and 0.51% for 14Res. The improvement in results demonstrates the effectiveness of the aforementioned components.

UABSA focuses on extracting (aspect, sentiment) pairs, similar to ASTE, but without extracting opinions. This task presents challenges in accurately classifying sentiments in sequence tagging and aligning sentiments in Semantic Alignment. The F1 results for the UABSA tasks are presented in Table 6. TAGS has achieved an average improvement of 1.27% compared to the previous model. This improvement demonstrates the effectiveness of the aforementioned components in enhancing the accuracy of sentiment analysis.

<table><tr><td>Model</td><td>16Res</td><td>14Lap</td><td>15Res</td><td>14Res</td></tr><tr><td>HAST+TOWE(Zhang et al., 2021b)</td><td>63.84</td><td>53.41</td><td>58.12</td><td>62.39</td></tr><tr><td>JERE-MHS(Zhang et al., 2021b)</td><td>67.65</td><td>52.34</td><td>59.64</td><td>66.02</td></tr><tr><td>SpanMlt(Zhao et al., 2020)</td><td>71.78</td><td>68.66</td><td>64.68</td><td>75.60</td></tr><tr><td>SDRN(Chen et al., 2020)</td><td>73.67</td><td>66.18</td><td>65.75</td><td>73.30</td></tr><tr><td>GAS(Zhang et al., 2021b)</td><td>74.54</td><td>68.08</td><td>67.19</td><td>74.12</td></tr><tr><td>LEGO(Gao et al., 2022)</td><td>77.6</td><td>69.7</td><td>71.4</td><td>78.1</td></tr><tr><td>EHG(Lv et al., 2023)</td><td>78.19</td><td>69.05</td><td>69.11</td><td>77.17</td></tr><tr><td>TAGS (Our)</td><td>80.47</td><td>71.20</td><td>72.23</td><td>78.61</td></tr></table>

Table 5: Main F1 results of the AOPE task. The best results are in bold, second best results are underlined.

<table><tr><td>Model</td><td>14Lap</td><td>16Res</td><td>15Res</td><td>14Res</td></tr><tr><td rowspan="5">BERT+GRU(Li et al., 2019b) SPAN-BERT(Hu et al., 2019) MN-BERT (Li et al., 2019b) RACL(Chen and Qian, 2020)</td><td>61.12</td><td>70.21</td><td>59.60</td><td>73.17</td></tr><tr><td>61.25</td><td></td><td>62.29</td><td>73.68</td></tr><tr><td>61.73</td><td></td><td>60.22</td><td>70.72</td></tr><tr><td>63.40</td><td></td><td>66.05</td><td>75.42</td></tr><tr><td>65.94</td><td></td><td>65.08</td><td>75.95</td></tr><tr><td>Dual-MRC(Mao et al., 2021) GAS(Zhang et al., 2021b)</td><td>67.37</td><td>71.87</td><td>65.75</td><td>75.77</td></tr><tr><td>EHG(Lv et al., 2023)</td><td>68.48</td><td>77.12</td><td>70.04</td><td>79.32</td></tr><tr><td>TAGS (Our)</td><td>71.37</td><td>78.11</td><td>70.76</td><td>79.80</td></tr></table>

Table 6: Main F1 results of the UABSA task. The best results are in bold, second best results are underlined.

These results demonstrate the effectiveness and generalization of TAGS across different tasks.

## 5 Related Work

ASTE employed sequence tagging methods, when it was first introduced by (Peng et al., 2020). Subsequent research efforts (Xu et al., 2020b; Wu et al., 2020; Chen et al., 2022a; Liang et al., 2022; Gou et al., 2023) have been focused on enhancing the sequence tagging schemes and model components to facilitate the integration and mutual interpretation of the triple elements. However, the sequence tagging technique in ASTE fails to capture the semantic information conveyed by the labels, which can lead to semantic mismatches in the predicted results(Zhang et al., 2021b). Generation methods were initially proposed by Zhang et al. (2021c). The generation-based approach in ASTE has achieved good performance by reducing potential error propagation present in pipeline methods and effectively utilizing the rich semantic information provided by labels(Paolini et al., 2021; Yu et al., 2023). They employed various targets for generation, such as sentiment element sequences (Zhang et al., 2021c,c; Hu et al., 2022b), natural language (Liu et al., 2021; Zhang et al., 2021a), and structured extraction patterns (Lu et al., 2022). Recently proposed models, LEGO-ABSA (Gao et al., 2022), UnifiedABSA (Wang et al., 2022) and Mvp (Gou et al., 2023), have focused on leveraging task prompts or guided design for multi-task processing.

## 6 Conclusion

In this work, we introduce a generation model called TAGS, which enhances the supervision of both the encoder and decoder through multipleperspective tagging assistance and label semantic representations. Specifically, TAGS utilizes sequence tagging to enhance the generation model in multiple aspects: Multitask Learning, Guided Generation, and Result Optimization. Additionally, TAGS employs a label self-decoding process to obtain semantic representations of labels and aligns the decoder’s hidden states with these representations, thereby providing enhanced semantic supervision for the decoder’s hidden states. These two components enhance the supervision of the encoder and decoder’s hidden states, resulting in improved generation quality. Extensive experiments demonstrate that our method significantly advances the state-of-the-art on benchmark datasets.

## 7 Limitations

Despite achieving state-of-the-art performance, our proposed methods still have some limitations that point to potential future directions.

1. Compared to conventional generation methods, our approach requires an additional generation step to obtain more accurate hidden states, namely semantic labels. As a result, there is an increase in training overhead.

2. Although we apply a simple yet effective aggregation strategy to combine the results of the sequence tagging task and generation task, more advanced strategies can be explored to further enhance performance.

3. We have indeed observed that the improvement of our model varies on different datasets, which may be due to the differences in the characteristics of these datasets.

4. Our work utilizes a relatively simple sequence tagging approach, specifically characterized by the absence of explicit pairing between extracted aspects and opinions. There is room for designing a more robust and sophisticated sequence tagging scheme that can also seamlessly integrate with generation models, thereby enhancing performance.

## 8 Ethics Statement

In all our experiments, we used existing datasets that have been widely used in previous scientific publications. When analyzing the experimental results, we strive to maintain fairness and honesty, ensuring that our work does not cause harm to anyone.

As for broader implications, this work may contribute to further research in sentiment analysis and the use of generation methods to simplify and automate the extraction of user opinions in realworld applications. However, it is important to note that this work involves fine-tuning large-scale pre-trained language models to generate sentiment triplets. Due to the nature of the Internet-based large-scale pre-training corpora, the predicted sentiment polarities may be influenced by unintended biases related to gender, race, and intersectional identities (Tan and Celis, 2019). LPMLs often inherit biases present in their training data, potentially leading to biased sentiment analysis results, particularly when assessing text from underrepresented or marginalized groups, thereby perpetuating and amplifying societal prejudices. Another limitation is the opacity of these models. Their complex architectures make it challenging to fully understand the reasoning behind their predictions, raising concerns about transparency and accountability. This lack of interpretability may hinder the identification and mitigation of harmful biases and ethical violations in sentiment analysis applications. It is crucial for the natural language processing community to consider these biases more extensively. Fortunately, these issues are actively being addressed within the research community, including efforts to standardize datasets and methodologies.

## 9 Acknowledgements

We would like to thank the anonymous reviewers for their insightful comments. This work is partially supported by National Natural Science Foundation of China (Grants No. 62176271), and Science and Technology Program of Guangzhou (Grant No. 202201011681).

## References

Hongjie Cai, Rui Xia, and Jianfei Yu. 2021. Aspectcategory-opinion-sentiment quadruple extraction with implicit aspects and opinions. In Proceedings of the 59th Annual Meeting of the Association for

Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 340–350, Online. Association for Computational Linguistics.

Hao Chen, Zepeng Zhai, Fangxiang Feng, Ruifan Li, and Xiaojie Wang. 2022a. Enhanced multi-channel graph convolutional network for aspect sentiment triplet extraction. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 2974–2985. Association for Computational Linguistics.

Shaowei Chen, Jie Liu, Yu Wang, Wenzheng Zhang, and Ziming Chi. 2020. Synchronous double-channel recurrent network for aspect-opinion pair extraction. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 6515– 6524, Online. Association for Computational Linguistics.

Shaowei Chen, Yu Wang, Jie Liu, and Yuelin Wang. 2021. Bidirectional machine reading comprehension for aspect sentiment triplet extraction. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications ofArtificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, pages 12666–12674. AAAI Press.

Yuqi Chen, Keming Chen, Xian Sun, and Zequn Zhang. 2022b. A span-level bidirectional network for aspect sentiment triplet extraction. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 4300–4309. Association for Computational Linguistics.

Zhuang Chen and Tieyun Qian. 2020. Relation-aware collaborative learning for unified aspect-based sentiment analysis. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 3685–3694, Online. Association for Computational Linguistics.

Zhifang Fan, Zhen Wu, Xin-Yu Dai, Shujian Huang, and Jiajun Chen. 2019. Target-oriented opinion words extraction with target-fused neural sequence labeling. In Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2509–2518, Minneapolis, Minnesota. Association for Computational Linguistics.

Hao Fei, Fei Li, Chenliang Li, Shengqiong Wu, Jingye Li, and Donghong Ji. 2022. Inheriting the wisdom of predecessors: A multiplex cascade framework for unified aspect-based sentiment analysis. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022, pages 4121–4128. ijcai.org.

Tianhao Gao, Jun Fang, Hanyu Liu, Zhiyuan Liu, Chao Liu, Pengzhang Liu, Yongjun Bao, and Weipeng Yan. 2022. LEGO-ABSA: A prompt-based task assemblable unified generative framework for multi-task aspect-based sentiment analysis. In Proceedings of the 29th International Conference on Computational Linguistics, pages 7002–7012, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.

Zhibin Gou, Qingyan Guo, and Yujiu Yang. 2023. Mvp: Multi-view prompting improves aspect sentiment tuple prediction. CoRR, abs/2305.12627.

Ruidan He, Wee Sun Lee, Hwee Tou Ng, and Daniel Dahlmeier. 2019. An interactive multi-task learning network for end-to-end aspect-based sentiment analysis. In Proceedings ofthe 57th Annual Meeting of the Association for Computational Linguistics, pages 504–515, Florence, Italy. Association for Computational Linguistics.

Mengting Hu, Yike Wu, Hang Gao, Yinhao Bai, and Shiwan Zhao. 2022a. Improving aspect sentiment quad prediction via template-order data augmentation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 7889–7900. Association for Computational Linguistics.

Mengting Hu, Yike Wu, Hang Gao, Yinhao Bai, and Shiwan Zhao. 2022b. Improving aspect sentiment quad prediction via template-order data augmentation. EMNLP.

Minghao Hu, Yuxing Peng, Zhen Huang, Dongsheng Li, and Yiwei Lv. 2019. Open-domain targeted sentiment analysis via span-based extraction and classification. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 537–546, Florence, Italy. Association for Computational Linguistics.

Samson Yu Bai Jian, Tapas Nayak, Navonil Majumder, and Soujanya Poria. 2021. Aspect sentiment triplet extraction using reinforcement learning. In CIKM ’21: The 30th ACM International Conference on Information and Knowledge Management, Virtual Event, Queensland, Australia, November 1 - 5, 2021, pages 3603–3607. ACM.

Xin Li, Lidong Bing, Piji Li, and Wai Lam. 2019a. A unified model for opinion target extraction and target sentiment prediction. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications ofArtificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019, pages 6714–6721. AAAI Press.

Xin Li, Lidong Bing, Wenxuan Zhang, and Wai Lam. 2019b. Exploiting BERT for end-to-end aspect-based

sentiment analysis. In Proceedings ofthe 5th Workshop on Noisy User-generated Text (W-NUT 2019), pages 34–41, Hong Kong, China. Association for Computational Linguistics.

Shuo Liang, Wei Wei, Xian-Ling Mao, Yuanyuan Fu, Rui Fang, and Dangyang Chen. 2022. STAGE: span tagging and greedy inference scheme for aspect sentiment triplet extraction. CoRR, abs/2211.15003.

Shuo Liang, Wei Wei, Xian-Ling Mao, Yuanyuan Fu, Rui Fang, and Dangyang Chen. 2023. STAGE: span tagging and greedy inference scheme for aspect sentiment triplet extraction. In Thirty-Seventh AAAI Conference on Artificial Intelligence, AAAI 2023, Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence, IAAI 2023, Thirteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2023, Washington, DC, USA, February 7-14, 2023, pages 13174–13182. AAAI Press.

Jian Liu, Zhiyang Teng, Leyang Cui, Hanmeng Liu, and Yue Zhang. 2021. Solving aspect category sentiment analysis as a text generation task. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 4406–4416, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Yaojie Lu, Qing Liu, Dai Dai, Xinyan Xiao, Hongyu Lin, Xianpei Han, Le Sun, and Hua Wu. 2022. Unified structure generation for universal information extraction. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5755–5772, Dublin, Ireland. Association for Computational Linguistics.

Haoran Lv, Junyi Liu, Henan Wang, Yaoming Wang, Jixiang Luo, and Yaxiao Liu. 2023. Efficient hybrid generation framework for aspect-based sentiment analysis. In Proceedings of the 17th Conference of the European Chapter ofthe Associationfor Computational Linguistics, EACL 2023, Dubrovnik, Croatia, May 2-6, 2023, pages 1007–1018. Association for Computational Linguistics.

Yue Mao, Yi Shen, Jingchao Yang, Xiaoying Zhu, and Longjun Cai. 2022. Seq2path: Generating sentiment tuples as paths of a tree. In Findings ofthe Association for Computational Linguistics: ACL 2022, pages 2215–2225.

Yue Mao, Yi Shen, Chao Yu, and Longjun Cai. 2021. A joint training dual-mrc framework for aspect based sentiment analysis. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, pages 13543–13551. AAAI Press.

Rajdeep Mukherjee, Tapas Nayak, Yash Butala, Sourangshu Bhattacharya, and Pawan Goyal. 2021. PASTE: A tagging-free decoding framework using

pointer networks for aspect sentiment triplet extraction. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 9279– 9291. Association for Computational Linguistics.

Giovanni Paolini, Ben Athiwaratkun, Jason Krone, Jie Ma, Alessandro Achille, Rishita Anubhai, Cícero Nogueira dos Santos, Bing Xiang, and Stefano Soatto. 2021. Structured prediction as translation between augmented natural languages. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Haiyun Peng, Lu Xu, Lidong Bing, Fei Huang, Wei Lu, and Luo Si. 2020. Knowing what, how and why: A near complete solution for aspect-based sentiment analysis. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications ofArtificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 8600–8607. AAAI Press.

Joseph Peper and Lu Wang. 2022. Generative aspectbased sentiment analysis with contrastive learning and expressive structure. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 6089–6095. Association for Computational Linguistics.

Maria Pontiki, Dimitris Galanis, Haris Papageorgiou, Ion Androutsopoulos, Suresh Manandhar, Mohammad AL-Smadi, Mahmoud Al-Ayyoub, Yanyan Zhao, Bing Qin, Orphée De Clercq, Véronique Hoste, Marianna Apidianaki, Xavier Tannier, Natalia Loukachevitch, Evgeniy Kotelnikov, Nuria Bel, Salud María Jiménez-Zafra, and Gül¸sen Eryigit.˘ 2016. SemEval-2016 task 5: Aspect based sentiment analysis. In Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016), pages 19–30, San Diego, California. Association for Computational Linguistics.

Maria Pontiki, Dimitris Galanis, Haris Papageorgiou, Suresh Manandhar, and Ion Androutsopoulos. 2015. SemEval-2015 task 12: Aspect based sentiment analysis. In Proceedings of the 9th International Workshop on Semantic Evaluation (SemEval 2015), pages 486–495, Denver, Colorado. Association for Computational Linguistics.

Maria Pontiki, Dimitris Galanis, John Pavlopoulos, Harris Papageorgiou, Ion Androutsopoulos, and Suresh Manandhar. 2014. SemEval-2014 task 4: Aspect based sentiment analysis. In Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014), pages 27–35, Dublin, Ireland. Association for Computational Linguistics.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21:140:1–140:67.

Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier. 2022. A contrastive framework for neural text generation. In NeurIPS.

Yi Chern Tan and L. Elisa Celis. 2019. Assessing social and intersectional biases in contextualized word representations. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 13209–13220.

Zengzhi Wang, Rui Xia, and Jianfei Yu. 2022. Unifiedabsa: A unified ABSA framework based on multitask instruction tuning. CoRR, abs/2211.10986.

Zhen Wu, Chengcan Ying, Fei Zhao, Zhifang Fan, Xinyu Dai, and Rui Xia. 2020. Grid tagging scheme for aspect-oriented fine-grained opinion extraction. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 2576–2585, Online. Association for Computational Linguistics.

Lu Xu, Yew Ken Chia, and Lidong Bing. 2021. Learning span-level interactions for aspect sentiment triplet extraction. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4755–4766, Online. Association for Computational Linguistics.

Lu Xu, Hao Li, Wei Lu, and Lidong Bing. 2020a. Position-aware tagging for aspect sentiment triplet extraction. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 2339–2349. Association for Computational Linguistics.

Lu Xu, Hao Li, Wei Lu, and Lidong Bing. 2020b. Position-aware tagging for aspect sentiment triplet extraction. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2339–2349, Online. Association for Computational Linguistics.

Hang Yan, Junqi Dai, Tuo Ji, Xipeng Qiu, and Zheng Zhang. 2021. A unified generative framework for aspect-based sentiment analysis. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2416–2429, Online. Association for Computational Linguistics.

Chengze Yu, Taiqiang Wu, Jiayi Li, Xingyu Bai, and Yujiu Yang. 2023. Syngen: A syntactic plug-andplay module for generative aspect-based sentiment analysis. CoRR, abs/2302.13032.

Li Yuan, Jin Wang, Liang-Chih Yu, and Xuejie Zhang. 2023. Encoding syntactic information into transformers for aspect-based sentiment triplet extraction. IEEE Transactions on Affective Computing, pages 1–15.

Chen Zhang, Qiuchi Li, Dawei Song, and Benyou Wang. 2020. A multi-task learning framework for opinion triplet extraction. In Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020, volume EMNLP 2020 of Findings of ACL, pages 819–828. Association for Computational Linguistics.

Mi Zhang and Tieyun Qian. 2020. Convolution over hierarchical syntactic and lexical graphs for aspect level sentiment analysis. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 3540–3549, Online. Association for Computational Linguistics.

Wenxuan Zhang, Yang Deng, Xin Li, Yifei Yuan, Lidong Bing, and Wai Lam. 2021a. Aspect sentiment quad prediction as paraphrase generation. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7- 11 November, 2021, pages 9209–9219. Association for Computational Linguistics.

Wenxuan Zhang, Xin Li, Yang Deng, Lidong Bing, and Wai Lam. 2021b. Towards generative aspect-based sentiment analysis. In Proceedings of the 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 2: Short Papers), Virtual Event, August 1-6, 2021, pages 504–510. Association for Computational Linguistics.

Wenxuan Zhang, Xin Li, Yang Deng, Lidong Bing, and Wai Lam. 2021c. Towards generative aspect-based sentiment analysis. In Proceedings of the 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 504–510, Online. Association for Computational Linguistics.

Yice Zhang, Yifan Yang, Yihui Li, Bin Liang, Shiwei Chen, Yixue Dang, Min Yang, and Ruifeng Xu. 2022a. Boundary-driven table-filling for aspect sentiment triplet extraction. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 6485– 6498. Association for Computational Linguistics.

Zhihao Zhang, Yuan Zuo, and Junjie Wu. 2022b. Aspect sentiment triplet extraction: A seq2seq approach with span copy enhanced dual decoder. IEEE ACM Trans. Audio Speech Lang. Process., 30:2729–2742.

He Zhao, Longtao Huang, Rong Zhang, Quan Lu, and Hui Xue. 2020. SpanMlt: A span-based multi-task

Table 7: Descriptions of the tagging scheme. It focuses on the aspect attribution of each word. The sequence tagging task is to classify each word to one of these tags.
<table><tr><td rowspan=1 colspan=1>tag</td><td rowspan=1 colspan=1>Meaning</td></tr><tr><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>not belong to aspect term or opinion term</td></tr><tr><td rowspan=1 colspan=1>A-POS</td><td rowspan=1 colspan=1>a part of aspect term with positive sentiment</td></tr><tr><td rowspan=1 colspan=1>A-NEG</td><td rowspan=1 colspan=1>a part of aspect term with negative sentiment</td></tr><tr><td rowspan=1 colspan=1>A-NEU</td><td rowspan=1 colspan=1>a part of aspect term with neural sentiment</td></tr><tr><td rowspan=1 colspan=1>O-POS</td><td rowspan=1 colspan=1>a part of opinion term with positive sentiment</td></tr><tr><td rowspan=1 colspan=1>O-NEG</td><td rowspan=1 colspan=1>a part of opinion term with negative sentiment</td></tr><tr><td rowspan=1 colspan=1>O-NEU</td><td rowspan=1 colspan=1>a part of opinion term with neural sentiment</td></tr></table>

learning framework for pair-wise aspect and opinion terms extraction. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 3239–3248, Online. Association for Computational Linguistics.

## A Additional details about the methodology

## A.1 Tagging Scheme

Specific details of the sequence tagging scheme and their explanations are presented in Table 7.

## A.2 Inference

Here’s an overview of the inference process:

1. Obtain the probabilities of each word in the generated sentence.

2. Conduct experiments on the development dataset to find a suitable probability threshold.

3. For each triplet in the generated result, check if all the words in the triplet have probabilities greater than the threshold. If $^ { \mathrm { s o , } }$ retain the triplet. Otherwise, proceed to the next step.

4. Compare the generated triplet with the triplet identified by the sequence tagging task. If the generated aspect is a subset of the sequence tagging aspect set, or vice versa, and the generated opinion is a subset of the sequence tagging opinion set, or vice versa, retain the triplet. Otherwise, discard the triplet.

The specific algorithm pseudocode is presented in algorithm 1.

## A.3 Other Attention

$$
\widetilde { \boldsymbol { a } } _ { t i } = \frac { e x p ( \boldsymbol { a } _ { t i } ) } { \sum _ { j = 1 } ^ { n } e x p ( \boldsymbol { a } _ { t j } ) }\tag{13}
$$

$$
\widetilde { \boldsymbol { a } } _ { t i } = \frac { e x p ( \widetilde { \boldsymbol { p } } _ { i } \cdot \boldsymbol { a } _ { t i } ) } { \sum _ { j = 1 } ^ { n } e x p ( \widetilde { \boldsymbol { p } } _ { j } \cdot \boldsymbol { a } _ { t j } ) }\tag{14}
$$

## B Additional Experiment

## B.1 case study

In this case study section, we compare our model with the Paraphrase model (Zhang et al., 2021a) to illustrate how our two components benefit the results. For the first example, the Paraphrase model fails to extract the triplet (barmenu, Disappointingly, NEG) because it is a relatively hard and implicit triplet. Additionally, thanks to the semantic alignment training of the decoder hidden state, our model can generate higher-quality results. Therefore, our model can extract this triplet successfully. In the second example, the Paraphrase model incorrectly extracts the triplet $( s t a f f )$ , supportive, NEG). However, during the inference stage, our model optimizes the generation results based on the sequence tagging output, resulting in the discarding of this incorrect triplet.

## B.2 Arithmetic Operations in Tag Attention

In the context of Tag Attention, we have explored several approaches to incorporating tagging probability into the cross-attention mechanism:

1. Multiplication before softmax: Multiply the attention scores by the probability weights and then apply softmax.

2. Multiplication is performed before softmax, but without adding 1 to $\widetilde { p _ { i } }$ . The attention foremula is given by Equation (14).

3. Softmax after multiplication: Apply softmax to the attention scores and then multiply them by the probability weights.

4. Addition: We directly add the probability information to the attention scores.

Through the evaluation of these various operations, our objective is to gain insights into their impact on the Tag Attention mechanism and their effectiveness in incorporating probability information. We conducted this experiment on a model without the "Inference" process because including the "Inference" process could potentially narrow down the performance gaps observed in these results. The results in Table 9 demonstrate that the first approach performs better. This is because it provides valuable information to the generation model while minimizing any disruptive effects on the original generation process. It can be regarded as a gentle process. The results of the second approach are worse compared to the first approach. One possible reason for this is that in the first approach, by adding 1 to the ${ \widetilde { p } } ,$ the attention is not esolely determined by the sequence tagging results. This helps mitigate the potential impact of tagging errors on the overall generation quality. Furthermore, we found that the performance of the last two arithmetic operations is worse.

Table 8: Case study. The ground truth represents the correct triplets. The aspect and opinion words of the same triplet are highlighted in the same color. The two examples in the table demonstrate how our model can avoid the errors made by the Paraphrase model.
<table><tr><td rowspan=1 colspan=1>Example</td><td rowspan=1 colspan=1>Ground Truth</td><td rowspan=1 colspan=1>Paraphrase (Zhang et al., 2021a)</td><td rowspan=1 colspan=1>Ours</td></tr><tr><td rowspan=1 colspan=1>Disappointingly, their wonderful Saketinihas been taken off the bar menu.</td><td rowspan=1 colspan=1>(Saketini,wonderful,POS)(bar menu,Disappointingly,NEG)</td><td rowspan=1 colspan=1>(Saketini,wonderful,POS)</td><td rowspan=1 colspan=1>(Saketini,wonderful,POS)(bar menu,Disappointingly,NEG)</td></tr><tr><td rowspan=1 colspan=1>Our waiter was friendly and it is a shamethat he didnt have a supportive staff to work with.</td><td rowspan=1 colspan=1>(waiter,friendly,POS)</td><td rowspan=1 colspan=1>(waiter,friendly,POS)(staff,supportive,NEG)</td><td rowspan=1 colspan=1>(waiter,friendly,POS)</td></tr></table>

Table 9: Result of different arithmetic operations
<table><tr><td></td><td>16res</td><td>15res</td><td>14lap</td><td>14res</td></tr><tr><td>Multiplication before softmax</td><td>76.61</td><td>67.90</td><td>64.53</td><td>75.05</td></tr><tr><td>Multiplication before softmax without adding 1  $\mathbf { t o } \widetilde { p }$ </td><td>75.81</td><td>66.71</td><td>63.23</td><td>74.09</td></tr><tr><td>Softmax after multiplication</td><td>73.92</td><td>65.05</td><td>61.78</td><td>72.38</td></tr><tr><td>Addition</td><td>73.06</td><td>64.58</td><td>61.43</td><td>72.12</td></tr></table>

## B.3 Loss Function for Semantic Alignment

We discuss loss function for semantic alignment in our approach. Specifically, we compare two different approaches:

1. Confining the similarity scores to the range of 0 to 1 and utilizing the Binary Cross Entropy (BCE) loss function.

2. Preserving the cosine similarity scores in the range of -1 to 1 and employing the margin ranking loss function to constrain the similarity.

The results in Table 10 indicate that in our method, the BCE loss function outperforms the margin rank loss function. From this, we can conclude that it is not necessary to push the similarity of incorrect hidden states to -1, i.e., there is no need to excessively move away from the negative hidden states associated with incorrect words. Since the hidden states are generated from the label sentences, even if some negative hidden states are incorrect, they remain relatively close to the correct hidden states. Moving too far away from negative hidden states may lead to an increase in the distance from the correct hidden state.

Table 10: Result of different loss function
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>16res</td><td rowspan=1 colspan=1>15res</td><td rowspan=1 colspan=1>14lap</td><td rowspan=1 colspan=1>14res</td></tr><tr><td rowspan=1 colspan=1>MarginRankLoss</td><td rowspan=1 colspan=1>75.55</td><td rowspan=1 colspan=1>67.12</td><td rowspan=1 colspan=1>64.19</td><td rowspan=1 colspan=1>73.91</td></tr><tr><td rowspan=1 colspan=1>BCELoss</td><td rowspan=1 colspan=1>76.61</td><td rowspan=1 colspan=1>67.90</td><td rowspan=1 colspan=1>64.53</td><td rowspan=1 colspan=1>75.05</td></tr></table>

## B.4 ABSA subtask Detail

The subtasks are described as follows:

1. Aspect Opinion Pair Extraction (AOPE) aims to extract aspect terms and their corresponding opinion terms as pairs (Zhang and Qian, 2020; Chen et al., 2020).

2. Unified ABSA (UABSA) is the task of extracting aspect terms and predicting their sentiment polarities at the same time (Li et al., 2019a; Chen and Qian, 2020). We also formulate it as an (aspect, sentiment polarity) pair extraction problem

For these tasks, we adopt the dataset used in (Zhang et al., 2021b).

For AOPE task, we compare our model with the following models: a multi-task learning model SpanMlt (Zhao et al., 2020), a synchronous double channel extraction model SDRN (Chen et al., 2020), HAST+TOWE and JERE-MHS model compared in (Zhang et al., 2021b), GAS (Zhang et al., 2021b), LEGO(Gao et al., 2022) and EHG(Lv et al., 2023).

For the UABSA task, we compare our model with the following models: a BERT base model BERT+GRU (Li et al., 2019b), a span-base extraction model SPAN-BERT (Hu et al., 2019), an interactive multi-task learning network LMN-BERT (He et al., 2019), a Relation-Aware Collaborative Learning (RACL) model RACL (Chen and Qian, 2020), a machine reading comprehension models Dual-MRC (Mao et al., 2021) , GAS (Zhang et al., 2021b) and EHG(Lv et al., 2023).

## B.5 Analysis on Potential Practical Applications

Time Complexity: The time complexity of the TAGS model is quadratic relative to the input data. The primary source of complexity in this quadratic time complexity is the attention operations within the transformer. It’s important to note that the additional modules introduced in our model, such as the sequence tagging classification layer and the label-driven semantic alignment module, have a linear time complexity relative to the input data. Consequently, the time complexity introduced by our additional modules remains exceedingly modest compared to that of the transformer. As such, the primary temporal overhead in our model stems from the transformer’s attention operations. Consequently, the complexity of the TAGS model closely aligns with the time complexity of baseline models that rely on transformers. Moreover, existing lightweight and acceleration-oriented designs based on the transformer can be readily assimilated into our model. Hence, although our model does introduce some additional time overhead, it does not impose a significant obstacle to the training process.

Space Complexity: Apart from the core model architecture and input data, the additional space utilization of the TAGS model primarily consists of a linear layer for sequence tagging classification and the semantic representation of label sentences. The additional space occupation amounts to 5.2 M, which is notably minor when compared to the parameter size of the T5 model, standing at 222 M. Additionally, the tag attention module does not introduce any additional parameters.

Based on the aforementioned explanation, it’s evident that the TAGS model demonstrates commendable scalability. As dataset volumes increase, the incremental rise in both time and space overheads within our model remains consistent.

Input: Generated triplets $T _ { 1 } ^ { \prime } = \{ ( a _ { i } , o _ { i } , s _ { i } ) _ { k } \} _ { k = 1 } ^ { | T _ { 1 } ^ { \prime } | } ,$   
Word probabilities $G { = } \{ g _ { k } = ( g _ { k 1 } , g _ { k 2 } , g _ { k 3 } ) _ { k } \} _ { k = 1 } ^ { | I _ { 1 } | } ,$   
Tagging Aspect Set $S _ { a s p e c t } = \left\{ a _ { k } \right\} _ { k = 1 } ^ { | S _ { a s p e c t } | }$   
Tagging Opinion Set $S _ { o p i n i o n } = \left\{ a _ { k } \right\} _ { k = 1 } ^ { | S _ { o p i n i o n } | }$   
Threshold threshold.   
Output: Result triplets   
Function Verify(gen\_element, sequenceTag):   
foreach tag\_element in sequenceTag do   
if gen\_element is a part oftag\_element OR tag\_element is a part ofgen\_element then   
return true   
end   
end   
return false   
begin   
Result $ \emptyset$   
foreach (triplet, wordProb) in $( T _ { 1 } ^ { \prime } , G )$ do   
if (wordProb threshold).all() then   
Result Result  triplet end   
else   
aspect triplet.aspect   
Averified Verify(aspect, $S _ { a s p e c t } )$   
opinion triplet.opinion   
Overified Verify(opinion, $ { S _ { o p i n i o n } } )$   
if Averified AND Overified then   
Result Result triplet   
end   
end   
end   
end   
return Result   
Algorithm 1: Inference