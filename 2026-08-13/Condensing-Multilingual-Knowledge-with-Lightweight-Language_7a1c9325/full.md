# Condensing Multilingual Knowledge with Lightweight Language-Specific Modules

Haoran Xu, Weiting Tan\*, Shuyue Stella Li\*, Yunmo Chen\*, Benjamin Van Durme, Philipp Koehn, Kenton Murray

Johns Hopkins University {hxu64,wtan12,sli136,yunmo,phi,kenton}@jhu.edu

## Abstract

Incorporating language-specific (LS) modules or Mixture-of-Experts (MoE) are proven methods to boost performance in multilingual model performance, but the scalability of these approaches to hundreds of languages or experts tends to be hard to manage. We present Language-specific Matrix Synthesis (LMS), a novel method that addresses the issue. LMS utilizes parameter-efficient and lightweight modules, reducing the number of parameters while outperforming existing methods, e.g., +1.73 BLEU over Switch Transformer on OPUS-100 multilingual translation. Additionally, we introduce Fuse Distillation (FD) to condense multilingual knowledge from multiple LS modules into a single shared module, improving model inference and storage efficiency. Our approach demonstrates superior scalability and performance compared to state-of-the-art methods.<sup>1</sup>

## 1 Introduction

Multilingual models confer the benefit of facilitating cross-lingual learning; however, they also grapple with the issue of language interference (Conneau et al., 2020; Wang et al., 2020a; Shaham et al., 2022). Recent studies aim to alleviate negative language interference through the introduction of language-specific (LS) modules (Zhang et al., 2020; Fan et al., 2020; Zhang et al., 2021; Fan et al., 2021; Pires et al., 2023). In this setup, each language batch is processed through its designated module rather than a shared module. Although this approach is promising and barely inflates the number of FLOPs like Mixture-of-Experts (MoE) (Shazeer et al., 2017; Lepikhin et al., 2021),<sup>2</sup> the number of parameters becomes difficult to manage and sometimes impractical when working with a large variety of languages. This is because the fundamental element forming LS or MoE modules is typically the full-rank weight matrix derived from a densely connected layer, which causes a rapid increase in the number of parameters with a large number of languages or experts.<sup>3</sup>

![](images/cc11e4d9e098f160b0a4a6a9284d65ccbe03adf83cb902e0064a97640642f2ad.jpg)  
Figure 1: We show the BLEU gains between the LMS method and the Switch Transformer as the model’s parameters increase in our multilingual translation ablation study. The LMS method notably outperforms the Switch Transformer with similar extra LS (expert) parameter counts, achieving comparable performance even with four to five times fewer parameters.

In this paper, we first scrutinize the parameter efficiency of language-specific modules from the perspective of using fewer parameters. Consequently, a necessary question arises (RQ1): can we approximate the original dense weight matrix using substantially fewer parameters? To answer this question, we propose novel and parameter-efficient method, Language-Specific

Matrix Synthesis (LMS), which can achieve similar performance to switch transformer even with three to four times smaller LS parameters (as shown in Figure 1).

Then, we further investigate parameter efficiency from the perspective of knowledge density in each LS module. Given recent discoveries that the performance improvement of sparsely activated models diminishes with an increase in the number of experts (Hoffmann et al., 2022; Gao et al., 2022; Xu et al., 2023), we hypothesize that knowledge in these experts (or LS modules) is over-estimated. Hence, we propose another question (RQ2): Could a single shared module encapsulate the same level of knowledge as language-specific modules? In addressing this question, we introduce the Fuse Distillation (FD) method to examine the feasibility of condensing the multilingual knowledge into a single module.

Our main contributions are summarized as follows:

• We propose the parameter-efficient and lightweight LMS method, which substantially outperforms previous LS methods or MoE with fewer than or the same number of parameters, e.g., +1.73 BLEU over Switch Transformer on OPUS-100 multilingual translation.

• We introduce FD to condense multilingual knowledge from LS modules into a shared module. FD is able to use only 2M more parameters (1% increase) to achieve the 65% of performance gains from Switch Transformer which use 760M more parameters (314% increase) during inference.

• LMS and FD show strong generalization performance among multiple tasks, including multilingual machine translation (MMT) (Zhang et al., 2020), multilingual named-entity recognition (MNER) (Pan et al., 2017), and multilingual question answering (MQA) (Artetxe et al., 2020).

## 2 Lightweight LS Modules

In this section, we address RQ1 by constructing LS modules with significantly fewer parameters.

## 2.1 Language-Specific Matrix Synthesis

Language-specific modules are typically composed of linear projections, whose weights are fullrank matrices in previous studies. We propose the Language-specific Matrix Synthesis (LMS) method to form low-rank matrices to approximate the full-rank ones. This is inspired by the concept of “intrinsic dimension” in pre-trained language models (Aghajanyan et al., 2021; Hu et al., 2021) and “intrinsic rank” in trainable matrices, leading to the idea that features are learned in a subspace. Specifically, as shown in Figure 2, our LS matrix is derived from the multiplication of an LS ‘vertical’ matrix with an $\mathrm { L S \ ^ { \cdot } f l a t ^ { \cdot } }$ matrix. Formally speaking, let $W \in \mathbb { R } ^ { r \times c }$ be a weight matrix in the model and we want to build parallel LS matrices which have the same size. Hence, for each language $l _ { i } , ~ i ~ \in ~ \{ 1 , 2 , \cdot \cdot \cdot , L \}$ with L being the number of languages, there exists an LS vertical matrix $W _ { v } ^ { l _ { i } } \in \mathbb { R } ^ { r \times d }$ and an LS flat matrix $W _ { f } ^ { l _ { i } } \in \mathbb { R } ^ { d \times c }$ $( d \ll \operatorname* { m i n } ( r , c ) )$ that we use to approximate the full-rank matrix. Here, we propose two synthesis methods: language-wise and pair-wise synthesis.

![](images/3050046d9726cb44dc156104c28c34bdb33b7d0745fba57edce80f73b5cd4325.jpg)  
Figure 2: The difference between pair- and languagewise synthesis. Language-wise synthesis constructs a low-rank matrix using both the vertical and flat matrices derived from the same language. Conversely, pairwise synthesis formulates the matrix by combining the vertical matrix from the source language with the flat matrix from the target language.

Language-Wise Synthesis Most multilingual tasks, such as conventional multilingual questionanswering, are characterized by a languagemonolithic nature: a single example only pertains to a single language, and examples from different languages build the multilingual data. Under such circumstances, a naive way to assemble a language-specific matrix for a given language, $l _ { i } ,$ is straightforwardly using its corresponding vertical and flat matrices, such that $W ^ { l _ { i } } = W _ { v } ^ { l _ { i } } W _ { f } ^ { l _ { i } }$

Pair-Wise Synthesis Cross-lingual tasks like MMT can also be accomplished using languagewise synthesis, wherein the encoder uses the source language matrix and the decoder uses the target language matrix. However, we posit that this is not the optimal strategy for MMT tasks due to the lack of learning bilingual information. Motivated by this, we introduce a pair-wise synthesis method to accommodate the bilingual context in each example in MMT. In this strategy, the language-specific matrix is a composition of the vertical matrix from the source language $l _ { i }$ and the flat matrix from the target language $l _ { j } \colon W ^ { l _ { i }  l _ { j } } = W _ { v } ^ { l _ { i } } W _ { f } ^ { l _ { j } }$ . The difference between the language-wise and pairwise synthesis approaches is depicted in Figure 2. In Section 5, we will demonstrate that the pair-wise synthesis approach is more effective.

After deriving a language-specific matrix, we incorporate it into the original full-rank matrix, as opposed to performing an isolated forward pass of the model like MoE and conventional LS methods. This approach stems from our hypothesis that the employment of low-rank matrices alone may not sufficiently facilitate the learning of features. Therefore, given an input $x _ { i }$ associated with a source language $l _ { i }$ and a target language ${ { l } _ { j } } \left( { { l } _ { i } } \right.$ and $l _ { j }$ are the same for language-monolithic tasks), our modified forward pass yields the output $x _ { o } \colon$

$$
\boldsymbol { x _ { o } } = ( W + W ^ { l _ { i }  l _ { j } } ) \boldsymbol { x _ { i } } = ( W + W _ { v } ^ { l _ { i } } W _ { f } ^ { l _ { j } } ) \boldsymbol { x _ { i } } .\tag{1}
$$

## 2.2 Where to Implement?

We primarily focus on incorporating languagespecific matrices generated using the LMS method into the linear projection of each feedforward network (FFN) layer in every transformer layer. Recall from earlier that r and c are the number of rows and columns in the matrix, and $L$ is the number of languages. Thus, the total number of language-specific parameters added is given by $2 L \cdot N \cdot d \cdot ( c + r )$ , where N represents the number of layers. We also conduct an ablation study to examine the performance when implementing LMS in attention layers in Section 6. For initialization, we employ a random Gaussian distribution for vertical matrices and zeros for flat matrices suggested by Hu et al. (2021).

## 3 Can We Fuse Multilingual Knowledge in A Single Module?

In this section, we introduce Fuse Distillation (FD) and use a preliminary experiment to answer RQ2: whether we can condense the multilingual knowledge from language-specific modules into a single module.

## 3.1 Fuse Distillation

Let us first consider a language- (or task-) level MoE (Kudugunta et al., 2021), where we replace a single FFN layer with L FFN modules. L is the number of languages, as defined previously. The slight difference from the original design is we discard the routing gate and make each expert language-specific, i.e., an expert only serves batches in its corresponding language. Given recent findings that model improvements diminish with an increasing number of experts (Hoffmann et al., 2022; Gao et al., 2022; Xu et al., 2023), we hypothesize that information contained in experts is sparse and can be condensed into a shared module. To fuse knowledge from L FFN layers to the shared one, we propose the following training scheme and name this method Fuse Distillation:

We first add an additional shared FFN parallel to an existing model with L FFN layers as shown in Figure 3. During training, each batch undergoes two forward passes and one backward pass. In the first forward pass, the batch is processed through its language-specific FFN module; in the second pass, the batch is routed through the shared FFN. To fuse the language-specific knowledge contained within the $L$ FFN modules into the shared FFN module, a distillation loss between the outputs from the two forward passes is also incorporated:

$$
{ \mathcal { L } } _ { f d } = \mathbb { K L } ( g ( p _ { l } ) \parallel p _ { s } ) .\tag{2}
$$

where $p _ { l }$ denotes the probability output for the LS pass, and $p _ { s }$ represents the shared pass output. The function $g ( \cdot )$ signifies that gradients will not be traced back, so only the shared module learns from LS modules but LS ones do not learn from this loss. The backward pass also involves optimizing the model by minimizing the Cross-Entropy loss (CE) between the target and predicted values (the regular training loss). Thus, the total loss is:

$$
\mathcal { L } = \frac { 1 } { 2 } ( \mathbb { C E } ( y \parallel p _ { l } ) + \mathbb { C E } ( y \parallel p _ { s } ) ) + \mathcal { L } _ { f d } ,\tag{3}
$$

where $y$ denotes gold labels.

Then, during the inference stage, we discard the LS modules. The model only forward passes the shared FFN for inference. To evaluate whether the shared FFN has effectively learned all LS information, we conduct a comparison between its results and those obtained via the routing through LS modules instead.

## 3.2 Preliminary Experiments

Our preliminary experiments are conducted under three settings:

(1) Naive MMT: A basic multilingual translation model is trained without any modifications.

(2) FD: This setting utilizes our proposed fuse distillation method.

(3) FD-LS: We train the model with the FD method, but during the inference stage, the input is processed through its language-specific FFN module instead of the shared module as the original language-level MoE did.

We carry out our experiments using the IWSLT benchmarks, focusing on the many-to-many translation model paradigm. Following Lin et al. (2021); Xu et al. (2022), we collect 8 Englishcentric language pairs from the IWSLT’14 dataset, with sizes ranging from 89K to 169K sentences. We train all methods with the same number of steps and leave detailed training settings in Appendix A. We report sacreBLEU scores (Papineni et al., 2002; Post, 2018) with the FLORES-200 tokenizer (NLLB Team et al., 2022).

## 3.3 Results and Analysis

Overview results of these 4 settings are shown in Table 1. The reported scores are the average of both xx en and en xx directions. As anticipated, after applying language-specific modules for each FFN layer, FD-LS has considerable enhancements over the naive MMT (+1.50 BLEU gains). Importantly, after discarding LS modules, FD only performs slightly worse than FD-LS (+1.17 vs. +1.50) with much fewer parameters for inference (48M vs. 149M). This observation underscores the feasibility of condensing multilingual knowledge into a single FFN module, thereby reducing the need of a large number of LS parameters for inference.

## 4 Combining LMS and FD

We have shown the success of multilingual information condensation by fuse distillation. We are interested in further reducing the parameters needed by utilizing the language-specific matrix synthesis method during inference, so we then attempt to incorporate the FD method within LMS. Similar to Section 3.1, apart from the LS vertical and flat matrices, we introduce shared vertical and flat matrices, denoted as W<sup>shared</sup> v and $W _ { f } ^ { \mathrm { s h a r e d } }$ , respectively. To employ the fuse distillation method, each batch is required to undergo two forward passes. The initial pass navigates through the LS matrix $W + W _ { v } ^ { l _ { i } } W _ { f } ^ { l _ { j } }$ while the subsequent pass traverses the shared matrix $W + W _ { v } ^ { \mathrm { s h a r e d } } W _ { f } ^ { \mathrm { s h a r e d } }$ These two passes generate two respective outputs, $p _ { l }$ and $p _ { s }$ . Given the common parameter W shared across both paths, we utilize symmetric KL divergence (Jiang et al., 2020) for distillation, as opposed to the traditional KL divergence:

![](images/5fceb28c31ac9038bd23c47c8431d575f91ab257e7450311f70adb04b0378c78.jpg)  
Figure 3: We utilize a language-level MoE architecture to verify the feasibility of fusing multilingual knowledge from all language-specific modules into a single shared module. During training, each batch goes through the LS module in the first forward pass and goes through the shared module in the second pass. Then, we conduct distillation between two outputs to condense the knowledge into the shared module. For inference, we discard the LS module and only use the shared module.

$$
\mathcal { L } _ { f d } ^ { \prime } = \frac { 1 } { 2 } ( \mathbb { K L } ( p _ { l } \parallel p _ { s } ) + \mathbb { K L } ( p _ { s } \parallel p _ { l } ) ) .\tag{4}
$$

Thus, the backward pass optimizes both the standard prediction loss and the fuse distillation

<table><tr><td rowspan="2">Methods</td><td rowspan="2">ar</td><td rowspan="2">de</td><td rowspan="2">es</td><td rowspan="2">fa</td><td rowspan="2">he</td><td rowspan="2">it</td><td rowspan="2">nl</td><td rowspan="2">pl</td><td rowspan="2">avg.</td><td colspan="2">#params</td></tr><tr><td>Training</td><td>Inference</td></tr><tr><td>Naive MMT</td><td>25.03</td><td>32.59</td><td>39.98</td><td>18.76</td><td>33.39</td><td>34.00</td><td>36.71</td><td>22.37</td><td>30.35</td><td>48M</td><td>48M</td></tr><tr><td>FD</td><td>+1.01</td><td>+1.15</td><td>+1.43</td><td>+0.64</td><td>+1.44</td><td>+1.19</td><td>+1.22</td><td>+1.22</td><td>+1.17</td><td>161M</td><td>48M</td></tr><tr><td>FD-LS</td><td>+1.30</td><td>+1.45</td><td>+1.72</td><td>+0.77</td><td>+2.08</td><td>+1.48</td><td>+1.41</td><td>+1.73</td><td>+1.50</td><td>161M</td><td>149M</td></tr></table>

Table 1: Average BLEU on IWSLT’14 many-to-many translation. Our proposed FD is able to fuse the majority of knowledge into a single module (+1.17 vs. +1.50) with the same parameters as the naive model during inference.

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Illustration</td><td rowspan=1 colspan=2>Complexity for Training</td><td rowspan=1 colspan=2>Complexity for Inference</td><td rowspan=6 colspan=1>= LS full-rank matrix= shared full-rank matrix_     Small matrices forlow-rank matrix</td></tr><tr><td rowspan=1 colspan=1>Conventional LS Module</td><td rowspan=1 colspan=1>L×</td><td rowspan=1 colspan=1>O(L·r·c)     62.9</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1> $\mathcal { O } ( L \cdot r \cdot c )$ </td><td rowspan=1 colspan=1>62.9M</td></tr><tr><td rowspan=1 colspan=1>Mixture-of-Experts</td><td rowspan=1 colspan=1>E×</td><td rowspan=1 colspan=1>O(E·r·c)    33.6</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>O(E·r · c)</td><td rowspan=1 colspan=1>33.6M</td></tr><tr><td rowspan=1 colspan=1>LMS (ours)</td><td rowspan=1 colspan=1>L×( +</td><td rowspan=1 colspan=1>O(L · d · (r + c))</td><td rowspan=1 colspan=1>2.5M</td><td rowspan=1 colspan=1>0(L · d · (r + c))</td><td rowspan=1 colspan=1>2.5M</td></tr><tr><td rowspan=1 colspan=1>Conventional LS Module+ FD (ours)</td><td rowspan=1 colspan=1>L×      +</td><td rowspan=1 colspan=1>O((L + 1) · r · c)  67.1</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>O(r·c)</td><td rowspan=1 colspan=1>4.2M</td></tr><tr><td rowspan=1 colspan=1>LMS + FD (ours)</td><td rowspan=1 colspan=1>L×( 十+    </td><td rowspan=1 colspan=1>O((L + 1) · d · (r + c))</td><td rowspan=1 colspan=1>; 2.6M</td><td rowspan=1 colspan=1> $\mathcal { O } ( d \cdot ( r + c ) )$ </td><td rowspan=1 colspan=1>0.2M</td></tr></table>

Figure 4: Suppose we incorporate additional language-specific (LS) linear projections into a layer. We compare the space complexity of the extra LS parameters (or experts) needed across all methods for both training and inference phases. Let’s denote L = 15 as the number of languages, r = 4096 as the output dimension, c = 1024 as the input dimension, E = 8 represents the number of experts for Mixture-of-Experts (MoE), and d = 32 signifies the rank for low-rank matrices. The number adjacent to the dashed line is the number of parameters calculated based on the given sample numbers. In this case, one can observe that the Language-Specific Matrix Synthesis (LMS) requires a significantly lower quantity of LS parameters compared to other methods during training, and fuse distillation (FD) demands a substantially reduced number of additional parameters during the inference stage.

loss.

In Figure 4, we provide a comprehensive comparison of space complexity for generating extra LS (or expert) modules, among conventional LS modules, Mixture-of-Experts, and our proposed methods. Notably, our methods demonstrate substantial reductions in parameter usage during both training and inference.

## 5 Experiments

We evaluate our LMS and LMS+FD methods using three tasks: MMT, MNER, and MQA. Similar to Section 3.2, we have two routing options for the LMS+FD method during inference time: 1) evaluating the model by passing the shared route (denoted as LMS+FD-Share, the default setting), or 2) passing the language-specific module (denoted as LMS+FD-LS). We present results for both routes to show the performance difference between using the condensed module and the original LS modules. Considering the computational cost for MMT, we run all methods once with the same random seed. For the other two tasks, we run experiments with 3 different random seeds and report the average scores. For ease of implementation, we build homogeneous batches (i.e., a batch only containing sentences in one language or one language direction) and only activate the corresponding LS module.<sup>4</sup>

## 5.1 Baselines

We compare our approaches against two strong baselines that incorporate additional parameters to mitigate language interference.

CLSR: The first baseline is Conditional Language-Specific Routing (CLSR) (Zhang et al., 2021), which employs LS linear projections following FFN or attention layer. Following their best settings, we set the budget p = 0.3 for LS routing. The original setting used shared LS projections across all encoder or decoder sublayers. We also consider a non-shared version, where each sublayer has its own LS projection, and denote it as CLSR\*.

Switch Transformer: We also consider Switch Transformer (Fedus et al., 2021) as the second strong baseline, which uses similar FLOPs as our methods.<sup>5</sup> We use 16 experts for every two layers with a gate balance loss with a weight of 0.01.

<table><tr><td rowspan="2">Methods</td><td rowspan="2">ar</td><td rowspan="2">de</td><td rowspan="2">es</td><td rowspan="2">fa</td><td rowspan="2">he</td><td rowspan="2">it</td><td rowspan="2">nl</td><td rowspan="2"> $\mathsf { p l }$ </td><td rowspan="2"> $\mathrm { a v g . }$ </td><td colspan="2">#params</td></tr><tr><td>Training</td><td>Inference</td></tr><tr><td>Naive MMT</td><td>25.03</td><td>32.59</td><td>39.98</td><td>18.76</td><td>33.39</td><td>34.00</td><td>36.71</td><td>22.37</td><td>30.35</td><td>48M</td><td>48M</td></tr><tr><td>Switch Transformer</td><td>+0.28</td><td>+0.40</td><td>+0.45</td><td>+0.04</td><td>+0.60</td><td>+0.59</td><td>+0.34</td><td>+0.67</td><td>+0.42</td><td>149M</td><td>149M</td></tr><tr><td>CLSR</td><td>+0.00</td><td>+0.48</td><td>+0.51</td><td>-0.23</td><td>+0.31</td><td>+0.50</td><td>+0.42</td><td>+0.30</td><td>+0.28</td><td>53M</td><td>53M</td></tr><tr><td>CLSR*</td><td>+0.66</td><td>+0.87</td><td>+1.16</td><td>+0.53</td><td>+0.99</td><td>+1.00</td><td>+0.87</td><td>+0.94</td><td>+0.88</td><td>105M</td><td>105M</td></tr><tr><td>LMS, lang-wise</td><td>+0.48</td><td>+0.53</td><td>+0.88</td><td>+0.83</td><td>+0.86</td><td>+0.91</td><td>+0.81</td><td>+0.91</td><td>+0.78</td><td>58M</td><td>58M</td></tr><tr><td>LMS</td><td>+0.87</td><td>+1.08</td><td>+1.04</td><td>+0.62</td><td>+1.37</td><td>+1.20</td><td>+1.04</td><td>+1.16</td><td>+1.05</td><td>58M</td><td>58M</td></tr><tr><td>LMS+FD-Share</td><td>+0.82</td><td>+0.93</td><td>+1.06</td><td>+0.34</td><td>+1.23</td><td>+0.92</td><td>+0.87</td><td>+0.83</td><td>+0.88</td><td>60M</td><td>49M</td></tr><tr><td>LMS+FD-LS</td><td>+1.23</td><td>+1.34</td><td>+1.44</td><td>+0.77</td><td>+1.51</td><td>+1.36</td><td>+1.24</td><td>+1.15</td><td>+1.26</td><td>60M</td><td>58M</td></tr></table>

Table 2: Overall BLEU results of on IWSLT’14 many-to-many translation. LMS outperforms all baselines. At inference, LMS+FD-Share utilizes extra 1M parameters to exceed baselines that enlarge the model size 2 or 3 times.
<table><tr><td rowspan="2">Methods</td><td colspan="5">en→xx</td><td colspan="5">xx→en</td><td colspan="2">#params</td></tr><tr><td>high</td><td>med</td><td>low</td><td>all</td><td>WR (%)</td><td>high</td><td>med</td><td>low</td><td>all</td><td>WR (%)</td><td>Training</td><td>Inference</td></tr><tr><td>Naive MMT</td><td>23.89</td><td>31.17</td><td>29.76</td><td>27.37</td><td></td><td>29.40</td><td>31.85</td><td>31.49</td><td>30.60</td><td></td><td>242M</td><td>242M</td></tr><tr><td>Switch Transformer</td><td>+1.87</td><td>+3.29</td><td>+3.51</td><td>+2.66</td><td>100</td><td>+1.18</td><td>+1.15</td><td>-0.31</td><td>+0.84</td><td>83</td><td>1002M</td><td>1002M</td></tr><tr><td>CLSR</td><td>+0.02</td><td>+0.00</td><td>+0.01</td><td>+0.02</td><td>52</td><td>+1.33</td><td>+2.00</td><td>+2.71</td><td>+1.83</td><td>91</td><td>443M</td><td>443M</td></tr><tr><td>LMS, lang-wise, d = 64</td><td>+2.12</td><td>+2.28</td><td>+1.77</td><td>+2.09</td><td>95</td><td>+1.85</td><td>+2.34</td><td>+2.30</td><td>+2.09</td><td>94</td><td>989M</td><td>989M</td></tr><tr><td>LMS, d = 64</td><td>+3.60</td><td>+3.82</td><td>+3.32</td><td>+3.60</td><td>99</td><td>+2.75</td><td>+3.74</td><td>+4.16</td><td>+3.35</td><td>95</td><td>989M</td><td>989M</td></tr><tr><td>LMS+FD-Share, d = 64</td><td>+0.49</td><td>+0.75</td><td>+1.29</td><td>+0.74</td><td>88</td><td>+0.64</td><td>+1.52</td><td>+2.08</td><td>+1.22</td><td>98</td><td>996M</td><td>250M</td></tr><tr><td>LMS+FD-LS, d = 64</td><td>+1.72</td><td>+2.03</td><td>+2.60</td><td>+2.01</td><td>100</td><td>+1.64</td><td>+2.82</td><td>+4.03</td><td>+2.52</td><td>99</td><td>996M</td><td>996M</td></tr><tr><td>LMS, d = 16</td><td>+2.45</td><td>+2.62</td><td>+2.56</td><td>+2.53</td><td>99</td><td>+1.75</td><td>+2.68</td><td>+3.40</td><td>+2.39</td><td>96</td><td>429M</td><td>429M</td></tr><tr><td>LMS+FD-Share, d = 16</td><td>+0.54</td><td>+1.13</td><td>+2.20</td><td>+1.09</td><td>94</td><td>+0.81</td><td>+1.26</td><td>+1.85</td><td>+1.17</td><td>94</td><td>431M</td><td>244M</td></tr><tr><td>LMS+FD-LS, d = 16</td><td>+1.28</td><td>+1.84</td><td>+2.74</td><td>+1.77</td><td>100</td><td>+1.35</td><td>+2.25</td><td>+3.53</td><td>+2.10</td><td>100</td><td>431M</td><td>431M</td></tr><tr><td>LMS, d = 4</td><td>+1.72</td><td>+2.05</td><td>+2.31</td><td>+1.95</td><td>99</td><td>+1.33</td><td>+1.80</td><td>+1.71</td><td>+1.55</td><td>93</td><td>289M</td><td>289M</td></tr></table>

Table 3: BLEU scores on OPUS-100 many-to-many translation. LMS with d = 64 outperforms all baselines on average. LMS+FD-Share with d = 16 uses 1% more parameters, and achieves 65% BLEU gains averaged by all directions, compared to the Switch Transformer which uses 314% more parameters.

## 5.2 Multilingual Machine Translation

Data and Training settings We concentrate on the many-to-many translation setting, with results reported from two benchmarks. The first is the English-centric IWSLT’14 dataset, as aforementioned in Section 3.2. Additionally, we examine the OPUS-100 dataset (Zhang et al., 2020), which encompasses 100 languages in total, including 94 development/test language pairs. We preprocess the data by sentencepiece (Kudo and Richardson, 2018), establishing a vocabulary size of 32K for the IWSLT’14 dataset and 64K for the OPUS-100 dataset. We utilize transformer<sub>small</sub> and $\mathrm { t r a n s f o r m e r _ { b i g } }$ for IWSLT’14 and OPUS-100, respectively. We fix the training steps for all methods for a fair comparison. For IWSLT’14, we use d = 32 as the rank for low-rank matrices. For OPUS-100, we consider three settings: (i) d = 64 to match the parameter size of the Switch Transformer, (ii) d = 16 to match the parameter size of CLSR, and (iii) d = 4 for very lightweight LS model construction. The default LMS setting for MMT tasks is pair-wise unless otherwise specified. We discuss more training details in Appendix A.

Evaluation We report results in terms of sacreBLEU (Post, 2018), tokenized by FLORES-200 tokenizer (NLLB Team et al., 2022), and win ratio (WR) (Zhang et al., 2020) which is the proportion of language pairs on which our method beats the baseline. For IWSLT’14, we report the scores averaged by xx en and en xx directions. For OPUS-100, we split the 94 test language pairs into three groups based on their training data size suggested by Zhang et al. (2020): high-resource (> 0.9M, 45 languages), low-resource (< 0.1M, 21 languages) and medium-resource (others, 28 languages), and report the averaged scores in each category. We use beam search with a width of 5 and use a length penalty of 1.

LMS performance: Light and Effective LS Module The primary results for IWSLT’14 and OPUS-100 are presented in Table 2 and Table 3, respectively. In the IWSLT’14 dataset, LMS significantly surpasses both the Switch Transformer and CLSR, despite having considerably fewer parameters. For OPUS-100, our methods and the baselines are evaluated with approximately equal extra parameters (e.g., 1002M in the Switch Transformer and 989M in LMS with $d \ : = \ : 6 4 )$ Compared with the gains from Switch transformer (+2.66 for en xx and +0.84 for xx en), our pairwise LMS method achieves substantially higher gains (+3.60 and +3.35). Similarly, our LMS method also outperforms CLSR (+0.02 and +1.83) with a comparable number of extra parameters. These results show the strong parameter efficiency of LMS for the MMT tasks. With merely 47M parameters (d = 4), our LMS method matches the Switch Transformer’s performance for en xx and the CLSR’s performance for xx en.

Language-Wise or Pair-Wise? We compare language- and pair-wise synthesis in both IWSLT’14 and OPUS-100 (d = 64) datasets. On average, pair-wise synthesis outperforms languagewise synthesis by 0.27 BLEU points on IWSLT’14 (+1.05 vs. +0.78). Moreover, the pair-wise method (+3.60 and +3.35) also shows superior performance on the OPUS-100 dataset compared with the language-wise one (+2.09 and + 2.09). Notably, pair-wise synthesis with d = 16 surpassed the performance of language-wise synthesis with $d = 6 4$ , even though the latter has 4 times more extra parameters. Hence, this discovery strongly advocates for the use of pair-wise synthesis over the language-wise approach.

FD performance: Can FD Fuse 95 Languages? On the IWSLT’14 8-language MMT dataset, we observe negligible differences between LMS and LMS+FD (+1.05 vs. +0.88), suggesting successful condensation of information from various language-specific modules into the shared module. In the 95-language (94 languages plus English) scenario of OPUS-100, FD with a dimensionality of 16 utilizes only an additional 2M parameters (less than 1% increase compared to the 242M naive model) to attain 65% of the performance improvements from Switch Transformer (+1.13 vs. +1.75 on average), which requires 760M additional parameters (a 314% increase). While FD may not condense all multilingual information due to restricted parameter capacity, its parameter efficiency is commendable.

<table><tr><td rowspan="2">Methods</td><td colspan="2">Sampled Language</td><td rowspan="2"> ${ \mathrm { a v g . } }$ </td><td rowspan="2">WR (%)</td><td colspan="2">#params</td></tr><tr><td>qu</td><td>vi</td><td>Tra.</td><td>Inf.</td></tr><tr><td>Naive MNER</td><td>76.79</td><td>92.60</td><td>89.20</td><td>-</td><td>270M</td><td>270M</td></tr><tr><td>LMS</td><td>+3.61</td><td>+0.28</td><td>+0.55</td><td>96</td><td>340M</td><td>340M</td></tr><tr><td>LMS+FD-Share</td><td>+3.22</td><td>+0.45</td><td>+0.33</td><td>88</td><td>343M</td><td>273M</td></tr><tr><td>LMS+FD-LS</td><td>+3.96</td><td>+0.57</td><td>+0.67</td><td>100</td><td>343M</td><td>340M</td></tr></table>

Table 4: The overall MNER results (F1 score) between baseline and our three proposed methods.

## 5.3 Multilingual Named-Entity Recognition

Data and Settings We evaluate our methods on Wikiann Named-Entity Recognition (Pan et al., 2017) dataset. We randomly select 24 languages to conduct experiments. The model architecture is based on pre-trained $\mathbf { X L M - R _ { b a s e } }$ , attached with a feed-forward token-level classifier. We set the dropout rate as 0.1 and run 20 epochs for all methods. We set $d = 3 2$ for low-rank matrices and report F1 scores.

Results The overall results are shown in Table 4. When applying LMS to each FFN layer for 24 languages, the model size increases by only 70M, while yielding a 0.55 F1 improvement. After implementing LMS+FD, the performance improves by 0.67 with the LS route and achieves a 0.33 gain with the shared route, which requires only an additional 3M parameters. Full results are shown in Appendix B.

## 5.4 Multilingual Question Answering

Data and Settings We pick 6 languages from TyDiQA (Typologically Diverse Question Answering)-Gold Passage to conduct the MQA experiments (Artetxe et al., 2020). Following Xu and Murray (2022), the representations of subwords in $\mathrm { X L M - R _ { b a s e } }$ are input to a span classification head; a linear layer computing the answer’s start and end. We set $d = 3 2$ for low-rank matrices, dropout rate = 0.1, and run 20 epochs.

Results The overall results are shown in Table 5. Upon the application of LMS and LMS+FD, all methods exhibit improved performance with a slight increase in parameters. Notably, LMS+FD-Share outperforms LMS+FD-LS. This suggests that FD may be more effective in fusing knowledge when the number of languages is relatively small. Full results are shown in Appendix C.

<table><tr><td rowspan="2">Methods</td><td colspan="2">Sampled Language</td><td rowspan="2">avg.</td><td rowspan="2">WR (%)</td><td colspan="2">#params</td></tr><tr><td>bn</td><td>SW</td><td>Tra.</td><td>Inf.</td></tr><tr><td>Naive MQA</td><td>77.69</td><td>80.97</td><td>75.31</td><td>=</td><td>270M</td><td>270M</td></tr><tr><td>LMS</td><td>-0.59</td><td>+0.93</td><td>+0.58</td><td>50</td><td>287M</td><td>287M</td></tr><tr><td>LMS+FD-Share</td><td>+1.39</td><td>+0.32</td><td>+1.22</td><td>100</td><td>290M</td><td>273M</td></tr><tr><td>LMS+FD-LS</td><td>+1.26</td><td>+0.38</td><td>+1.15</td><td>100</td><td>290M</td><td>287M</td></tr></table>

Table 5: The overall MQA results (F1 score) between baseline and our three proposed methods.

## 6 Ablation Study

## 6.1 Is LMS Parameter-Efficient?

Here, we examine the parameter efficiency of the LMS method, i.e., whether an increase in extra parameters yields a proportional enhancement in model performance. We conduct experiments with d ranging from 4 to 60 in increments of 8 to observe the resulting performance variations. For comparison, we examine the Switch Transformer with 4, 8, 12, 16 experts to assess its parameter efficiency. We focus on the MMT task using the OPUS-100 dataset. Due to computational demands, we limit experiments to randomly selected 15 languages from OPUS-100, designated as OPUS-15. We leave training details in Appendix D.

We report the average BLEU gains over all translation directions in Figure 1. The plot reveals that the LMS curve is steeper compared to that of the Switch Transformer, indicating a higher parameter efficiency for our method, i.e., it achieves greater model performance with fewer additional parameters. Compared with a 16-expert Switch Transformer, LMS with d = 52 yields similar performance by using 3.7 times smaller parameters (51M vs. 189M). Numeric results are in Appendix E.

## 6.2 Applying LMS to The Attention Layer

In our default design, the LMS is solely applied to FFN layers. We are interested in assessing the potential benefits of extending LMS to the attention layer (in each K, Q, V, output projection). We consider three model variants: (1) LMS applied only to FFN layers (default design), (2) LMS applied only to the attention layers, and (3) LMS applied to both FFN and attention layers. We conduct experiments on OPUS-15, with a fixed rank value of d = 20.

We show the averaged BLEU of all translation directions of the three designs in Table 6. LMS applied only to attention layers yields inferior performance compared to LMS applied only to FFN layers with a similar number of extra parameters. Moreover, applying LMS to both FFN and attention layers results in a marginal improvement over its application solely to FFN layers. This outcome suggests that LS information is primarily situated in FFN layers, aligning with the previous findings of Wang et al. (2020b).

<table><tr><td>Methods</td><td>avg. BLEU</td><td>WR (%) #params</td><td></td></tr><tr><td>Naive MMT</td><td>28.05</td><td>一</td><td>61M</td></tr><tr><td>LMS, ffn only (default)</td><td>+2.10</td><td>100</td><td>80M</td></tr><tr><td>LMS, att only</td><td>+1.32</td><td>100</td><td>77M</td></tr><tr><td>LMS, att+ffn</td><td>+2.14</td><td>100</td><td>96M</td></tr></table>

Table 6: The average BLEU gains with three different LMS designs with a fixed rank d = 20.

## 7 Related Work

Language-Specific Modules To mitigate language interference, previous studies incorporate language-specific modules into models, such as additional language-aware linear projections (Zhang et al., 2020; Fan et al., 2020; Zhang et al., 2021; Fan et al., 2021), LS layer normalization (Zhang et al., 2020). Feed-Forward Networks (Kwon and Chung, 2023), or even entire languagedependent transformer layers (Escolano et al., 2021; Wang and Zhang, 2022; Pires et al., 2023). Similar to LS modules, Mixture-of-Experts (MoE) are also able to reduce language interference (Shazeer et al., 2017; Lepikhin et al., 2021; Fedus et al., 2021; Xu et al., 2023). However, the parameter count of LS (or expert) drastically increases when scaling to numerous languages. Zhang et al. (2021) address this issue by sharing all LS modules across all encoder or decoder layers. However, this does not fundamentally resolve the problem, given that the complexity of constructing LS modules remains unaltered and that different layers may need to learn varying types of LS information.

Lightweight Modules Our proposed techniques draw inspiration from another research line, lightweight fine-tuning, wherein the model undergoes fine-tuning on a parameter subset significantly smaller than that of the original model, such as prefix tuning (Li and Liang, 2021), prompt tuning (Lester et al., 2021), multitask prompt tuning (Wang et al., 2023), LoRA (Hu et al., 2021). In the multilingual machine translation setting, previous studies use language-pair adapters (Bapna and Firat, 2019) to fine-tune a specific direction. This approach also extends to languagewise adapters (Philip et al., 2020), languagefamily adapters (Chronopoulou et al., 2023), hyperadapters (Baziotis et al., 2022) to facilitate the cross-lingual learning. In light of the efficient lightweight modules, we propose LMS to help LS modules scale to hundreds of languages.

## 8 Conclusion

The construction of language-specific modules (or experts) using full-rank matrices tends to be parameter-intensive and inefficient, especially as the number of languages (or experts) increases. To address this, we have introduced the Language-Specific Matrix Synthesis (LMS) method that approximates the original full-rank matrix. Notably, pair-wise synthesis, a variant of the LMS methods, exhibits commendable performance in MMT tasks. Further, we have proposed the Fuse Distillation (FD) approach to condense multilingual information into a shared module, thereby further diminishing parameter requirements during inference. Our methods outperform CLSR and Switch Transformer in MMT tasks and also demonstrate their effectiveness in MNER and MQA tasks.

## Limitations

One limitation of our LMS method is that it necessitates the construction of homogeneous batches, i.e., batches containing sentences exclusively in one language or language direction. However, this limitation could potentially be addressed by implementing ALLToALL communications amongst devices, a strategy that is already widely employed in Mixture of Experts (MoE) models (Lepikhin et al., 2021), which is a topic we intend to explore in future research. In each forward pass of an FFN layer, we need an additional step to multiply two small matrices, creating the low-rank large matrix. The additional cost of this operation is negligible, as the computational complexity of the FLOPs/tok for a Feedforward linear projection, given an input dimension c and output dimension r, is $\mathcal { O } ( r \cdot c )$ while the complexity for constructing the low-rank matrix with rank d is $O ( d \cdot ( r + c ) )$ ). For example, in our ablation study, when $r = 2 0 4 8 , c = 5 1 2$ and $d \ = \ 2 0 ,$ the difference in computational load can be $\frac { 2 0 4 8 \times 5 1 2 } { 2 0 \times ( 5 1 2 + 2 0 4 8 ) } \approx 2 0$ times less. In terms of actual training time, no significant differences were observed; the discrepancy was less than 1 second per 100 updates. Additionally, a potentially effective strategy to enhance multilingual information encapsulation in FD could involve using a larger shared module relative to other lightweight LS modules. This could be an intriguing avenue for future research.

## Acknowledgements

We thank anonymous reviewers for their insightful feedback. We also extend our gratitude to Lingfeng Shen, Hieu Hoang, Young Jin Kim, Hany Hassan Awadalla, Stephen Rawls, and Amr Sharaf for their valuable suggestions. This work was supported in part by IARPA BETTER (#2019-19051600005). The views and conclusions contained in this work are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, or endorsements of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein. This work is also supported in part by an Amazon Initiative for Artificial Intelligence (AI2AI) Faculty Research Award.

## References

Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. 2021. Intrinsic dimensionality explains the effectiveness of language model finetuning. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 7319–7328.

Roee Aharoni, Melvin Johnson, and Orhan Firat. 2019. Massively multilingual neural machine translation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3874–3884, Minneapolis, Minnesota. Association for Computational Linguistics.

Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2020. On the Cross-lingual Transferability of Monolingual Representations. In Proceedings of ACL 2020.

Ankur Bapna and Orhan Firat. 2019. Simple, scalable adaptation for neural machine translation. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th

International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1538–1548, Hong Kong, China. Association for Computational Linguistics.

Christos Baziotis, Mikel Artetxe, James Cross, and Shruti Bhosale. 2022. Multilingual machine translation with hyper-adapters. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 1170–1185, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Alexandra Chronopoulou, Dario Stojanovski, and Alexander Fraser. 2023. Language-family adapters for low-resource multilingual neural machine translation. In Proceedings of the The Sixth Workshop on Technologiesfor Machine Translation of Low-Resource Languages (LoResMT 2023), pages 59–72, Dubrovnik, Croatia. Association for Computational Linguistics.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 8440–8451, Online. Association for Computational Linguistics.

Carlos Escolano, Marta R. Costa-jussà, José A. R. Fonollosa, and Mikel Artetxe. 2021. Multilingual machine translation: Closing the gap between shared and language-specific encoder-decoders. In Proceedings ofthe 16th Conference ofthe European Chapter of the Association for Computational Linguistics: Main Volume, pages 944–948, Online. Association for Computational Linguistics.

Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, et al. 2020. Beyond english-centric multilingual machine translation. arXiv preprint arXiv:2010.11125.

Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, et al. 2021. Beyond english-centric multilingual machine translation. The Journal of Machine Learning Research, 22(1):4839–4886.

William Fedus, Barret Zoph, and Noam Shazeer. 2021. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity.

Ze-Feng Gao, Peiyu Liu, Wayne Xin Zhao, Zhong-Yi Lu, and Ji-Rong Wen. 2022. Parameter-efficient mixture-of-experts architecture for pre-trained language models. arXiv preprint arXiv:2203.01104.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. 2022. An empirical analysis of compute-optimal large language model training. In Advances in Neural Information Processing Systems.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Tuo Zhao. 2020. SMART: Robust and efficient fine-tuning for pretrained natural language models through principled regularized optimization. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 2177–2190, Online. Association for Computational Linguistics.

Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

Taku Kudo and John Richardson. 2018. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 66–71, Brussels, Belgium. Association for Computational Linguistics.

Sneha Kudugunta, Yanping Huang, Ankur Bapna, Maxim Krikun, Dmitry Lepikhin, Minh-Thang Luong, and Orhan Firat. 2021. Beyond distillation: Task-level mixture-of-experts for efficient inference. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 3577–3599.

Yoohwan Kwon and Soo-Whan Chung. 2023. Mole: Mixture of language experts for multi-lingual automatic speech recognition. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE.

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2021. {GS}hard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on

Natural Language Processing (Volume 1: Long Papers), pages 4582–4597, Online. Association for Computational Linguistics.

Zehui Lin, Liwei Wu, Mingxuan Wang, and Lei Li. 2021. Learning language specific sub-network for multilingual machine translation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 293–305, Online. Association for Computational Linguistics.

Marta R NLLB Team, Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al. 2022. No language left behind: Scaling human-centered machine translation. arXiv preprint arXiv:2207.04672.

Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Nothman, Kevin Knight, and Heng Ji. 2017. Crosslingual name tagging and linking for 282 languages. In Proceedings of the 55th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1946–1958, Vancouver, Canada. Association for Computational Linguistics.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Jerin Philip, Alexandre Berard, Matthias Gallé, and Laurent Besacier. 2020. Monolingual adapters for zero-shot neural machine translation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4465– 4470.

Telmo Pessoa Pires, Robin M Schmidt, Yi-Hsiu Liao, and Stephan Peitz. 2023. Learning language-specific layers for multilingual machine translation. arXiv preprint arXiv:2305.02665.

Matt Post. 2018. A call for clarity in reporting bleu scores. WMT 2018, page 186.

Uri Shaham, Maha Elbayad, Vedanuj Goswami, Omer Levy, and Shruti Bhosale. 2022. Causes and cures for interference in multilingual translation. arXiv preprint arXiv:2212.07530.

Noam Shazeer, \*Azalia Mirhoseini, \*Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations.

Qian Wang and Jiajun Zhang. 2022. Parameter differentiation based multilingual neural machine translation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 10, pages 11440–11448.

Zhen Wang, Rameswar Panda, Leonid Karlinsky, Rogerio Feris, Huan Sun, and Yoon Kim. 2023. Multitask prompt tuning enables parameter-efficient transfer learning. In The Eleventh International Conference on Learning Representations.

Zirui Wang, Zachary C. Lipton, and Yulia Tsvetkov. 2020a. On negative interference in multilingual models: Findings and a meta-learning treatment. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4438–4450, Online. Association for Computational Linguistics.

Zirui Wang, Zachary C Lipton, and Yulia Tsvetkov. 2020b. On negative interference in multilingual models: Findings and a meta-learning treatment. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4438–4450.

Haoran Xu, Maha Elbayad, Kenton Murray, Jean Maillard, and Vedanuj Goswami. 2023. Towards being parameter-efficient: A stratified sparsely activated transformer with dynamic capacity. arXiv preprint arXiv:2305.02176.

Haoran Xu, Philipp Koehn, and Kenton Murray. 2022. The importance of being parameters: An intradistillation method for serious gains. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 170–183, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Haoran Xu and Kenton Murray. 2022. Por qué não utiliser alla språk? mixed training with gradient optimization in few-shot cross-lingual transfer. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 2043–2059, Seattle, United States. Association for Computational Linguistics.

Biao Zhang, Ankur Bapna, Rico Sennrich, and Orhan Firat. 2021. Share or not? learning to schedule language-specific capacity for multilingual translation. In International Conference on Learning Representations.

Biao Zhang, Philip Williams, Ivan Titov, and Rico Sennrich. 2020. Improving massively multilingual neural machine translation and zero-shot translation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1628–1639, Online. Association for Computational Linguistics.

## A Training Details for IWSLT’14 and OPUS-100

To balance the training data, we also over-sample low-resource languages with a temperature of $T = 5$ (Aharoni et al., 2019) for the OPUS-100 data and $T \ = \ 2$ for the $\mathrm { I W S L T }$ 14 data. We preprocess the data by sentencepiece (Kudo and Richardson, 2018), establishing a vocabulary size of 32K for the IWSLT’14 dataset and 64K for the OPUS-100 dataset. We pre-pend a special language id symbol at the beginning of the source sentence to indicate the target language. We build homogeneous batches (i.e., a batch only containing sentences in one language direction) and only activate the corresponding language-specific matrix. We set the dropout rate as 0.1 for both datasets. For the IWSLT’14 dataset, we fix the training steps at 150K with 8K warm-up steps for all methods, with a batch size of 4096 tokens. For OPUS, we fix the training steps at 100K with 8K warm-up steps for all methods, with a batch size of 4096 tokens but accumulating gradients 4 times. We train all models on 4 RTX 6000 GPUs. For the IWSLT’14 dataset, we employ the transformer model (with an FFN dimension of 1024 and an embedding dimension of 512), while the $\mathrm { t r a n s f o r m e r } _ { \mathrm { b i g } }$ model (with an FFN dimension of 4096 and an embedding dimension of 1024) is utilized for training the OPUS-100 dataset. The maximum learning rate is 0.0005. The optimizer is Adam (Kingma and Ba, 2014) with inverse\_sqrt learning rate scheduler and weight decay of 0. We use beam search with a width of 5 and use a length penalty of 1.

## B Full Results for MNER

We show the full results of MNER in Table 7.

## C Full Results for MQA

We show the full results of MQA in Table 8.

## D Training Details for The Ablation Study

We randomly pick 15 languages from the OPUS-100 data to build a smaller 15-language data (OPUS-15) for the ablation study: eu, pt, bg, sk, zh, sl, de, hr, nb, ga, rw, as, fy, mr, se. We conduct the ablation study under the many-to-many translation settings. To balance the training data, we sample the data with a temperature of $T = 5$

We preprocess the data by sentencepiece (Kudo and Richardson, 2018), establishing a vocabulary size of 32K vocabulary. we fix the training steps at 50K with 8K warm-up steps for all methods, with a batch size of 4096 tokens. We employ the transformer<sub>base</sub> model (with an FFN dimension of 2048 and an embedding dimension of 512) for training the OPUS-15 dataset. The other settings are the same as Appendix A.

## E Numeric Results for The Ablation Study

Figure 1 shows the averaged BLEU over all directions. Here, We show the detailed numeric results in Figure 9.

<table><tr><td>Methods</td><td>az</td><td>pt</td><td>ms</td><td>af</td><td>kk</td><td>ar</td><td>qu</td><td>te</td><td>vi</td><td>my</td><td>tl</td><td>fr</td><td>hi</td></tr><tr><td>Naive NER</td><td>90.12</td><td>92.56</td><td>94.7</td><td>91.59</td><td>88.25</td><td>89.64</td><td>76.79</td><td>82.42</td><td>92.60</td><td>73.22</td><td>96.65</td><td>90.47</td><td>90.63</td></tr><tr><td>LMS</td><td>90.47</td><td>92.76</td><td>94.87</td><td>92.95</td><td>88.45</td><td>89.62</td><td>80.4</td><td>83.15</td><td>92.88</td><td>75.92</td><td>97.00</td><td>90.69</td><td>90.87</td></tr><tr><td>LMS-FD-Share</td><td>90.67</td><td>92.79</td><td>94.91</td><td>92.29</td><td>87.98</td><td>89.74</td><td>80.01</td><td>82.61</td><td>93.05</td><td>73.18</td><td>96.84</td><td>90.61</td><td>91.24</td></tr><tr><td>LMS-FD-LS</td><td>90.90</td><td>93.15</td><td>95.13</td><td>93.05</td><td>88.25</td><td>89.87</td><td>80.75</td><td>83.33</td><td>93.17</td><td>74.04</td><td>96.94</td><td>90.78</td><td>91.54</td></tr><tr><td></td><td>ro</td><td>eu</td><td>tr</td><td>zh</td><td>et</td><td>hu</td><td>nl</td><td>id</td><td>el</td><td>he</td><td>en</td><td>avg.</td><td>WR (%)</td></tr><tr><td>Naive NER</td><td>94.90</td><td>92.17</td><td>93.49</td><td>77.26</td><td>92.06</td><td>93.24</td><td>92.18</td><td>93.64</td><td>92.01</td><td>86.23</td><td>83.97</td><td>89.20</td><td></td></tr><tr><td>LMS</td><td>95.01</td><td>92.42</td><td>93.75</td><td>77.32</td><td>92.71</td><td>93.56</td><td>92.46</td><td>93.84</td><td>92.07</td><td>86.59</td><td>84.20</td><td>89.75</td><td>96%</td></tr><tr><td>LMS-FD-Share</td><td>94.88</td><td>92.31</td><td>93.65</td><td>77.78</td><td>92.39</td><td>93.40</td><td>92.41</td><td>93.79</td><td>92.07</td><td>85.67</td><td>84.33</td><td>89.53</td><td>88</td></tr><tr><td>LMS-FD-LS</td><td>95.03</td><td>92.63</td><td>93.83</td><td>77.99</td><td>92.67</td><td>93.75</td><td>92.67</td><td>94.02</td><td>92.22</td><td>86.88</td><td>84.35</td><td>89.87</td><td>100%</td></tr></table>

Table 7: Full results for the NMER task. We report F1 scores.

<table><tr><td>Methods</td><td>bn</td><td>en</td><td>fi</td><td>id</td><td>ko</td><td>SW</td><td>avg.</td></tr><tr><td>Naive MQA</td><td>77.69</td><td>70.36</td><td>78.26</td><td>83.00</td><td>61.60</td><td>80.97</td><td>75.31</td></tr><tr><td>LMS</td><td>77.1</td><td>71.7</td><td>78.18</td><td>82.76</td><td>63.70</td><td>81.90</td><td>75.89</td></tr><tr><td>LMS+FD-LS</td><td>78.95</td><td>73.47</td><td>78.80</td><td>84.27</td><td>61.90</td><td>81.35</td><td>76.46</td></tr><tr><td>LMS+FD-Share</td><td>79.08</td><td>73.44</td><td>78.86</td><td>84.34</td><td>62.15</td><td>81.29</td><td>76.53</td></tr></table>

Table 8: Full results for the MQA task. We report F1 scores.

<table><tr><td rowspan="2">Methods</td><td colspan="5">en→xx</td><td colspan="5">xx→en</td><td rowspan="2">extra #params</td></tr><tr><td>high</td><td>med</td><td>low</td><td>all</td><td>WR (%)</td><td>high</td><td>med</td><td>low</td><td>all</td><td>WR (%) Training</td></tr><tr><td>Naive MMT</td><td>20.94</td><td>42.3</td><td>22.72</td><td>26.99</td><td>=</td><td>25.45</td><td>37.25</td><td>27.95</td><td>29.1</td><td>=</td><td></td></tr><tr><td>Switch Transformer, E = 4</td><td>21.94</td><td>45.00</td><td>25.76</td><td>28.85</td><td>100</td><td>26.21</td><td>39.35</td><td>29.12</td><td>30.30</td><td>100</td><td>38M</td></tr><tr><td>Switch Transformer, E = 8</td><td>22.36</td><td>45.11</td><td>27.47</td><td>29.45</td><td>100</td><td>26.37</td><td>40.02</td><td>29.26</td><td>30.59</td><td>93</td><td>88M</td></tr><tr><td>Switch Transformer, E = 12</td><td>22.66</td><td>45.50</td><td>27.19</td><td>29.65</td><td>100</td><td>26.52</td><td>40.32</td><td>29.55</td><td>30.81</td><td>100</td><td>138M</td></tr><tr><td>Switch Transformer, E = 16</td><td>23.05</td><td>46.25</td><td>28.61</td><td>30.35</td><td>100</td><td>26.82</td><td>40.33</td><td>30.31</td><td>31.12</td><td>100</td><td>189M</td></tr><tr><td>LMS, d = 4</td><td>21.61</td><td>40.55</td><td>24.24</td><td>27.19</td><td>87</td><td>26.16</td><td>38.52</td><td>29.21</td><td>30.07</td><td>100</td><td>4M</td></tr><tr><td>LMS, d = 12</td><td>22.20</td><td>44.10</td><td>25.12</td><td>28.63</td><td>100</td><td>26.56</td><td>39.40</td><td>28.65</td><td>30.40</td><td>100</td><td>12M</td></tr><tr><td>LMS, d = 20</td><td>22.57</td><td>45.19</td><td>25.85</td><td>29.26</td><td>100</td><td>26.86</td><td>39.89</td><td>30.34</td><td>31.03</td><td>100</td><td>20M</td></tr><tr><td>LMS, d = 28</td><td>22.82</td><td>43.56</td><td>26.13</td><td>29.01</td><td>93</td><td>27.07</td><td>39.88</td><td>30.27</td><td>31.13</td><td>100</td><td>28M</td></tr><tr><td>LMS, d = 36</td><td>23.10</td><td>43.89</td><td>26.3</td><td>29.28</td><td>93</td><td>27.24</td><td>40.07</td><td>30.31</td><td>31.27</td><td>100</td><td>36M</td></tr><tr><td>LMS, d = 44</td><td>23.32</td><td>43.61</td><td>26.52</td><td>29.37</td><td>93</td><td>27.30</td><td>40.53</td><td>30.81</td><td>31.53</td><td>100</td><td>43M</td></tr><tr><td>LMS, d = 52</td><td>23.36</td><td>45.05</td><td>26.64</td><td>29.80</td><td>93</td><td>27.36</td><td>40.75</td><td>30.72</td><td>31.60</td><td>100</td><td>51M</td></tr><tr><td>LMS, d = 60</td><td>23.50</td><td>45.63</td><td>26.94</td><td>30.09</td><td>100</td><td>27.51</td><td>40.88</td><td>31.20</td><td>31.81</td><td>100</td><td>59M</td></tr></table>

Table 9: The numeric results for the Figure 1.