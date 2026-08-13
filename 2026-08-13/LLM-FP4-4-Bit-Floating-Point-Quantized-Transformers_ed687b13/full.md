# LLM-FP4: 4-Bit Floating-Point Quantized Transformers

Shih-yang Liu\*1, Zechun Liu\*2, Xijie Huang1, Pingcheng Dong1, Kwang-Ting Cheng1

1Hong Kong University of Science and Technology, 2Meta Reality Labs

zechunliu@meta.com

{sliuau, xhuangbs, pingcheng.dong} @connect.ust.hk

timcheng@ust.hk

## Abstract

We propose LLM-FP4 for quantizing both weights and activations in large language models (LLMs) down to 4-bit floating-point values, in a post-training manner. Existing posttraining quantization (PTQ) solutions are primarily integer-based and struggle with bit widths below 8 bits. Compared to integer quantization, floating-point (FP) quantization is more flexible and can better handle long-tail or bell-shaped distributions, and it has emerged as a default choice in many hardware platforms. One characteristic of FP quantization is that its performance largely depends on the choice of exponent bits and clipping range. In this regard, we construct a strong FP-PTQ baseline by searching for the optimal quantization parameters. Furthermore, we observe a high interchannel variance and low intra-channel variance pattern in activation distributions, which adds activation quantization difficulty. We recognize this pattern to be consistent across a spectrum of transformer models designed for diverse tasks, such as LLMs, BERT, and Vision Transformer models. To tackle this, we propose per-channel activation quantization and show that these additional scaling factors can be reparameterized as exponential biases of weights, incurring a negligible cost. Our method, for the first time, can quantize both weights and activations in the LLaMA-13B to only 4-bit and achieves an average score of 63.1 on the common sense zero-shot reasoning tasks, which is only 5.8 lower than the full-precision model, significantly outperforming the previous stateof-the-art by 12.7 points. Code is available at: https://github.com/nbasy1/LLM-FP4.

## 1 Introduction

Since the introduction of transformer architecture (Vaswani et al., 2017), transformers have superseded recursive neural networks, emerging as the dominant architecture in numerous natural language processing (NLP) tasks (Kenton and

Toutanova, 2019; Lewis et al., 2020). The transformative impact of the transformer has been further propelled by the emergence of models like GPT (Brown et al., 2020; OpenAI, 2023), catapulting the popularity of this architecture to new heights. Meanwhile, the versatility of transformers extends beyond NLP, encompassing diverse domains such as vision (Dosovitskiy et al.; Touvron et al., 2021), audio (Akbari et al., 2021), etc. This trend towards a unified architecture for different modalities represents a groundbreaking development within the realm of deep learning.

However, the advancements in transformer performance are accompanied by a corresponding increase in model size and computational costs (Kaplan et al., 2020). This poses significant challenges when attempting to leverage the full potential of transformer models in use cases where memory or computational resources are limited. Despite the extensive research and widespread adoption of transformers, the field of transformer compression remains relatively underexplored. To address this gap, our study focuses on the compression of transformers, especially through floating-point post-training quantization techniques.

Post-training quantization (PTQ) offers the advantages of simple to use with minimal fine-tuning requirements (Nagel et al., 2020; Cai et al., 2020). Existing PTQ solutions for transformers primarily focus on integer (INT) quantization (Liu et al., 2021; Yuan et al., 2022), which can be effective in certain scenarios but often break down when bit widths are below 8 bit. On the other hand, floatingpoint (FP) quantization has gained significant traction as a more flexible alternative, capable of better accommodating various activation and weight distributions. In fact, FP8 has emerged as the default choice in various hardware platforms, including the NVIDIA H100.

Different from integer (INT) quantization, a particular challenge in floating-point (FP) quantization is how to select appropriate exponent bits and scale parameters. Improper parameter choices can lead to subpar or divergent quantization results. To tackle this challenge, we introduce a robust recipe for FP quantization, which leverage layer-wise reconstruction to jointly search for optimal exponent bits and maximum values. Compared to previous approaches that utilize gradient updates for exponent bits (Kuzmin et al., 2022), our search-based method proves to be more stable and consistently delivers desirable quantization results, which establishes a strong baseline for FP-PTQ.

Furthermore, our investigation uncovers an intriguing pattern of activation distributions in transformers, characterized by high inter-channel variance and low intra-channel variance. Similar patterns are also observed in previous works (Xiao et al., 2022; Dettmers et al., 2022), while we argue that this pattern is inherent to transformer architectures and not limited to specific tasks, as we have observed consistent patterns not only in large language models but also in BERT model and even vision transformers. Motivated by these findings, we introduce a novel pre-shifted exponent bias for FP quantization of transformers. Concretely, we leverage the per-channel activation variance computed from calibration data and reparameterize these scales as the exponential bias of the corresponding FP quantized weight vectors. This approach effectively addresses the challenge posed by high inter-channel variance while incurring negligible computational cost.

In summary, we study floating-point posttraining quantization (PTQ) for transformer architectures, and the contribution of this paper includes: • We propose a search-based framework for determining the optimal exponent bias and maximal quantization value. This method outperforms existing techniques in terms of stability and performance, establishing a strong baseline for floatingpoint post-training quantization.

• We propose a novel technique, pre-shifted exponent bias, which effectively addresses the challenge of high inter-channel variance in the transformer with negligible computational overhead.

• Experimental results demonstrate that the proposed method yields the first usable FP4 weight and activation quantized LLaMA-13B model with mere 5.8-point degradation in zero-shot reasoning tasks against the full-precision model, reducing the gap by ～70% compared to the previous SoTA.

• We further extend our method to BERT and vision transformers. It surpasses the previous best 4- bit quantized BERT by 7.8 points on GLUE dataset and achieves 31.4 points higher accuracy compared to the previous SoTA ViT quantization method for 4-bit DeiT-S on ImageNet dataset.

## 2 Related Works

## 2.1 Post-Training Quantization

Model quantization can be mainly categorized into quantization-aware training (QAT) and posttraining quantization (PTQ), depending on whether it involves additional training for weight finetuning or not. Most PTQ studies are primarily focused on convolutional neural networks (CNNs) (Nagel et al., 2020; Li et al., 2021; Wu et al., 2020; Cai et al., 2020; Nagel et al., 2019). However, with the growing popularity of transformer-based models, only a limited number of works (Bondarenko et al., 2021; Yuan et al., 2022; Ding et al., 2022) have been conducted to realize PTQ on transformers. Moreover, the existing works primarily focus on visual transformer models and exhibit inferior performance when the bit width is below 8. Therefore, in this work, we delve into the challenges of the low-bit PTQ for language transformers.

## 2.2 Floating-Point Quantization

Floating-point (FP) quantization has emerged as a promising alternative to integer quantization due to its ability to handle long-tail distributions, and offers increased flexibility (Kuzmin et al., 2022). Additionally, modern GPUs such as H100 (Micikevicius et al., 2022) now support FP quantization. Nonetheless, minimal research has been conducted on FP quantization. Only (Kuzmin et al., 2022) proposes a general FP8 quantization scheme primarily for vision tasks, and (Zhang et al., 2023) adopts a mixture of FP and INT formats quantization for LLMs. In this work, we propose FPQ baseline as a general guideline for low-bit floating-point PTQ to compress language transformer models.

## 3 Preliminaries

## 3.1 Formulation of Floating-Point Variables

A standard floating-point number is represented as:

$$
X _ { \mathrm { F P } } = ( - 1 ) ^ { s } 2 ^ { p - b } ( 1 + \frac { d _ { 1 } } { 2 } + \frac { d _ { 2 } } { 2 ^ { 2 } } + . . . + \frac { d _ { m } } { 2 ^ { m } } )\tag{1}
$$

where $s \in \{ 0 , 1 \}$ is the sign bit. $d _ { i } \in \{ 0 , 1 \}$ is $i ^ { t h }$ mantissa bit, m denoted number of mantissa bits.

![](images/a3e25756bfeb13146ff353031229dfa06e020609084058713523a73016bf0460.jpg)  
Figure 1: An illustration of floating-point (FP) quantization process using FP5 (E2M2) positive axis. The real-valued clipped $X _ { \mathrm { R } } ^ { \prime \prime }$ in Eq. 5 is rescaled by the real-valued scaling factor α. Then, the quantization step-size v is determined by the range $[ 2 ^ { p } , 2 ^ { p } + 1 )$ in which $\frac { X _ { \mathrm { R } } ^ { \prime \prime } } { { \tilde { \alpha } } }$ falls $( \operatorname { E q . 9 } )$ . Here, $p \in \{ 0 , 1 , . . . , 2 ^ { e - 1 } \}$ is the exponent bit value. Lastly, X can be quantized to low-bit floating point values simply by $\begin{array} { r } { X _ { \mathrm { F P } } = \tilde { \alpha } \cdot v \cdot \left\lfloor \frac { X _ { \mathrm { R } } ^ { \prime \prime } } { \tilde { \alpha } \cdot v } \right\rceil ( \mathrm { E q . } 8 ) } \end{array}$

$p$ is an integer in $[ 0 , 2 ^ { e } - 1 ]$ , and e denotes number of exponent bits. b is an integer exponent bias. A floating point with j number exponent bits and k mantissa bits is denoted as FP format EjMk.

## 3.2 Floating-Point Quantization Process

In integer quantization, the real-valued variable $X _ { \mathrm { R } }$ is quantized to an integer $X _ { \mathrm { I N T } }$ with the following formula:

$$
X _ { \mathrm { I N T } } = \alpha { \left\lfloor { \mathrm { C l i p } } { \left( { \frac { X _ { \mathrm { R } } } { \alpha } } , Q _ { m i n } , Q _ { m a x } \right) } \right\rceil }\tag{2}
$$

where $\lfloor \cdot \rceil$ is the rounding function. $X _ { \mathrm { R } }$ is the real-valued variable, α represents the full-precision scaling factor, and $Q _ { m i n } , Q _ { m a x }$ are the min/max value of the quantization range. Similarly, a realvalued variable $X _ { \mathrm { R } }$ can be converted to floatingpoint $X _ { \mathrm { F P } }$ in two steps.

(1) Scale and clip. In FP quantization, we also scale and clip the real-valued variable before quantization as:

$$
X _ { \mathrm { R } } ^ { \prime } = \mathrm { C l i p } ( X _ { \mathrm { R } } , Q _ { m i n } , Q _ { m a x } )\tag{3}
$$

where the min/max value range of signed floatingpoint quantization can be calculated from Eq.1:

$$
Q _ { m a x } = - Q _ { m i n } = ( 2 - 2 ^ { - m } ) 2 ^ { 2 ^ { e } - b - 1 }\tag{4}
$$

Here the integer exponent bias b is another adjustable hyperparameter controlling $Q _ { m a x }$ and $Q _ { m i n }$ , which has similar functionality as α. Therefore, for simplicity, we reformulate Eq. 3 as:

$$
X _ { \mathrm { R } } ^ { \prime \prime } = \mathrm { C l i p } \Big ( X _ { \mathrm { R } } , \tilde { Q } _ { m i n } , \tilde { Q } _ { m a x } \Big ) ,\tag{5}
$$

where

$$
\begin{array} { c } { { \tilde { Q } _ { m a x } = \alpha Q _ { m a x } = \alpha \cdot ( 2 - 2 ^ { - m } ) 2 ^ { 2 ^ { e } - b - 1 } } } \\ { { { } } } \\ { { = \alpha \cdot 2 ^ { - b } \cdot ( 2 - 2 ^ { - m } ) 2 ^ { 2 ^ { e } - 0 - 1 } } } \\ { { { } } } \\ { { = 2 ^ { - \tilde { b } } \cdot ( 2 - 2 ^ { - m } ) 2 ^ { 2 ^ { e } - 0 - 1 } } } \end{array}\tag{6}
$$

Note that we combine the tensor-wise real-valued scaling factor α with integer exponent bias b to form a new scaling factor $\tilde { \alpha } = 2 ^ { - \tilde { b } } = 2 ^ { - b } \cdot \alpha$ Here $\tilde { b }$ denotes a relaxed tensor-wise real-valued exponent, and we can derive $\tilde { b }$ from the desired clipping value $\tilde { Q } _ { m a x }$ from Eq. 6 as:

$$
\tilde { b } = 2 ^ { e } - \log _ { 2 } \tilde { Q } _ { m a x } + \log _ { 2 } ( 2 - 2 ^ { - m } ) - 1\tag{7}
$$

(2) Compare and quantize. Different from integer quantization, which simply utilizes the rounding function to convert the real-valued variables to quantized ones, in floating-point quantization, there is an additional step of comparing $X _ { \mathrm { R } } ^ { \prime \prime }$ with quantization levels and then quantize:

$$
X _ { \mathrm { F P } } = \tilde { \alpha } \cdot v \cdot \left\lfloor \frac { X _ { \mathrm { R } } ^ { \prime \prime } } { \tilde { \alpha } \cdot v } \right\rceil\tag{8}
$$

where $X _ { \mathrm { R } } ^ { \prime \prime }$ is clipped real-valued variable $( \operatorname { E q . 5 } )$ $\tilde { \alpha }$ is the tensor-wise floating-point scaling factor, and v is an integer power of 2.

$$
v = { \displaystyle \{ 2 ^ { \lfloor \log _ { 2 } \lvert { \bf X } _ { \mathrm { R } } ^ { \prime \prime } \rvert + \tilde { { \mathrm { b } } } \rfloor - \mathrm { m } } { \mathrm { i f } \ \lfloor \log _ { 2 } \lvert { \bf X } _ { \mathrm { R } } ^ { \prime \prime } \rvert + \tilde { { \mathrm { b } } } \rfloor \geq 1 } } \\tag{9}
$$

Here we select the quantization level v according to the magnitude of $\mathrm { ~ \frac { ~ X _ { R } ^ { \prime \prime } ~ } { ~ \tilde { \alpha } ~ } ~ }$ , which equals to $X _ { \mathrm { R } } ^ { \prime \prime } \cdot 2 ^ { b }$ Then the floating-point quantized variables can be derived with Eq.8. The illustration of the quantization process is in Fig. 1, detailed explanation can also be found in (Micikevicius et al., 2022).

## 3.3 Floating-Point Matrix Multiplication

With the floating-point quantized variables, the matrix multiplication is formulated as:

$$
\mathbf { O } _ { o u t } ^ { i , k } = \mathbf { X } _ { \mathrm { F P } } ^ { i , : } \mathbf { W } _ { \mathrm { F P } } ^ { : , k } = \tilde { \alpha } _ { \mathbf { x } } \tilde { \alpha } _ { \mathbf { w } } ^ { k } \tilde { \mathbf { X } } _ { \mathrm { F P } } ^ { i , : } \tilde { \mathbf { W } } _ { \mathrm { F P } } ^ { : , k }\tag{10}
$$

Here in per-tensor activation quantization and perchannel weight quantization, $\bar { \mathbf { X } } _ { \mathrm { F P } } ^ { i , : }$ denotes $i ^ { t h }$ row in the activation matrix and $\mathbf { W } _ { \mathrm { F P } } ^ { : , k }$ denotes $k ^ { t h }$ column in the weight matrix, such that each element $\mathbf { O } _ { o u t } ^ { i , k }$ in the output matrix is computed by the product of two real-valued scalars $\tilde { \alpha } _ { \mathbf { x } }$ and $\tilde { \alpha } _ { \mathbf { w } } ^ { k }$ times the corresponding quantized activation and weight vectors. We depict all the possible quantization granularity options that support such efficient matrix multiplication in Appendix D.

## 4 Method

In this section, we begin by introducing our joint format and max value search, which establishes our strong baseline and already achieves state-ofthe-art results at 8-bit and 6-bit quantization. Then we present an efficient pre-shifted exponent bias to tackle the catastrophic high inter-channel activation variance in transformer models and push the quantization limit to 4-bit.

## 4.1 Joint Format and Max Value Search

The objective of post-training quantization is to minimize the perturbation $( \delta \mathbf { X } = \mathbf { X } _ { \mathrm { F P } } - \mathbf { X } _ { \mathrm { R } } )$ introduced by quantization to the pre-trained realvalued network:

$$
\operatorname* { m i n } \mathbb { E } [ \mathcal { L } ( \mathbf { X } _ { \mathrm { R } } + \delta \mathbf { X } ) - \mathcal { L } ( \mathbf { X } _ { \mathrm { R } } ) ]\tag{11}
$$

In this study, we adopt the setting presented in (Choukroun et al., 2019; Wu et al., 2020), which assumes a positive correlation between the change in the intermediate output of the quantized model and Eq. 11. Therefore, minimizing the distance between the intermediate output of the quantized layer (Ô) and the output of the original layer (O) leads to minimize Eq. 11. Hence, the objective loss metric is formulated as:

$$
\operatorname* { m i n } { ( \hat { \mathbf { O } } - \mathbf { O } ) ^ { 2 } }\tag{12}
$$

which is used to search for the optimal FP quantization function in the following proposed framework.

The challenges in FP quantization arise from its sensitivity to the quantization format and clipping range. Undesirable format selection will result in a catastrophic error rate. In addition, we observe that the optimal clipping range varies depending on the format used. Previous work (Kuzmin et al., 2022) on floating-point (FP) quantization-aware training (QAT) proposed to learn both the FP format and maximum value with gradients. However, we find this method suffers from over-fitting in PTQ, with accuracy being even worse than naïve MinMax method, details can be found in Appendix E. Instead, we propose a search-based algorithm that jointly determines the optimal format and its associated clipping range to address this challenge.

The searching process is conducted layer by layer with the metric of minimizing Eq. 12. The output of matrix multiplication corresponding to each sub-module is denoted as $\mathbf { O } = \mathbf { X } \mathbf { Y }$ , where Y can be either a weight tensor W or another activation tensor.

The search space of q-bit FP format includes all formats except for the format with an exponent bit equal to 0, as the quantization of the format with an exponent bit equal to 1 already degenerates to INT quantization. We search for the real-valued exponent bias ${ \tilde { b } } ,$ which equals to the logarithm of the scaling factor. We initialize $\tilde { b } _ { \mathbf { x } }$ and $\bar { \tilde { b } } _ { \mathbf { v } }$ from Eq. 7 with $Q _ { m a x }$ equals the maximum value of $\lvert \mathbf { X } _ { \mathrm { R } } \rvert$ and $\lvert \mathbf { Y } _ { \mathrm { { R } } } \rvert$ , respectively. We then define the search space of $b _ { \mathbf { x } }$ and $\tilde { b } _ { \mathbf { Y } }$ by linearly dividing $[ \gamma _ { _ 1 } \tilde { b } _ { \mathbf { x } } ^ { i n i t } , \gamma _ { _ 2 } \hat { b } _ { \mathbf { x } } ^ { i n i t } ]$ and $[ \gamma _ { _ 1 } \tilde { b } _ { \mathbf { Y } } ^ { i n i t } , \gamma _ { _ 2 } \tilde { b } _ { \mathbf { Y } } ^ { i n i t } ]$ into k intervals, where $\gamma _ { 1 }$ and $\gamma _ { 2 }$ are empirically set to 0.01 and 1.2, and $k = 1 0 0$

The search process is outlined in Alg.1. We search the quantization scheme in all the matrix multiplication layers in parallel following (Yuan et al., 2022; Bai et al., 2022). The algorithm can be divided into two parts. (1) Do forward propagation to store the intermediate raw output of each layer l. (2) Iteratively update the optimal format and biases for each layer for three rounds by minimizing the reconstruction metric (Eq. 12). We name this search-based framework as Floating Point Quantization Baseline (FPQ baseline), and it can already achieve state-of-the-art results on both 8-bit and 6- bit settings.

## 4.2 Pre-Shifted Exponent Bias

In transformer architectures, we observed an intriguing phenomenon of high inter-channel variance. As shown in Fig.2, the magnitudes of values within the same channel are close to each other but exhibit significant differences across different channels. This phenomenon is not only observed in language models (i.e., LLaMA and BERT) but also significant in vision transformer models. Since outlier channels are often orders of magnitude bigger than the rest, they will dominate the quantization precision of the quantized tensor, resulting in less representation capacity for those channels with smaller magnitudes (Xiao et al., 2022). This makes tensor-wise or token-wise scaling factor insufficient for accurate activations quantization.

Algorithm 1 FPQ baseline   
1: Input: Calibration dataset, Full-precision Model M,   
Quantization format search space Rx (e.g., $\begin{array} { r l } { R _ { X } } & { { } = } \end{array}$   
{E3M0, E2M1, E1M2} for FP4), number of round   
$n = 3 ,$   
2: Output: FP q Quantized model   
3: for l in $1 ^ { s t }$ to $L ^ { \mathit { \hat { t } h } }$ layer in M do   
4: Forward & collect raw output $O ^ { l } = X ^ { l } Y ^ { l }$ of layer l;   
5: end for   
6: for l in $1 ^ { s t }$ to $L ^ { t h }$ layer in M do   
7: Initialize the FP format search space w.r.t $X ^ { l }$ and $Y ^ { l }$   
as $R _ { \mathbf { x } } = \{ r _ { \mathbf { x } } ^ { 1 } , r _ { \mathbf { x } } ^ { 2 } , . . . , r _ { \mathbf { x } } ^ { t } \}$ and $R _ { \mathbf { Y } } { \dot { } } = \{ r _ { \mathbf { Y } } ^ { 1 } , r _ { \mathbf { Y } } ^ { 2 } , . . . . r _ { \mathbf { Y } } ^ { t } \}$   
8: Initialize bias $\tilde { b } _ { \mathbf { x } } ^ { i } , \tilde { b } _ { \mathbf { x } } ^ { i }$ with Eq.7 for each format can  
didate $r _ { X } ^ { i } \in R _ { \mathbf { x } }$ and $r _ { \mathbf { Y } } ^ { i } \in R _ { \mathbf { Y } } .$   
9: Generate search space of $\tilde { b } _ { \mathbf { x } }$ in t formats to be   
[γ1 $\tilde { b } _ { \mathbf { x } } ^ { i n i t } , \gamma _ { 2 } \tilde { b } _ { \mathbf { x } } ^ { i n i t } ]$ and $\tilde { b } _ { \mathbf { Y } }$ to be $[ \gamma _ { 1 } \tilde { b } _ { \mathbf { Y } } ^ { i n i t } , \gamma _ { 2 } \tilde { b } _ { \mathbf { Y } } ^ { i n i t } ]$   
10: for 0 to n do   
11: Search for $\tilde { b } _ { \mathbf { x } } ^ { i }$ w.r.t each $r _ { \mathbf { x } } ^ { i }$ that minimizes Eq.12   
12: Search for $r _ { \mathbf { x } } ^ { i } \in R _ { \mathbf { x } }$ that minimizes Eq.12   
13: Search for $\tilde { b } _ { \mathbf { Y } } ^ { i }$ w.r.t each $r _ { \mathbf { Y } } ^ { i }$ that minimizes Eq.12   
14: Search for $r _ { \mathbf { Y } } ^ { i } \in R _ { \mathbf { Y } }$ that minimizes Eq.12   
15: end for   
16: end for

However, applying per-channel scaling factors for activations poses challenges to efficient matrix multiplication, because the scaling factor is not a shared constant along the multiplication direction and cannot be extracted as Eq. 10. To address this challenge, we introduce pre-shifted exponent bias, which allows us to calculate per-channel scaling factors from activations. These scaling factors are then re-parameterized as the exponent biases of the corresponding weights. This method effectively handles high inter-channel variance while maintaining nearly identical efficiency to per-tensor quantization.

Recalling in Eq. 7, we extracted the tensor-wise integer exponent bias b and times it with realvalued scaling factor α and becomes a new scaling factor $\tilde { \alpha } = 2 ^ { - \bar { b } } = 2 ^ { - b } \cdot \alpha$ . Then, the floating-point quantization formula in Eq. 13 becomes:

$$
X _ { \mathrm { F P } } { = } 2 ^ { - \tilde { b } } ( - 1 ) ^ { s } 2 ^ { p - 0 } ( 1 { + } \frac { d _ { 1 } } { 2 } { + } \frac { d _ { 2 } } { 2 ^ { 2 } } { + } . . . { + } \frac { d _ { m } } { 2 ^ { m } } )\tag{13}
$$

We note that after the bias is absorbed in the scaling factor, the original bias term $( b ^ { o r i } )$ in the FP formula is always zero. In dealing with the interchannel variance, we devise an innovative usage of this integer exponent bias: we set it to be a perchannel variant $( \mathbf { b } ^ { o r i } \in \mathbb { Z } ^ { c } )$ 1

Then the calculation of the channel-wise integer bias vector $( \mathbf { b } ^ { o r i } )$ is very straightforward. We first calculate the initial per-channel real-valued scaling factor $( 2 ^ { - \tilde { \mathbf { b } } _ { j } } )$ from the per-channel maximum

![](images/8cb0c5093d51241a2e5223dfedba9d1d260d81e8c0e36be80b9d3382827d029a.jpg)  
Figure 2: Magnitude of the output activations of the feed-forward network blocks in LLaMA-7B, BERT, and DeiT.

values:

$$
\tilde { \mathbf { b } } _ { j } = 2 ^ { e } - \mathrm { l o g } _ { 2 } ( \mathrm { m a x } ( | \mathbf { X } _ { \mathrm { R } } ^ { : , j } | ) ) + \mathrm { l o g } _ { 2 } ( 2 - 2 ^ { - m } ) - 1\tag{14}
$$

Here ${ \bf X } _ { \mathrm { R } } ^ { : , j }$ denotes the $j ^ { t h }$ channel in the activation matrix. Then we separate b to a tensor-wise realvalued scaling factor plus a channel-wise integer scaling factor:

$$
\begin{array} { l } { { \tilde { \mathbf { b } } = \tilde { \rho } + \mathbf { b } ^ { o r i } } } \\ { { \mathbf { \sigma } = \tilde { \rho } + c l i p ( \lfloor \tilde { \mathbf { b } } - \tilde { \rho } \rceil , 0 , 2 ^ { e - 1 } ) } } \end{array}\tag{15}
$$

where $\tilde { \rho } \in \mathbb R ^ { 1 } , \mathbf { b } ^ { o r i } \in \mathbb Z ^ { c }$ . Then the formula for one of the entries in the $j ^ { t h }$ channel of X can be rewrote as follows:

$$
\begin{array} { l } { { \displaystyle X _ { \mathrm { F P } } = 2 ^ { - \tilde { \mathbf { b } } _ { j } } ( - 1 ) ^ { s } 2 ^ { p - 0 } ( 1 + \frac { d _ { 1 } } { 2 } + \ldots + \frac { d _ { m } } { 2 ^ { m } } ) } } \\ { { \displaystyle ~ = 2 ^ { - \tilde { \rho } } ( - 1 ) ^ { s } 2 ^ { p - { \mathbf b } _ { j } ^ { o r i } } ( 1 + \frac { d _ { 1 } } { 2 } + \ldots + \frac { d _ { m } } { 2 ^ { m } } ) } } \end{array}\tag{16}
$$

Note that the bias $\mathbf { b } ^ { o r i }$ is constrained to integers within $[ 0 , 2 ^ { e } - 1 ]$ , compatible with the standard floating-point number calculation. Nevertheless, adding different biases for each channel during inference may still cause some extra hardware operations. Thus, we re-parameterized the perchannel activation bias into a weight tensor and pre-computed the weights using the calibration set. This way, the exponent biases shifting only happens in the calibration stage. Then, an element in ${ \bar { j } } ^ { t h }$ channel of activation tensors X becomes:

$$
X _ { \mathrm { F P } } { = } 2 ^ { - \tilde { \rho } } ( - 1 ) ^ { s } 2 ^ { p - 0 } ( 1 { + } \frac { d _ { 1 } } { 2 } { + } . . . { + } \frac { d _ { m } } { 2 ^ { m } } )\tag{17}
$$

![](images/a52fd795b576c447f48747b11222ebe9e4cc5f304313e6d229fc2cb950f63dbc.jpg)  
Figure 3: Overview of pre-shifted exponent bias method: (a) Search phase: The real-valued channel-wise scaling exponent bias for activations $( \tilde { \mathbf { b } } _ { j } )$ is partitioned into a real-valued tensor-wise exponent bias (ρ), and the integer-based channel-wise exponent bias $( \tilde { \mathbf { b } } _ { j } ^ { o r i } )$ . (b) Reparameterization and weight pre-computation: Once the optimal values are determined on the calibration set, $\tilde { \mathbf { b } } _ { j } ^ { o r i }$ are re-parameterized into the weight tensor. The weights are pre-computed to apply the bias, therefore this is a one-time cost. (c) Inference phase: The method leverages efficient matrix multiplication between low-bit floating-point matrices.

and the corresponding weight element in $j ^ { t h }$ row of the weight tensor W becomes:

$$
W _ { \mathrm { F P } } = 2 ^ { - \tilde { \mathbf { b } } ^ { W } } ( - 1 ) ^ { s } 2 ^ { p - \mathbf { b } _ { j } ^ { o r i } } ( 1 + \frac { d _ { 1 } } { 2 } + . . . + \frac { d _ { m } } { 2 ^ { m } } )\tag{18}
$$

As result, efficient matrix multiplication in Eq.10 is reformulated as:

$$
{ \bf O } _ { o u t } ^ { i , k } = { \bf X } _ { \mathrm { F P } } ^ { i , : } { \bf W } _ { \mathrm { F P } } ^ { : , k } = \tilde { \alpha } _ { \bf x } \tilde { \alpha } _ { \bf w } ^ { k } \tilde { \bf X } _ { \mathrm { F P } } ^ { i , : } ( \beta \odot \tilde { \bf W } _ { \mathrm { F P } } ^ { : , k } )\tag{19}
$$

where  is the element-wise multiplication, $\beta$ $2 ^ { - \mathbf { b } ^ { o r i } }$ and $( \beta \odot \tilde { \mathbf { W } } _ { \mathrm { F P } } ^ { : , k } )$ can be pre-calculated and stored in low-bit FP format. We depict the overall pre-shifted exponent bias method in Fig.3. This method applies to quantizing all the fullyconnected layers. During the search process, we initialize $\tilde { \rho } _ { \mathbf { x } }$ as the $\mathrm { m i n } _ { j } ( \tilde { \mathbf { b } } _ { j } )$ . Then, we fixed $\tilde { \mathbf { b } } _ { \mathbf { x } }$ to be the bias calculated from the Eq. 14 and search for the optimal $\tilde { \rho } _ { \mathbf { x } }$ from $[ \gamma _ { 1 } \tilde { \rho } _ { \mathbf { X } } ^ { \ i n i t } , \gamma _ { 2 } \tilde { \rho } _ { \mathbf { X } } ^ { \ i n i t } ]$

Combining pre-shifted exponent bias method with the joint format and max-value search framework(FPQ baseline), we name our method as (FPQ), short for Floating Point Quantization.

## 5 Experiments

To validate the effectiveness of the proposed method, we conduct experiments on LLaMA (Touvron et al., 2023) and BERT (Devlin et al., 2019) models in 5.2.1 and Sections 5.2.2. Further, in Section 5.2.3 we show that our method also generalizes well to vision transformer architectures. We present ablation studies on the calibration size and search range in Section 5.3, and analyze the hardware costs of implementing FP operators in Section 5.4.

## 5.1 Experiments Details

We adopt per-tensor quantization for activation and per-channel quantization for weight. We employ layer reconstruction following the settings of (Yuan et al., 2022; Nagel et al., 2020), and parallel quantization based on the approach outlined in (Bai et al., 2022; Yuan et al., 2022). A more detailed discussion regarding our implementation decisions can be found in Appendix F. For LLaMA models, we quantize all the weight and activation tensors in fully-connected layers for a fair comparison with previous work (Xiao et al., 2022; Liu et al., 2023). For BERT and ViT models, both fully-connected layers and activation-activation multiplication tensors in the self-attention module are quantized. Note that for FPQ on BERT (Devlin et al., 2019) and ViTs models, the reconstruction metric Eq. 12 is substituted with a Hessian approximation loss metric. This substitution is further detailed in Appendix A.

## 5.2 Main Results

## 5.2.1 LLM Zero-Shot Reasoning

We evaluate the effectiveness of FPQ for LLaMA-7B/ LLaMA-13B (Touvron et al., 2023) on common sense zero-shot reasoning tasks. For the calibration data, we sample 32 random segments with 2048 tokens length from the C4 (Raffel et al., 2020)

<table><tr><td>Quant Method</td><td>|#Bits (E/W/A) | # Calib</td><td></td><td>BoolQ</td><td>PIQA</td><td>HellaSwag</td><td>WinoGrande</td><td>e ARC-e</td><td>ARC-c</td><td>Avg.</td></tr><tr><td>LLaMA-7B Full-precision</td><td>16/16/16</td><td></td><td>75.1</td><td>78.7</td><td>56.9</td><td>69.9</td><td>75.3</td><td>41.9</td><td>66.3</td></tr><tr><td>MinMax INT Quant MinMax FP Quant (E4M3)</td><td>8/8/8</td><td>32</td><td>64.3</td><td>66.8</td><td>40.5</td><td>57.4</td><td>59.0</td><td>29.6</td><td>52.9</td></tr><tr><td></td><td>8/8/8</td><td>32</td><td>74.9</td><td>78.6</td><td>56.8</td><td>69.5</td><td>75.5</td><td>41.6</td><td>66.1</td></tr><tr><td>SmoothQuant (Xiao et al., 2022)</td><td>16/8/8</td><td>512</td><td>74.0</td><td>77.5</td><td>55.0</td><td>69.6</td><td>74.4</td><td>37.4</td><td>64.6</td></tr><tr><td>FPQ baseline FPQ</td><td>8/8/8 8/8/8</td><td>32 32</td><td>75.8</td><td>78.3 78.2</td><td>55.9</td><td>69.5</td><td>75.6</td><td>41.3</td><td>66.1</td></tr><tr><td></td><td></td><td></td><td>75.6</td><td></td><td>56.6</td><td>70.2</td><td>74.6</td><td>40.7</td><td>66.0</td></tr><tr><td>MinMax INT Quant</td><td>4/4/16</td><td>32</td><td>64.1</td><td>76.1</td><td>51.6</td><td>66.3</td><td>72.4</td><td>40.0</td><td>61.7</td></tr><tr><td>MinMax FP Quant (E2M1)</td><td>4/4/16</td><td>32</td><td>73.0</td><td>77.9</td><td>55.2</td><td>69.1</td><td>73.6</td><td>40.9</td><td>64.9</td></tr><tr><td>GPTQ (Frantar et al., 2023) FPQ baseline</td><td>4/4/16 4/4/16</td><td>128</td><td>73.3</td><td>77.9</td><td>54.9</td><td>67.9</td><td>72.7</td><td>37.4</td><td>64.0</td></tr><tr><td>FPQ</td><td>4/4/16</td><td>32 32</td><td>74.8 74.2</td><td>77.9</td><td>55.6</td><td>69.5</td><td>75.2</td><td>41.0</td><td>65.7</td></tr><tr><td>MinMax INT Quant</td><td></td><td></td><td></td><td>77.8</td><td>55.8</td><td>69.9</td><td>74.9</td><td>40.4</td><td>65.5</td></tr><tr><td>MinMax FP Quant (E2M1/E4M3)</td><td>4/4/8</td><td>32</td><td>50.4</td><td>56.5</td><td>27.9</td><td>46.5</td><td>36.1</td><td>21.2</td><td>39.7</td></tr><tr><td>FPQ baseline</td><td>4/4/8 4/4/8</td><td>32</td><td>73.0</td><td>77.5</td><td>55.0</td><td>69.3</td><td>73.6</td><td>40.9</td><td>64.9</td></tr><tr><td>FPQ</td><td>4/4/8</td><td>32 32</td><td>75.0 75.0</td><td>77.6 77.7</td><td>55.9</td><td>69.9</td><td>74.3</td><td>39.4</td><td>65.3</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>55.5</td><td>69.8</td><td>74.5</td><td>39.9</td><td>65.4</td></tr><tr><td>MinMax INT Quant</td><td>4/4/4</td><td>32 32</td><td>54.1</td><td>51.7</td><td>25.6</td><td>49.8</td><td>24.7</td><td>22.9</td><td>38.1</td></tr><tr><td>MinMax FP Quant (E2M1)</td><td>4/4/4 16/4/4</td><td></td><td>47.3</td><td>53.1</td><td>25.7</td><td>50.7</td><td>25.1</td><td>22.4</td><td>37.4</td></tr><tr><td>SmoothQuant (Xiao et al., 2022) LLM-QAT (Liu et al., 2023)</td><td>16/4/4</td><td>512</td><td>54.1 63.5</td><td>62.8</td><td>41.5</td><td>52.6</td><td>50.6</td><td>32.9</td><td>49.1</td></tr><tr><td>FPQ baseline</td><td>4/4/4</td><td>(QAT) 32</td><td>57.4</td><td>64.3 56.6</td><td>55.6 30.2</td><td>52.9 51.1</td><td>50.3</td><td>30.2</td><td>52.8</td></tr><tr><td>FPQ</td><td>4/4/4</td><td>32</td><td>64.2</td><td>73.5</td><td>47.8</td><td>63.7</td><td>37.7 65.9</td><td>23.2</td><td>42.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>33.6</td><td>58.1</td></tr><tr><td>LLaMA-13B Full-precision</td><td>16/16/16</td><td>-</td><td>77.9</td><td>79.2</td><td>59.9</td><td>72.6</td><td>77.4</td><td>46.4</td><td>68.9</td></tr><tr><td>MinMax INT Quant</td><td>8/8/8</td><td>32</td><td>60.6</td><td>69.6</td><td>46.0</td><td>61.5</td><td>63.3</td><td>32.8</td><td>55.6</td></tr><tr><td>MinMax FP Quant (E4M3)</td><td>8/8/8</td><td>32</td><td>78.0</td><td>79.1</td><td>60.0</td><td>72.3</td><td>77.2</td><td>47.1</td><td>68.9</td></tr><tr><td>SmoothQuant (Xiao et al., 2022)</td><td>16/8/8</td><td>512</td><td>76.5</td><td>78.0</td><td>58.0</td><td>72.1</td><td>76.3</td><td>45.5</td><td>68.2</td></tr><tr><td>FPQ baseline</td><td>8/8/8</td><td>32</td><td>78.0</td><td>79.1</td><td>59.9</td><td>72.3</td><td>77.2</td><td>47.1</td><td>68.9</td></tr><tr><td>FPQ</td><td>8/8/8</td><td>32</td><td>78.1</td><td>78.5</td><td>59.1</td><td>72.4</td><td>76.4</td><td>46.1</td><td>68.4</td></tr><tr><td>MinMax INT Quant</td><td>4/4/8</td><td>32</td><td>52.1</td><td>65.0</td><td>36.4</td><td>53.9</td><td>52.3</td><td>29.0</td><td>48.1</td></tr><tr><td>MinMax FP Quant (E2M1/E4M3)</td><td>4/4/8</td><td>32</td><td>78.0</td><td>78.9</td><td>58.0</td><td>71.6</td><td>76.0</td><td>44.8</td><td>67.9</td></tr><tr><td>FPQ baseline</td><td>4/4/8</td><td>32</td><td>76.2</td><td>78.2</td><td>57.9</td><td>71.9</td><td>75.1</td><td>43.9</td><td>67.2</td></tr><tr><td>FPQ MinMax INT Quant</td><td>4/4/8</td><td>32</td><td>76.4</td><td>78.5</td><td>58.2</td><td>72.1</td><td>75.2</td><td>44.7</td><td>67.5</td></tr><tr><td></td><td>4/4/4</td><td>32</td><td>54.5</td><td>52.7</td><td>25.5</td><td>51.1</td><td>25.3</td><td>22.1</td><td>38.5</td></tr><tr><td>MinMax FP Quant (E2M1)</td><td>4/4/4</td><td>32</td><td>45.8</td><td>51.7</td><td>25.5</td><td>49.5</td><td>25.0</td><td>22.8</td><td>36.7</td></tr><tr><td>SmoothQuant (Xiao et al., 2022)</td><td>16/4/4</td><td>512</td><td>57.6</td><td>61.3</td><td>56.0</td><td>52.6</td><td>49.9</td><td>25.1</td><td>50.4</td></tr><tr><td>FPQ baseline</td><td>4/4/4</td><td>32</td><td>54.3</td><td>57.7</td><td>35.7</td><td>52.2</td><td>41.1</td><td>25.7</td><td>44.5</td></tr><tr><td>FPQ</td><td>4/4/4</td><td>32</td><td>71.9</td><td>74.8</td><td>53.3</td><td>66.7</td><td>71.7</td><td>39.9</td><td>63.1</td></tr></table>

Table 1: Zero-shot performance on common sense reasoning tasks with LLaMA (Touvron et al., 2023) models. We denote E/W/A as the bit-width of word embeddings, model weight and activations, respectively.

dataset following the setting of GPTQ (Frantar et al., 2023). The data preprocessing and score calculation are based on EleutherAI evaluation harness¹. In Table 1, we compare FPQ to the floatingpoint PTQ baselines, and state-of-the-art PTQ and QAT methods, including SmoothQuant (Xiao et al., 2022) and GPTQ (Frantar et al., 2023), and LLM-QAT (Liu et al., 2023).

In general, all methods, except for the naïve Min-Max INT Quantization, produce comparable outcomes in the 8-bit setting on both LLaMA-7B and LLaMA-13B. Additionally, we observe that the naïve MinMax FP Quantization achieves nearly lossless results and even surpasses the state-ofthe-art integer post-training quantization method, SmoothQuant (Xiao et al., 2022), which indicates that floating-point quantization naturally has a strong capability in handling the distributions in transformers. However, both MinMax FP Quant and FPQ baseline fail when pushing the quantization precision to ultra-low 4/4/4 bit setting, with 28.9% and 23.8% accuracy degradation on LLaMA-7B, respectively. In this extreme case, the previous state-of-the-art PTQ and QAT methods, SmoothQuant (Xiao et al., 2022) and LLM-QAT (Liu et al., 2023) also suffer severe accuracy downgrade. In comparison, FPQ demonstrates a strong capability of handling extra-low bit settings and achieves only 8.2/5.8% accuracy drop on LLaMA-7B/13B with 4/4/4 bit-width, outperforming SmoothQuant (Xiao et al., 2022) by a large margin, yet with less bit-width and smaller calibration size. Moreover, FPQ even achieves 5.3% accuracy improvements compared to LLM-QAT (Liu et al., 2023) in the 4/4/4 setting and 1.5% over GPTQ (Frantar et al., 2023) in the 4/4/16 configuration on LLaMA-7B.

For practitioners, a crucial consideration is determining the appropriate quantization methods for various bit-widths. Therefore, based on our findings, we offer two recommendations that balance the trade-off between accuracy and search/optimization efficiency. First of all, since the difference between MinMax FP Quant and the rest of the methods is marginal for the 8/8/8 setting, we recommend simply using the MinMax FP Quant method for the 8/8/8 setting as the MinMax method does not involve search process. However, for more demanding scenarios, especially with activation quantization to 4 bits, we recommend employing FPQ for minimizing accuracy degradation with negligible inference overhead.

<table><tr><td>Quant Method</td><td>|#Bits (E/W/A)</td><td>|# Calib</td><td>|MNLI -m</td><td>QQP</td><td>QNLI</td><td>SST-2</td><td>CoLA</td><td>STS-B</td><td>MRPC</td><td>RTE</td><td>Avg.</td></tr><tr><td>(Full-precision)</td><td>32-32-32</td><td></td><td>84.9</td><td>91.4</td><td>92.1</td><td>93.2</td><td>59.7</td><td>90.1</td><td>86.3</td><td>72.2</td><td>83.7</td></tr><tr><td>MinMax INT Quant</td><td>8/8/8</td><td>128</td><td>77.0</td><td>89.9</td><td>88.9</td><td>92.9</td><td>51.8</td><td>88.2</td><td>83.8</td><td>71.5</td><td>80.5</td></tr><tr><td>MinMax FP Quant (E2M5)</td><td>8/8/8</td><td>128</td><td>78.9</td><td>90.8</td><td>88.6</td><td>92.9</td><td>52.7</td><td>88.4</td><td>84.3</td><td>69.0</td><td>80.7</td></tr><tr><td>MinMax FP Quant (E3M4)</td><td>8/8/8</td><td>128</td><td>84.5</td><td>90.9</td><td>91.5</td><td>93.2</td><td>58.3</td><td>89.3</td><td>87.7</td><td>71.8</td><td>83.4</td></tr><tr><td>MinMax FP Quant (E4M3)</td><td>8/8/8</td><td>128</td><td>84.7</td><td>90.9</td><td>91.7</td><td>93.0</td><td>58.6</td><td>89.3</td><td>86.5</td><td>72.2</td><td>83.4</td></tr><tr><td>MinMax FP Quant (E5M2)</td><td>8/8/8</td><td>128</td><td>84.1</td><td>90.9</td><td>91.4</td><td>93.6</td><td>58.1</td><td>89.2</td><td>87.5</td><td>71.8</td><td>83.3</td></tr><tr><td>FPQ baseline</td><td>8/8/8</td><td>128</td><td>84.6</td><td>90.9</td><td>91.7</td><td>93.1</td><td>58.6</td><td>89.3</td><td>88.0</td><td>72.2</td><td>83.5</td></tr><tr><td>FPQ</td><td>8/8/8</td><td>128</td><td>84.6</td><td>91.0</td><td>91.6</td><td>93.3</td><td>58.8</td><td>89.3</td><td>88.0</td><td>72.2</td><td>83.6</td></tr><tr><td>MinMax INT Quant</td><td>6/6/6</td><td>128</td><td>31.9</td><td>62.0</td><td>52.8</td><td>58.8</td><td>0.0</td><td>12.7</td><td>32.1</td><td>52.7</td><td>37.9</td></tr><tr><td>MinMax FP Quant (E2M3)</td><td>6/6/6</td><td>128</td><td>43.5</td><td>85.4</td><td>79.4</td><td>90.5</td><td>45.2</td><td>86.0</td><td>66.9</td><td>59.9</td><td>69.6</td></tr><tr><td>MinMax FP Quant (E3M2)</td><td>6/6/6</td><td>128</td><td>83.9</td><td>90.8</td><td>90.8</td><td>92.2</td><td>58.2</td><td>88.6</td><td>87.0</td><td>72.2</td><td>83.0</td></tr><tr><td>MinMax FP Quant (E4M1)</td><td>6/6/6</td><td>128</td><td>84.4</td><td>90.2</td><td>90.1</td><td>92.2</td><td>58.2</td><td>89.2</td><td>85.3</td><td>69.7</td><td>82.4</td></tr><tr><td>FPQ baseline</td><td>6/6/6</td><td>128</td><td>84.6</td><td>90.9</td><td>91.2</td><td>93.2</td><td>58.8</td><td>88.7</td><td>87.5</td><td>70.8</td><td>83.2</td></tr><tr><td>FPQ</td><td>6/6/6</td><td>128</td><td>84.5</td><td>90.8</td><td>91.6</td><td>93.1</td><td>57.3</td><td>89.3</td><td>88.7</td><td>71.8</td><td>83.2</td></tr><tr><td>MinMax INT Quant</td><td>4/4/8</td><td>128</td><td>33.1</td><td>63.8</td><td>60.1</td><td>49.3</td><td>0.0</td><td>44.0</td><td>50.2</td><td>49.1</td><td>43.7</td></tr><tr><td>MinMax FP Quant (E2M1)</td><td>4/4/8</td><td>128</td><td>60.6</td><td>70.9</td><td>77.4</td><td>79.9</td><td>5.5</td><td>78.6</td><td>46.8</td><td>56.6</td><td>59.5</td></tr><tr><td>MREM-S (Bai et al., 2022)</td><td>4/4/8</td><td>4096</td><td>83.5</td><td>90.2</td><td>91.2</td><td>91.4</td><td>55.1</td><td>89.1</td><td>84.8</td><td>71.8</td><td>82.1</td></tr><tr><td>MREM-P (Bai et al., 2022)</td><td>4/4/8</td><td>4096</td><td>83.4</td><td>90.2</td><td>91.0</td><td>91.5</td><td>54.7</td><td>89.1</td><td>86.3</td><td>71.1</td><td>82.2</td></tr><tr><td>FPQ baseline</td><td>4/4/8</td><td>128</td><td>84.4</td><td>90.6</td><td>91.4</td><td>92.9</td><td>58.6</td><td>83.7</td><td>88.2</td><td>73.3</td><td>82.9</td></tr><tr><td>FPQ</td><td>4/4/8</td><td>128</td><td>84.5</td><td>90.6</td><td>91.1</td><td>92.7</td><td>58.8</td><td>89.3</td><td>88.7</td><td>73.3</td><td>83.6</td></tr><tr><td>MinMax INT Quant</td><td>4/4/4</td><td>128</td><td>31.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MinMax FP Quant (E2M1)</td><td>4/4/4</td><td>128</td><td>33.6</td><td>39.7 54.0</td><td>50.5 50.6</td><td>49.1 50.8</td><td>0.0</td><td>6.7</td><td>31.6 31.6</td><td>54.5 52.0</td><td>32.9 34.1</td></tr><tr><td>BrecQ (Li et al., 2021)</td><td>8/4/4</td><td>4096</td><td>31.9</td><td></td><td>50.7</td><td></td><td>0.0</td><td>0.0</td><td></td><td></td><td>35.8</td></tr><tr><td></td><td></td><td>4096</td><td></td><td>62.3</td><td></td><td>50.9</td><td>0.9</td><td>6.4</td><td>31.7</td><td>52.3</td><td></td></tr><tr><td>QDrop (Wei et al., 2022)</td><td>8/4/4</td><td></td><td>71.4</td><td>79.0</td><td>76.8</td><td>88.1</td><td>40.9</td><td>81.9</td><td>79.2</td><td>60.7</td><td>72.3</td></tr><tr><td>FPQ baseline FPQ</td><td>4/4/4 4/4/4</td><td>128 128</td><td>38.9 82.3</td><td>68.3 89.2</td><td>55.3 86.6</td><td>83.6 91.5</td><td>10.6 52.6</td><td>0.0 85.5</td><td>43.8 83.8</td><td>55.2 69.0</td><td>44.5 80.1</td></tr></table>

Table 2: Results on the GLUE development set with BERT (Bai et al., 2022) model. We denote E/W/A as the bit-width of word embeddings, model weight and activations, respectively.

## 5.2.2 BERT Model

We evaluate the proposed quantization techniques for BERT model on GLUE tasks (Wang et al., 2019). Full-precision BERT-base models finetuned on GLUE datasets are obtained from Huggingface public repository². We randomly sample 128 data from the training set as the calibration set. In Table 2, FPQ demonstrates remarkable performance, achieving absolute average accuracy improvements of 44.3% compared to BrecQ (Li et al., 2021) and 7.9% over QDrop (Wei et al., 2022) with 4/4/4 bit setting. Further, with 4-bit weight and 8-bit activation, MREM-S/MREM-P (Bai et al., 2022) present a 1.6/1.5% accuracy gap to the fullprecision model with 4096 calibration data, while FPQ achieves almost no accuracy loss with only

128 calibration data points.

## 5.2.3 Generalizability on Vision Transformer

Based on our findings that vision transformers also exhibit a consistent activation distribution pattern as language transformers, characterized by high inter-channel variance and low intra-channel variance, as detailed in Fig. 2, we extended our proposed methods to ViT and compared FPQ with floating-point PTQ baselines and state-of-the-art PTQ method for ViT on the ImageNet classification task. Table 3 shows that findings on ViT are consistent with that on language models: previous state-of-the-art integer-based methods struggled to maintain reasonable accuracy when quantizing the transformer to lower bits. In comparison, the proposed FPQ outperformed both PTQ4ViT and APQ-ViT on 6 bits, and also achieved 40.9% and 31.5% absolute accuracy improvement over PTQ4ViT and APQ-ViT on DeiT-S in the 4-bit configuration.

## 5.3 Ablation Study

In this section, we first compare the influence of different calibration sizes on FPQ. We vary the calibration size in {32, 64, 128, 256} and test on MNLI, QQP, and CoLA. Table 4 shows that the evaluation on MNLI and QQP is more robust to different settings, and the variance is more significant on CoLA. We observe that FPQ performs well with a calibration set size of 128 data points. However, we also find that it remains robust and maintains competitive accuracy even with limited access to calibration data, such as when using as few as 32 data points.

<table><tr><td>W/A</td><td>Quant Method</td><td>Deit-S</td><td>Deit-B</td><td>ViT-S</td></tr><tr><td>Full-prec |</td><td></td><td>79.9</td><td>81.8 80.3</td><td>81.4</td></tr><tr><td>6/6 6/6 6/6 6/6 6/6</td><td>PTQ4ViT(Yuan et al., 2022) APQ-ViT(Ding et al., 2022) MinMax FP Quant (E3M2) FPQ baseline FPQ</td><td>76.3 77.8 79.3 79.43 79.5</td><td>80.4 81.7 81.7 81.8</td><td>78.6 79.2 80.7 80.9 81.1</td></tr><tr><td>4/4 4/4 4/4 4/4 4/4</td><td>PTQ4ViT(Yuan et al., 2022) APQ-ViT (Ding et al., 2022) MinMax FP Quant (E2M1) FPQ baseline FPQ</td><td>34.1 43.6 0.4 6.57 75.0</td><td>64.4 67.5 0.1 0.71 79.4</td><td>42.6 48.0 0.1 0.3 73.2</td></tr></table>

Table 3: Comparison on the ImageNet dataset with vision transformer structures.

<table><tr><td rowspan=1 colspan=1>E/W/A</td><td rowspan=1 colspan=2>#Calib | MNLI-MQQPCoLA</td></tr><tr><td rowspan=1 colspan=1>4/4/4</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>81.5   89.4 44.4</td></tr><tr><td rowspan=2 colspan=1>4/4/44/4/44/4/4</td><td rowspan=1 colspan=1>64</td><td rowspan=2 colspan=1>81.8   89.4 47.982.3   89.2 52.681.9   89.0 52.9</td></tr><tr><td rowspan=1 colspan=1>128256</td></tr><tr><td rowspan=2 colspan=1>6/6/66/6/6</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>84.8   90.8 55.0</td></tr><tr><td rowspan=1 colspan=1>64</td><td rowspan=3 colspan=1>84.7   90.9 58.284.5   90.8 57.384.6   90.8 57.6</td></tr><tr><td rowspan=1 colspan=1>6/6/6</td><td rowspan=1 colspan=1>128</td></tr><tr><td rowspan=1 colspan=1>6/6/6</td><td rowspan=1 colspan=1>256</td></tr></table>

Table 4: Ablation studies of different calibration sizes.

We investigate the robustness of FPQ to different search ranges (γ1, γ2). Table 5 presents the results of FPQ using three sets of (γ1, γ2): (0.01, 1.2), (0.1, 1.2), (0.5, 1.5), on MNLI, QQP, and CoLA. It is observed that no single search range outperforms the others consistently across all tasks. For instance, the search range (0.01, 1.2) performs better than (0.5, 1.5) on MNLI and QQP, but slightly worse on CoLA in the 4-bit configuration. Overall, FPQ exhibits robustness to various γ1 and γ2, as long as the search range is not overly aggressive.

## 5.4 Hardware Cost

We further examine the hardware utilization of lowbit INT, FP, and mixed-format FP multiplication operators, including adder, multiplier, and multiplyaccumulate (MAC) units, in terms of hardware area. Mixed-format FP refers to the multiplication of floating-point numbers with different formats, e.g., E2M1 multiplies with E1M2. We implemented the MAC operator by Verilog HDL and utilized Cadence Genus to obtain the synthesized area under TSMC 40nm technology and 0.5GHz clock frequency.

Table 6 illustrates the hardware cost of the INT and FP operators, with the multiplier being the primary cost for INT and the adder for FP. Notably, the disparity between FP4 and INT4 adders is small, while INT has twice the hardware cost for the multiplier. Moreover, the mixed-format FP4 operator has comparable hardware area as the standard FP4 operator. These findings indicate that the proposed FPQ approach imposes negligible overhead in terms of hardware implementation when compared to the standard FP operators and the hardware cost for FP is comparable with INT.

<table><tr><td>E/W/A</td><td>γ1, γ2</td><td>MNLI-M</td><td>QQP</td><td>CoLA</td></tr><tr><td>4/4/4 4/4/4</td><td>0.01, 1.2</td><td>82.3</td><td>89.2 89.1</td><td>52.6</td></tr><tr><td>4/4/4</td><td>0.1, 1.2 0.5, 1.5</td><td>82.2 82.3</td><td>88.4</td><td>53.6 52.8</td></tr><tr><td>6/6/6</td><td>0.01, 1.2</td><td>84.5</td><td>90.8</td><td>57.3</td></tr><tr><td>6/6/6</td><td>0.1,1.2</td><td>84.7</td><td>90.8</td><td>57.5</td></tr><tr><td>6/6/6</td><td>0.5,1.5</td><td>84.7</td><td>90.8</td><td>57.8</td></tr></table>

Table 5: Ablation studies of different search range.

<table><tr><td>Format</td><td colspan="3">Adder(µm2) Multiplier(µm2) MAC(µm2)</td></tr><tr><td>INT4</td><td>93</td><td>182</td><td>410</td></tr><tr><td>INT6</td><td>132</td><td>340</td><td>529</td></tr><tr><td>E2M1</td><td>111</td><td>92</td><td>443</td></tr><tr><td>E3M2</td><td>223</td><td>138</td><td>498</td></tr><tr><td>E2M1*E1M2</td><td>105</td><td>107</td><td>432</td></tr></table>

Table 6: Area differences of INT, FP and mixed Format FP operators across different bit-widths.

## 6 Conclusion

This paper presents the first successful demonstration of 4-bit floating-point post-training quantization for weights, activations, and embeddings in natural language transformer architectures, including both large language models and BERT model. We also extend our method to vision transformers and observe its robust generalization ability. Our approach involves a practical search-based technique which establishes a strong baseline and achieves state-of-the-art results for 6-bit and 8-bit quantization. Furthermore, we address the challenge of high inter-channel variance in transformers by proposing pre-shifted exponent bias, which proves highly effective in achieving accurate 4-bit quantization.

## Acknowledgement

This research is supported by National Natural Science Foundation of China/ HKSAR Research Grants Council Joint Research Scheme under Grant NH KUST627/20, and Foshan HKUST Projects under Grant FSUST21 – H KUST10E.

## Limitations

Our experiments were conducted on publicly available datasets with finite sentence lengths, and the generalizability of our method to extremely long sequences or streaming data has not been verified and may require further investigation. In addition, it remains to be seen how our proposed method can generalize to other domains beyond language and vision, such as audio. It would also be interesting to see the applicability of our method to generative tasks and other applications.

## References

Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong. 2021. Vatt: Transformers for multimodal selfsupervised learning from raw video, audio and text. Advances in Neural Information Processing Systems, 34:24206-24221.

Haoli Bai, Lu Hou, Lifeng Shang, Xin Jiang, Irwin King, and Michael Lyu. 2022. Towards efficient posttraining quantization of pre-trained language models. In Advances in Neural Information Processing Systems.

Yelysei Bondarenko, Markus Nagel, and Tijmen Blankevoort. 2021. Understanding and overcoming the challenges of efficient transformer quantization.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami, Michael W Mahoney, and Kurt Keutzer. 2020. Zeroq: A novel zero shot quantization framework. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13169–13178.

Yoni Choukroun, Eli Kravchik, Fan Yang, and Pavel Kisilev. 2019. Low-bit quantization of neural networks for efficient inference.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. 2022. Llm.int8(): 8-bit matrix multiplication for transformers at scale. Advances in Neural Information Processing Systems, 35:30318–30332.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding.

Yifu Ding, Haotong Qin, Qinghua Yan, Zhenhua Chai, Junjie Liu, Xiaolin Wei, and Xianglong Liu. 2022. Towards accurate post-training quantization for vision transformer. In Proceedings of the 30th ACM

International Conference on Multimedia, MM '22, page 5380–5388, New York, NY, USA. Association for Computing Machinery.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2023. GPTQ: Accurate post-training compression for generative pretrained transformers. In International Conference on Learning Representations.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT, pages 4171–4186.

Andrey Kuzmin, Mart Van Baalen, Yuwei Ren, Markus Nagel, Jorn Peters, and Tijmen Blankevoort. 2022. Fp8 quantization: The power of the exponent. Advances in Neural Information Processing Systems, 35:14651–14662.

Jemin Lee, Yongin Kwon, Jeman Park, Misun Yu, and Hwanjun Song. 2023. Q-hyvit: Post-training quantization for hybrid vision transformer with bridge block reconstruction.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7871–7880.

Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, and Shi Gu. 2021. Brecq: Pushing the limit of post-training quantization by block reconstruction. arXiv preprint arXiv:2102.05426.

Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. 2023. Llm-qat: Data-free quantization aware training for large language models. arXiv preprint arXiv:2305.17888.

Zhenhua Liu, Yunhe Wang, Kai Han, Wei Zhang, Siwei Ma, and Wen Gao. 2021. Post-training quantization for vision transformer. Advances in Neural Information Processing Systems, 34:28092–28103.

Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi, Stuart Oberman, Mohammad Shoeybi, Michael Siu, and Hao Wu. 2022. Fp8 formats for deep learning.

Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos, and Tijmen Blankevoort. 2020. Up or down? adaptive rounding for post-training quantization. In International Conference on Machine Learning, pages 7197–7206. PMLR.

Markus Nagel, Mart van Baalen, Tijmen Blankevoort, and Max Welling. 2019. Data-free quantization through weight equalization and bias correction.

OpenAI. 2023. Gpt-4 technical report. ArXiv, abs/2303.08774.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551.

Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jé- gou. 2021. Training data-efficient image transformers & distillation through attention. In International conference on machine learning, pages 10347–10357. PMLR.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019. Glue: A multi-task benchmark and analysis platform for natural language understanding.

Xiuying Wei, Ruihao Gong, Yuhang Li, Xianglong Liu and Fengwei Yu. 2022. QDrop: Randomly dropping quantization for extremely low-bit post-training quantization. In International Conference on Learning Representations.

Di Wu, Qi Tang, Yongle Zhao, Ming Zhang, Ying Fu, and Debing Zhang. 2020. Easyquant: Post-training quantization via scale optimization.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Julien Demouth, and Song Han. 2022. Smoothquant: Accurate and efficient post-training quantization for large language models. arXiv preprint arXiv:2211.10438.

Zhihang Yuan, Chenhao Xue, Yiqi Chen, Qiang Wu, and Guangyu Sun. 2022. Ptq4vit: Post-training quantization for vision transformers with twin uniform quantization. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XII, pages 191–207. Springer.

Yijia Zhang, Lingran Zhao, Shijie Cao, Wenqiang Wang, Ting Cao, Fan Yang, Mao Yang, Shanghang Zhang, and Ningyi Xu. 2023. Integer or floating point? new outlooks for low-bit quantization on large language models.

## A Hessian-Based Loss Metric

The objective of post-training quantization is to minimize the perturbation $( \delta \mathbf { X } = \mathbf { X } _ { \mathrm { F P } } - \mathbf { X } _ { \mathrm { R } } )$ introduced by quantization to the pre-trained realvalued network:

$$
\operatorname* { m i n } \mathbb { E } [ \mathcal { L } ( \mathbf { X } _ { \mathrm { R } } + \delta \mathbf { X } ) - \mathcal { L } ( \mathbf { X } _ { \mathrm { R } } ) ]\tag{20}
$$

Following the Taylor series expansion, we have

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } ( \mathbf { X } _ { \mathrm { R } } + \delta \mathbf { X } ) - \mathcal { L } ( \mathbf { X } _ { \mathrm { R } } ) ] } \\ & { \approx \delta \mathbf { X } ^ { T } \bar { \mathbf { g } } ^ { ( \mathbf { X } ) } + \frac { 1 } { 2 } \delta \mathbf { X } ^ { T } \bar { \mathbf { H } } ^ { ( \mathbf { X } ) } \delta \mathbf { X } } \\ & { \approx \frac { 1 } { 2 } \delta \mathbf { X } ^ { T } \bar { \mathbf { H } } ^ { ( \mathbf { X } ) } \delta \mathbf { X } } \end{array}\tag{21}
$$

Here, $\bar { \bf g } ^ { ( { \bf X } ) }$ is the gradients and $\bar { \bf H } ^ { ( { \bf X } ) }$ is the Hessian matrix. Since the pre-trained model is wellconverged, we can assume that $\bar { \bf g } ^ { ( { \bf X } ) }$ has near zero value in every element, and thus term $\delta \mathbf { X } ^ { T } \bar { \mathbf { g } } ^ { ( \mathbf { X } ) }$ can be neglected.

The Hessian matrix $\bar { \bf H } ^ { ( { \bf X } ) }$ is computed as:

$$
\bar { \mathbf { H } } ^ { ( \mathbf { X } ) } = \mathbf { J } _ { \mathbf { O } } ^ { T } ( \mathbf { X } ) \bar { \mathbf { H } } ^ { ( \mathbf { O } ) } \mathbf { J } _ { \mathbf { O } } ( \mathbf { X } )\tag{22}
$$

where $\mathbf { J } _ { \mathbf { O } } ( \mathbf { X } )$ denotes the Jacobian matrix of the layer output O w.r.t X, and $\bar { \bf { H } } ^ { ( { \bf { O } } ) }$ is the Hessian matrix w.r.t O. We then substitute the above equation back to equation 21 :

$$
\begin{array} { r l } & { \delta \mathbf { X } ^ { T } \bar { \mathbf { H } } ^ { ( \mathbf { X } ) } \delta \mathbf { X } } \\ & { = ( \mathbf { J } _ { \mathbf { O } } ( \mathbf { X } ) \delta \mathbf { X } ) ^ { T } \bar { \mathbf { H } } ^ { ( \mathbf { O } ) } ( \mathbf { J } _ { \mathbf { O } } ( \mathbf { X } ) \delta \mathbf { X } ) } \\ & { \approx ( \hat { \mathbf { O } } - \mathbf { O } ) ^ { T } \bar { \mathbf { H } } ^ { ( \mathbf { O } ) } ( \hat { \mathbf { O } } - \mathbf { O } ) } \end{array}\tag{23}
$$

Here $\hat { \bf O }$ is the intermediate output of the quantized layer and O is the original layer output. Note that under the assumption that δX is relatively small (Li et al., 2021), we can approximate $( \hat { \mathbf O } - \mathbf O )$ as ${ \mathbf { J } } _ { \mathbf { O } } ( \mathbf { X } ) \delta \mathbf { X }$ using first-order Taylor expansion.

Nevertheless, the calculation of $\bar { \bf H } ^ { ( \bar { \bf O } ) }$ is still burdensome, therefore, we use the diagonal entries of the Fisher Information Matrix of O to substitute $\bar { \bf { H } } ^ { ( \bf { O } ) }$ following (Li et al., 2021; Yuan et al., 2022), and the new Hessian-based metric becomes:

$$
\mathbb { E } [ ( \hat { \mathbf { O } } - \mathbf { O } ) ^ { T } d i a g ( ( \frac { \partial L } { \partial \mathbf { O } _ { 1 } } ) ^ { 2 } , . . . , ( \frac { \partial L } { \partial \mathbf { O } _ { n } } ) ^ { 2 } ( \hat { \mathbf { O } } - \mathbf { O } ) ]\tag{24}
$$

Here, each entry of O is assumed to be independent and n denoted the total number of elements in O. In this study, this hessian-based metric is used as the reconstruction metric to search for the optimal FP quantization function for both the weight and activation when performing layer-wise reconstruction in BERT and Vision Transformer models.

## B Quantization Error of Different Floating-Point Formats

Figure 4 compares the quantization error of different formats in 8-bit quantization, including INT8, E2M5, E3M4, E4M3, and E5M2. We apply these formats to different BERT modules in the first, fifth, and last layers. The figures demonstrate that the optimal FP formats differs depending on the specific module that we are quantizing.

## C Inter-Channel Variance Visualization

Figure 5 and 6 depict the output of different fullyconnected layers in BERT for the MNLI task, DeiT-S for the ImageNet-1K task, and LLaMA-7B for the zero-shot reasoning task. The visualizations reveal a noticeable inter-channel variance presented in both language and vision transformers.

## D Efficient Matrix Multiplication

Figure 7 displays a comprehensive list of all the granularity options that allow for efficient matrix multiplication. While per-token quantization theoretically provides greater precision in terms of quantization granularity, the accuracy gains achieved through this method are minimal and do not justify the additional computational overhead required. As a result, we have opted to use pertensor quantization when quantizing activations.

## E Learning Format and Maximum Value

We compare the previous gradient-based method (Kuzmin et al., 2022) with the proposed search-based method for finding the optimal format and maximum value. On DeiT-S, the learnable method only achieves 74.38% accuracy for an 8-bit quantized model on ImageNet, in contrast, FPQ can attain an almost loss-less result of 79.88%. We analyze the gradients for the number of exponent bits e derived in (Kuzmin et al., 2022) and observe that each time the exponent bits change, the gradients experience exponential variations, leading to high instability. Based on this observation, we assert that employing a search-based method to determine the optimal formats is crucial in post-training quantization (PTQ).

## F Reconstruction Choices

The previous works on integer post-training quantization involves breaking down the target model into sub-modules and reconstructing them separately (Nagel et al., 2020; Li et al., 2021; Bai et al., 2022; Yuan et al., 2022). This addresses the problem of over-fitting, given that only a limited amount of unlabeled calibration data is available. In this study we find the layer-wise reconstruction and parallel quantization works best for floating-point PTQ:

Layer Reconstruction: Recent research (Li et al., 2021; Bai et al., 2022) suggests increasing the reconstruction granularity from layer reconstruction (Nagel et al., 2020) to block reconstruction (Li et al., 2021) or even larger granularity (Lee et al., 2023). This is achieved by jointly optimizing all the linear layers or matrix multiplication components within each module to prevent the propagation of reconstruction errors among the layers. Despite this, we have observed that increasing the reconstruction granularity does not improve the accuracy of FPQ baseline or sometimes even lead to worse results. Therefore, we choose layer reconstruction.

Parallel Quantization: Sequential quantization is the most commonly used approach (Wu et al., 2020; Nagel et al., 2020; Li et al., 2021) where modules are quantized consecutively based on their sequential order, and the input for the current calibrating module is generated using all the previously quantized modules. However, some recent works (Yuan et al., 2022; Bai et al., 2022) proposed a new parallel quantization framework. This framework uses the raw output of the full-precision modules as input and makes the calibration of each module independent from one another. In this work, we use parallel quantization, as it yields better results than its sequential counterparts.

![](images/d67b038ba1850753e2408aba6b59767d081eeac7d9444ce03f17496c78077d05.jpg)

![](images/989a5c7bbf7dd9c89e3f763b234704f69c0ec8f1fb075c4b16377ec27869861e.jpg)

![](images/700da8f84383162de625caf44dc465f88f949ae7bb9d857a9a5e44cd76be9c96.jpg)  
Figure 4: Quantization error of different formats for BERT layers.

Bert  
![](images/c3b0c50f69635d3511bd848772c8ff53e4c3fbc5fe2b1d8ff14eb8d5ce30fac8.jpg)

![](images/ade3dd453c5f4598e4825a9cfd0b6fcd09e776b75212b351da9e0329d9702478.jpg)  
(a)

![](images/8d231923638dfdda478e365913fe17ccd6b1e5b460c3ade41a4d1f86e830f627.jpg)

![](images/20d6446548ef918546cfefb3833c0e38989c8335e6cde8d8a91bdc3817a8ef7a.jpg)

![](images/138d226e0fde756e7cc83663c92cf2756c72fccb5d00528dfa97c020639d4145.jpg)  
(b)

![](images/059cb96bc0345a5d92cc44744024532df4bd823be3a6f2cfcd3b1c289c054562.jpg)

layer.0.intermediate.denselayer.5.intermediate.denselayer.11.intermediate.dense  
![](images/fe9189021006d6fdcea13bdb9a90b0d459750326bef90867f1e039d17f3f5e62.jpg)

![](images/575a4cc746ccaa8fa47437fcf3c6749e0d9fbb4122611c48cb0d1b34e4e87fd2.jpg)  
(c)

![](images/79abb7e833a3168cca53f802d8d72e40e0755a8ef453981eba7c67b834c8b3e7.jpg)

![](images/3d4fc1199089c3f917f462a64d6b7b7f9a1c1c5b86990090eef8db3d95d72ccb.jpg)

![](images/832fd28859a58bc1daaacf3a95ac8deb0bb3cc19880648cfd878d79038271835.jpg)  
(d)

![](images/41b4c6febd35b2891e2fa34b679eced99743d8b9ce9d9cd6da069dd70b87a604.jpg)

![](images/7865c76de3712d6d601f3f5f1e32596220a413c5dd3f5cbebc1297fd12675c44.jpg)

![](images/8f94bd4a3242925dd0770001f8824e69557961c384e54b083165d27f609532e1.jpg)  
(e)

![](images/764a1e011acc738158b0738e7b55f8b6bf2d3d8c4161735cdaec01cdd2bb6a7b.jpg)

![](images/52aea57f82eb5a7dc072a0650d4e2bc327a97071ff49c135766663453894a646.jpg)

![](images/4330f7a71b2f2704fb2fb98ff4be152c3d0d2f264abec62cfebba15587c4e31c.jpg)  
(f)

![](images/5f07e9b85a22f8fb7a96d0cdb3aa7d39fb8315af965608fc14e233ae4999ab0f.jpg)  
Figure 5: Magnitude of the output activations of different modules in BERT (left column), and DeiT-S (right column).

Per-token  
layer.0.self\_attn.q\_proj  
![](images/176c762138e1fe85a54f2e24738822ebe33087dcd9f29885317e3d199b39c1a7.jpg)

LLaMa-7B  
layer.5.self\_attn.q\_proj  
![](images/c89588df3c7ea8db33bdc29f530e947cd7f24fefd24fa10f0825113bbedb39f1.jpg)  
(a)

layer.11.self\_attn.q\_proj  
![](images/53317bb75e66d7fe6c0746d7d6afbd8b01769c22b351da7b303cba25c24ec087.jpg)

layer.0.mlp.down\_proj  
![](images/2d91b2e1267c234648c181ccf9c469dd714eb97214f8cf4087afaf84d17f7ee8.jpg)

layer.5.mlp.down\_proj  
![](images/1666407ea79b765090be6a8a17fb1395d2645df7618920594dba5fc98e875d5d.jpg)  
(b)

layer.11.mlp.down\_proj  
![](images/ff8e9bd23abcc43d70b4d6e43a30d404b89ce3b58833f0a0f67919fafafa0eaa.jpg)

layer.0.self\_attn.o\_proj  
![](images/29f81108aa0195de5ace7867fefbf13c5e50ac4bb577ba20e0afe4430e28d6d6.jpg)

layer.5.self\_attn.o\_proj  
![](images/16f30a7a9c144bcf8f917d131844d58579b11056c1af9f7d3dd50209c3af86c9.jpg)  
(c)

layer.11.self\_attn.o\_proj  
![](images/4bf6b94e4929ed90bd92ac997e6405f8943b0ab9025b8e1b9661e0d59d837b04.jpg)  
Figure 6: Magnitude of the output activations of different modules in LLaMA-7B.

![](images/d24c774e2a17e5b9ad3848a3039afaf9d82c133ec384e9964b10d0aab5234169.jpg)

![](images/3636bf3c1b72c4201950c3ee762401d6d09537900877b0bd2ba315102aaffc85.jpg)

![](images/e39088418f720658fde40c743b4c22b4420f9c9b75c04046baa2d3b9930c0b79.jpg)

![](images/892e7f5b8ece84f1fbd38c840140049a9eae73b1e1df157bc927e804d002a20a.jpg)  
Figure 7: Quantization granularity options that support efficient matrix multiplication. The dimensions that share the same scaling factor are indicated with red dotted frames