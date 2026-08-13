# Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis

Haoyu Zhang<sup>1,2</sup>,Yu Wang<sup>2</sup>,Guanghao Yin<sup>2</sup>,Kejun Liu<sup>2</sup>,Yuanyuan Liu<sup>2,3</sup>,Tianshu Yu<sup>1,4</sup>∗

<sup>1</sup>The Chinese University of Hong Kong, Shenzhen

<sup>2</sup>China University of Geosciences, Wuhan

<sup>3</sup>Nanyang Technological University

4 <sup>4</sup> Shenzhen Institute of Artificial Intelligence and Robotics for Society <sup>1,4</sup>{zhanghaoyu, yutianshu}@cuhk.edu.cn

<sup>2</sup>{zhanghaoyu, vvy190701, ygh2, liukejun, liuyy}@cug.edu.cn <sup>3</sup>scse-yyliu@ntu.edu.sg

## Abstract

Though Multimodal Sentiment Analysis (MSA) proves effective by utilizing rich information from multiple sources (e.g., language, video, and audio), the potential sentiment-irrelevant and conflicting information across modalities may hinder the performance from being further improved. To alleviate this, we present Adaptive Language-guided Multimodal Transformer (ALMT), which incorporates an Adaptive Hyper-modality Learning (AHL) module to learn an irrelevance/conflict-suppressing representation from visual and audio features under the guidance of language features at different scales. With the obtained hyper modality representation, the model can obtain a complementary and joint representation through multimodal fusion for effective MSA. In practice, ALMT achieves state-of-the-art performance on several popular datasets (e.g., MOSI, MOSEI and CH-SIMS) and an abundance of ablation demonstrates the validity and necessity of our irrelevance/conflict suppression mechanism.

## 1 Introduction

Multimodal Sentiment Analysis (MSA) focuses on recognizing the sentiment attitude of humans from various types of data, such as video, audio, and language. It plays a central role in several applications, such as healthcare and human-computer interaction (Jiang et al., 2020; Qian et al., 2019). Compared with unimodal methods, MSA methods are generally more robust by exploiting and exploring the relationships between different modalities, showing significant advantages in improving the understanding of human sentiment.

Most recent MSA methods can be grouped into two categories: representation learning-centered methods (Hazarika et al., 2020; Yang et al., 2022;

![](images/26bf2702950098b847dae1a9eb60d22af68981b196c536933e6720bba8d879bc.jpg)  
Figure 1: In multimodal sentiment analysis, language modality usually is a dominant modality in all modalities, while audio and visual modalities not contributing as much to performance as language modality.

Yu et al., 2021; Han et al., 2021; Guo et al., 2022) and multimodal fusion-centered methods (Zadeh et al., 2017; Liu et al., 2018; Tsai et al., 2019a; Huang et al., 2020). The representation learningcentered methods mainly focus on learning refined modality semantics that contains rich and varied human sentiment clues, which can further improve the efficiency of multimodal fusion for relationship modelling. On the other hand, the multimodal fusion-centered methods mainly focus on directly designing sophisticated fusion mechanisms to obtain a joint representation of multimodal data. In addition, some works and corresponding ablation studies (Hazarika et al., 2020; Rahman et al., 2020; Guo et al., 2022) further imply that various modalities contribute differently to recognition, where language modality stands out as the dominant one. We note, however, information from different modalities may be ambiguous and conflicting due to sentiment-irrelevance, especially from non-dominating modalities (e.g., lighting and head pose in video and background noise in audio). Such disruptive information can greatly limit the performance of MSA methods. We have observed this phenomenon in several datasets (see Section 4.5.1) and an illustration is in Figure 1. To the best of our knowledge, there has never been prior work explicitly and actively taking this factor into account.

Motivated by the above observation, we propose a novel Adaptive Language-guided Multimodal Transformer (ALMT) to improve the performance of MSA by addressing the adverse effects of disruptive information in visual and audio modalities. In ALMT, each modality is first transformed into a unified form by using a Transformer with initialized tokens. This operation not only suppresses the redundant information across modalities, but also compresses the length of long sequences to facilitate efficient model computation. Then, we introduce an Adaptive Hyper-modality Learning (AHL) module that uses different scales of language features with dominance to guide the visual and audio modalities to produce the intermediate hyper-modality token, which contains less sentiment-irrelevant information. Finally, we apply a cross-modality fusion Transformer with language features serving as query and hyper-modality features serving as key and value. In this sense, the complementary relations between language and visual and audio modalities are implicitly reasoned, achieving robust and accurate sentiment predictions. In summary, the major contributions of our work can be summarized as:

• We present a novel multimodal sentiment analysis method, namely Adaptive Languageguided Multimodal Transformer (ALMT), which for the first time explicitly tackles the adverse effects of redundant and conflicting information in auxiliary modalities (i.e., visual and audio modalities), achieving a more robust sentiment understanding performance.

• We devise a novel Adaptive Hyper-modality Learning (AHL) module for representation learning. The AHL uses different scales of language features to guide the visual and audio modalities to form a hyper modality that complements the language modality.

• ALMT achieves state-of-the-art performance in several public and widely adopted datasets. We further provide in-depth analysis with rich empirical results to demonstrate the validity and necessity of the proposed approach.

## 2 Related Work

In this part, we briefly review previous work from two perspectives: multimodal sentiment analysis and Transformers.

## 2.1 Multimodal Sentiment Analysis

As mentioned in the section above, most previous MSA methods are mainly classified into two categories: representation learning-centered methods and multimodal fusion-centered methods.

For representation learning-centered methods, Hazarika et al. (2020) and Yang et al. (2022) argued representation learning of multiple modalities as a domain adaptation task. They respectively used metric learning and adversarial learning to learn the modality-invariant and modality-specific subspaces for multimodal fusion, achieving advanced performance in several popular datasets. Han et al. (2021) proposed a framework named MMIM that improves multimodal fusion with hierarchical mutual information maximization. Rahman et al. (2020) and Guo et al. (2022) devised different architectures to enhance language representation by incorporating multimodal interactions between language and non-verbal behavior information. However, these methods do not pay enough attention to sentiment-irrelevant redundant information that is more likely to be present in visual and audio modalities, which limits the performance of MSA.

For multimodal fusion-centered methods, Zadeh et al. (2017) proposed a fusion method (TFN) using a tensor fusion network to model the relationships between different modalities by computing the cartesian product. Tsai et al. (2019a) and Huang et al. (2020) introduced a multimodal Transformer to align the sequences and model long-range dependencies between elements across modalities. However, these methods directly fuse information from uni-modalities, which is more accessible to the introduction of sentiment-irrelevant information, thus obtaining sub-optimal results.

## 2.2 Transformer

Transformer is an attention-based building block for machine translation introduced by Vaswani et al. (2017). It learns the relationships between tokens by aggregating data from the entire sequence, showing an excellent modeling ability in various tasks, such as natural language processing, speech processing, and computer vision, etc. (Kenton and

![](images/a0cecac721e8ca3eb25cbf44635caf8ae0174cbcff23dae4a641648420e9f5b3.jpg)  
Figure 2: Processing pipeline of the proposed ALMT for multimodal sentiment analysis (MSA). With the multimodal input, we first apply three Transformer layers to embed modality features with low redundancy. Then, we employ a Hyper-modality Learning (AHL) module to learn a hyper-modality representation from visual and audio modalities under the guidance of language features at different scales. Finally, a Cross-modality Fusion Transformer is applied to incorporate hyper-modality features based on their relations to the language features, thus obtaining a complementary and joint representation for MSA.

Toutanova, 2019; Carion et al., 2020; Chen et al., 2022; Liu et al., 2023a). In MSA, this technique has been widely used for feature extraction, representation learning, and multimodal fusion (Tsai et al., 2019a; Huang et al., 2020; Liu et al., 2023b; Yuan et al., 2021).

## 3 Method

## 3.1 Overview

The overall processing pipeline of the proposed Adaptive Language-guided Multimodal Transformer (ALMT) for robust multimodal sentiment analysis is in Figure 2. As shown, ALMT first extracts unified modality features from the input. Then, Adaptive Hyper-Modality Learning (AHL) module is employed to learn the adaptive hypermodality representation with the guidance of language features at different scales. Finally, we apply a Cross-modality Fusion Transformer to synthesize the hyper-modality features with language features as anchors, thus obtaining a language-guided hypermodality network for MSA.

## 3.2 Multimodal Input

Regarding the multimodal input, each sample consists of language (l), audio (a), and visual (v) sources. Referring to previous works, we first obtain pre-computed sequences calculated by BERT (Kenton and Toutanova, 2019), Librosa (McFee et al., 2015), and OpenFace (Baltrusaitis et al., 2018), respectively. Then, we denote these sequence inputs as $U _ { m } ~ \in ~ \mathbb { R } ^ { T _ { m } \times d _ { m } }$ , where $m \in$ $\{ l , v , a \} , T _ { m }$ is the sequence length and $d _ { m }$ is the vector dimension of each modality. In practice, $T _ { m }$ and $d _ { m }$ are different on different datasets. For example, on the MOSI dataset, $T _ { v } , T _ { a } , T _ { l } , d _ { a } , d _ { v }$ and $d _ { l }$ are 50, 50, 50, 5, 20, and 768, respectively.

## 3.3 Modality Embedding

With multimodal input $U _ { m }$ , we introduce three Transformer layers to unify features of each modality, respectively. More specifically, we randomly initialize a low-dimensional token $H _ { m } ^ { 0 } \in \mathbb { R } ^ { T \times d _ { m } }$ for each modality and use the Transformer to embed the essential modality information to these to-

![](images/a52f01c40ad1f7e0ebc30c5508a853517f2c7ed1bfc3f2aeca6f40fa4d828582.jpg)  
Figure 3: An example of the Adaptive Hyper-modality Learning (AHL) Layer.

kens :

$$
H _ { m } ^ { 1 } = \mathrm { E } _ { \mathrm { m } } ^ { 0 } ( \mathrm { c o n c a t } ( H _ { m } ^ { 0 } , U _ { m } ) , \theta _ { E _ { m } ^ { 0 } } ) \in \mathbb { R } ^ { T \times d }\tag{1}
$$

where $H _ { m } ^ { 1 }$ is the unified feature of each modality m with a size of $T \times d , \mathrm { E _ { m } ^ { 0 } }$ and $\theta _ { E _ { m } ^ { 0 } }$ respectively represent the modality feature extractor and corresponding parameters, concat( ) represent the concatenation operation.

In practice, T and d are set to 8 and 128, respectively. The structure of the transformer layer is designed as the same as the Vision Transformer (VIT) (Dosovitskiy et al., 2021) with a depth setting of 1. Moreover, it is worth noting that transferring the essential modality information to initialized lowdimensional tokens is beneficial to decrease the redundant information that is irrelevant to human sentiment, thus achieving higher efficiency with lesser parameters.

## 3.4 Adaptive Hyper-modality Learning

After modality embedding, we further employ an Adaptive Hyper-modality Learning (AHL) module to learn a refined hyper-modality representation that contains relevance/conflict-suppressing information and highly complements language features. The AHL module consists of two Transformer layers and three AHL layers, which aim to learn language features at different scales and adaptively learn a hyper-modality feature from visual and audio modalities under the guidance of language features. In practice, we found that the language features significantly impact the modeling of hyper-modality (with more details in section 4.5.4).

## 3.4.1 Construction of Two-scale Language Features

We define the feature $H _ { l } ^ { 1 }$ as low-scale language feature. With the feature, we introduce two Transformer layers to learn language features at middlescale and high-scale (i.e. $H _ { l } ^ { \bar { 2 } }$ and $H _ { l } ^ { 3 } )$ . Different from the Transformer layer in the modality embedding stage that transfers essential information to an initialized token, layers in this stage directly model the language features:

$$
H _ { l } ^ { i } = { \mathrm { E } } _ { l } ^ { i } ( H _ { l } ^ { i - 1 } , \theta _ { E _ { l } ^ { i } } ) \in \mathbb { R } ^ { T \times d }\tag{2}
$$

where $i \in \{ 2 , 3 \} , H _ { l } ^ { i }$ is language features at different scales with a size of $T \times d , \mathrm { E } _ { l } ^ { i }$ and $\theta _ { E _ { I } ^ { i } }$ represents the i-th Transformer layer for language features learning and corresponding parameters. In practice, we used 8-head attention to model the information of each modality.

## 3.4.2 Adaptive Hyper-modality Learning Layer

With the language features of different scales $H _ { l } ^ { i }$ , we first initialize a hyper-modality feature $\dot { H _ { h y p e r } ^ { 0 } } \in \mathbb { R } ^ { T \times d }$ , then update $H _ { h y p e r } ^ { 0 }$ by calculating the relationship between obtained language features and two remaining modalities using multihead attention (Vaswani et al., 2017). As shown in Figure 3, using the extracted $H _ { l } ^ { i }$ as query and $H _ { a } ^ { 1 }$ as key, we can obtain the similarity matrix α between language features and audio features :

$$
\begin{array} { l } { \displaystyle \alpha = \mathrm { s o f t m a x } ( \frac { Q _ { l } K _ { a } ^ { T } } { \sqrt { d _ { k } } } ) } \\ { \displaystyle \ = \mathrm { s o f t m a x } ( \frac { H _ { l } ^ { i } W _ { Q _ { l } } W _ { K _ { a } } ^ { T } H _ { a } ^ { 1 T } } { \sqrt { d _ { k } } } ) \in \mathbb { R } ^ { T \times T } } \end{array}\tag{3}
$$

where softmax represents weight normalization operation, $W _ { Q _ { l } } \in \mathbb { R } ^ { d \times d _ { k } }$ and $W _ { K _ { a } } \in \mathbb { R } ^ { d \times d _ { k } }$ are learnable parameters, $d _ { k }$ is the dimension of each attention head. In practice, we used 8-head attention and set $d _ { k }$ to 16.

Similar to $\alpha , \beta$ represents the similarity matrix between language modality and visual modality:

$$
\begin{array} { r l } { \beta } & { = \mathrm { s o f t m a x } ( \frac { Q _ { l } K _ { v } ^ { T } } { \sqrt { d _ { k } } } ) } \\ & { = \mathrm { s o f t m a x } ( \frac { H _ { l } ^ { i } W _ { Q _ { l } } W _ { K _ { v } } ^ { T } H _ { v } ^ { 1 T } } { \sqrt { d _ { k } } } ) \in \mathbb { R } ^ { T \times T } } \end{array}\tag{4}
$$

where $W _ { K _ { v } } \in \mathbb { R } ^ { d \times d _ { k } }$ is learnable.

Then the hyper-modality features $H _ { h y p e r } ^ { j }$ can be updated by weighted audio features and weighted visual features as:

$$
\begin{array} { r l } & { H _ { h y p e r } ^ { j } = H _ { h y p e r } ^ { j - 1 } + \alpha V _ { a } + \beta V _ { v } } \\ & { \qquad = H _ { h y p e r } ^ { j - 1 } + \alpha H _ { a } ^ { 1 } W _ { V _ { a } } + \beta H _ { v } ^ { 1 } W _ { V _ { v } } } \end{array}\tag{5}
$$

where $j \in \{ 1 , 2 , 3 \}$ and $H _ { h y p e r } ^ { j } \in \mathbb { R } ^ { T \times d }$ respectively represent the j-th AHL layer and corresponding output hyper-modality features, $W _ { V _ { a } } \in \mathbb { R } ^ { d \times d _ { k } }$ and $W _ { V _ { v } } \in \mathbb { R } ^ { d \times d _ { k } }$ are learnable parameters.

## 3.5 Multimodal Fusion and Output

In the Multimodal Fusion, we first obtain a new language feature $H _ { l }$ and $H _ { h y p e r }$ and a new hypermodality feature by respectively concatenating initialized a token $\dot { H _ { 0 } } \in \mathbb { R } ^ { 1 \times d }$ with $H _ { h y p e r } ^ { 3 }$ and $H _ { l } ^ { 3 }$ Then we apply Cross-modality Fusion Transformer to transfer the essential joint and complementary information to these tokens. In practice, the Crossmodality Fusion Transformer fuse the language features $H _ { l }$ (serving as the query) and hyper-modality features $H _ { h y p e r }$ (serving as the key and value), thus obtaining a joint multimodal representation $H \in \mathbb { R } ^ { 1 \times d }$ for final sentiment analysis. We denote the Cross-modality Fusion Transformer as

CrossTrans, so the fusion process can be written as:

$$
H _ { l } = \operatorname { C o n c a t } ( H _ { 0 } , H _ { l } ^ { 3 } ) \in \mathbb { R } ^ { ( T + 1 ) \times d }\tag{6}
$$

$$
H _ { h y p e r } = \mathrm { C o n c a t } ( H _ { 0 } , H _ { h y p e r } ^ { 3 } ) \in \mathbb { R } ^ { ( T + 1 ) \times d }\tag{7}
$$

$$
H = \mathrm { C r o s s } \mathrm { T r a n s } ( H _ { l } , H _ { h y p e r } ) \in \mathbb { R } ^ { 1 \times d }\tag{8}
$$

After the multimodal fusion, we obtain the final sentiment analysis output $\hat { y }$ by applying a classifier on the outputs of Cross-modality Fusion Transformer $H .$ . In practice, we also used 8-head attention to model the relationships between language modality and hyper-modality. For more details of the Cross-modality Fusion Transformer, we refer readers to Tsai et al. (2019a).

## 3.6 Overall Learning Objectives

To summarize, our method only involves one learning objective, $i . e .$ , the sentiment analysis learning loss ${ \mathcal { L } } ,$ which is:

$$
\mathcal { L } = \frac { 1 } { N _ { b } } \sum _ { n = 0 } ^ { N _ { b } } \| y ^ { n } - \hat { y } ^ { n } \| _ { 2 } ^ { 2 }\tag{9}
$$

where $N _ { b }$ is the number of samples in the training set, $y ^ { n }$ is the sentiment label of the n-th sample. $\hat { y } ^ { n }$ is the prediction of our ALMT.

In addition, thanks to our simple optimization goal, compared with advanced methods (Hazarika et al., 2020; Yu et al., 2021) with multiple optimization goals, ALMT is much easier to train without tuning extra hyper-parameters. More details are shown in section 4.5.10.

## 4 Experiments

## 4.1 Datasets

We conducted extensive experiments on three popular trimodal datasets (i.e., MOSI (Zadeh et al., 2016), MOSEI (Zadeh et al., 2018), and CH-SIMS (Yu et al., 2020)).

MOSI. The dataset comprises 2,199 multimodal samples encompassing visual, audio, and language modalities. Specifically, the training set consists of 1,284 samples, the validation set contains 229 samples, and the test set encompasses 686 samples. Each individual sample is assigned a sentiment score ranging from -3 (indicating strongly negative) to 3 (indicating strongly positive).

MOSEI. The dataset comprises 22,856 video clips collected from YouTube with a diverse factors (e.g., spontaneous expressions, head poses, occlusions, illuminations). This dataset has been categorized into 16,326 training instances, 1,871 validation instances, and 4,659 test instances. Each instance is meticulously labeled with a sentiment score ranging from -3 to 3. And the sentiment scores from -3 to 3 indicate most negative to most positive.

Table 1: Comparison on MOSI and MOSEI. Note: the best result is highlighted in bold; \* represents the result is from Hazarika et al. (2020); represents the result is from Mao et al. (2022) and its corresponding GitHub page<sup>1</sup>.
<table><tr><td rowspan="2">Method</td><td colspan="6">MOSI</td><td colspan="6">MOSEI</td></tr><tr><td>Acc-7</td><td>Acc-5</td><td>Acc-2</td><td>F1</td><td>MAE</td><td>Corr</td><td>Acc-7</td><td>Acc-5</td><td>Acc-2</td><td>F1</td><td>MAE</td><td>Corr</td></tr><tr><td>TFN*</td><td>34.9</td><td></td><td>-/80.8</td><td>- /80.7</td><td>0.901</td><td>0.698</td><td>50.2</td><td></td><td>- / 82.5</td><td>- / 82.1</td><td>0.593</td><td>0.700</td></tr><tr><td>LMF*</td><td>33.2</td><td></td><td>- / 82.5</td><td>- / 82.4</td><td>0.917</td><td>0.695</td><td>48.0</td><td></td><td>- / 82.0</td><td>- / 82.1</td><td>0.623</td><td>0.677</td></tr><tr><td>MFM*</td><td>35.4</td><td></td><td>- / 81.7</td><td>-/81.6</td><td>0.877</td><td>0.706</td><td>51.3</td><td></td><td>- / 84.4</td><td>- / 84.3</td><td>0.568</td><td>0.717</td></tr><tr><td>MulT</td><td>40.0</td><td></td><td>- / 83.0</td><td>- / 82.8</td><td>0.871</td><td>0.698</td><td>51.8</td><td></td><td>- / 82.5</td><td>- / 82.3</td><td>0.580</td><td>0.703</td></tr><tr><td>MISA</td><td>42.3</td><td></td><td>81.8 / 83.4</td><td>81.7 / 83.6</td><td>0.783</td><td>0.761</td><td>52.2</td><td></td><td>83.6 / 85.5</td><td>83.8 / 85.3</td><td>0.555</td><td>0.756</td></tr><tr><td>PMR</td><td>40.6</td><td></td><td>-/83.6</td><td>-/83.4</td><td></td><td></td><td>52.5</td><td></td><td>-/ 83.3</td><td>- / 82.6</td><td></td><td></td></tr><tr><td>MAG-BERT</td><td>43.62</td><td></td><td>82.37 / 84.43</td><td>82.50 / 84.61</td><td>0.727</td><td>0.781</td><td>52.67</td><td></td><td>82.51 / 84.82</td><td>82.77 / 84.71</td><td>0.543</td><td>0.755</td></tr><tr><td>Self-MM</td><td>45.79</td><td></td><td>82.54 / 84.77</td><td>83.68 / 84.91</td><td>0.712</td><td>0.795</td><td>53.46</td><td></td><td>82.68 / 84.96</td><td>82.95 / 84.93</td><td>0.529</td><td>0.767</td></tr><tr><td>MMIM</td><td>46.65</td><td></td><td>84.14 / 86.06</td><td>84.00 / 85.98</td><td>0.700</td><td>0.800</td><td>54.24</td><td></td><td>82.24 / 85.97</td><td>82.66 / 85.94</td><td>0.526</td><td>0.772</td></tr><tr><td>FDMER</td><td>44.1</td><td></td><td>- /84.6</td><td>- /84.7</td><td>0.724</td><td>0.788</td><td>54.1</td><td></td><td>-/ 86.1</td><td>- / 85.8</td><td>0.536</td><td>0.773</td></tr><tr><td>CHFN</td><td>48.6</td><td></td><td>84.3 / 86.4</td><td>84.2 / 86.2</td><td>0.689</td><td>0.809</td><td>54.3</td><td></td><td>83.7 / 86.2</td><td>83.9 / 86.1</td><td>0.525</td><td>0.778</td></tr><tr><td>MulT†</td><td></td><td>42.68</td><td>-1-</td><td>-1-</td><td></td><td></td><td></td><td>54.18</td><td>-1-</td><td>-/-</td><td></td><td></td></tr><tr><td>MISA†</td><td></td><td>47.08</td><td>-1-</td><td>-1-</td><td></td><td></td><td></td><td>53.63</td><td>-1-</td><td>-1-</td><td></td><td></td></tr><tr><td>Self-MM†</td><td></td><td>53.47</td><td>-/-</td><td>-/-</td><td></td><td></td><td></td><td>55.53</td><td>-1-</td><td>-1-</td><td></td><td></td></tr><tr><td>ALMT</td><td>49.42</td><td>56.41</td><td>84.55 / 86.43</td><td>84.57 / 86.47</td><td>0.683</td><td>0.805</td><td>54.28</td><td>55.96</td><td>84.78 / 86.79</td><td>85.19 / 86.86</td><td>0.526</td><td>0.779</td></tr></table>

Table 2: Comparison results on CH-SIMS. Note: the best result is highlighted in bold; represents the result is from Mao et al. (2022) and its corresponding GitHub page<sup>1</sup>.
<table><tr><td>Method</td><td>Acc-5</td><td>Acc-3</td><td>Acc-2</td><td>F1</td><td>MAE</td><td>Corr</td></tr><tr><td>TFN†</td><td>39.30</td><td>65.12</td><td>78.38</td><td>78.62</td><td>0.432</td><td>0.591</td></tr><tr><td>LMF†</td><td>40.53</td><td>64.68</td><td>77.77</td><td>77.88</td><td>0.441</td><td>0.576</td></tr><tr><td>MFM†</td><td></td><td></td><td>75.06</td><td>75.58</td><td>0.477</td><td>0.525</td></tr><tr><td>MuLT†</td><td>37.94</td><td>64.77</td><td>78.56</td><td>79.66</td><td>0.453</td><td>0.564</td></tr><tr><td>MISA†</td><td></td><td></td><td>76.54</td><td>76.59</td><td>0.447</td><td>0.563</td></tr><tr><td>MAG-BERT†</td><td></td><td></td><td>74.44</td><td>71.75</td><td>0.492</td><td>0.399</td></tr><tr><td>Self-MM†</td><td>41.53</td><td>65.47</td><td>80.04</td><td>80.44</td><td>0.425</td><td>0.595</td></tr><tr><td>ALMT</td><td>45.73</td><td>68.93</td><td>81.19</td><td>81.57</td><td>0.404</td><td>0.619</td></tr></table>

CH-SIMS. It is a Chinese multimodal sentiment dataset that comprises 2,281 video clips collected from variuous sources, such as different movies and TV serials with spontaneous expressions, various head poses, etc. It is divided into 1,368 training samples, 456 validation samples, and 457 test samples. Each sample is manually annotated with a sentiment score from -1 (strongly negative) to 1 (strongly positive).

## 4.2 Evaluation Criteria

Following prior works (Yu et al., 2020), we used several evaluation metrics, i.e., binary classification accuracy (Acc-2), F1, three classification accuracy (Acc-3), five classification accuracy (Acc-5), seven classification accuracy (Acc-7), mean absolute error (MAE), and the correlation of the model’s prediction with human (Corr). Moreover, on MOSI and MOSEI, agreeing with prior works (Hazarika et al., 2020), we calculated Acc-2 and F1 in two ways: negative/non-negative and negative/positive on MOSI and MOSEI datasets, respectively.

## 4.3 Baselines

To comprehensively validate the performance of our ALMT, we make a fair comparison with the several advanced and state-of-the-art methods, i.e., TFN (Zadeh et al., 2017), LMF (Liu et al., 2018), MFM (Tsai et al., 2019b), MuLT (Tsai et al., 2019a), MISA (Hazarika et al., 2020), PMR (Lv et al., 2021), MAG-BERT (Rahman et al., 2020), Self-MM (Yu et al., 2021), MMIM (Han et al., 2021), FDMER (Yang et al., 2022) and CHFN (Guo et al., 2022).

## 4.4 Performance Comparison

Table 1 and Table 2 list the comparison results of our proposed method and state-of-the-art methods on the MOSI, MOSEI, and CH-SIMS, respectively.

As shown in the Table 1, the proposed ALMT obtained state-of-the-art performance in almost all metrics. On the task of more difficult and finegrained sentiment classification (Acc-7), our model achieves remarkable improvements. For example, on the MOSI dataset, ALMT achieved a relative improvement of 1.69% compared to the secondbest result obtained by CHFN. It demonstrates that eliminating the redundancy of auxiliary modalities is essential for effective MSA.

Moreover, it is worth noting that the scenarios in

SIMS are more complex than MOSI and MOSEI. Therefore, it is more challenging to model the multimodal data. However, as shown in the Table 2, ALMT achieved state-of-the-art performance in all metrics compared to the sub-optimal approach. For example, compared to Self-MM, it achieved relative improvements with 1.44% on Acc-2 and 1.40% on the corresponding F1, respectively. Achieving such superior performance on SIMS with more complex scenarios demonstrates ALMT’s ability to extract effective sentiment information from various scenarios.

## 4.5 Ablation Study and Analysis

## 4.5.1 Effects of Different Modalities

To better understand the influence of each modality in the proposed ALMT, Table 3 reports the ablation results of the subtraction of each modality to the ALMT on the MOSI and CH-SIMS datasets, respectively. It is shown that, if the AHL is removed based on the subtraction of each modality, the performance decreases significantly in all metrics. This phenomenon demonstrates that AHL is beneficial in reducing the sentiment-irrelevant redundancy of visual and audio modalities, thus improving the robustness of MSA.

In addition, we note that after removing the video and audio inputs, the performance of ALMT remains relatively high. Therefore, in the MSA task, we argue that eliminating the sentimentirrelevant information that appears in auxiliary modalities $( i . e . ,$ , visual and audio modalities) and improving the contribution of auxiliary modalities in performance should be paid more attention to.

Table 3: Effects of different modalities. Note: the best result is highlighted in bold.
<table><tr><td rowspan="2">Method</td><td colspan="2">MOSI</td><td colspan="2">CH-SIMS</td></tr><tr><td>Acc-7</td><td>MAE</td><td>Acc-5</td><td>MAE</td></tr><tr><td>ALMT</td><td>49.42</td><td>0.683</td><td>45.73</td><td>0.404</td></tr><tr><td>w/o Audio</td><td>48.69</td><td>0.705</td><td>45.08</td><td>0.416</td></tr><tr><td>w/o Video</td><td>47.96</td><td>0.704</td><td>44.64</td><td>0.403</td></tr><tr><td>w/o Audio &amp; AHL</td><td>46.91</td><td>0.724</td><td>43.54</td><td>0.407</td></tr><tr><td>w/o Video &amp; AHL</td><td>47.08</td><td>0.726</td><td>43.76</td><td>0.406</td></tr><tr><td>w/o Video &amp; Audio</td><td>46.79</td><td>0.752</td><td>40.26</td><td>0.405</td></tr></table>

## 4.5.2 Effects of Different Components

To verify the effectiveness of each component of our ALMT, in Table 4, we present the ablation result of the subtraction of each component on the MOSI and CH-SIMS datasets, respectively. We observe that deactivating the AHL (replaced with feature concatenation) greatly decreases the performance, demonstrating the language-guided hypermodality representation learning strategy is effective. Moreover, after the removal of the fusion Transformer and Modality Embedding, the performance drops again, also supporting that the fusion Transformer and Modality embedding can effectively improve the ALMT’s ability to explore the sentiment information in each modality.

Table 4: Effects of different components. Note: the best result is highlighted in bold.
<table><tr><td rowspan="2">Method</td><td colspan="2">MOSI</td><td colspan="2">CH-SIMS</td></tr><tr><td>Acc-7</td><td>MAE</td><td>Acc-5</td><td>MAE</td></tr><tr><td>ALMT</td><td>49.42</td><td>0.683</td><td>45.73</td><td>0.404</td></tr><tr><td>w/o AHL</td><td>34.40</td><td>0.952</td><td>38.29</td><td>0.444</td></tr><tr><td>w/o Fusion Transformer</td><td>48.69</td><td>0.703</td><td>43.76</td><td>0.410</td></tr><tr><td>w/o Modality Embedding</td><td>47.96</td><td>0.701</td><td>43.11</td><td>0.429</td></tr></table>

## 4.5.3 Effects of Different Query, Key, and Value Settings in Fusion Transformer

Table 5 presents the experimental results of different query, key, and value settings in Transformer on the MOSI and MOSEI datasets, respectively. We observed that ALMT can obtain better performance when aligning hyper-modality features to language features (i.e., using $H _ { l } ^ { 3 }$ as query and using $H _ { h y p e r } ^ { 3 }$ as key and value). We attribute this phenomenon to the fact that language information is relatively clean and can provide more sentimentrelevant information for effective MSA.

Table 5: Effect of different Query, Key, and Value settings in Fusion Transformer.
<table><tr><td rowspan="2">Q</td><td rowspan="2">K&amp;V</td><td colspan="2">MOSI</td><td colspan="2">CH-SIMS</td></tr><tr><td>Acc-7</td><td>MAE</td><td>Acc-5</td><td>MAE</td></tr><tr><td> $H _ { h y p e r } ^ { 3 }$ </td><td> $H _ { l } ^ { 3 }$ </td><td>48.10</td><td>0.707</td><td>44.64</td><td>0.410</td></tr><tr><td> $H _ { l } ^ { 3 }$ </td><td> $H _ { h y p e r } ^ { 3 }$ </td><td>49.42</td><td>0.683</td><td>45.73</td><td>0.404</td></tr></table>

## 4.5.4 Effects of the Guidance of Different Language Features in AHL

To discuss the effect of the guidance of different language features in AHL, we show the ablation result of different guidance settings on MOSI and CH-SIMS in Table 6. In practice, we replace the AHL layer that do not require language guidance with MLP layer. Obviously, we can see that the ALMT can obtain the best performance when all scals of language features $( i . e . , H _ { l } ^ { 1 } , H _ { l } ^ { 2 } , H _ { l } ^ { 3 } )$ involve the guidance of hyper-modality learning.

In addition, we found that the model is more difficult to converge when AHL is removed. It indicates that sentiment-irrelevant and conflicting information visual and audio modalities may limit the improvement of the model.

Table 6: Effects of different guidance of different language features in AHL. Note: the best result is highlighted in bold.
<table><tr><td rowspan="2"> $H _ { l } ^ { 1 }$ </td><td rowspan="2"> $H _ { l } ^ { 2 }$ </td><td rowspan="2"> $H _ { l } ^ { 3 }$ </td><td colspan="2">MOSI</td><td colspan="2">CH-SIMS</td></tr><tr><td>Acc-7</td><td>MAE</td><td>Acc-5</td><td>MAE</td></tr><tr><td rowspan="3"> $\checkmark$ </td><td rowspan="3">√</td><td></td><td>34.40</td><td>0.952</td><td>38.29</td><td>0.444</td></tr><tr><td></td><td>47.38</td><td>0.704</td><td>43.54</td><td>0.412</td></tr><tr><td></td><td>48.10</td><td>0.709</td><td>43.11</td><td>0.415</td></tr><tr><td rowspan="4">√ √</td><td></td><td>√</td><td>48.54</td><td>0.711</td><td>43.98</td><td>0.412</td></tr><tr><td>√</td><td></td><td>46.36</td><td>0.736</td><td>45.51</td><td>0.417</td></tr><tr><td></td><td>√</td><td>48.10</td><td>0.707</td><td>44.20</td><td>0.409</td></tr><tr><td>√</td><td>√</td><td>47.81</td><td>0.729</td><td>43.76</td><td>0.416</td></tr><tr><td>√</td><td>√</td><td>√</td><td>49.42</td><td>0.683</td><td>45.73</td><td>0.404</td></tr></table>

## 4.5.5 Effects of Different Fusion Techniques

To analyze the effects of different fusion techniques, we conducted some experiments, whose results are shown in the table 7. Obviously, on the MOSI dataset, the use of our Cross-modality Fusion Transformer to fuse language features and hyper-modality features is the most effective. On the CH-SIMS dataset, although TFN achieves better performance on the MAE metric, its Acc-5 is lower. Overall, using Transformer for feature fusion is an effective way.

Table 7: Effects of different fusion techniques.
<table><tr><td rowspan="2">Fusion Technique</td><td colspan="2">MOSI</td><td colspan="2">CH-SIMS</td></tr><tr><td>Acc-7</td><td>MAE</td><td>Acc-5</td><td>MAE</td></tr><tr><td>Concatenation</td><td>48.69</td><td>0.703</td><td>43.76</td><td>0.410</td></tr><tr><td>Addition</td><td>46.36</td><td>0.706</td><td>42.45</td><td>0.411</td></tr><tr><td>GRU</td><td>47.81</td><td>0.710</td><td>44.86</td><td>0.414</td></tr><tr><td>Tensor Fusion (TFN)</td><td>47.23</td><td>0.710</td><td>44.20</td><td>0.403</td></tr><tr><td>Low-rank Fusion (LMF)</td><td>46.65</td><td>0.715</td><td>45.08</td><td>0.408</td></tr><tr><td>Ours</td><td>49.42</td><td>0.683</td><td>45.73</td><td>0.404</td></tr></table>

## 4.5.6 Analysis on Model Complexity

As shown in Table 8, we compare the parameters of ALMT with other state-of-the-art Transformer-based methods. Due to the different hyper-parameter configurations for each dataset may lead to a slight difference in the number of parameters calculated. We calculated the model parameters under the hyper-parameter settings on the MOSI. Obviously, our ALMT obtains the best performance (Acc-7 of 49.42 %) with a second computational cost (2.50M). It shows that ALMT achieves a better trade-off between accuracy and computational burden.

Table 8: Analysis on model complexity. Note: the parameter of other Transformer-based methods was calculated by authors from open source code with default hyper-parameters on MOSI.
<table><tr><td>Method</td><td>Parameter</td><td>Acc-7 on MOSI</td></tr><tr><td>MuLT</td><td>2.57M</td><td>40.00</td></tr><tr><td>MISA</td><td>3.10M</td><td>42.3</td></tr><tr><td>MAG-BERT</td><td>1.22M</td><td>43.62</td></tr><tr><td>ALMT</td><td>2.50M</td><td>49.42</td></tr></table>

## 4.5.7 Visualization of Attention in AHL

In Figure 4, we present the average attention matrix $( i . e . ,$ α and $\beta )$ on CH-SIMS. As shown, ALMT pays more attention to the visual modality, indicating that the visual modality provides more complementary information than the audio modality. In addition, from Table 3, compared to removing audio input, the performance of ALMT decreases more obviously when the video input is removed. It also demonstrates that visual modality may provide more complementary information.

![](images/e9c0a955586fd55ef50fc3ad6d9c69a0b2c3a7bf91f68efd8b0aeec9cb774a07.jpg)  
(a)

![](images/54c9b8a7ef7fed15d9cb8fff0f80afcad40407667d9a8e0917f55fdfa4722b42.jpg)  
(b)  
Figure 4: Visualization of average attention weights from the last AHL layer on CH-SIMS dataset. (a) Average attention matrix α between language and audio modalities; (b) average attention matrix $\beta$ between language and visual modalities. Note: darker colors indicate higher attention weights for learning.

## 4.5.8 Visualization of Robustness of AHL

To test the AHL’s ability to perceive sentimentirrelevant information, as shown in Figure 5, we visualize the attention weights (β) of the last AHL layer between language features $( H _ { l } ^ { 3 } )$ and visual features $( H _ { v } ^ { 1 } )$ on CH-SIMS. More specifically, we first randomly selected a sample from the test set. Then we added random noise to a peak frame (marked by the black dashed boxes) of $H _ { v } ^ { 1 }$ , and finally observed the change of attention weights between $H _ { l } ^ { 3 }$ and $H _ { v } ^ { 1 }$ . It is seen that when the random noise is added to the peak frame, the attention weights between language and the corresponding peak frame show a remarkable decrease. This phenomenon demonstrates that AHL can suppress sentiment-irrelevant information, thus obtaining a more robust hyper-modality representation for multimodal fusion.

![](images/59ed5f6eacc1f47f6eedd7216809bb527117c7a5d4d5d7aa4d1d35e2056284b0.jpg)

![](images/a888b87ebf19bb3e358c4fb35ffd0b1fb8df6d1700efb9565b4f3c528a904aba.jpg)  
Figure 5: Visualization of the attention weights between language and visual modalities learned by the AHL for a randomly selected sample with and without a random noise on the CH-SIMS dataset. (a) The attention weights without a random noise; (b) the attention weights with a random noise. Note: darker colors indicate higher attention weights for learning.

## 4.5.9 Visualization of Different Representations

In Figure 6, we visualized the hyper-modality representation $H _ { h y p e r } ^ { 3 }$ , visual representation $H _ { 1 } ^ { v }$ and audio representation $H _ { 1 } ^ { a }$ in a 3D feature space by using t-SNE (Van der Maaten and Hinton, 2008) on CH-SIMS. Obviously, there is a modality distribution gap existing between audio and visual features, as well as within their respective modalities. However, the hyper-modality representations learned from audio and visual features converge in the same distribution, indicating that the AHL can narrow the difference of inter-/intra modality distribution of audio and visual representations, thus reducing the difficulty of multimodal fusion.

## 4.5.10 Visualization of Convergence Performance

In Figure 7, we compared the convergence hehavior of ALMT with three state-of-the-art methods $( i . e .$ , MulT, MISA and Self-MM) on CH-SIMS. We choose the MAE curve for comparison as MAE indicates the model’s ability to predict fine-grained sentiment. Obviously, on the training set, although

![](images/efc15ad8181bedf5bab72dc080cb9ec36f69dc03490ae0001def9892876c74fa.jpg)  
Figure 6: Visualization of different representations in 3D space by using t-SNE.

Self-MM converges the fastest, its MAE of convergence is larger than ALMT at the end of the epoch. On the validation set, ALMT seems more stable compared to other methods, while the curves of other methods show relatively more dramatic fluctuations. It demonstrates that the ALMT is easier to train and has a better generalization capability.

![](images/0b4302af2608753e0990b531a963b5d59fb5480394860a9e98026a0d3fd091ac.jpg)  
(a) MAE on Training Set

![](images/7082e03d596660627a7689db7a7def201047f05bc19edf7a4fc3a517382d8e7c.jpg)  
(b) MAE on Validation Set  
Figure 7: Visualization of convergence performance on the train and validation sets of CH-SIMS. (a) The comparison of MAE curves on the training set; (b) the comparison of MAE curves on the validation set. Note: the results of other methods reproduced by authors from open source code with default hyper-parameters.

## 5 Conclusion

In this paper, a novel Adaptive Language-guided Multimodal Transformer (ALMT) is proposed to better model sentiment cues for robust Multimodal Sentiment Analysis (MSA). Due to effectively suppressing the adverse effects of redundant information in visual and audio modalities, the proposed method achieved highly improved performance on several popular datasets. We further present rich indepth studies investigating the reasons behind the effectiveness, which may potentially advise other researchers to better handle MSA-related tasks.

## Limitations

Our AMLT which is a Transformer-based model usually has a large number of parameters. It requires comprehensive training and thus can be subjected to the size of the training datasets. As current sentiment datasets are typically small in size, the performance of AMLT may be limited. For example, compared to classification metrics, such as Acc-7 and Acc-2, the more fine-grained regression metrics (i.e., MAE and Corr) may need more data for training, resulting in relatively small improvements compared to other advanced methods.

## References

Tadas Baltrusaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency. 2018. Openface 2.0: Facial behavior analysis toolkit. In 13th IEEE International Conference on Automatic Face & Gesture Recognition, FG 2018, Xi’an, China, May 15-19, 2018, pages 59–66. IEEE Computer Society.

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. 2020. End-to-end object detection with transformers. In Proceedings of the 16th European Conference on Computer Vision, volume 12346, pages 213–229.

Weidong Chen, Xiaofen Xing, Xiangmin Xu, Jianxin Pang, and Lan Du. 2022. Speechformer: A hierarchical efficient framework incorporating the characteristics of speech. In Proceedings ofthe 23rd Annual Conference ofthe International Speech Communication Association, pages 346–350.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR.

Jiwei Guo, Jiajia Tang, Weichen Dai, Yu Ding, and Wanzeng Kong. 2022. Dynamically adjust word representations using unaligned multimodal information. In Proceedings ofthe 30th ACM International Conference on Multimedia, MM ’22, page 3394–3402. Association for Computing Machinery.

Wei Han, Hui Chen, and Soujanya Poria. 2021. Improving multimodal fusion with hierarchical mutual information maximization for multimodal sentiment analysis. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 9180–9192, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Devamanyu Hazarika, Roger Zimmermann, and Soujanya Poria. 2020. MISA: modality-invariant and -specific representations for multimodal sentiment analysis. In Proceedings of the 28th ACM international conference on multimedia, pages 1122–1131. ACM.

Jian Huang, Jianhua Tao, Bin Liu, Zheng Lian, and Mingyue Niu. 2020. Multimodal transformer fusion for continuous emotion recognition. In 2020 IEEE International Conference on Acoustics, Speech and Signal Processing(ICASSP), pages 3507–3511. IEEE.

Yingying Jiang, Wei Li, M. Shamim Hossain, Min Chen, Abdulhameed Alelaiwi, and Muneer Al-Hammadi. 2020. A snapshot research and implementation of multimodal information fusion for data-driven emotion recognition. Information Fusion, 53:209–221.

Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT, pages 4171–4186.

Yuanyuan Liu, Wenbin Wang, Chuanxu Feng, Haoyu Zhang, Zhe Chen, and Yibing Zhan. 2023a. Expression snippet transformer for robust video-based facial expression recognition. Pattern Recognition, 138:109368.

Yuanyuan Liu, Haoyu Zhang, Yibing Zhan, Zijing Chen, Guanghao Yin, Lin Wei, and Zhe Chen. 2023b. Noise-resistant multimodal transformer for emotion recognition. arXiv preprint arXiv:2305.02814.

Zhun Liu, Ying Shen, Varun Bharadhwaj Lakshminarasimhan, Paul Pu Liang, AmirAli Bagher Zadeh, and Louis-Philippe Morency. 2018. Efficient lowrank multimodal fusion with modality-specific factors. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2247–2256, Melbourne, Australia. Association for Computational Linguistics.

Fengmao Lv, Xiang Chen, Yanyong Huang, Lixin Duan, and Guosheng Lin. 2021. Progressive modality reinforcement for human multimodal emotion recognition from unaligned multimodal sequences. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2554–2562. Computer Vision Foundation / IEEE.

Huisheng Mao, Ziqi Yuan, Hua Xu, Wenmeng Yu, Yihe Liu, and Kai Gao. 2022. M-sena: An integrated platform for multimodal sentiment analysis. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pages 204–213.

Brian McFee, Colin Raffel, Dawen Liang, Daniel P. W. Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. 2015. librosa: Audio and music signal analysis in python. In Proceedings of the 14th Python in Science Conference 2015 (SciPy 2015), Austin, Texas, July 6 - 12, 2015, pages 18–24. scipy.org.

Yongfeng Qian, Yin Zhang, Xiao Ma, Han Yu, and Limei Peng. 2019. EARS: emotion-aware recommender system based on hybrid information fusion. Information Fusion, 46:141–146.

Wasifur Rahman, Md Kamrul Hasan, Sangwu Lee, Amir Zadeh, Chengfeng Mao, Louis-Philippe Morency, and Ehsan Hoque. 2020. Integrating multimodal information in large pretrained transformers. In Proceedings of the conference. Association for Computational Linguistics. Meeting, volume 2020, page 2359. NIH Public Access.

Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, J. Zico Kolter, Louis-Philippe Morency, and Ruslan Salakhutdinov. 2019a. Multimodal transformer for unaligned multimodal language sequences. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 6558– 6569, Florence, Italy. Association for Computational Linguistics.

Yao-Hung Hubert Tsai, Paul Pu Liang, Amir Zadeh, Louis-Philippe Morency, and Ruslan Salakhutdinov. 2019b. Learning factorized multimodal representations. In ICLR.

Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-sne. Journal of machine learning research, 9(11).

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, pages 5998–6008.

Dingkang Yang, Shuai Huang, Haopeng Kuang, Yangtao Du, and Lihua Zhang. 2022. Disentangled representation learning for multimodal emotion recognition. In Proceedings ofthe 30th ACM International Conference on Multimedia, pages 1642–1651. ACM.

Wenmeng Yu, Hua Xu, Fanyang Meng, Yilin Zhu, Yixiao Ma, Jiele Wu, Jiyun Zou, and Kaicheng Yang. 2020. CH-SIMS: A chinese multimodal sentiment analysis dataset with fine-grained annotation of modality. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 3718–3727. Association for Computational Linguistics.

Wenmeng Yu, Hua Xu, Ziqi Yuan, and Jiele Wu. 2021. Learning modality-specific representations with selfsupervised multi-task learning for multimodal sentiment analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 10790–10797.

Ziqi Yuan, Wei Li, Hua Xu, and Wenmeng Yu. 2021. Transformer-based feature reconstruction network for robust multimodal sentiment analysis. In Proceedings of the 29th ACM International Conference on Multimedia, pages 4400–4407. ACM.

Amir Zadeh, Minghai Chen, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. 2017. Tensor fusion network for multimodal sentiment analysis. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 1103–1114, Copenhagen, Denmark. Association for Computational Linguistics.

Amir Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. 2018. Multimodal language analysis in the wild: CMU-MOSEI dataset and interpretable dynamic fusion graph. In Proceedings ofthe 56th Annual Meeting ofthe Association for Computational Linguistics, pages 2236–2246.

Amir Zadeh, Rowan Zellers, Eli Pincus, and Louis-Philippe Morency. 2016. Multimodal sentiment intensity analysis in videos: Facial gestures and verbal messages. IEEE Intelligent Systems, 31(6):82–88.

(a)

(b)

(a)

## A Hyper-parameters

In this section, we show the selection of some key hyper-parameters on the validation set of CH-SIMS.

## A.1 Overview

We used PyTorch to implement our method. The experiments were conducted on a PC with Intel(R) Xeon(R) 6240C CPU at 2.6GHz and 128GB memory and NVIDIA GeForce RTX 3090. The key parameters are shown in Table 9. We see that most hyper-parameters are the same across these datasets, demonstrating the ALMT does not require complex hyper-parameters adjustment.

Table 9: Hyper-parameters of ALMT we use on the different datasets
<table><tr><td></td><td>MOSI</td><td>MOSEI</td><td>CH-SIMS</td></tr><tr><td>Modality Feature Length T</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Vector Dimension d</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Modality Embedding Depth</td><td>1</td><td>1</td><td>1</td></tr><tr><td>AHL Depth</td><td>3</td><td>3</td><td>3</td></tr><tr><td>Fusion Transformer Depth</td><td>2</td><td>4</td><td>4</td></tr><tr><td>Batch Size</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Initial Learning Rate</td><td>1e-4</td><td>1e-4</td><td>1e-4</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Epochs</td><td>200</td><td>200</td><td>200</td></tr><tr><td>Warm Up</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Cosine Annealing</td><td>√</td><td>√</td><td>√</td></tr></table>

## A.2 Effects of Length Settings of Modality Feature

In Figure 8, we show the effect of the sequence length $T$ of the token $H _ { m } ^ { 0 }$ in modality embedding on the CH-SIMS dataset. It is observed that there are significant performance changes when the hyper-parameter is changed. And a similar phenomenon occurred on the MOSI and MOSEI datasets. Although the MAE is not the best when the Acc-5 is highest, e.g., T is set to 32. Considering that the model computation rises when the T increases, we set T to 8, which is beneficial for ALMT to obtain the best performance with a relatively lower computational cost.

## A.3 Effects of Depth Settings of AHL

Figure 9 presents the effect of AHL depth settings for MSA. Obviously, the ALMT achieves the best performance on the two most difficult evaluation metrics, i.e., Acc-5 and MAE. Hence, in this study, we set the depth of AHL to 3. Moreover, on the MOSI and MOSEI datasets, we set it to 3 as the similar phenomenon is also observed.

![](images/526a840e423a4a849c2c738d7b31164b58f37ef26a5201120161b4965538d2fe.jpg)

![](images/67788cb0ad21a2d682056e6e7bc3a3107e3707bf070429142a94d62fc958b327.jpg)

Figure 8: Effects of Token Length Settings in Modality Embedding. (a) Accuracy curve of different Token sequence lengths on the CH-SIMS; (b) MAE curve of different Token sequence lengths on the CH-SIMS  
![](images/e753513e41519924943e4c3858a7d6cda49ebb6abaab057835d2a3dbb29317f9.jpg)

![](images/d07b81041c9a790084c54f87e965cbeae43a85cca068cf50e4947755accd0a19.jpg)  
Figure 9: Effects of Depth Settings of AHL. (a) Accuracy curve of different AHL depth on the CH-SIMS; (b) MAE curve of different AHL depth on the CH-SIMS

## A.4 Effects of Depth Settings of Fusion Transformer

In Figure 10, we presents the effects of depth settings of cross-modality fusion Transformer on CH-SIMS. We observed that the ALMT can obtain the best result on Acc-5 and MAE when the depth is set to 3 and 5, respectively. However, to balance performance and model computation, we set the depth to 4 on the CH-SIMS. Following the similar rule, we set the depth of the cross-modality fusion Transformer to 2 and 4 on the MOSI and MOSEI, respectively.

![](images/89ed0af986d8ff4091f6ac1c8d6c8c127f1b926a3d50bd4156024f7ff689a71d.jpg)

![](images/e4a616e146be1b10e4e998f7e37978eda5d7719bbc5c7032618a04429101c65d.jpg)  
Figure 10: Effects of Depth Settings of Cross-modality Fusion Transformer. (a) Accuracy curve of different depth settings on the CH-SIMS; (b) MAE curve of different depth settings on the CH-SIMS