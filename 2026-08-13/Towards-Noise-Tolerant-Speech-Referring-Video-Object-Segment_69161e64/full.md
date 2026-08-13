# Towards Noise-Tolerant Speech-Referring Video Object Segmentation: Bridging Speech and Text

Xiang Li<sup>1</sup> Jinglu Wang<sup>2</sup> Xiaohao Xu<sup>3</sup> Muqiao Yang<sup>1</sup>

Fan Yang<sup>4</sup> Yizhou Zhao<sup>1</sup> Rita Singh<sup>1</sup> Bhiksha Raj<sup>1,5</sup>

<sup>1</sup> Carnegie Mellon University <sup>2</sup> Microsoft <sup>3</sup> University of Michigan Ann Arbor <sup>4</sup> Ohio State University <sup>5</sup> Mohamed bin Zayed University of Artificial Intelligence xl6@andrew.cmu.edu

## Abstract

Linguistic communication is prevalent in Human-Computer Interaction (HCI). Speech (spoken language) serves as a convenient yet potentially ambiguous form due to noise and accents, exposing a gap compared to text. In this study, we investigate the prominent HCI task, Referring Video Object Segmentation (R-VOS), which aims to segment and track objects using linguistic references. While text input is well-investigated, speech input is underexplored. Our objective is to bridge the gap between speech and text, enabling the adaptation of existing text-input R-VOS models to accommodate noisy speech input effectively. Specifically, we propose a method to align the semantic spaces between speech and text by incorporating two key modules: 1) Noise-Aware Semantic Adjustment (NSA) for clear semantics extraction from noisy speech; and 2) Semantic Jitter Suppression (SJS) enabling R-VOS models to tolerate noisy queries. Comprehensive experiments conducted on the challenging AVOS benchmarks reveal that our proposed method outperforms state-of-the-art approaches.

## 1 Introduction

Recent advances in vision-language learning have significantly advanced Human-Computer Interactions (HCI). A demanding task within HCI is referring video object segmentation (R-VOS), which involves segmenting and tracking objects in videos based on textual references. The successful development of R-VOS techniques has paved the way for diverse real-world applications such as video editing (Li et al., 2022d) and augmented reality (Huang et al., 2022). Notably, recent R-VOS methods (Luo et al., 2023) have shown unprecedented progress, propelled by the rapid advancement of multimodal foundation models such as CLIP (Radford et al., 2021). These R-VOS models enable various text-referred scenarios, allowing referring segmentation for generalized textual expressions even in complex visual scenes.

![](images/cd9f2122f3180a37fc2070b7249327490fc84af28df8bc4fac6a5b6060a5dcd6.jpg)  
Figure 1: (a) Referring video object segmentation (R-VOS) employs text as a query for segmentation. (b) Speech-referring video object segmentation. Compared to written language (text), spoken language (speech) is a more noisy form, potentially involving greater information loss and disturbance due to background noises. We propose a plug-and-play STBridge module, which seamlessly extends a frozen text-conditioned R-VOS model to accommodate noisy speech inputs.

However, a more challenging scenario arises in the prevalent speech dialogue system, where the aim is to refer to specific targets using spoken language, i.e., speech. The inherent nature of speech introduces vulnerabilities to disturbances from background noises (sound except for the referring speech). Consequently, crucial information within the spoken content can be distorted or even lost, which poses extra challenges in maintaining effective segmentation when referring to targets verbally. Though previous R-VOS methods achieve remarkable performance with textual queries, their performance with real-world spoken language is rarely discussed. Hence, it is crucial to develop an effective approach to bridge the text and speech to adapt well-trained R-VOS methods (frozen) for speech inputs.

Yet, bridging speech to R-VOS methods introduces new challenges. A straightforward solution involves utilzing automatic speech recognition (ASR) (Li et al., 2022a) to convert speech to text, followed by text-conditioned referring segmentation. However, this can result in suboptimal performance for two primary reasons. (1) Noise in language queries ofR-VOS. Existing R-VOS models rely on clean text-video pairs, wherein the textual expression unambiguously identifies the target object. Nonetheless, information extracted from speech may be incomplete or distorted due to background noise and ASR errors, leading to inadequate references to the target object. To maintain robust segmentation quality, it is crucial to adapt R-VOS models to handle perturbed referring queries. (2) Noise in speech understanding. In practice, speech and background noise are closely intertwined, making it difficult to accurately comprehend semantic information from speech. Considering the diverse types of noise, an effective noise-tolerant speech understanding approach is vital for achieving robust speech-referring video object segmentation.

In this paper, we present STBridge, a novel approach that enables R-VOS models trained on clean text-video pairs to adapt to noisy speech as referring guidance, maintaining robust performance even amidst background noises. As illustrated in Fig. 1, the proposed STBridge links the well-trained R-VOS model with speech input, incorporating two core considerations to improve the model’s robustness: (1) enhancing the welltrained R-VOS model to accept incomplete guidance, and (2) providing the noise-tolerant capability for speech understanding. On the one hand, we introduce a semantic-jitter suppression (SJS) module to help the R-VOS model understand noisy information from referring guidance. The SJS module generates object queries with randomly jittered textual features, allowing the model to learn from incomplete referring guidance under proper supervision. On the other hand, we introduce a noise-aware semantic adjustment (NSA) module, which generates noise-adaptive filters to enhance the speech representation. This differs from traditional speech enhancement, as it focuses solely on encoded semantics during speech understanding, while discarding low-level information, i.e., waveforms.

We further introduce a slack semantic alignment to align text and speech queries, enabling the integration of speech input with well-trained R-VOS models. Notably, our method incorporates additional modules without any retraining of R-VOS models, which is essential for numerous real-world applications. In summary, our contributions are as follows:

• We propose STBridge, a novel approach to bridge speech input to referring segmentation models, enabling segmenting objects with spoken language.

• We introduce semantic-jitter suppression and noise-aware semantic adjustment modules to enable the noise-tolerant capability for speech queries.

• We conduct extensive experiments on speechreferring segmentation benchmarks and the results of which show our approach performs favorably over prior arts.

## 2 Related Works

Video segmentation. Video segmentation (Wang et al., 2021c; Li et al., 2022b,c, 2023a; Yan et al., 2023; Li et al., 2023c) is a fundamental task to enable video editing. Semi-supervised video object segmentation (VOS) which leverages a firstframe mask to assign the target object is among the most popular video segmentation tasks due to its high segmentation quality. Some recent works (Yang et al., 2020, 2021) propagate masks by exploring matches among adjacent frames. Space-Time-Memory networks (STM) (Oh et al., 2019) builds a memory bank for matches. Several works follow the paradigm used in STM and improve the memory construction policy (Xie et al., 2021; Liang et al., 2020; Wang et al., 2021a) or enhance the memory reading strategy (Cheng et al., 2021a; Seong et al., 2020; Hu et al., 2021; Cheng et al., 2021b; Seong et al., 2021; Yang et al., 2021). Recently, Yan et al.(Yan et al., 2023) introduced a two-shot setting for VOS tasks which enables highperformance segmentation with limited annotated frames. Since the VOS task is primarily used for video editing which requires human involvement, to reduce the labor in assigning the target object, referring video object segmentation (R-VOS) is introduced. Specifically, R-VOS aims to segment an object in a video sequence given a linguistic description as the query. ReferFormer (Wu et al., 2022) and MTTR (Botach et al., 2022) are two pioneering works that utilize transformers to decode or fuse multimodal features. Recently, R<sup>2</sup>-VOS (Li et al., 2023b) introduces a cyclic structural consistency to enhance the robustness of R-VOS. And

![](images/af7f2d5d1c7e1250f3619480fbc48a602ac6f06f1576ee62e581c71c175c6067.jpg)  
Figure 2: Overview of STBridge. We extend frozen R-VOS model (in blue) with trainable STBridge (in red) modules to adapt to noisy speech as input. (I) Training: A speech encoder is utilized to extract noisy speech $g _ { n s }$ and noise $g _ { n }$ embeddings from noisy speech S. On one hand, a noise-aware semantic adjustment (NSA) module is utilized to mitigate the noise influence, which derives the cleaner speech embedding $g _ { s } .$ . On the other hand, to enhance the noise-tolerant capability of R-VOS model, we first generate perturbation to the text embedding g<sub>t</sub> and then equip a semantic jitter suppression (SJS) module to suppress the noises. Moreover, semantic alignment constraints are introduced to align the text $g _ { t }$ and speech $g _ { s }$ embeddings. (II) Inference: After aligning the text and speech embeddings during training, we can directly discard the text branch and leverage speech embedding $g _ { s }$ as the input to the SJS module.

OnlineRefer (Wu et al., 2023) employs the query propagation module to enable the online R-VOS.

Spoken language understanding. Spoken language, $i . e .$ , speech, enables a more natural way for humans to refer to a certain object than using text-based language. Thanks to the emergence of datasets with paired images and speech, $e . g .$ , Flicker8K (Harwath and Glass, 2015) and AVOS (Pan et al., 2022), more works (Chrupała, 2022; Harwath et al., 2020; Kano et al., 2021; Seo et al., 2023) started to research on the representation of speech and explore the synergy between speech and other modalities, $e . g .$ , image and video. For example, LAVISH (Lin et al., 2023) incorporates a small set of latent tokens to align the visual and audio representation, and VisualVoice (Gao and Grauman, 2021) conducts speech separation with the speaker’s facial appearance as a conditional prior. Later, research on speech has also moved towards finer granularity tasks. Some works (Lei et al., 2021) focus on the mono-modal impact of speech to study the subtle semantic information of spoken language to better understand human speech, while others (Jiang et al., 2021) study how to introduce the knowledge of speech understanding to create more natural human-computer interaction applications, e.g. talking head (Hwang et al., 2023; Li et al., 2023c; Qu et al., 2023).

## 3 Method

To ground objects verbally, we start from a frozen text-referring video object segmentation (R-VOS) model (shown as blue modules in Fig. 2), including frozen video-text encoders and a mask decoder. We introduce a speech encoder, a semantic jitter suppression (SJS) module, a noise-aware semantic adjustment (NSA) module, and a semantic alignment constraint to bridge text and speech (shown as pink modules in Fig. 2). During training, STBridge leverages video V , text T, and noisy speech S triplets to align the query spaces between text and speech. Thereafter, we can discard the text branch and directly query the objects with speech during inference.

## 3.1 Encoders

Frozen video and text encoders. We consider a generic referring segmentation framework that equips a video encoder $\mathcal { E } _ { v }$ and a text encoder $\mathcal { E } _ { t }$ to extract visual and textual features. Let us denote the extracted visual feature as $f = \mathcal { E } _ { v } ( V ) \in$ $\mathbb { R } ^ { C _ { v } \times L _ { v } \times H \times W }$ and extracted text embeddings as $g _ { t } = \mathcal { E } _ { t } ( T ) \in \mathbb { R } ^ { C \times L _ { t } }$ , where $C _ { v } , C$ and $L _ { v } , L _ { t }$ are the channel and length of visual and text embeddings respectively. We freeze the video and text encoders during both training and inference.

Speech encoder. We leverage a transformerbased speech encoder, Wav2Vec2 (Baevski et al., 2020) to extract speech features. We additionally augment two linear layers on top of the last hidden state of Wav2Vec2 to predict noise type. Thereby, each speech embedding corresponds to a noise embedding to describe the noise information. We denote the extracted noisy speech embedding as $g _ { n s } \in \mathbb { R } ^ { C \times L _ { s } }$ and noise embedding as $g _ { n } \in \mathbb { R } ^ { C \times L _ { s } }$ . C and $L _ { s }$ are the channel and length of embeddings.

## 3.2 Semantic Jitter Suppression

To equip the R-VOS model, which is typically trained on clean data samples, with the noisetolerance capability, we first mimic noisy text embeddings $g _ { t } ^ { \prime }$ by applying semantic jitters to the original text embeddings $g _ { t }$ . After that, we introduce a learnable semantic jitter suppression block $\varphi ( \cdot )$ to suppress the jitter and generate proper object query q for the following mask decoding.

Specifically, we implement the semantic jitter with a linear perturbation function where $g _ { t } ^ { \prime } = m \cdot$ O $g _ { t } + \delta$ . Here, $m \in \{ 0 , 1 \} ^ { C \times L _ { t } }$ is a binary masking operation at either word-level (along $L _ { t }$ dimension) or channel-level (along $C$ dimension); $\boldsymbol { \delta } \in \mathbb { R } ^ { C \times L _ { t } }$ is a random noise; denotes the Hadamard product. Besides, the jitter suppression block is constructed by cascading a transformer encoder and a global average pooling layer which pools along the word dimension. Formally, the final object query $q \in$ $\mathbb { R } ^ { C \times 1 }$ can be generated as

$$
q = \varphi ( m \circ g _ { t } + \delta ) .\tag{1}
$$

## 3.3 Noise-aware Semantic Adjustment

We introduce noise-aware semantic adjustment (NSA) to adjust inaccurate semantics introduced by noises, which consists of two components: a bi-directional cross-attention for noise-speech interaction and a noise-guided modulation for speech embedding adjustment.

Bi-directional cross-attention (BCA). In BCA, Noise-to-Speech (N-S) and Speech-to-Noise (S-N) cross-attention layers are involved to compute noise-aware speech embeddings $g _ { n } ^ { \prime }$ and speechaware noise embedding $g _ { n } ^ { \prime }$ . Formally, they take the form:

$$
h _ { n  s } = \mathrm { S o f t m a x } ( Q _ { n } ^ { \mathrm { T } } K _ { s } / \sqrt { d } ) V _ { s }\tag{2}
$$

![](images/ca3af999c998774d8b47a05523c964656f4fedeb4025fea7874f0ec3e96e8465.jpg)

Figure 3: Illustration of Noise-aware Semantic Adjustment. Noisy speech and noise embeddings, $i . e . , g _ { n s }$ and $g _ { n }$ , first interact with each other via the bi-directional cross-attention mechanism. Then, the fused noise embedding $g _ { n } ^ { \prime }$ is used to modulate the fused speech embedding $g _ { n s } ^ { \prime }$ to make it more noise-aware.

$$
h _ { s  n } = \mathrm { S o f t m a x } ( Q _ { s } ^ { \mathrm { T } } K _ { n } / \sqrt { d } ) V _ { n } ,\tag{3}
$$

where $h _ { n  s }$ and $h _ { s  n }$ are outputs of N-S and S-N attention. $K , Q .$ , and V are derived by applying linear projections on the original speech or noise embedding. d is the dimension of $K$ and $Q$ . The $h _ { n  s }$ and $h _ { s  n }$ are fused back to their paths with residual connections (He et al., 2016). We denote the fused embeddings as $g _ { n s } ^ { \prime }$ and $g _ { n } ^ { \prime }$

Noise-guided modulation (NGM). To incorporate noise information into speech embeddings, we propose a noise-guided feature modulation with channel-wise attention (Tian et al., 2020). Different from attention in BCA acting along the time dimension, channel-wise attention directly acting on feature channels is more efficient to exploit semantically meaningful correlations (Wang et al., 2021b; Tian et al., 2020), especially for instance-level correlations (Tian et al., 2020; Cao et al., 2020; Bolya et al., 2019). Given the speech-aware noise embedding $g _ { n s } ^ { \prime }$ , we first apply a fully connected layer on it to form the dynamic filters $\Theta = \{ \theta _ { i } \} _ { i = 1 } ^ { L _ { s } }$ . Here, each filter $\theta _ { i } \in \mathbb { R } ^ { C \times 1 }$ represents the noise information for each timestep and modulates the speech embeddings according to their category and amplitude. Then we utilize channel-wise attention to modulate the noise-aware speech feature $g _ { s } ^ { \prime } .$ , which is given by:

$$
g _ { s | n } = \Theta \circ g _ { n s } ^ { \prime }\tag{4}
$$

where $g _ { s | n }$ is the modulated speech embeddings and represents Hadmard product. We fuse the $g _ { s | n }$ back to $g _ { n s } ^ { \prime }$ with a residual connection, which derives the final output $g _ { s } = g _ { n s } ^ { \prime } + g _ { s | n }$

## 3.4 Frozen Mask Decoder

Referring segmentation methods typically leverage a query-based mask decoder $\mathcal { D } ( q , f )$ (Wu et al.,

2022; Botach et al., 2022) that takes an object query $q \in \mathbb { R } ^ { C \times 1 }$ encoding the object information and a video feature $f$ as inputs to predict the object masks $M \in \mathbb { R } ^ { \mathbf { \bar { \cal { N } } } \times L _ { v } \times \mathbf { \bar { \cal { H } } } _ { o } \times W _ { o } }$ , object bounding boxes $\boldsymbol { B } \in \mathbb { R } ^ { N \times L _ { v } \times 4 }$ and confidence scores $S \in \mathbb R ^ { N \times L _ { v } \times 1 }$ across video frames. N is the object candidate number. Here, we omit the detailed structure (available in Appendix) for simplicity. It is worth mentioning that the object query for the decoder is simply an averaged text embedding for recent popular R-VOS methods, which takes the form: $q = \mathrm { p o o l } ( g _ { t } )$ . The well-trained mask decoder keeps frozen in our method.

## 3.5 Training Objectives

We utilize a semantic alignment loss $\lambda _ { a l i g n }$ to align speech and text queries, a noise classification loss $\mathcal { L } _ { n o i s e }$ to facilitate speech understanding, and a segmentation loss $\mathcal { L } _ { m a t c h }$ to segment objects:

$$
\mathcal { L } = \lambda _ { a l i g n } \mathcal { L } _ { a l i g n } + \lambda _ { n o i s e } \mathcal { L } _ { n o i s e } + \mathcal { L } _ { m a t c h }\tag{5}
$$

where $\lambda _ { a l i g n }$ and $\lambda _ { n o i s e }$ are constants.

Semantic alignment. To bridge the text and speech queries, we conduct semantic alignment between text and speech embeddings. As the object query q requires sentence-level semantics (each sentence describes one object), it is not necessary to enforce a tight sequence-to-sequence alignment between text and speech embeddings (Tan et al., 2023). Instead, we align text and speech embeddings with a loose constraint. Specifically, given a text embedding $g _ { t }$ and a speech embedding $g _ { s } ,$ we first pool them among word and time dimensions correspondingly. After that, an alignment constraint is applied between them

$$
\mathcal { L } _ { a l i g n } = \| \mathrm { p o o l } ( g _ { t } ) - \mathrm { p o o l } ( g _ { s } ) \| _ { 2 }\tag{6}
$$

where $\| \cdot \| _ { 2 }$ is the $\mathrm { L _ { 2 } } .$ -Norm.

Noise classification. We augment the clean speech with different categories of audio, e.g., dog barking, as noise. We apply a noise classification head on top of noise embedding $g _ { n }$ to predict noise categories. Let us denote the predicted probabilities as $p \in \mathbb { R } ^ { N _ { c } \times 1 }$ and the ground truth class as $c ,$ where $N _ { c }$ is the noise type number. The noise classification loss $\mathcal { L } _ { n o i s e }$ can be computed as

$$
\mathcal { L } _ { n o i s e } = - \mathrm { l o g } p [ c ]\tag{7}
$$

where $p [ c ]$ denotes the probability of class c.

Object segmentation. Following the object segmentation methods (Wu et al., 2022; Wang et al., 2021c), we assign each mask prediction with a ground-truth label and then apply a set of loss functions between them to optimize the segmentation mask quality. Given a set of predictions $y =$ $\{ y _ { i } \} _ { i = 1 } ^ { N }$ and ground-truth $\hat { y } ~ = ~ \mathsf { \bar { \{ B } }  _ { l } , \hat { S } _ { l } , \hat { M } _ { l } \mathsf { \bar { \{ } }  _ { l = 1 } ^ { L _ { v } }$ where $y _ { i } = \{ B _ { i , l } , S _ { i , l } , M _ { i , l } \} _ { l = 1 } ^ { L _ { v } }$ , we search for an assignment $\sigma ~ \in ~ \mathcal { P } _ { N }$ with the highest similarity where $\mathcal { P } _ { N }$ is a set of permutations of $N$ elements. The similarity can be computed as

$$
\begin{array} { r } { \mathcal { L } _ { m a t c h } ( y _ { i } , \hat { y } ) = \lambda _ { b o x } \mathcal { L } _ { b o x } + \lambda _ { c o n f } \mathcal { L } _ { c o n f } } \\ { + \lambda _ { m a s k } \mathcal { L } _ { m a s k } } \end{array}\tag{8}
$$

where $\lambda _ { b o x } , \lambda _ { c o n f }$ and $\lambda _ { m a s k }$ are constant numbers to balance the losses. Following previous works (Ding et al., 2021; Wang et al., 2021c), we leverage a combination of Dice (Li et al., 2019) and BCE loss as $\mathcal { L } _ { m a s k }$ , focal loss (Lin et al., 2017b) as $\mathcal { L } _ { c o n f }$ , and GIoU (Rezatofighi et al., 2019) and L1 loss as $\mathcal { L } _ { b o x }$ . The best assignment σˆ is solved by the Hungarian algorithm (Kuhn, 1955).

## 3.6 Inference

During inference, we only keep speech and video as inputs. We first pool the speech embedding $g _ { s } ^ { \prime }$ into a fixed size and then utilize it to replace the text embedding $g _ { t }$ . Thereby, noisy speech can replace text to query the visual object. Please note that the text branch stays functional as we froze the R-VOS model during the training of STBridge.

## 4 Experiment

## 4.1 Datasets and Metrics

Datasets. We conduct experiments on the largescale speech-referring video object segmentation dataset, AudioGuided-VOS (AVOS) (Pan et al., 2022) which augments three R-VOS benchmarks with speech guidance: Ref-YoutubeVOS (Seo et al., 2020), A2D-sentences (Xu et al., 2015) and JHMDB-sentences (Jhuang et al., 2013). Specifically, it involves 18,811 pairs of video sequences and speech audio, which is divided into the training, validation, and test set in a ratio of 0.75, 0.1, and 0.15, respectively. The AVOS test set only contains Ref-YoutubeVOS samples. The A2D-sentences and JHMDB-sentences test sets are evaluated on their original test splits with speech as queries. Based on the AVOS dataset, we synthesize noisy speech by combining randomly picked audio from

<table><tr><td rowspan="2">Method</td><td rowspan="2">Query</td><td colspan="2">No Noise</td><td colspan="2">Noise (30 dB)</td><td colspan="2">Noise (20 dB)</td><td colspan="2">Noise (10 dB)</td></tr><tr><td>J</td><td>F</td><td>J</td><td>F</td><td>J</td><td>F</td><td>J</td><td>F</td></tr><tr><td colspan="9">Text-referring Video Object Segmentation (for reference)</td></tr><tr><td>ReferFormer (Wu et al., 2022)</td><td>Text</td><td>66.3</td><td>68.1</td><td>=</td><td></td><td>-</td><td>-</td><td>–</td><td></td></tr><tr><td>MTTR (Botach et al., 2022)</td><td>Text</td><td>63.2</td><td>64.7</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Noisy Speech-referring Video Object Segmentation</td></tr><tr><td>ReferFormer (Wu et al., 2022)</td><td>Text (ASR)</td><td>56.1</td><td>60.7</td><td>55.3</td><td>58.9</td><td>53.3</td><td>58.1</td><td>50.3</td><td>56.4</td></tr><tr><td>MTTR (Botach et al., 2022)</td><td>Text (ASR)</td><td>52.4</td><td>56.8</td><td>51.8</td><td>56.5</td><td>49.4</td><td>55.4</td><td>47.8</td><td>53.4</td></tr><tr><td>STBridge (Ours)</td><td>Speech</td><td>63.7</td><td>67.4</td><td>63.5</td><td>67.1</td><td>62.1</td><td>66.1</td><td>59.9</td><td>64.8</td></tr></table>

Table 1: Quantitative results for noisy speech-referring object segmentation in videos. The noise loudness (dB) is measured by the signal-noise ratio (SNR) to the clean speech.

Audioset (Gemmeke et al., 2017). Specifically, a noise ranging from 0 to 40 dB signal-noise ratio (SNR) to the clean speech is sampled during training. For validation and testing, we create noisy speech under 10 dB, 20 dB, and 30 dB SNR for comprehensive evaluation.

Metrics. We leverage the region similarity  and contour accuracy (Pont-Tuset et al., 2017) metrics for the evaluation of speech-referring video object segmentation. The overall evaluation metric $\mathcal { I } \& \mathcal { F }$ is the average of score and score. Both the and $\mathcal { F } \uparrow$ scores are the larger the better.

## 4.2 Implementation Details

We implement our method in PyTorch. Without losing generality, we leverage ReferFormer (Wu et al., 2022) as our frozen R-VOS model (can be replaced with any query-based model). We train our model for 2 epochs with a learning rate of 1e-4. All experiments are run on 8 NVIDIA V100 GPUs. We adopt batchsize 8 and an AdamW (Loshchilov and Hutter, 2017) optimizer with weight decay 5 $1 0 ^ { - 4 }$ . Images are cropped to have the longest side 640 and the shortest side 360 during training and evaluation. In Eq. 1, we utilize the random noise δ Uniform( 0.5, 0.5) and a masking ratio of 0.1 for m as default. Please refer to the Appendix for more details.

## 4.3 Quantitative Results

Segmentation with clean speech. Table 2 compares the proposed STBridge with previous methods using the ResNet-50 (He et al., 2016) backbone. To better analyze the performance of STbridge, we introduce two popular R-VOS baselines (with text query), i.e., ReferFormer (Wu et al., 2022) and MTTR (Botach et al., 2022), and leverage Wav2Vec (same as our speech encoder) (Baevski et al., 2020) to conduct ASR to adapt them to speech input. We notice that ASR-converted text will degrade the baseline models’ performance even without noise impact. We consider this can result from word errors in the converted text from speech. For example, if the target object ‘cat’ is wrongly recognized as ‘cap’ by ASR, the R-VOS model will inevitably segment the wrong object.

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Query</td><td rowspan=1 colspan=1>J&amp;F</td><td rowspan=1 colspan=1>J    F</td></tr><tr><td rowspan=1 colspan=4>Text-referring Video Object Segmentation (for reference)</td></tr><tr><td rowspan=1 colspan=1>ReferFormerMTTR</td><td rowspan=1 colspan=1>TextText</td><td rowspan=1 colspan=1>67.264.0</td><td rowspan=1 colspan=1>66.3  68.163.2  64.7</td></tr><tr><td rowspan=1 colspan=4>Speech-referring Video Object Segmentation</td></tr><tr><td rowspan=1 colspan=1>ReferFormerMTTR</td><td rowspan=1 colspan=1>Text (ASR)Text (ASR)</td><td rowspan=1 colspan=1>58.454.6</td><td rowspan=1 colspan=1>56.1  60.752.4  56.8</td></tr><tr><td rowspan=1 colspan=1>STBridge (Ours)</td><td rowspan=1 colspan=1>Speech</td><td rowspan=1 colspan=1>65.5</td><td rowspan=1 colspan=1>63.7  67.4</td></tr></table>

Table 2: Quantitative results for speech-referring object segmentation in videos. Both the and scores are the larger the better. & is the average of and as convention.

Segmentation with noisy speech. As shown in Table 1, we compare the performance of STBridge to previous text-queried methods with noisy speech as inputs. We modify the signal-noise ratio (SNR) of noisy speech to comprehensively evaluate the noise influence. We notice that ASR-based methods suffer severe performance drops compared to clean speech. In contrast, STBrigde shows a more robust performance with only slight degradation when noise becomes loud.

## 4.4 Visualization

In Fig. 4, we show the qualitative comparisons between our method, i.e., STBridge, and a cascade of Wav2Vec2 (Baevski et al., 2020) (ASR model) and ReferFormer (Wu et al., 2022) (RVOS model). Note that the ASR model is fairly chosen to have a goose standing in the second from the right a young women seating on top of a brown horse in water on the far right

![](images/95685cc04428e94ddd67f6e303261d2e9d6904b0744b229693e6d87c1915c6de.jpg)  
Figure 4: Qualitative comparison between STBridge (our speech-referring VOS model) and ASR-assisted Refer-Former (an assembly of ASR (Baevski et al., 2020) and text-referring VOS (Wu et al., 2022) models).

![](images/f6525bae2fac64c1ab0a005b8368be3bdd13e2025c0c481d4ef1dcdd61ebbf62.jpg)  
(a) Original

![](images/c4ec2a074a597004b9cb5130e258863bb0c9c5d936046d1f4fb9fd5b8a22aeef.jpg)  
(b) Jittered

![](images/f794bc050406d7a9b773def7f71147e548bdd7e12ba030775786fe9e29972412.jpg)  
(c) After SJS  
Figure 5: (a) Original embedding. (b) Semantic-jitterinjected embedding. (c) Jitter-suppressed embedding.

the same speech encoder as STBridge. Notably, our method successfully refers to the correct object while ASR-assisted ReferFormer fails to understand the speech input and predicts wrongly.

Fig. 5 demonstrates the function of the SJS module. By comparing Fig. 5 (b) and Fig. 5 (c), we can notice that SJS module effectively suppresses noises in the jittered embeddings.

## 4.5 Ablation Experiments

We conduct ablation studies to show the impact of different modules. Unless otherwise specified, all experiments are conducted on the AVOS test set.

Module effectiveness. We conduct experiments to validate the effectiveness of our proposed modules. We add the proposed modules step-by-step as shown in Table 3. (1) With semantic alignment between textual and speech representation, STBridge achieves 61.0 and 55.7 $\mathcal { I } \& \mathcal { F }$ with clean and noisy speech queries, respectively. (2) After equipping the SJS module, STBridge can better handle noises thus boosting the performance with noisy speech to 59.7 $\mathcal { I } \& \mathcal { F }$ while only marginal improvement is achieved with clean speech queries. (3) The equipping of NSA module benefits the performance with both clean and noisy speech queries. We consider the reason is that the two types of attention in NSA better filter out irrelevant features and help the speech embedding focus on the target object. With all modules, STBridge achieves 65.6 and 62.4 & for clean and noisy speech correspondingly.

Semantic jitter type. During training, STBridge generates semantic jitter and then learns to suppress it using the SJS module to enhance the noisetolerant capability of the R-VOS model. We conduct an ablation study to investigate the influence of different semantic jitter types. Specifically, the implemented semantic jitter on the text embedding g<sub>t</sub> has a form of $m \circ g _ { t } + \delta _ { \cdot }$ , where m and δ are binary mask and random noise respectively. As shown in Table 4, we notice masking among both the word-level and channel-level shows an improvement in performance. Random noise δ brings an additional 0.5  & to the final performance.

Design choices. We conduct experiments to ablate the design choices in STBridge and their impacts on the segmentation performance. (1) We first study the effect of frame window size selected from the entire video sequence during training. We notice that the window size only shows a marginal impact on the performance, which can be due to the visual encoder and mask decoder being frozen during training. As shown in Table 5a, we notice a window size of 5 achieves the best performance. (2) After that, we ablate on the loss type for semantic alignment in Table 5b. We leverage L<sub>1</sub>, L<sub>2</sub>, and Cosine loss to align the text and speech embeddings. We notice that $L _ { 2 }$ loss achieves the best performance among them. (3) Semantic jitter suppression is an essential component in STBridge for noise tolerance. We conduct ablation studies to demonstrate the impact of different masking ratios and random noise amplitude. Table 5c demonstrates the performance with different masking ratios of $m \in [ 0 , 1 ] ^ { C \times L _ { t } }$ (calculated as $1 - \frac { \mathrm { s u m } ( m ) } { C \times L _ { t } } )$ . Small masking ratios cannot provide enough perturbation to the inputs while large ratios may lose the semantics to the target object. We find a masking ratio of 0.1 is a good trade-off as shown in Table 5c. (4) We ablate the amplitude of noise δ added as a semantic jitter in Table 5d. We notice that an amplitude of 0.5 leads to the best performance.

![](images/77a3dcf90fd3dba713490174f0816f9e53d61b89c3326687fad52d6675fb52a7.jpg)

<table><tr><td colspan="2">Module</td><td>Clean</td><td>Noisy</td></tr><tr><td>SA</td><td>SJS NSA</td><td>J&amp;F</td><td>J&amp;F</td></tr><tr><td>V V</td><td rowspan="5">V</td><td>61.0</td><td>55.7</td></tr><tr><td>V</td><td>61.3</td><td>59.7</td></tr><tr><td>V</td><td>63.9</td><td>61.4</td></tr><tr><td>V 65.5</td><td>62.4</td><td></td></tr><tr><td></td><td></td><td></td></tr></table>

<table><tr><td colspan="3">Semantic Jitter Type</td><td rowspan="2">J&amp;F</td></tr><tr><td>Word Mask</td><td>Channel Mask</td><td>Random Noise</td></tr><tr><td></td><td></td><td></td><td>61.2</td></tr><tr><td rowspan="4">√</td><td></td><td></td><td></td></tr><tr><td>√</td><td></td><td>61.4</td></tr><tr><td>√</td><td></td><td>61.9</td></tr><tr><td>√</td><td>√</td><td>62.4</td></tr></table>

Table 3: Module effectiveness with clean and noisy speech as queries. SA: semantic alignment. SJS: semantic jitter suppression. NSA: noise-aware semantic adjustment.  
Table 4: Ablation on semantic jitter types. We conduct ablations on the impact of created semantic jitter types in the SJS module. We conduct this experiment with noisy speech (10 dB).

Table 5: Design choices for STBridge. We report the performance with the noisy speech queries on AVOS test set. (a) We ablate the window size (input frame number) during training. (b) We ablate the semantic alignment loss types. (c) We ablate the making ratio for m in creating semantic jitters. The ratio is calculated by $1 - \bar { \frac { \mathrm { s u m } ( m ) } { C \times L _ { t } } }$ . (d) We ablate the amplitude of the random noise δ. The noise δ  Uniform( Amp, Amp).  
![](images/f2001ede2922f880ba232a7023d792c75015affb6c2e3251437941dccee38c18.jpg)  
Figure 6: Analysis of the impact of noise categories.

Noise category. We conduct an experiment to investigate the impact of different noise categories. We additionally synthesize noisy speech queries by mixing clean speech from AVOS test set with different categories of audio recordings from Audioset (Gemmeke et al., 2017). As shown in Fig. 6, we illustrate the results of queries with different noise categories. We notice that sustained and loud noises, e.g., ambulance siren, can lead to a severe performance drop compared to short-lived and faint noises, e.g., horse clip-clop.

## 5 Conclusion

In conclusion, this paper presents STBridge, a novel approach that enables R-VOS models trained on clean text-video pairs to adapt to noisy speech as referring guidance, maintaining robust performance. The approach incorporates semantic jitter suppression (SJS) and noise-aware semantic adjustment (NSA) modules to enhance noise tolerance in speech queries. Experimental results demonstrate the effectiveness of STBridge, outperforming previous methods on three benchmarks. STBridge expands the applicability of R-VOS models, enabling robust speech-referred video object segmentation in real-world scenarios.

Limitation. In spite of STBrdge’s high performance on existing benchmarks, we only consider the scenario that text and speech queries are in the same language. Bridging text and speech in different languages can impose more challenges as the semantic spaces may suffer more divergence, which will be our future focus.

## References

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33:12449–12460.

Daniel Bolya, Chong Zhou, Fanyi Xiao, and Yong Jae Lee. 2019. Yolact: Real-time instance segmentation. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 9157–9166.

Adam Botach, Evgenii Zheltonozhskii, and Chaim Baskin. 2022. End-to-end referring video object segmentation with multimodal transformers. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4985–4995.

Jiale Cao, Rao Muhammad Anwer, Hisham Cholakkal, Fahad Shahbaz Khan, Yanwei Pang, and Ling Shao. 2020. Sipmask: Spatial information preservation for fast image and video instance segmentation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIV 16, pages 1–18. Springer.

Ho Kei Cheng, Yu-Wing Tai, and Chi-Keung Tang. 2021a. Modular interactive video object segmentation: Interaction-to-mask, propagation and difference-aware fusion. arXiv preprint arXiv:2103.07941.

Ho Kei Cheng, Yu-Wing Tai, and Chi-Keung Tang. 2021b. Rethinking space-time networks with improved memory coverage for efficient video object segmentation. arXiv preprint arXiv:2106.05210.

Grzegorz Chrupała. 2022. Visually grounded models of spoken language: A survey of datasets, architectures and evaluation techniques. Journal ofArtificial Intelligence Research, 73:673–707.

Zihan Ding, Tianrui Hui, Shaofei Huang, Si Liu, Xuan Luo, Junshi Huang, and Xiaoming Wei. 2021. Progressive multimodal interaction network for referring video object segmentation. The 3rd Large-scale Video Object Segmentation Challenge, page 7.

Ruohan Gao and Kristen Grauman. 2021. Visualvoice: Audio-visual speech separation with cross-modal consistency. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15490–15500. IEEE.

Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. 2017. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 776–780. IEEE.

David Harwath and James Glass. 2015. Deep multimodal semantic embeddings for speech and images.

In 2015 IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU), pages 237–244. IEEE.

David Harwath, Wei-Ning Hsu, and James Glass. 2020. Learning hierarchical discrete linguistic units from visually-grounded speech. In International Conference on Learning Representations.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770– 778.

Li Hu, Peng Zhang, Bang Zhang, Pan Pan, Yinghui Xu, and Rong Jin. 2021. Learning position and target consistency for memory-based video object segmentation. arXiv preprint arXiv:2104.04329.

Shijia Huang, Yilun Chen, Jiaya Jia, and Liwei Wang. 2022. Multi-view transformer for 3d visual grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15524–15533.

Geumbyeol Hwang, Sunwon Hong, Seunghyun Lee, Sungwoo Park, and Gyeongsu Chae. 2023. Discohead: Audio-and-video-driven talking head generation by disentangled control of head pose and facial expressions. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE.

Hueihan Jhuang, Juergen Gall, Silvia Zuffi, Cordelia Schmid, and Michael J Black. 2013. Towards understanding action recognition. In Proceedings of the IEEE international conference on computer vision, pages 3192–3199.

Yuming Jiang, Ziqi Huang, Xingang Pan, Chen Change Loy, and Ziwei Liu. 2021. Talk-to-edit: Fine-grained facial editing via dialog. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13799–13808.

Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. 2021. Mdetr-modulated detection for end-to-end multi-modal understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1780–1790.

Takatomo Kano, Sakriani Sakti, and Satoshi Nakamura. 2021. Transformer-based direct speech-to-speech translation with transcoder. In 2021 IEEE Spoken Language Technology Workshop (SLT), pages 958– 965. IEEE.

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. 2014. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787– 798.

Harold W Kuhn. 1955. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83–97.

Yi Lei, Shan Yang, and Lei Xie. 2021. Fine-grained emotion strength transfer, control and prediction for emotional speech synthesis. In 2021 IEEE Spoken Language Technology Workshop (SLT), pages 423– 430. IEEE.

Jinyu Li et al. 2022a. Recent advances in end-to-end automatic speech recognition. APSIPA Transactions on Signal and Information Processing, 11(1).

Xiang Li, Haoyuan Cao, Shijie Zhao, Junlin Li, Li Zhang, and Bhiksha Raj. 2023a. Panoramic video salient object detection with ambisonic audio guidance. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 1424–1432.

Xiang Li, Jinglu Wang, Xiao Li, and Yan Lu. 2022b. Hybrid instance-aware temporal fusion for online video instance segmentation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 36, pages 1429–1437.

Xiang Li, Jinglu Wang, Xiao Li, and Yan Lu. 2022c. Video instance segmentation by instance flow assembly. IEEE Transactions on Multimedia.

Xiang Li, Jinglu Wang, Xiaohao Xu, Xiao Li, Bhiksha Raj, and Yan Lu. 2023b. Robust referring video object segmentation with cyclic structural consensus. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 22236–22245.

Xiang Li, Jinglu Wang, Xiaohao Xu, Xiulian Peng, Rita Singh, Yan Lu, and Bhiksha Raj. 2023c. Rethinking audiovisual segmentation with semantic quantization and decomposition. arXiv preprint arXiv:2310.00132.

Xiaoya Li, Xiaofei Sun, Yuxian Meng, Junjun Liang, Fei Wu, and Jiwei Li. 2019. Dice loss for data-imbalanced nlp tasks. arXiv preprint arXiv:1911.02855.

Zhen Li, Cheng-Ze Lu, Jianhua Qin, Chun-Le Guo, and Ming-Ming Cheng. 2022d. Towards an end-to-end framework for flow-guided video inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17562–17571.

Yongqing Liang, Xin Li, Navid Jafari, and Jim Chen. 2020. Video object segmentation with adaptive feature bank and uncertain-region refinement. Advances in Neural Information Processing Systems, 33.

Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. 2017a. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2117–2125.

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2017b. Focal loss for dense object detection. In Proceedings ofthe IEEE international conference on computer vision, pages 2980–2988.

Yan-Bo Lin, Yi-Lin Sung, Jie Lei, Mohit Bansal, and Gedas Bertasius. 2023. Vision transformers are parameter-efficient audio-visual learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2299–2309.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Zhuoyan Luo, Yicheng Xiao, Yong Liu, Shuyan Li, Yitong Wang, Yansong Tang, Xiu Li, and Yujiu Yang. 2023. Soc: Semantic-assisted object cluster for referring video object segmentation. arXiv preprint arXiv:2305.17011.

Seoung Wug Oh, Joon-Young Lee, Ning Xu, and Seon Joo Kim. 2019. Video object segmentation using space-time memory networks. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 9226–9235.

Wenwen Pan, Haonan Shi, Zhou Zhao, Jieming Zhu, Xiuqiang He, Zhigeng Pan, Lianli Gao, Jun Yu, Fei Wu, and Qi Tian. 2022. Wnet: Audio-guided video object segmentation via wavelet-based cross-modal denoising networks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1320–1331.

Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc Van Gool. 2017. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675.

Liao Qu, Xianwei Zou, Xiang Li, Yandong Wen, Rita Singh, and Bhiksha Raj. 2023. The hidden dance of phonemes and visage: Unveiling the enigmatic link between phonemes and facial features. arXiv preprint arXiv:2307.13953.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR.

Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. 2019. Generalized intersection over union: A metric and a loss for bounding box regression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 658–666.

Paul Hongsuck Seo, Arsha Nagrani, and Cordelia Schmid. 2023. Avformer: Injecting vision into frozen speech models for zero-shot av-asr. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22922–22931.

Seonguk Seo, Joon-Young Lee, and Bohyung Han. 2020. Urvos: Unified referring video object segmentation network with a large-scale benchmark. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XV 16, pages 208–223. Springer.

Hongje Seong, Junhyuk Hyun, and Euntai Kim. 2020. Kernelized memory network for video object segmentation. In European Conference on Computer Vision, pages 629–645. Springer.

Hongje Seong, Seoung Wug Oh, Joon-Young Lee, Seongwon Lee, Suhyeon Lee, and Euntai Kim. 2021. Hierarchical memory matching network for video object segmentation.

Yi Xuan Tan, Navonil Majumder, and Soujanya Poria. 2023. Sentence embedder guided utterance encoder (segue) for spoken language understanding. Proc. Interspeech 2023.

Zhi Tian, Chunhua Shen, and Hao Chen. 2020. Conditional convolutions for instance segmentation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, pages 282–298. Springer.

Haochen Wang, Xiaolong Jiang, Haibing Ren, Yao Hu, and Song Bai. 2021a. Swiftnet: Real-time video object segmentation. arXiv preprint arXiv:2102.04604.

Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. 2021b. Max-deeplab: Endto-end panoptic segmentation with mask transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5463–5474.

Yuqing Wang, Zhaoliang Xu, Xinlong Wang, Chunhua Shen, Baoshan Cheng, Hao Shen, and Huaxia Xia. 2021c. End-to-end video instance segmentation with transformers. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8741–8750.

Dongming Wu, Tiancai Wang, Yuang Zhang, Xiangyu Zhang, and Jianbing Shen. 2023. Onlinerefer: A simple online baseline for referring video object segmentation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 2761–2770.

Jiannan Wu, Yi Jiang, Peize Sun, Zehuan Yuan, and Ping Luo. 2022. Language as queries for referring video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4974–4984.

Haozhe Xie, Hongxun Yao, Shangchen Zhou, Shengping Zhang, and Wenxiu Sun. 2021. Efficient regional memory network for video object segmentation. arXiv preprint arXiv:2103.12934.

Chenliang Xu, Shao-Hang Hsieh, Caiming Xiong, and Jason J Corso. 2015. Can humans fly? action understanding with multiple classes of actors. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 2264–2273.

Kun Yan, Xiao Li, Fangyun Wei, Jinglu Wang, Chenbin Zhang, Ping Wang, and Yan Lu. 2023. Two-shot video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2257–2267.

Zongxin Yang, Yunchao Wei, and Yi Yang. 2020. Collaborative video object segmentation by foregroundbackground integration. In European Conference on Computer Vision, pages 332–348. Springer.

Zongxin Yang, Yunchao Wei, and Yi Yang. 2021. Associating objects with transformers for video object segmentation. Advances in Neural Information Processing Systems, 34.

## A More Implementation Details

Training details. Following the previous referring segmentation methods (Wu et al., 2022; Botach et al., 2022; Kamath et al., 2021), we leverage loss weight coefficients $\lambda _ { d i c e }$ and λfocal to balance Dice (Li et al., 2019) and focal (Lin et al., 2017b) losses in $\mathcal { L } _ { m a s k }$ . And $\lambda _ { g i o u }$ and $\lambda _ { L 1 }$ to balance GIoU (Rezatofighi et al., 2019) and L1 losses in $\mathcal { L } _ { b o x }$ . During training, we set $\lambda _ { n o i s e } =$ $\lambda _ { a l i g n } = 1 , \lambda _ { c o n f } = \lambda _ { g i o u } = \lambda _ { d i c e } = 2$ , and $\lambda _ { L 1 } = \lambda _ { f o c a l } = 5$ . We set the layer number in the transformer encoder in φ as 3. The frozen ReferFormer (Wu et al., 2022) is pre-trained on Ref-COCO/+/g (Kazemzadeh et al., 2014) for 12 epochs and then finetuned on the AVOS training set while with text queries as input for 6 epochs. For the text, speech, and video triplets used in STBridge training, we only ensure the text and speech describe the same object while the words in the text and speech may differ slightly.

During STBridge training, the noisy speech input is generated by mixing the original clean speech with randomly picked audio, which is injected as noise, from AudioSet with an SNR ranging from 0 to 40. The noise categories include ’ambulance siren’,’ baby laughter’, ’gun shooting’, ’cat meowing’, ’chainsawing trees’, ’coyote howling’, ’dog barking’, ’driving buses’, ’helicopter’, ’horse clip-clop’, ’lawn mowing’, ’lions roaring’, ’bird singing’, ’guitar’, ’glockenspiel’, ’piano’, ’tabla’, ’ukulele’, ’violin’, ’race car’, ’typing keyboard’. The sampling probabilities for each category are the same.

## B Detailed Structure of Mask Decoder

We demonstrate the detailed mask decoding process in Figure A. Except for the visual feature $f _ { t }$ and object query q as defined in the main paper, we additionally enroll the prototype masks $\{ P _ { t } \} _ { t = 1 } ^ { N }$ and instance embedding $\{ e _ { t } \} _ { t = 1 } ^ { T }$ . Given the object query $q \in \mathbb { R } ^ { C \times 1 }$ from STBridge, we first repeat it $N$ times to form the input to the transformer decoder TrD where N is the object candidate number (the final output is selected from object candidates based on confidence score). After that, we generate instance embedding $\{ e _ { t } \} _ { t = 1 } ^ { T }$ for each time step separately using a shared transformer decoder TrD with encoded memory $\{ f _ { t } \} _ { t = 1 } ^ { T }$ from visual encoder. The instance embedding here encodes the instance information and is leveraged to guide the mask decoding process. The mask prediction

$M _ { t }$ for each time step t is derived by a dynamic convolution between prototype mask $P _ { t }$ and dynamic weights which are learned from instance embedding $e _ { t }$ by two fully connected layers. The prototype masks $\{ P _ { t } \} _ { t = 1 } ^ { T }$ is generated by feature pyramid network (FPN) (Lin et al., 2017a) with visual feature $\{ f _ { t } \} _ { t = 1 } ^ { T }$

## C Inference Details of R-VOS models

To obtain the final segmentation result, we select the mask (among N candidates) with highest confidence throughout time as:

$$
\begin{array} { c } { \hat { M } _ { t } = M _ { \hat { s } , t } , } \\ { \hat { s } = \arg \operatorname* { m a x } \{ S _ { i , 1 } + \cdot \cdot \cdot + S _ { i , T } \} _ { i = 1 } ^ { N } } \end{array}\tag{9}
$$

where $\{ \hat { M } _ { t } \} _ { t = 1 } ^ { T }$ is the masks of referred object. $S _ { i , t }$ and $M _ { i , t }$ represent the i-th slot in $S _ { t }$ and $M _ { t }$ respectively. sˆ is the slot with the highest confidence to be the target object. Box predictions can help training procedures but are not used during inference.

<table><tr><td>Frozen</td><td>J&amp;F</td><td> $\overline { { \mathcal { I } } }$ </td><td>F</td></tr><tr><td>X</td><td>65.5</td><td>63.7</td><td>67.4</td></tr><tr><td>√</td><td>58.9</td><td>57.3</td><td>60.4</td></tr></table>

Table A: Ablation study on updating R-VOS parameters during training.

## D Training with Trainable R-VOS Model

We conduct additional ablation studies to show the results of training with updating parameters in R-VOS model. As shown in Table A, we notice that updating R-VOS parameters during the adaptation to noisy speech inputs will result in severe performance degradation. We consider this because 1) the information in noisy speech input is not enough to accurately refer to the object resulting in noises in the training process, and 2) the S-VOS dataset is smaller than the R-VOS dataset leading to overfitting.

## E More Visualization.

As shown in Fig. B, we demonstrate more visualizations of the proposed method. We notice that our method can correctly refer to the target object and help the R-VOS model segment temporally consistent object masks across frames.

![](images/24c80427d104cbd772847a1c516f30d4b83aa8fe169486f500d82a4c6f86ccf1.jpg)  
Figure A: Illustration of mask decoder, which derives mask predictions $\{ M _ { t } \} _ { t = 1 } ^ { T }$ from the visual features $\{ f _ { t } \} _ { t = 1 } ^ { T }$ the object query $q .$ Given the query feature $q$ from STBridge, we first repeat it $N$ times to form the input to the transformer decoder TrD where $N$ is the object candidate number (the final output is selected from object candidates based on confidence score). After that, we generate instance embedding $\{ e _ { t } \} _ { t = 1 } ^ { T }$ for each time step separately using a shared transformer decoder TrD with visual feature $\{ f _ { t } \} _ { t = 1 } ^ { T }$ from visual encoder. The mask prediction $M _ { t }$ for each time step t is derived by a dynamic convolution between prototype masks $P _ { t }$ and dynamic weights which are learned from instance embedding $e _ { t }$ by two fully connected layers. The prototype masks $\{ P _ { t } \} _ { t = 1 } ^ { T }$ is generated by feature pyramid network (FPN) (Lin et al., 2017a) with visual feature $\{ f _ { t } \} _ { t = 1 } ^ { T }$

![](images/06efa7dfefea4b3f4f61545460f902eb029760d3ce5ea2dd97333c65d76b67c5.jpg)

a white and brown owl standing next to a white cat

![](images/9cb9c190282a82841f71277dd98c04f62da4b5cb02a8b02dcc584a473007533d.jpg)

![](images/d1ebbc6f19d03873e18e1f58cb1b8b53124b85bb8b672e114bc09d39ba6a16f5.jpg)

![](images/c211d455623a84b995604dc1f83f829844b1e745accecfbd0c22bd618316c2fe.jpg)

![](images/c4868dc2e210b7a39e3ff72ded9a036ac7bc21c8373c5a4767b993f7df0bd380.jpg)

![](images/235e9583992cf70bf604d99efa0a72279782bad61c514d80747b4c2b06e57e6b.jpg)

![](images/49e33366238e3748633d60d3646612cae00745a28461564b8e4ad06244407f57.jpg)

![](images/fd849240974b6e8b762fa927b1d4c056c600fd673ede3f2431d9fc673c11fdc3.jpg)

![](images/35d4c3059d4b24cf73141f39642faed05acb348842cafc6038f3fdcf7ff02eac.jpg)

![](images/045ea3350c38f6d2a65e0ac8d76f9c1fdb8bc30a763fee33ba3c436b1a6f9f88.jpg)

![](images/5eb3ab7cf7118eb09b16f7ec7d1224195fc1045a21e500a4a38d045f1a23d99d.jpg)

the giraffe walking around

![](images/be6ba83ea4a98995aca9c08042c57bfd0bfcc1db646424c5272273e9114bd037.jpg)

![](images/2221572ad896a98aa00a9af93835157303707648aba8b30f65e24e67eafbda59.jpg)

![](images/23c272281dbcde13bd7fd76bb254a3785f3a01026bfc863b39346878ff938674.jpg)

![](images/47142a8cfe76f28b312943aa4946ecdee1147b65c343368ffc075e20ef4edaed.jpg)

![](images/76dae6b5f1e19ada600f3f9d41293a726085d6263f53b62f84f2011aba0c3c4e.jpg)

![](images/4b25e7d6baeeb6ae22eb37c60e40e748c25f213b907ca32169a5a69939739b7d.jpg)

![](images/4fe804d97bc59d0a74790513a28365b0e3bb75af37297736d5bc6432f9e2a55a.jpg)

![](images/fb437a61ddbbfdab42ed307744a2071a292f105519a21edea3028ba2ff2ca283.jpg)

![](images/7d32e57c6a9fa74f5cde301c64271a45a0381ef5e4e34c4c2598ca3a41c8f132.jpg)

![](images/b5a7fb02e338ab79f447a59fcf8036d27f95bef2538fc9862b815f4a2096cfe4.jpg)

a lizard in a small enclosure with two others outside and eat

![](images/a6893feb8c7eda7406ecedcd239f9078a4082fc6b7faddd2822eb98f600f129b.jpg)

![](images/e13b262588811e0a96a841073c87c6dca77498b8117ae6c438c8f29bf00901ed.jpg)

![](images/db6b96f3719d7827fc3fe2d9699fb7d26294fe8e3a446999715334d5330f2e34.jpg)

![](images/6f720911f5d8ffbe05e29a64ecc35857a4452e9eb6152022dcf9e31cb334d01d.jpg)

![](images/a37fb05c4c1fb787a6c0e2d4e0d353d8bc7169dcab57d325bf7ca9b08dea42ed.jpg)

![](images/314d7b433685c090679dcb0d114ed82d377c9fdc8f5208cde5cf248d226c3c34.jpg)

![](images/e8699f5a81aa9b18da57f45c1cac3fcc509bb8cb424156f1814e9a1493ebbfa7.jpg)

![](images/deb22b3d8c4e1230e0a00e20296aadb16f872162a3bbde571a8476f1b93b4968.jpg)

![](images/038ec7979cad60458f5e6ba27d5eeacbe1f52126901c0672758375f516b25173.jpg)

![](images/ed40b93add2cb7b73cc0c474b37d462c8d0c2371a223ef282f50b7564ed58d85.jpg)

![](images/53c7b1c61455cb4a167882d3e838b6a80dcaad20aac1ee9bff268de1316400c0.jpg)

a person grabbing a crocodile

![](images/b641ad00d88e544aa0ab1d074f4aecd32e67d7de23ce81762e71a930a1bbb374.jpg)

![](images/c8215190d52ba5e8d28e60f6afb0b03d5615888c4ac90519822bdee03690ba22.jpg)

![](images/d74ff9931c17cfc793a9b6251f5a3be6c232121a0b196de9d501337bba7763e8.jpg)

![](images/212bb840ab2ca5c7ef19631b18fc9ffdbbb4044778d4c200b2dca726fca62aaf.jpg)

![](images/d2d62e82c5cdf3c66f2ee5584214a8287e6874e9c649abd66430565d6fc0018d.jpg)

![](images/2eef3aea81febff6c418fb458e51e2dda528c805766adc8870b9a624c11db5ad.jpg)

![](images/cc408c6ddf2a6d7bdc18846e6a90a0cf9fe3f5ddae13ac969fd6b50df271a925.jpg)

![](images/e05d72dae53b77df3e3b5315069cbc6f038aac5cbe87c1eb906c321f71c11772.jpg)

![](images/b1365cab4ef345a368066657fb0ba3f1f7b0e82d0ce61843e2d3f6a6fdbc33bd.jpg)

![](images/e9a4e244decc8bee0e59eb7896f91dd626bd347c74b02f6b1b6c611e82ef7e92.jpg)

a zebra is standing to the left of view

![](images/22a3e1d8fa1c51ea395b6774107c5b5815ebb27e36907a49dce2a10139e07c64.jpg)

![](images/a9cbd8f09d42f500ef01efbe30b4c0ee6a2fb62b6bd28cd7e94cf9c5562f9de8.jpg)

![](images/d0b9589dd9e8fe66ab77a398f307a08445f21f28bebd758f5f4d1683cfe58f6a.jpg)

![](images/aa280afc56140c1055af0a67ae3ea0365ef584e1cb5f3c248ee1a38e5b6f421c.jpg)

![](images/318270f7217430f38c8ccc1cae065653e2925ae4a9abeeb4f23c0a39c3b2e516.jpg)

![](images/9387651de74ae98d62d859176e618cb29031ffe0cd9dc7b918028839b83cde28.jpg)

![](images/bd64ba25b92bdae285877770079220cdf4a406f3a2473524e3c69a11e1cc6430.jpg)

![](images/b651b86612a54d90b7f0fa7fef37686475878d3b6b42d9d29576880e4144f895.jpg)

![](images/b66890336b4a0be30282b27319fdd2f914d29c9a5fac9c62bad3c1e14d5361d5.jpg)  
Figure B: More visualization of our method on AVOS test set. 2296

![](images/07150b25ba49e083ca1a5248ba40e988cf573dcb0f8b85d43837593dce46ff4b.jpg)