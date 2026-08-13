# Coarse-to-Fine Contrastive Learning in Image-Text-Graph Space for Improved Vision-Language Compositionality

Harman Singh<sup>1</sup>†, Pengchuan Zhang<sup>1</sup>, Qifan Wang<sup>1</sup>, Mengjiao Wang<sup>1</sup>, Wenhan Xiong<sup>1</sup>, Jingfei Du<sup>1</sup>†, Yu Chen<sup>2</sup>†

<sup>1</sup>Meta AI <sup>2</sup>Anytime.AI

harmansingh.iitd@gmail.com

{pengchuanzhang, wqfcr, mengjiaow, xwhan, jingfeidu}@meta.com ychen@anytime-ai.com

## Abstract

Contrastively trained vision-language models have achieved remarkable progress in vision and language representation learning. However, recent research has highlighted severe limitations of these models in their ability to perform compositional reasoning over objects, attributes, and relations. Scene graphs have emerged as an effective way to under stand images compositionally. These are graphstructured semantic representations of images that contain objects, their attributes, and relations with other objects in a scene. In this work, we consider the scene graph parsed from text as a proxy for the image scene graph and propose a graph decomposition and augmentation framework along with a coarse-to-fine contrastive learning objective between images and text that aligns sentences of various complexities to the same image. We also introduce novel negative mining techniques in the scene graph space for improving attribute binding and relation understanding. Through extensive experiments, we demonstrate the effectiveness of our approach that significantly improves attribute binding, relation understanding, systematic generalization, and productivity on multi ple recently proposed benchmarks (For example, improvements up to 18% for systematic generalization, 16.5% for relation understanding over a strong baseline), while achieving similar or better performance than CLIP on various general multimodal tasks.

## 1 Introduction

Recent progress in contrastive learning using largescale image-text data for joint image-text representation learning has led to Vision-Language models (VLMs) like CLIP (Radford et al., 2021) and ALIGN (Jia et al., 2021) that show remarkable zeroshot classification and retrieval capabilities. However, recent works have shown that these models struggle at compositional reasoning (Yuksekgonul et al., 2022; Thrush et al., 2022; Ma et al., 2022). In particular, they struggle with binding correct attributes to the correct objects, understanding relations between objects, generalizing systematically to unseen combinations of concepts and to larger and more complex sentences.

![](images/2937a6659b865d47b6ea2d82ce64c4b9a7a8bd2de380b095222a1b45f40e7eac.jpg)

![](images/516cfaf6f2e6295cc0b00c9787df5bc0158d5cbeac5456e4124c5edc0f02e218.jpg)

(b)  
The green plant and the white bird The white plant and the green bird CLIP NegCLIP MosaiCLIP✓  
(a)  
![](images/4fe6bdf09248589afdcd5942e6d95c61820a9791b9ba95fe2012a24881035c1c.jpg)  
(c)  
Figure 1: (Left) a) A typical example from the ARO benchmark for testing attribute understanding of VLMs. VLMs struggle with matching the image to the correct caption (in green). (Right) Average scores of MosaiCLIP (our method) compared with NegCLIP and CLIP on prominent compositionality benchmarks for measuring b) Systematic Generalization c) Attribute, Relation, and Word Order understanding.

Some works have made progress on this problem. Yuksekgonul et al. (2022) show that hard negative mining of images and text during fine-tuning is a promising first step to improving compositionality. However, performance gains are highly dependent on how clean the training data is, and generalizing to unseen combinations of concepts remains a challenge. Doveh et al. (2023) use LLMs for hard negative mining and Cascante-Bonilla et al. (2023) explore using synthetic datasets to improve compositional understanding in VLMs. Synthetic datasets lead to a domain gap compared to natural datasets. We aim to develop a general-purpose approach for improving compositionality of all such contrastively trained VLMs.

In this paper, we consider a scene graph representation of the image and text. We observe that multiple sub-graphs of the text scene graph with different semantic complexities can be matched with the same image. Performing this matching improves fine-grained and hierarchical understanding of text and thereby, of images. We achieve this by developing a scene graph-based text decomposition strategy that creates a scene graph for any given text, decomposing it into sub-graphs, and matching an image to multiple sentences derived from these sub-graphs (See Fig. 2 for an overview). Each sub-graph represents a distinct part of the image, aligning well with CLIP’s original imagetext matching objective. Focused on improving attribute binding and relation understanding, we develop novel hard negative graph creation strategies which helps VL contrastive learning. We provide a novel Image-to-Multi-Text contrastive loss for matching individual images to multiple sentences. Our approach of matching texts of different complexity (from coarse-grained to fine-grained) to the image leads to fine-grained and hierarchical text understanding. Our resulting model is MosaiCLIP.

Our approach leads to significant improvements across compositionality benchmarks. For example, Figure 1 b) and c) shows that MosaiCLIP improves performance by 11.5% and 9.1% on CREPE and ARO dataset over a strong baseline and by > 20% over CLIP. Our contributions encompass:

• A novel graph-based text decomposition and augmentation framework and a coarse-to-fine contrastive learning objective for matching images to text sub-graphs of varying complexity.

• Hard-negative mining techniques using graph transformations of the text scene graphs, that are seamlessly coupled with our text decomposition strategy, and applied over any text.

• A thorough analysis for understanding why MosaiCLIP improves vision-language compositionality, disentangling the effect of image and text encoders and providing a novel tree-score based analysis showing that MosaiCLIP exhibits improved hierarchical text understanding.

• Extensive experiments over three model architectures, two pre-training datasets, three fine-tuning datasets and test over four compositionality benchmarks (11 datasets) to prove the efficacy of MosaiCLIP for improving compositionality.

## 2 Related Work

Contrastive Vision-Language Pre-training: Large-scale contrastive learning for Vision and Language is utilized to create models like CLIP (Radford et al., 2021) and ALIGN (Jia et al., 2021). These models showcase impressive performance on a variety of tasks, including image classification, text and image retrieval, image captioning (Mokady et al., 2021), object detection (Zhong et al., 2022; Li et al., 2022c) etc.

Visio-Linguistic Compositionality: Various studies have introduced benchmarks for assessing the compositional reasoning abilities of vision-language foundation models (VLMs). For instance, Winoground (Thrush et al., 2022) is a handpicked collection of 400 test cases, each comprising two images and two sentences. Sentences have the same word content and differ in word-order. Diwan et al. (2022) show that the Winoground dataset tests additional challenges along with compositionality, including handling ambiguous image-text pairs and unusual examples. Yuksekgonul et al. (2022) proposed the ARO benchmark for probing VLMs ability to understand Attribute, Relations, and Word-Order. Ma et al. (2022) proposed CREPE for measuring two aspects of compositionality: systematic generalization and productivity. All benchmarks suggest that contrastively trained VLMs have severe difficulty in compositional reasoning. As a remedy, NegCLIP (Yuksekgonul et al., 2022) and Teaching SVLC (Doveh et al., 2023) create targeted rule-based and LLM-guided hard negative sentences, SyViC (Cascante-Bonilla et al., 2023) fine-tunes CLIP with million scale synthetic images-text pairs, for improving relational and attribute understanding. We observe that previous methods are either highly dependent on how clean the training data is, use expensive LLM’s for data augmentation or use synthetic datasets that require special solutions to resolve the synthetic-to-real domain gap. We hence develop a coarse-to-fine contrastive learning framework that matches images with texts of multiple complexities, which serves as a general-purpose solution to improve fine-grained and hierarchical text understanding, thereby improving compositionality.

Scene Graphs are structured representations of visual scenes, consisting of objects, their attributes, and relationships between objects. Scene graphs are beneficial for a range of tasks including image retrieval (Wu et al., 2019; Johnson et al., 2015), image captioning (Yang et al., 2019), and image generation (Johnson et al., 2018) among others.

## 3 Methodology

## 3.1 Overview

Here we present the key high-level ideas of our approach. We first present a graph-centric view of the standard image-text matching objective in CLIP, which serves as a motivation to develop our approach (Sec. 3.2). We create scene graphs derived from the text, decompose them into multiple sub-graphs (Sec. 3.3) and apply augmentations on these sub-graphs to create negative sub-graphs (Sec. 3.4) which are used as hard negatives in a batch. Sec. 3.5 formally defines the Image-to-Multi-Text and Text-to-Image losses used for a batch of V-L inputs which is key for learning from multiple positive and negative texts derived from sub-graphs. Matching images with coarse-to-fine sub-graphs results in improved fine-grained and hierarchical understanding of text. Sec. 3.6 provides a twostage curriculum learning strategy for improved fine-tuning performance.

## 3.2 Image-Text-Graph Alignment

Our approach builds on the idea that the standard image-text contrastive learning in CLIP can be viewed as a matching between an image scene graph and its sub-graph. Formally, given an imagetext pair $( I , T )$ , the image can be viewed by its scene graph, $\mathcal { G } _ { I } = ( \mathcal { V } _ { I } , \mathcal { E } _ { I } )$ . The text scene graph is given by $\mathcal { G } _ { T } ~ = ~ ( \nu _ { T } , \mathcal { E } _ { T } )$ Then $\mathcal G _ { T } ~ \subset ~ \mathcal G _ { I }$ According to this assumption, during contrastive learning in CLIP, we implicitly bring the representation of the image scene graph close to one of its sub-graph (the text scene graph). Now, let $S _ { \mathcal { G } } = \{ g | g \subset \mathcal { G } \}$ represent the set of sub-graphs of a graph $\mathcal { G }$ . According to the assumption above, $g \in S _ { \mathcal { G } _ { T } } \Rightarrow g \in S _ { \mathcal { G } _ { I } }$ . Hence $\forall g \in S _ { { \mathcal { G } } _ { T } } , ( g , { \mathcal { G } } _ { I } )$ becomes a correct matching pair during contrastive learning. We match multiple sub-graphs of the text scene graph to the same image, while also including hard negative sub-graphs in the batch. Matching between graphs is an implicit concept, and all graphs are first converted to text via templates, converted to embeddings using transformerbased (text) encoders, and matched to image embeddings.

## 3.3 Scene Graph Guided Text Decomposition

Scene graphs are succinct representations of images. However, an image scene graph generator used for generating a scene graph for any given input image is expensive to train since it requires supervised scene graph annotations for training (Li et al., 2017; Xu et al., 2017; Zhang et al., 2019), and also leads to issues like low coverage or biased generations against the long tail nature of objects and relationship annotations. We instead use the text scene graph created using an off-the-shelf text scene graph parser<sup>1</sup> (Wu et al., 2019). This serves as a proxy for the scene graph of (part of) the image and is assumed to be a sub-graph of the image scene graph, as also depicted by Figure 2.

Let the text scene graph obtained be $G _ { T } =$ $( V _ { T } , E _ { T } )$ , where $V _ { T }$ represent the nodes of the graph, which are either objects or their attributes. $E _ { T }$ are the edges of the graph that represent relations between objects. See Fig. 2 for an example of a text scene graph. As shown in the figure, we decompose this scene graph into multiple positive sub-graphs $P _ { g } = \{ g _ { 1 } , g _ { 2 } , g _ { 3 } , \cdot \cdot \cdot , g _ { k } \} , k \le M$ where M is the max number of decomposed subgraphs and is a hyperparameter. Each sub-graph is a representation of a part of the image. We then convert sub-graphs to sentences so that they can be easily processed by transformer-based (text) encoders commonly used to train CLIP. For this, we use a simple template-based approach. For e.g., we create templates of the form $" \{ N _ { 1 } \} \{ R \} \{ N _ { 2 } \} "$ if we need to convert a graph having two nodes $( N _ { 1 } , N _ { 2 } )$ and a relation $R ,$ into a sentence format. Corresponding to each sub-graph, we obtain one positive text for the image, creating a positive text set $P _ { t } = \{ t _ { 1 } , t _ { 2 } , t _ { 3 } , \cdot \cdot \cdot , t _ { k } \}$

## 3.4 Negative Sub-Graph Creation

Corresponding to sub-graphs in $P _ { g } .$ , we create negative sub-graphs $N _ { g } = \{ { } ^ { n } g _ { 1 } , { } ^ { n } g _ { 2 } , { } ^ { n } g _ { 3 } , \cdot \cdot \cdot \}$ . Subgraphs in $N _ { g }$ are a minimally perturbed versions of the positive sub-graphs in $P _ { g }$ . Similar to positive sub-graphs, we convert sub-graphs in $N _ { g }$ to text using the same template-based approach, and obtain $N _ { t } = \{ { } ^ { n } t _ { 1 } , { } ^ { n } t _ { 2 } , { } ^ { n } t _ { 3 } , \cdot \cdot \cdot \}$ . Texts in $N _ { t }$ serve as hard negative texts in a given batch, see Fig. 2. We focus on creating negative sub-graphs that improve the attribute binding and relation understanding capabilities of the model, for which we use the following strategies for negative graph creation: We first consider an external set of objects ( ), attributes ( ), and relations ( ).

![](images/3944febb087b981acbaff190394819dffc90ca4b49f6cf49a5670c12e634f5ef.jpg)  
(a) Text scene graph creation and decomposition (Sec. 3.3).

![](images/6da4fa6a584291520d2efb7b467e1e4707afa308379ed813a0c40483c010b151.jpg)  
(b) Hard negative sub-graph creation from positive sub-graphs (Sec. 3.4).

![](images/441e057aa0d22a84b6dc481294bfc82953918844c1144d2d711e44f451becb52.jpg)  
Figure 2: Overview of our approach. a) Depiction of the scene graph of an image (hypothetical) and a scene graph parsed from text. The text scene graph is a sub-graph of the image scene graph. The text scene graph is decomposed into sub-graphs from which b) minimally perturbed hard-negative sub-graphs are created. c) The Ground truth similarity matrix used for a batch of data during contrastive learning. Solid boxes represent a match between the image and the corresponding text. Different from CLIP, each image can be matched to multiple texts in our method.

1) Node Swapping and Replacement: We swap nodes in sub-graphs, these can be swaps of nodes which are attributes or objects. We also replace nodes with external nodes from , based on their type. 2) Edge Replacement: We replace edges with randomly sampled edges from the external relations set, . 3) Connecting Sub-graphs: Here we join two sub-graphs. For this, we use one sub-graph from $P _ { g } ,$ and another random graph created using nodes and edges sampled from external sets $\mathcal { N } , \mathcal { A } , \mathcal { R }$ . This creates an overall hard negative graph. Sub-graphs are joined by simply joining nodes from both graphs through a randomly sampled edge from . These strategies result in minimally perturbed hard negative subgraphs for improving attribute and relation understanding. We define multiple graph transformations $\{ f _ { g } : \mathbb { G } \longrightarrow P ( \mathbb { G } ) \} - f _ { r e l } , f _ { a t t r } , f _ { o b j }$ using the above techniques and create hard negative subgraphs. See Appendix Sec. B for more details regarding negative sub-graph creation.

## 3.5 Coarse-to-Fine Contrastive Learning in Image-Text-Graph Space

Given an image-text batch during training $\boldsymbol { B } =$ $\{ ( \pmb { x } _ { i } , \pmb { t } _ { i } ) \} _ { i = 1 } ^ { n }$ , consider separately the batch of images $B _ { I } = \{ { \pmb x } _ { i } \} _ { i = 1 } ^ { n }$ and a batch of texts $\begin{array} { l } { { \cal { B } } _ { { \cal { T } } } = } \end{array}$ $\{ \pmb { t } _ { i } \} _ { i = 1 } ^ { n }$ . The sentences in the text batch are first converted to scene graphs to obtain a batch of scene graphs $B _ { G } = \{ \mathcal { G } _ { i } \} _ { i = 1 } ^ { n }$ , followed by decomposition to sub-graphs to obtain the positive sub-graph batch $B _ { g } ^ { p o s } = \{ { \pmb g } _ { i } \} _ { i = 1 } ^ { m } , m > n .$ r negative subgraphs are sampled and added to the batch to obtain $B _ { g } = \{ { \pmb g } _ { i } \} _ { i = 1 } ^ { m + r }$ . We convert these sub-graphs to text to obtain the final text batch ${ \cal B } _ { t } = \{ t _ { i } ^ { g } \} _ { i = 1 } ^ { m + r }$

Consider an image encoder model $f _ { \theta }$ parameterized by θ, a text encoder $f _ { \phi }$ parameterized by $\phi .$ For any image x, text t, $\tilde { \boldsymbol { u } } = f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ is the unnormalized image feature, and $\tilde { v } = f _ { \phi } ( t )$ is the unnormalized text feature. As common practice, the features are normalized to obtain ${ \pmb u } = \tilde { { \pmb u } } / \| \tilde { { \pmb u } } \|$ and ${ \pmb v } = \tilde { { \pmb v } } / \| \tilde { { \pmb v } } \|$ . The Image-to-Multi-Text contrastive loss is given by:

$$
\mathcal { L } _ { i 2 t } ^ { \mathrm { { u c } } } = - \sum _ { i = 1 } ^ { | \mathcal { B } _ { I } | } \frac { 1 } { | \mathcal { P } ( i ) | } \sum _ { k \in \mathcal { P } ( i ) } \log \frac { \exp ( \tau \pmb { u } _ { i } ^ { T } \pmb { v } _ { k } ) } { \sum _ { j = 1 } ^ { | \mathcal { B } _ { t } | } \exp ( \tau \pmb { u } _ { i } ^ { T } \pmb { v } _ { j } ) }
$$

where $\mathcal { P } ( i ) = \{ k | k \in [ 1 , | \mathcal { B } _ { t } ^ { p o s } | ] , \mathbf { \{ }  g _ { k } \subseteq \mathcal { G } _ { i } \} .$ The Text-to-Image contrastive loss is only calcu-

lated for the positive texts. It is given by:

$$
\mathcal { L } _ { t 2 i } ^ { \mathrm { { M C } } } = - \sum _ { j = 1 } ^ { | \mathcal { B } _ { t } ^ { p o s } | } \log \frac { \exp ( \tau \pmb { \mathscr { u } } _ { p ( j ) } ^ { T } \pmb { \mathscr { v } } _ { j } ) } { \sum _ { i = 1 } ^ { | \mathcal { B } _ { I } | } \exp ( \tau \pmb { \mathscr { u } } _ { i } ^ { T } \pmb { \mathscr { v } } _ { j } ) }
$$

where $\pmb { g } _ { p ( j ) } \subseteq \mathcal { G } _ { j } . \ B _ { t } = [ \ B _ { t } ^ { p o s } ; B _ { t } ^ { n e g } ]$ , in which $B _ { t } ^ { p o s } , B _ { t } ^ { n e \ddot { g } }$ represent the texts in $B _ { t }$ , obtained from positive and negative sub-graphs respectively. The overall loss is $\mathcal { L } _ { \mathrm { M o s a i C L I P } } = ( \mathcal { L } _ { t 2 i } ^ { \mathrm { M C } } + \mathcal { L } _ { i 2 t } ^ { \mathrm { M C } } ) / 2$

## 3.6 Curriculum and Robust Fine-tuning

For fine-tuning experiments, we develop a twostage curriculum learning strategy motivated by recent work (Goyal et al., 2022; Wortsman et al., 2022; Kumar et al., 2022) that show how finetuning can distort pre-trained features and closely mimicking the contrastive pre-training objective while fine-tuning CLIP can help mitigate this problem (Goyal et al., 2022). However, our coarse-tofine contrastive learning objective naturally deviates from pre-training in two ways. a) Existence of hard negative texts in the batch, and b) Having multiple positive and negative texts for an im age. This can lead to a gap in pre-training vs finetuning objective, and a lower than optimal performance after fine-tuning. To solve this, our twostage curriculum learning strategy first fine-tunes the model while sampling (at max) a single positive and negative sub-graph per image, followed by fine-tuning it with multiple positive and negative sub-graphs. The hardness of data in this curriculum learning setup is defined by the amount of difference the fine-tuning setup has as compared to the pre-training setup. According to this intuition, it is easier for the model to first learn to handle hard negatives in a batch and then learn to handle multiple positive and hard negative sentences at once. We see consistent improvements using this strategy compared to a direct one-step fine-tuning, which we term as $\mathbf { M o s a i C L I P _ { N o C u r r i c } }$ in our ablations. For better performance on non-compositonal tasks, we use the robust fine-tuning approach (Wortsman et al., 2022) of weight space ensembling of the vision encoder, before and after fine-tuning. This model is called MosaiCLIP<sub>WiSE-FT</sub>

## 4 Experiments

Evaluation Datasets: We test MosaiCLIP and baselines on large scale benchmarks that require compositional reasoning: CREPE-Systematicity (Ma et al., 2022) measures systematic generalization, ARO (Yuksekgonul et al., 2022) measures attribute, relation and word-order understanding, SVO (Hendricks and Nematzadeh, 2021) measures verb (relation) understanding, VL-Checklist (Zhao et al., 2022) measures relation, attribute and object understanding. We use CREPE-Productivity (Ma et al., 2022) for measuring model’s ability to productively generalize to more complex and long sentences. Methods for improving compositionality should be tested on general downstream tasks used to evaluate the quality of learned representations of language and vision. For this, we utilize the popular ELEVATER benchmark (Li et al., 2022a) consisting of 20 datasets and ImageNet (Deng et al., 2009) following prior work (Doveh et al., 2023). Baselines: We compare with all recent techniques used for improving compositionality of CLIP style models including NegCLIP (Yuksekgonul et al., 2022), Teaching SVLC (Doveh et al., 2023) and Syn-CLIP (Cascante-Bonilla et al., 2023) along with CLIP (Radford et al., 2021) as well as CLIP-FT (fine-tuned) on datasets we use. See Appendix Sec. F for more details.

## Training and Evaluation Details:

Fine-tuning: NegCLIP (Yuksekgonul et al., 2022) was developed by fine-tuning CLIP on the COCO dataset (Lin et al., 2014), however, COCO images might overlap with benchmarks like CREPE and ARO which may lead to confounding of results. Hence we consider 2 additional similar sized finetuning datasets randomly sampled from CC-12M (Sharma et al., 2018; Changpinyo et al., 2021) and YFCC-15M (Thomee et al., 2016) and call them CC-FT, YFCC-FT. We also use CC3M (Sharma et al., 2018) for comparing with recent baselines. We fine-tune the commonly used OpenAI CLIP-ViT-B32 model and report results on all datasets, except for CREPE dataset which tests the systematic generalization for which we used OpenCLIP (Ilharco et al., 2021) models pre-trained on {CC-12M, YFCC-15M}, fine-tune them on {CC-FT, YFCC-FT}, and report results on {CC-12M,YFCC-15M} splits of CREPE. See Appendix E.3 for more information on evaluation datasets.

Pre-training: We pre-train MosaiCLIP, NegCLIP and CLIP on two prominent large-scale pre-training datasets, CC-12M and YFCC-15M, and use two different backbones (ResNet-50 and Swin-Tiny) following prior work (Yang et al., 2022) and report zero-shot performance on all test datasets. See Appendix H.1 for hyperparameters details.

<table><tr><td>FineTun. data →</td><td colspan="4">COCO</td><td colspan="7"></td><td colspan="7">YFCC-FT</td><td></td></tr><tr><td>Benchmark</td><td></td><td>ARO</td><td>VLC</td><td>SVO</td><td></td><td>ARO</td><td></td><td></td><td>CREPE</td><td></td><td>VLC SVO</td><td></td><td>ARO</td><td></td><td></td><td>CREPE</td><td>VLC</td><td>SVO</td><td>Meta</td></tr><tr><td>Method ↓</td><td>Rel.</td><td>Attr. Ord.</td><td>Avg.</td><td>Avg.</td><td>Rel.</td><td>Attr.</td><td>Ord.</td><td>CU</td><td>AU</td><td>Avg.</td><td>Avg.</td><td>Rel.</td><td>Attr.</td><td>Ord.</td><td>CU</td><td>AU</td><td>Avg.</td><td>Avg.</td><td>Avg.</td></tr><tr><td>Random</td><td>50.0</td><td>50.0</td><td>20.0 50.0</td><td>50.00</td><td>50.0</td><td>50.0</td><td>20.0</td><td>14.3</td><td>20.0</td><td>50.0</td><td>50.00</td><td>50.0</td><td>50.0</td><td>20.0</td><td>14.3</td><td>20.0</td><td>50.0</td><td>50.00</td><td>38.35</td></tr><tr><td>CLIP</td><td>59.8</td><td>63.2</td><td>53.3</td><td>70.8</td><td>83.58</td><td>59.8 63.2</td><td>53.3</td><td>45.1</td><td>35.0</td><td>70.8</td><td>83.58</td><td>59.8</td><td>63.2</td><td>53.3</td><td>39.8</td><td>39.5</td><td>70.8</td><td>83.58</td><td>60.60</td></tr><tr><td>CLIP-FT</td><td>58.9</td><td>65.3</td><td>38.4 71.3</td><td>90.15</td><td></td><td>58.1 63.3</td><td>42.7</td><td>45.8</td><td>35.6</td><td>70.1</td><td>88.56</td><td>51.4</td><td>63.1</td><td>25.3</td><td>36.4</td><td>38.3</td><td>68.9</td><td>85.27</td><td>57.73</td></tr><tr><td>NegCLIP</td><td>81.7</td><td>72.7</td><td>85.7 75.6</td><td></td><td>90.20</td><td>71.5 65.4</td><td>84.5</td><td>53.1</td><td>37.5</td><td>72.4</td><td>88.36</td><td>57.8</td><td>63.1</td><td>52.1</td><td>38.8</td><td>39.0</td><td>70.4</td><td>83.90</td><td>67.57</td></tr><tr><td>MosaiCLIP</td><td>82.6</td><td>78.0</td><td>87.1</td><td>81.4</td><td>90.67</td><td>80.4 69.8</td><td>85.5</td><td>72.4</td><td>40.9</td><td>77.6</td><td>88.73</td><td>74.3</td><td>66.9</td><td>84.4</td><td>48.8</td><td>41.5</td><td>75.1</td><td>85.36</td><td>74.29</td></tr></table>

Table 1: Fine-tuning results on the ARO, CREPE - Systematicity, VL-Checklist (VLC) and SVO benchmark (total 10 datasets). Abbreviations – Rel.:= VG-Relation, Attr.:= VG-Attribution, Ord:=Average of ARO-Flickr and ARO-COCO Order results, CU: HN-Comp-Unseen, AU: HN-Atom-Unseen. See Sec. 4.1 for more details.

<table><tr><td rowspan="3">FineTun. data → Benchmark → Method</td><td colspan="2">CC3M</td></tr><tr><td>VL-Checklist</td><td>ARO</td></tr><tr><td>Obj. Attr. Rel.</td><td>Rel. Attr. Ord.</td></tr><tr><td>CLIP CLIP-FT</td><td>81.6 67.6 63.1 59.9 79.0 64.7 54.3 41.7</td><td>63.6 53.3 59.3 25.2</td></tr><tr><td>Syn-CLIP†1</td><td>70.4 69.4</td><td>71.4 66.9 65.1</td></tr><tr><td>Teaching SVLC‡2</td><td>85.0 72.0 69.0</td><td></td></tr><tr><td>MosaiCLIP</td><td></td><td></td></tr><tr><td></td><td>86.4 73.7 71.9</td><td>83.7 78.0 79.4</td></tr><tr><td colspan="3">1(Cascante-Bonilla et al., 2023)²(Doveh et al., 2023)</td></tr></table>

Table 2: MosaiCLIP vs contemporary works that use †synthetic data or ‡LLM’s. See Appx. D.1 for details.

## 4.1 Results

In this section we provide experimental results in both pre-training and fine-tuning settings to show the efficacy of our approach. These are as follows:

Fine-tuning: Main fine-tuning results are shown in Table 1 and 2, where we fine-tune CLIP models using our method and compare it to baselines. Notably, we see that the generalization performance on unseen compounds and atoms as measured by the CREPE dataset is up to 18% higher than NegCLIP. Additionally MosaiCLIP shows upto 16.5%, 5.3%, 32.3% of improvement over NegCLIP in understanding relations, attributes and word order respectively. MosaiCLIP also shows consistent improvements in the verb understanding task as measured by the SVO dataset. Additional Comparisons: We also compare with latest contemporary works in Table 2 and Appendix Sec. D.1. We find significant improvements (upto 14% on ARO) over models that use LLMs or synthetic data for making CLIP more compositonal.

Pre-training: Table 3 shows pre-training results over all benchmarks. CREPE results show a significant gain in ability to systematically generalize to unseen combinations of concepts. Across pre-training settings, MosaiCLIP improves over NegCLIP by up to 42.5%, 4.9% when evaluated against HN-Comp (CU), HN-Atom (AU) hard negatives respectively. Significant improvements are observed in attribute and relation understanding, giving gains of up to 8.3%, 12.0% respectively across pretraining settings. We also note that order understanding of MosaiCLIP is worse than that of NegCLIP for the CC-12M pre-training dataset, while better than NegCLIP for the YFCC-15M dataset. Notably, there is a large variance in NegCLIP’s performance across pre-training datasets as seen in Table 3, and it also performs poorly when the pre-training dataset has higher noise (e.g. YFCC-15M). MosaiCLIP is fairly consistent and more robust to the change in the pre-training dataset. In Appendix C.5 we find that MosaiCLIP can provide improvements over NegCLIP while using as low as 0.3x of the total pre-training or fine-tuning data.

![](images/233c1cf94922b539912ba40409fe36c7c48d947fe9cea450ced429f15ea75bb0.jpg)  
Figure 3: MosaiCLIP’s average score difference with NegCLIP on 20 datasets from ELEVATER benchmark.

Results on classification and retrieval: On average, MosaiCLIP achieves +3.3%, +6.3% better performance on the ELEVATER classification benchmark compared to NegCLIP and CLIP while pre-training and maintains similar accuracy as CLIP while fine-tuning. We also try using our method along with the robust fine-tuning technique (WiSE-FT) so that performance degradation during fine-tuning is minimal, as shown in Appendix Table 9. See Fig. 3 for average results on ELEVATER over four training settings and Table 4 for results on retrieval benchmarks where we see a +5.4 point improvement over NegCLIP. We use the popular Karpathy splits having a 5K and 1K sized test set for obtaining the COCO and Flickr30k retrieval scores respectively. Hence MosaiCLIP’s training strategy improves or maintains the quality of learned representations while improving compositonality. Figures 11-14 show detailed results on ELEVATER.

<table><tr><td colspan="2">Pre-training data →</td><td colspan="7">CC-12M</td><td colspan="7">YFCC-15M</td><td rowspan="2">Meta</td></tr><tr><td rowspan="2">Arch. Method ↓</td><td rowspan="2">Benchmark →</td><td colspan="3">ARO</td><td colspan="2">CREPE</td><td colspan="2">VLC SVO</td><td colspan="2">ARO</td><td colspan="2">CREPE</td><td colspan="2">VLC</td><td>SVO</td></tr><tr><td>Rel.</td><td>Attr.</td><td>Ord.</td><td>CU</td><td>AU</td><td>Avg.</td><td>Avg.</td><td>Rel.</td><td>Attr.</td><td>Ord.</td><td>CU</td><td>AU</td><td>Avg.</td><td>Avg.</td><td>Avg.</td></tr><tr><td rowspan="2"></td><td>Random</td><td>50.0</td><td>50.0</td><td>20.0</td><td>14.3</td><td>20.0</td><td>50.0</td><td>50.00</td><td>50.0</td><td>50.0</td><td>20.0</td><td>14.3</td><td>20.0</td><td>50.0</td><td>50.00</td><td>36.33</td></tr><tr><td>CLIP</td><td>51.0</td><td>56.6</td><td>25.5</td><td>44.1</td><td>37.3</td><td>65.6</td><td>82.21</td><td>53.8</td><td>56.2</td><td>18.4</td><td>39.6</td><td>41.7</td><td>66.2</td><td>76.27</td><td>51.03</td></tr><tr><td rowspan="2">Swi-T</td><td>NegCLIP</td><td>82.4</td><td>66.8</td><td>59.7</td><td>80.3</td><td>39.6</td><td>70.0</td><td>82.04</td><td>73.6</td><td>58.9</td><td>35.5</td><td>47.1</td><td>41.5</td><td>66.0</td><td>76.10</td><td>62.82</td></tr><tr><td>MosaiCLIP</td><td>84.3</td><td>76.8</td><td>55.5</td><td>92.1</td><td>44.5</td><td>72.4</td><td>85.62</td><td>74.7</td><td>66.1</td><td>35.8</td><td>89.6</td><td>45.3</td><td>71.8</td><td>77.87</td><td>69.46</td></tr><tr><td rowspan="2">RN-50</td><td>CLIP</td><td>52.9</td><td>59.7</td><td>22.6</td><td>42.9</td><td>36.7</td><td>66.2</td><td>82.13</td><td>57.8</td><td>55.1</td><td>18.3</td><td>38.9</td><td>38.9</td><td>64.8</td><td>75.60</td><td>50.90</td></tr><tr><td>NegCLIP</td><td>80.5</td><td>66.5</td><td>60.5</td><td>82.0</td><td>41.4</td><td>69.5</td><td>82.03</td><td>68.0</td><td>58.5</td><td>37.1</td><td>67.2</td><td>41.5</td><td>66.1</td><td>75.18</td><td>64.00</td></tr><tr><td></td><td>MosaiCLIP</td><td>82.0</td><td>78.5</td><td>55.4</td><td>92.6</td><td>44.4</td><td>72.6</td><td>83.86</td><td>76.3</td><td>68.9</td><td>38.2</td><td>90.2</td><td>45.0</td><td>72.3</td><td>77.42</td><td>69.83</td></tr></table>

Table 3: Pre-training results on all compositionality benchmarks (4 benchmarks, 10 datasets) over four expt. settings (two pre-training datasets, two backbones). See Table 1 for abbreviations and Sec. 4.1 for more details.

![](images/9ee44b437d3f195298c2879821c6e7c0a53c205b05353add79ee22bc82ae330a.jpg)  
(a) CREPE-Productivity

![](images/1089178004952c88ce088a6fe99d14ebc42911609b51cb1a386b615e423622af.jpg)  
(b) Tree Scores

![](images/775f30c7ed7f9ebf43590e0ec7a5fc3e0696f079dfe1731bdf02078e5b1358eb.jpg)  
(c) ARO-Relation

![](images/316e4e8fecacc6bbb705a69bb209aef7deebe5f81f65194f1bd408fcdb401312.jpg)  
(d) ARO-Attribute

Figure 4: a) Results on CREPE-Productivity dataset b) Tree-score comparison of MosaiCLIP with NegCLIP: MosaiCLIP shows improved hierarchical understanding of language. c) and d) Selectively fine-tuning of image, text encoders and measure performance on different datasets. Also see similar results for SVO in Figure 10.
<table><tr><td rowspan="2">Model</td><td>COCO</td><td></td><td>Flickr30K</td><td rowspan="2">AVG.</td></tr><tr><td>I2T</td><td>T2I I2T</td><td>T2I</td></tr><tr><td>CLIP</td><td>20.7</td><td>13.1</td><td>36.2 24.1</td><td>23.5</td></tr><tr><td>NegCLIP</td><td>20.1</td><td>12.9</td><td>38.6 23.3</td><td>23.7</td></tr><tr><td>MosaiCLIP</td><td>25.9</td><td>16.5</td><td>44.5 29.5</td><td>29.1</td></tr></table>

Table 4: Comparison of Recall@1 scores of MosaiCLIP with NegCLIP and CLIP. All models are pre-traind on YFCC-15M with swin-Tiny backbone

Productivity: As defined by Ma et al. (2022), a productive VL model can handle arbitrarily long and complex sentences and is an important aspect of compositionality. Although we do not explicitly train our models for generalization to longer sentences, the improved hierarchical language understanding using our methods lead to an emergent behavior such that MosaiCLIP generalizes better than NegCLIP and CLIP to more complex sentences. We can see this effect in Fig. 4 a) and Appendix Fig. 8 and 9. We report the average of retrieval over swap and atom splits and find MosaiCLIP significantly improves over NegCLIP by upto 15% across different text complexities (4-12).

Application to more advanced VLMs: While our focus in this work has been on CLIP style, dual encoder models due to their various benefits, we believe our methods are model agnostic and aimed at improving contrastive learning through our coarse-to-fine learning framework and negative mining techniques. In this section we test our model on an advanced VLM, BLIP. We modified BLIP’s original image-text contrastive learning objective and create two variants, one called BLIP+NegCLIP where we use NegCLIP style hard negatives and the other BLIP+MosaiCLIP which uses our methods of scene graph guided text decomposition and negative sub-graph creation. We fine-tune BLIP model taken from the official BLIP repository and use the “BLIP w/ ViT-B and CapFilt-L model (pre-trained on 129M examples)” as our base model. Results for fine-tuning experiment using COCO dataset is shown in Table 5. We use the hyperparameters used by the official codebase (for the task of fine-tuning on COCO dataset for image-text retrieval). For each setting, we report performance of four models, namely BLIP (before fine-tuned version), BLIP-FT (vanilla fine-tuned version), BLIP+NegCLIP, BLIP+MosaiCLIP. The model are evaluated on the ARO dataset to measure attribute, relation and word-order understanding, using the evaluation scripts provided by the authors of the dataset (Yuksekgonul et al., 2022). We find that

<table><tr><td>Model</td><td>Rel</td><td>Attr</td><td>Ord</td><td>Avg</td></tr><tr><td>BLIP</td><td>53.5</td><td>91.0</td><td>53.5</td><td>66.0</td></tr><tr><td>BLIP-FT</td><td>58.9</td><td>88.4</td><td>58.9</td><td>68.7</td></tr><tr><td>BLIP+NegCLIP</td><td>63.6</td><td>90.7</td><td>63.6</td><td>72.6</td></tr><tr><td>BLIP+MosaiCLIP</td><td>69.9</td><td>91.1</td><td>69.9</td><td>77.0</td></tr></table>

Table 5: Comparison of BLIP (Li et al., 2022b) and finetuned version of BLIP with BLIP models that have integrated NegCLIP and MosaiCLIP methodology while training. Fine-tuning has been performed on COCO.

compared to vanilla fine-tuning, both NegCLIP and MosaiCLIP methodologies bring improvements to relation and word order understanding, while maintaining or improving performance on attribute understanding. The MosaiCLIP methodology significantly improves relational reasoning performance and word-order understanding compared to the NegCLIP methodology, up to 6.3%. Attribute understanding performance remains nearly the same as the baseline BLIP performance, with the MosaiCLIP methodology bringing in slight gains over NegCLIP’s methodology. On average MosaiCLIP’s methodology brings more improvements to BLIP than NegCLIP or vanilla fine-tuning.

## 4.2 Analysis

We provide a detailed analysis of our models and baselines, across different dimensions as follows:

Disentangling MosaiCLIP improvements: We quantify the relative importance of the vision and language side by freezing the language and vision encoder individually while fine-tuning all models. See Fig. 4 c,d for the results. Notably, we find that 1) Language encoder has significant scope for improvement over NegCLIP’s language encoder, and MosaiCLIP is able to successfully exploit this potential and deliver an enhanced compositional understanding of language, which is evident by performance increase of +3.7, +6.9% over NegCLIP when only the language encoder is fine-tuned, as shown in Fig. 4 c,d. 2) Improvements brought by MosaiCLIP over NegCLIP in the text encoder are always higher than improvements in the image encoder. This is evident from Fig. 4 c,d where the performance increase over NegCLIP when only the language encoder is fine-tuned is always higher as compared to when only the image encoder is fine-tuned; for example, $3 . 7 \% > \ 0 . 0 \% , \ 6 . 9 \% > 1 . 8 \%$ for ARO-Relation, ARO-Attribution. 3) MosaiCLIP brings significant improvements on the image encoder side (higher than NegCLIP) without using any image negative mining, unlike NegCLIP.

MosaiCLIP improves hierarchical text understanding: For further understanding MosaiCLIP’s improved compositional understanding, we provide a novel analysis by considering the recently proposed Tree-Score (Murty et al., 2022) that measures the degree to which a transformer (text) encoder processes text in a hierarchical manner. We hypothesize that having tree-like hierarchical computation over language can be one leading factor for explaining the compositionality (or lack thereof) of CLIP-like models. Along with this, we have previously shown that the language encoder has the most prominent effect in improving compositionality in the case of MosaiCLIP . These two reasons motivate the use of tree-score to compare the language encoder’s hierarchical understanding capability. Fig. 4 a) shows that MosaiCLIP’s language encoder has higher tree-scores than NegCLIP’s language encoder, suggesting that MosaiCLIP performs more tree-like computations. This explains the improved language compositionality of MosaiCLIP since a hierarchical tree-structured computation allows the language encoder to better understand input text compositionally, thereby improving vision-language compositionality. This is in line with the hypothesis that human’s semantic understanding of sentences involves a hierarchical (tree-structured) computation which has significant evidence (Crain and Nakayama, 1987; Hale et al.,

![](images/a503cb822af53db82c07a83059c8b1a4ae5138fbb20f45f987353439b503b156.jpg)  
Figure 5: Qualitative analysis on ARO dataset (Top:ARO-Attribution, Bottom: ARO-Relation). Models highlighted in blue match the image to the correct sentence (in green) while the models in white match the image to the incorrect sentence (in red). Here, models are taken from our fine-tuning experiments on COCO from Table 1.

2018; Pallier et al., 2011) and this leads to their compositional generalization capability.

MosaiCLIP is Robust: Noisy texts often have meaningful sub-texts which can be exploted by MosaiCLIP, hence MosaiCLIP often achieves consistent performance increase regardless of noise in the pre-training or fine-tuning dataset. For example, NegCLIP achieves significantly low performance on ARO when fine-tuned with YFCC-FT (having more noise in text) as compared CC-FT or COCO as shown in Table 1. NegCLIP takes a > 10% hit in performance across various ARO datasets when the fine-tuning dataset is changed from COCO to YFCC, whereas, MosaiCLIP achieves similar performance using both datasets. Appendix Sec. D.3 shows that pre-trained MosaiCLIP is robust to natural distributon shifts.

Qualitative Analysis: We take MosaiCLIP, NegCLIP and CLIP fine-tuned on COCO and filter out examples from the ARO dataset where MosaiCLIP and NegCLIP’s disagree. Some notable examples in Fig. 5 include cases where NegCLIP and CLIP often struggle to understand simple concepts like understanding the color of the cat and table (top-left Fig. 5 or understanding the "is holding" relation b/w sandwich and the box in bottom-right Fig. 5.

## 4.3 Ablations

Table 6 and Appendix Tables 8,9 show the effect of curriculum learning and robust fine-tunining where we find that curriculum learning can bring consistent improvements of up to 1.2% on average and robust-finetuning (WiSE-FT) technique performs the best on zero-shot tasks (i.e. minimal forgetting while fine-tuning), while still improving over NegCLIP by about 5% on compositional reasoning tasks. Table 7 shows the effects of different kinds of sub-graphs sampled during training. More details including the effect of sampling larger number

of sub-graphs are presented in Appendix Sec. C.
<table><tr><td>Benchmark →</td><td colspan="3">ARO</td><td colspan="2">CREPE</td><td>VLC</td><td>SVO | Meta</td><td></td></tr><tr><td>Method ↓</td><td>Rel.</td><td>Attr.</td><td>Ord.</td><td>CU</td><td>AU</td><td>Avg.</td><td>Avg.</td><td>Avg.</td></tr><tr><td>MosaiCLIP</td><td>80.4</td><td>69.8</td><td>85.5</td><td>72.4</td><td>40.9</td><td>77.6</td><td>88.73</td><td>73.6</td></tr><tr><td>MosaiCLIPNoCurric</td><td>79.0</td><td>69.6</td><td>80.6</td><td>71.1</td><td>40.2</td><td>77.7</td><td>88.91</td><td>72.4</td></tr><tr><td>MosaiCLIPwiSE-FT</td><td>78.8</td><td>69.4</td><td>82.6</td><td>67.5</td><td>41.2</td><td>76.4</td><td>88.08</td><td>72.0</td></tr></table>

Table 6: Effect of Curriculum learning and Robust Finetuning (MosaiCLIP<sub>WiSE-FT</sub>) using CC-FT data.

<table><tr><td>Fine-tuning data → Method↓</td><td>COCO Rel. Attr.</td><td>CC-FT Rel. Attr.</td><td></td><td>YFCC-FT Rel. Attr.</td></tr><tr><td>MosaiCLIP</td><td>82.6 78.0</td><td>80.4</td><td>69.8</td><td>74.3 66.9</td></tr><tr><td>without frel</td><td>81.7 76.6</td><td>78.8</td><td>68.7</td><td>73.5 66.2</td></tr><tr><td>without fattr</td><td>77.7</td><td>73.2 70.5</td><td>68.2</td><td>69.0 65.9</td></tr><tr><td>without  $f _ { r e l } , f _ { a t t r }$ </td><td>79.0 70.4</td><td>68.8</td><td>64.9</td><td>57.4 63.6</td></tr></table>

Table 7: Effect of different positive-negative sub-graph types sampled while training. Results are presented on the ARO benchmark.

## 5 Conclusion

We present a method to improve the compositional reasoning capabilities of contrastively trained large vision-language models. In particular, we provide a coarse-to-fine contrastive learning framework and a scene graph-based text decomposition strategy for matching subgraphs of the text scene graph having varying complexity to an image during contrastive learning. We also develop hard negative graph creation strategies focused on improving attribute binding and relation understanding capabilities. Our techniques leads to significant improvements in compositional reasoning capabilities. We investigate the reasons for improved compositionality and present a novel finding based on language encoder tree-scores, suggesting that our models learn improved fine-grained and hierarchical text understanding, which is likely the key reason for improved vision and language compositionality of MosaiCLIP as compared to baselines.

## 6 Limitations

Computational Cost: Although MosaiCLIP leads to significant performance increase on several benchmarks that test compositional reasoining, it requires a higher per-batch computational cost while training. For this we give a detailed analysis on the computational cost in Appendix C.6 and show that simply providing more compute to prior methods in the form of larger batch sizes does not improve compositional reasoning. We also show ways to tackle this computational cost, by using less data in Appendix C.5, since MosaiCLIP is data efficient and can provide improvements over baselines with as low as 0.3x of the total data. This along with our ablations in Appendix C.1 gives some control to any practitioner to vary either the training dataset size or the number of sub-graphs in our method, and obtain a clean tradeoff between accuracy and compute. As future work we would like to develop a coarse-to-fine grained objective requiring minimal extra computation cost per batch. Future work should also look at decreasing the extra computational cost incurred by contemporary methods like Syn-CLIP (Cascante-Bonilla et al., 2023) and Teaching SVLC (Doveh et al., 2023).

Other Vision Language Models: In our current work we primarily aim to improve the compositionality of CLIP-Style, dual-tower models trained using large scale contrastive learning, since they severely lacked compostional reasoning capabilities as shown by (Yuksekgonul et al., 2022). Many other VLMs exist such as those that undergo cross modal interactions between vision and language such as BLIP (Li et al., 2022b), X-VLM (Zeng et al., 2021), LXMERT (Tan and Bansal, 2019). Although our methods show promise in improving more advanced VLMs like BLIP as shown in Section 4 and Table 5, a more thorough analysis will be beneficial to study the extent to which our methods can improve vision-language contrastive learning for these models.

Sentence Templates: For simplicity, we currently use manually curated templates to convert subgraphs to sentences, however, this can lead to similar looking and synthetic sentences. Large language models like GPT-4 (OpenAI, 2023), BLOOM (Mitchell et al., May 2021-May 2022) should be looked into for developing sentences from scene-graphs, by directly giving the LLM a scene-graph as input and requiring it to generate a sentence. This approach might be effective but may also lead to higher computational cost while training.

## Acknowledgements

We thank anonymous reviewers for their insightful suggestions that helped in greatly improving our paper. We also thank Aditi Khandelwal, Animesh Sinha, Abhishek Kadian for their helpful comments and suggestions on this work.

## References

Paola Cascante-Bonilla, Khaled Shehada, James Seale Smith, Sivan Doveh, Donghyun Kim, Rameswar Panda, Gül Varol, Aude Oliva, Vicente Ordonez, Rogerio Feris, and Leonid Karlinsky. 2023. Going beyond nouns with vision & language models using synthetic data.

Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. 2021. Conceptual 12M: Pushing webscale image-text pre-training to recognize long-tail visual concepts. In CVPR.

Stephen Crain and Mineharu Nakayama. 1987. Structure dependence in grammar formation. Language, 63(3):522–543.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee.

Anuj Diwan, Layne Berry, Eunsol Choi, David Harwath, and Kyle Mahowald. 2022. Why is winoground hard? investigating failures in visuolinguistic compositionality. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 2236–2250, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Sivan Doveh, Assaf Arbelle, Sivan Harary, Eli Schwartz, Roei Herzig, Raja Giryes, Rogerio Feris, Rameswar Panda, Shimon Ullman, and Leonid Karlinsky. 2023. Teaching structured vision & language concepts to vision & language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2657–2668.

Chuang Gan, Jeremy Schwartz, Seth Alter, Damian Mrowca, Martin Schrimpf, James Traer, Julian De Freitas, Jonas Kubilius, Abhishek Bhandwaldar, Nick Haber, Megumi Sano, Kuno Kim, Elias Wang, Michael Lingelbach, Aidan Curtis, Kevin Tyler Feigelis, Daniel Bear, Dan Gutfreund, David Daniel Cox, Antonio Torralba, James J. DiCarlo, Joshua B. Tenenbaum, Josh Mcdermott, and Daniel LK Yamins. 2021. ThreeDWorld: A platform for interactive

multi-modal physical simulation. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1).

Sachin Goyal, Ananya Kumar, Sankalp Garg, Zico Kolter, and Aditi Raghunathan. 2022. Finetune like you pretrain: Improved finetuning of zero-shot vision models.

John Hale, Chris Dyer, Adhiguna Kuncoro, and Jonathan Brennan. 2018. Finding syntax in human encephalography with beam search. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2727–2736, Melbourne, Australia. Association for Computational Linguistics.

Lisa Anne Hendricks and Aida Nematzadeh. 2021. Probing image-language transformers for verb understanding. In Findings ofthe Associationfor Computational Linguistics: ACL-IJCNLP 2021, pages 3635–3644, Online. Association for Computational Linguistics.

Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, and Justin Gilmer. 2021a. The many faces of robustness: A critical analysis of outof-distribution generalization. ICCV.

Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. 2021b. Natural adversarial examples. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15262–15271.

Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. 2021. OpenCLIP.

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. 2021. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, pages 4904–4916. PMLR.

Justin Johnson, Agrim Gupta, and Li Fei-Fei. 2018. Image generation from scene graphs. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1219–1228.

Justin Johnson, Ranjay Krishna, Michael Stark, Li-Jia Li, David Shamma, Michael Bernstein, and Li Fei-Fei. 2015. Image retrieval using scene graphs. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3668–3678.

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, Michael Bernstein, and Li Fei-Fei. 2016. Visual genome: Connecting language and vision using crowdsourced dense image annotations.

Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, and Percy Liang. 2022. Fine-tuning can distort pretrained features and underperform outof-distribution. arXiv preprint arXiv:2202.10054.

Chunyuan Li, Haotian Liu, Liunian Li, Pengchuan Zhang, Jyoti Aneja, Jianwei Yang, Ping Jin, Houdong Hu, Zicheng Liu, Yong Jae Lee, et al. 2022a. Elevater: A benchmark and toolkit for evaluating language-augmented visual models. Advances in Neural Information Processing Systems, 35:9287– 9301.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022b. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR.

Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. 2022c. Grounded language-image pre-training. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10965– 10975.

Yikang Li, Wanli Ouyang, Bolei Zhou, Kun Wang, and Xiaogang Wang. 2017. Scene graph generation from objects, phrases and region captions. In Proceedings of the IEEE international conference on computer vision, pages 1261–1270.

Yong-Lu Li, Liang Xu, Xinpeng Liu, Xijie Huang, Yue Xu, Mingyang Chen, Ze Ma, Shiyi Wang, Hao-Shu Fang, and Cewu Lu. 2019. Hake: Human activity knowledge engine. arXiv preprint arXiv:1904.06539.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer.

Zixian Ma, Jerry Hong, Mustafa Omer Gul, Mona Gandhi, Irena Gao, and Ranjay Krishna. 2022. Crepe: Can vision-language foundation models reason compositionally? arXiv preprint arXiv:2212.07796.

Margaret Mitchell, Giada Pistilli, Yacine Jernite, Ezinwanne Ozoani, Marissa Gerchick, Nazneen Rajani, Sasha Luccioni, Irene Solaiman, Maraim Masoud, Somaieh Nikpoor, Muñoz Ferrandis Carlos, Stas Bekman, Christopher Akiki, Danish Contractor, David Lansky, Angelina McMillan-Major, Tristan Thrush, Suzana Ilic, Gérard Dupont, Shayne Long-´ pre, Manan Dey, Stella Biderman, Douwe Kiela, Emi Baylor, Teven Le Scao, Aaron Gokaslan, Julien Launay, and Niklas Muennighoff. May 2021-May 2022. Bigscience, bigscience language open-science openaccess multilingual (bloom) language model. International.

Ron Mokady, Amir Hertz, and Amit H Bermano. 2021. Clipcap: Clip prefix for image captioning. arXiv preprint arXiv:2111.09734.

Shikhar Murty, Pratyusha Sharma, Jacob Andreas, and Christopher D Manning. 2022. Characterizing intrinsic compositionality in transformers with tree projections. arXiv preprint arXiv:2211.01288.

OpenAI. 2023. Gpt-4 technical report.

Christophe Pallier, Anne-Dominique Devauchelle, and Stanislas Dehaene. 2011. Cortical representation of the constituent structure of sentences. Proceedings ofthe National Academy ofSciences, 108(6):2522– 2527.

Khoi Pham, Kushal Kafle, Zhe Lin, Zhihong Ding, Scott Cohen, Quan Tran, and Abhinav Shrivastava. 2021. Learning to predict visual attributes in the wild. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13018–13028.

Sarah Pratt, Mark Yatskar, Luca Weihs, Ali Farhadi, and Aniruddha Kembhavi. 2020. Grounded situation recognition. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part IV 16, pages 314–332. Springer.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR.

Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. 2019. Do imagenet classifiers generalize to imagenet? In International conference on machine learning, pages 5389–5400. PMLR.

Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. 2018. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, Melbourne, Australia. Association for Computational Linguistics.

Hao Tan and Mohit Bansal. 2019. LXMERT: Learning cross-modality encoder representations from transformers. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5100–5111, Hong Kong, China. Association for Computational Linguistics.

Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. 2016. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73.

Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. 2022. Winoground: Probing vision and language models for visio-linguistic compositionality. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5238– 5248.

Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. 2019. Learning robust global representations by penalizing local predictive power. Advances in Neural Information Processing Systems, 32.

Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, et al. 2022. Robust fine-tuning of zero-shot models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7959–7971.

Hao Wu, Jiayuan Mao, Yufeng Zhang, Yuning Jiang, Lei Li, Weiwei Sun, and Wei-Ying Ma. 2019. Unified visual-semantic embeddings: Bridging vision and language with structured meaning representations. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6609–6618.

Danfei Xu, Yuke Zhu, Christopher B Choy, and Li Fei-Fei. 2017. Scene graph generation by iterative message passing. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5410–5419.

Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Bin Xiao, Ce Liu, Lu Yuan, and Jianfeng Gao. 2022. Unified contrastive learning in image-text-label space. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19163– 19173.

Xu Yang, Kaihua Tang, Hanwang Zhang, and Jianfei Cai. 2019. Auto-encoding scene graphs for image captioning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10685–10694.

Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. 2014. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Associationfor Computational Linguistics, 2:67–78.

Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. 2022. When and why vision-language models behave like bag-of-words models, and what to do about it? arXiv preprint arXiv:2210.01936.

Yan Zeng, Xinsong Zhang, and Hang Li. 2021. Multi-grained vision language pre-training: Aligning texts with visual concepts. arXiv preprint arXiv:2111.08276.

Ji Zhang, Kevin J Shih, Ahmed Elgammal, Andrew Tao, and Bryan Catanzaro. 2019. Graphical contrastive losses for scene graph parsing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11535–11543.

Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. 2021. Vinvl: Revisiting visual representations in vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5579–5588.

Tiancheng Zhao, Tianqi Zhang, Mingwei Zhu, Haozhan Shen, Kyusong Lee, Xiaopeng Lu, and Jianwei Yin. 2022. Vl-checklist: Evaluating pre-trained visionlanguage models with objects, attributes and relations.

Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, et al. 2022. Regionclip: Region-based language-image pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16793–16803.

## Appendix

## A Background

Contrastive Language-Image pre-training (Radford et al., 2021) (CLIP) aims to learn generalpurpose representations of vision and language using paired image-text data. This is achieved using contrastive learning in the image-text space. In particular consider a pre-training dataset of size $n , \mathcal { D } \subset \mathcal { X } \times \mathcal { T } , \mathcal { D } = \{ \pmb { x } _ { i } , \pmb { t } _ { i } \} _ { i = 1 } ^ { n }$ . Here $\mathcal { X }$ and $\tau$ are the space of images and text, respectively, and ${ \bf \mathit { x } } _ { i } , t _ { i }$ are images and text in the dataset. Also, consider access to image and text encoders, that we represent by $f _ { \theta } : \mathcal { X }  \mathbb { R } ^ { d }$ and $f _ { \phi } : \mathcal { T }  \mathbb { R } ^ { d }$ respectively. To learn distributed representations for images and text, the following contrastive losses are used:

$$
\mathcal { L } _ { t 2 i } = - \left. \frac { 1 } { | \mathcal { B } | } \sum _ { j = 1 } ^ { | \mathcal { B } | } \log \frac { \exp ( \tau \pmb { u } _ { i } ^ { T } \pmb { v } _ { j } ) } { \sum _ { i = 1 } ^ { | \mathcal { B } | } \exp ( \tau \pmb { u } _ { i } ^ { T } \pmb { v } _ { j } ) } \right.\tag{1}
$$

$$
\mathcal { L } _ { i 2 t } = - \frac { 1 } { | \mathcal { B } | } \sum _ { i = 1 } ^ { | \mathcal { B } | } \log \frac { \exp ( \tau \pmb { u } _ { j } ^ { T } \pmb { v } _ { j } ) } { \sum _ { j = 1 } ^ { | \mathcal { B } | } \exp ( \tau \pmb { u } _ { i } ^ { T } \pmb { v } _ { j } ) }\tag{2}
$$

Where  represents the batch during one iteration of training. ${ \mathbf { } } u _ { i } , v _ { i }$ are the $\ell _ { 2 }$ normalized embeddings of $\tilde { { \mathbf { } } u _ { i } } , \tilde { { \mathbf { } } v _ { i } }$ , where $\tilde { { \mathbf { u } } _ { i } } = f _ { \theta } ( \mathbf { x } _ { i } ) , \tilde { { \mathbf { v } } _ { i } } = f _ { \phi } ( { \mathbf { t } _ { i } } )$ $\tau$ is the temperature parameter and is trainable. The overall loss is $\mathcal { L } _ { c l i p } = ( \mathcal { L } _ { t 2 i } + \mathcal { L } _ { i 2 t } ) / 2$

## B Scene Graph Decomposition

Here we provide additional details for text scene graph decomposition. Denote the text scene graph obtained from the scene graph parser by $G _ { T } =$ $( V _ { T } , E _ { T } )$ , where $V _ { T }$ represent the nodes of the graph, which are either objects or their attributes. $E _ { T }$ are the edges of the graph that represent relations between objects. Let G denote the set of all possible scene graphs. We first consider an external set of objects $( \mathcal { N } )$ , attributes $( \mathcal { A } )$ , and relations $( { \mathcal { R } } )$ that we use for creating negative sub-graphs. In practice, we create this set from Visual Genome (VG) dataset (Krishna et al., 2016). Following Zhang et al. (2021), we sample a total of 1594 entities that have 30 instances of them in the VG dataset. The attribute and Relation list contains 524, and 50 unique instances, respectively. Hence $| \mathcal { N } | = 1 5 9 4 , | \mathcal { A } | = 5 2 4 , | \mathcal { R } | = 5 0$ We first sample all possible sub-graphs having one or two objects in them, and these can have multiple attributes for the objects. We develop and use scene graph transformations that take a sub-graph as input and return a (set of) modified versions of the graph (minimally-perturbed negative sub-graphs for the image). For this, we define three graph transformations as follows:

$f _ { o b j } : \mathbb { G } \longrightarrow P ( \mathbb { G } )$ takes input a single object scene graph, where the object has attributes $A _ { o }$ . For each attribute, $a \in A _ { o }$ , a random attribute $a ^ { \prime }$ is sampled uniformly at random from ${ \mathcal { A } } .$ We finally obtain a set of sub-graphs $G _ { o b j } \in P ( \mathbb { G } )$ where $P ( . )$ denotes the power set. Each $g \in G _ { o b j }$ contains one object node connected with an attribute node which is sampled from .

$f _ { r e l } : \mathbb { G } \longrightarrow P ( \mathbb { G } )$ takes input sub-graphs having one relation edge and gives output a set of sub-graphs $G _ { r e l } \in P ( \mathbb { G } )$ where each $g \in G _ { r e l }$ has either object nodes shuffled, replaced by an external object node $n ^ { \prime }$ sampled uniformly at random from ${ \mathcal N }$ , and/or relation replaced by external relation $r ^ { \prime }$ sampled uniformly at random from $\mathcal { R }$ . Along with this, we also join the input positive sub-graph with a random sub-graph created by sampling random nodes and edges from $\mathcal { N } , A , \mathcal { R }$

$f _ { a t t r } : \mathbb { G } \longrightarrow P ( \mathbb { G } )$ takes input sub-graphs having one relation edge and gives output a set of sub-graphs $G _ { a t t r } \in P ( \mathbb { G } )$ where each $g \in G _ { a t t r }$ has attribute nodes shuffled, and/or replaced by an external attribute node $a ^ { \prime }$ sampled uniformly at random from ${ \mathcal { A } } .$

$f _ { o b j } , f _ { a t t r }$ broadly aims at improving the model’s attribute understanding, while $f _ { r e l }$ broadly targets improved relation understanding. For each positive sub-graph, we sample all possible negative subgraphs using $f _ { o b j } , f _ { r e l } , f _ { a t t r }$ and make positive-negative sub-graph pairs $( g _ { p o s _ { i } } , \{ g _ { n e g _ { i } } \} )$ These pairs can be classified into three categories $C = \{ c _ { o b j } , c _ { r e l } , c _ { a t t r } \}$ according to the transformation that created the negative sub-graphs. We sample sub-graph pairs from these categories according to probabilities $p _ { i } , i \in \{ 1 , 2 , 3 \}$ corresponding to the three categories respectively, and $\sum p _ { i } = 1$ These probabilities are hyperparameters; see $\mathsf { A p - }$ pendix Section H.1 for more details. Multiple subgraph pairs can have common positive or negative sub-graphs, and sampling these pairs would result in duplication, hence for each image, we make sure to deduplicate sub-graphs so that all sub-graphs, and therefore the text made from them are unique for a given image in a batch. After sampling, all sub-graphs are transformed to text using simple templates, as explained in Section 3.3.

## C Ablations and Model Analysis

## C.1 Sampling more subgraphs

We analyze the effect of increasing the maximum number of sub-graphs sampled for any given image in a batch of data during training. See Figures 6 and 7, in which we test the performance on ARO and CREPE benchmarks (averaged over three finetuning datasets considered in this work), as we increase the max positive and negative sub-graphs per image. We find that as we increase both positive and negative sub-graphs for an image, the performance steadily increases up to a point for all datasets, after which the performance can either flatten out, increase, or even decrease in some of the datasets. This is intuitive since a larger number of positive and negative sub-graphs per image leads to a gap w.r.t the pre-training stage as described in Sec. 3.6. Also, different compositional splits require different reasoning skills, and as we keep sampling positive and negative sub-graphs for an image, it is natural for certain types of positive and negative sub-graphs to be more pronounced, depending on the dataset statistics, and this can have varied effects on different datasets.

![](images/70230eb5881f351bad39893efc78f12a4e0a8e045d9a23f71b9eb049b55928d1.jpg)

![](images/7787aa3240dd1dfb6fe4857967c96dc8a5cc659a9b08d7f4480273e4e45a9275.jpg)  
Number of (pos, neg) subgraphs  
Figure 6: Effect of increasing the number of positive and negative subgraphs on ARO benchmark when finetuning MosaiCLIP. Results are averaged over 3 finetuning datasets considered in this work

## C.2 Effect of different sub-graph types

Here we analyze the effect of sampling different kinds of sub-graphs from the original scene graph of the text. In particular, we measure the effect of graph transformations that we define in Appendix

![](images/ea76aa2291ee4ea8ca1c43f0dcac5c78e487feecbe8ca48b5609ae14962802ad.jpg)

![](images/f9abf9e1afbafe260962c6b35ad92ffcab8df0563d8410bfb83207f0066564da.jpg)  
Number of (pos, neg) subgraphs  
Figure 7: Effect of increasing the number of positive and negative subgraphs on CREPE - Systematicity benchmark, when fine-tuning MosaiCLIP (Here we use Open-CLIP RN-50 model pre-trained on CC-12M and finetune it on CC-FT).

Sec. B. Results are presented in Table 7. We observe that both $f _ { r e l }$ and $f _ { a t t r }$ as described in Appendix Sec. B, are useful for improving relation and attribute understanding (as measured on the ARO benchmark), across fine-tuning datasets.

## C.3 Effect of curriculum training

As shown in Table 8, in all fine-tuning results, we can see consistent improvements when using our curriculum learning strategy, such as upto 2% on systematic generalization, and sometimes more than 6% as seen for ARO-Order results when the fine-tuning dataset is YFCC-FT.

## C.4 Effect of robust fine-tuning

Among many other techniques developed for mitigating forgetting in large models when they are fine-tuned, one prominent one is robust fin-tuning-WiSE-FT, (Wortsman et al., 2022). Following Wortsman et al. (2022) we perform weight-space ensembling on the image encoder before and after fine-tuning using our method and call this model MosaiCLIP<sub>WiSE-FT</sub>. The results on compositionality benchmarks can be seen in Table 8 while results on 21 multimodal tasks from ELEVATER and ImageNet can be seen in Table 9. We find that MosaiCLIP<sub>WiSE-FT</sub> has a slight performance decrease on some compositonal benchmarks as compared to MosaiCLIP, however, it is significantly better than NegCLIP on most benchmarks. The real benefit of using MosaiCLIP<sub>WiSE-FT</sub> is that it leads to least forgetting, and there is little to no performance degradation on 21 tasks as showin in Table 9.

<table><tr><td>FineTun. data →</td><td></td><td colspan="5">COCO</td><td colspan="5">CC-FT</td><td colspan="5"></td><td colspan="5">YFCC-FT</td><td></td></tr><tr><td>Benchmark</td><td>→</td><td></td><td>ARO</td><td></td><td>VLC</td><td>SVO</td><td>ARO</td><td></td><td></td><td>CREPE</td><td></td><td>VLC</td><td>SVO</td><td></td><td>ARO</td><td></td><td></td><td>CREPE</td><td></td><td>VLC</td><td>SVO</td><td>Meta</td></tr><tr><td>Method ↓</td><td>Rel.</td><td></td><td>Attr.</td><td>Ord. Avg.</td><td></td><td>Avg.</td><td>Rel. Attr.</td><td>Ord.</td><td></td><td>CU</td><td>AU</td><td>Avg.</td><td>Avg.</td><td></td><td>Rel. Attr.</td><td>Ord.</td><td>CU</td><td></td><td>AU</td><td>Avg.</td><td>Avg.</td><td>Avg.</td></tr><tr><td>CLIP</td><td>59.8</td><td>63.2</td><td>53.3</td><td>70.8</td><td></td><td>83.58</td><td>59.8 63.2</td><td>53.3</td><td>45.1</td><td>35.0</td><td></td><td>70.8</td><td>83.58</td><td>59.8</td><td>63.2</td><td>53.3</td><td>39.8</td><td>39.5</td><td></td><td>70.8</td><td>83.58</td><td>60.60</td></tr><tr><td>NegCLIP</td><td>81.7</td><td></td><td>72.7</td><td>85.7</td><td>75.6</td><td>90.20</td><td>71.5</td><td>65.4 84.5</td><td></td><td>53.1</td><td>37.5</td><td>72.4</td><td>88.36</td><td>57.8</td><td>63.1</td><td>52.1</td><td>38.8</td><td></td><td>39.0</td><td>70.4</td><td>83.90</td><td>67.57</td></tr><tr><td>MosaiCLIP</td><td>82.6</td><td>78.0</td><td></td><td>87.1</td><td>81.4</td><td>90.67</td><td>80.4 69.8</td><td>85.5</td><td>72.4</td><td></td><td>40.9</td><td>77.6</td><td>88.73</td><td>74.3</td><td>66.9</td><td>84.4</td><td>48.8</td><td>41.5</td><td></td><td>75.1</td><td>85.36</td><td>74.29</td></tr><tr><td>MosaiCLIPNoCurric</td><td>81.6</td><td>76.8</td><td></td><td>87.4</td><td>81.4</td><td>90.20</td><td>79.0 69.6</td><td>80.6</td><td>71.1</td><td></td><td>40.2</td><td>77.7</td><td>88.91</td><td></td><td>74.1 67.2</td><td>77.8</td><td>46.6</td><td>40.5</td><td></td><td>75.7</td><td>84.97</td><td>73.23</td></tr><tr><td>MosaiCLIPwiSE-FT</td><td>82.5</td><td>76.2</td><td></td><td>86.6</td><td>80.3</td><td>89.65</td><td>78.8</td><td>69.4 82.6</td><td>67.5</td><td></td><td>41.2</td><td>76.4</td><td>88.08</td><td></td><td>69.4 67.0</td><td>79.4</td><td>48.1</td><td></td><td>43.6</td><td>74.2</td><td>83.71</td><td>72.88</td></tr></table>

Table 8: Ablating the effect of Curriculum learning and Robust fine-tuning. MosaiCLIP refers to the version of our model without any curriculum learning. MosaiCLIP refers to the version where the image encoder of the final model (after fine-tuning) and before fine-tuning are weight-space ensembled. CLIP and NegCLIP scores are also shown for reference. See Appendix Sec. C.3.

<table><tr><td>Method</td><td>ZS(21) Compositional Score (Meta Avg.)</td></tr><tr><td>CLIP</td><td>56.4 60.60</td></tr><tr><td>NegCLIP</td><td>56.8 67.57</td></tr><tr><td>MosaiCLIPNoCurric</td><td>55.8 73.23</td></tr><tr><td>MosaiCLIPwiSE-FT</td><td>56.8 72.88</td></tr><tr><td>MosaiCLIP 55.7</td><td>74.29</td></tr></table>

Table 9: Zero Shot accuracy on 21 multimodal datasets from ELEVATER and ImageNet. Results are average of the three fine-tuning datasets. MosaiCLIP has negligible drop in performance in general (compared to the gains on compositionality benchmarks), and one can boost performance by using MosaiCLIP<sub>WiSE-FT</sub> which has equal performance as compared to NegCLIP on 21 muldimodal datasets. Meta Avg. Compositional Score is taken from Table 8. Second best results are underlined. Conclusion: One can use MosaiCLIP for getting the best compositional reasoning capabilities with minimal performance degradation on multimodal tasks, and use MosaiCLIP<sub>WiSE-FT</sub> for no degradation in performance on multimodal tasks, while still performing well on compositional reasoning.

## C.5 Data efficiency

We find that our technique leads to significant data efficiency requiring about 0.3x-0.6x fo the total fine-tuning or pre-training data to match or exceed NegCLIP performance. Results are shown in Tables 10 and 11.

## C.6 Computational cost

Even though MosaiCLIP uses the same global batch size of image-text pairs, it requires more compute as compared to NegCLIP or CLIP owing to the fact that decomposing sub-graph leads to a larger effective text-batch size and hence a larger contrastive learning matrix. It is a common practice in literature to trade-off larger compute for improving

<table><tr><td rowspan="2">Method</td><td rowspan="2">Fraction of data</td><td colspan="2">ARO</td><td>SVO</td></tr><tr><td>Rel.</td><td>Attr.</td><td>Avg.</td></tr><tr><td>NegCLIP</td><td>Full</td><td>73.6</td><td>58.9</td><td>76.10</td></tr><tr><td rowspan="5">MosaiCLIP</td><td>0.3x</td><td>71.6</td><td>60.6</td><td>70.82</td></tr><tr><td>0.5x</td><td>74.3</td><td>60.8</td><td>74.04</td></tr><tr><td>0.6x</td><td>74.7</td><td>63.8</td><td>75.76</td></tr><tr><td>0.8x</td><td>77.0</td><td>66.3</td><td>77.22</td></tr><tr><td>Full</td><td>74.7</td><td>66.1</td><td>77.87</td></tr></table>

Table 10: Data efficiency of MosaiCLIP during pretraining. Numbers in blue are lowest numbers that are within 1% or greater than NegCLIP performance. Pre-Training dataset: YFCC-15M.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Fraction of data</td><td colspan="2">ARO</td><td>SVO</td></tr><tr><td>Rel.</td><td>Attr.</td><td>Avg.</td></tr><tr><td>NegCLIP</td><td>Full</td><td>71.5</td><td>65.4</td><td>88.36</td></tr><tr><td rowspan="5">MosaiCLIP</td><td>0.3x</td><td>70.8</td><td>67.7</td><td>88.70</td></tr><tr><td>0.5x</td><td>74.5</td><td>68.6</td><td>88.80</td></tr><tr><td>0.6x</td><td>75.3</td><td>69.3</td><td>88.76</td></tr><tr><td>0.8x</td><td>78.2</td><td>69.8</td><td>88.98</td></tr><tr><td>Full</td><td>79.0</td><td>69.6</td><td>88.91</td></tr></table>

Table 11: Data efficiency of MosaiCLIP during finetuning. Numbers in blue are lowest numbers that are within 1% or greater than NegCLIP performance. Finetuning dataset: CC-FT. Curriculum learning has not been used for these experiments.

CLIP’s compositionality, as also done by previous methods Syn-CLIP (Cascante-Bonilla et al., 2023) that generate data using external graphics engines, and Teaching-SVLC (Doveh et al., 2023) which use LLMs requiring massive compute even during inference.

Providing NegCLIP with more compute: One can argue that providing more compute to Neg-CLIP can lead to better performance, however, on the contrary we found that NegCLIP’s performance decreases as batch size is scaled (from 256 to 4096, much beyond MosaiCLIP’s text or image batch size), as shown in Table 12.

Performance-Compute Tradeoff: It is to be noted that MosaiCLIP performance continues to increase up to a threshold, as sub-graphs are increased as shown in Table 7 and 6 hence this provides a clean tradeoff between number of sub-graphs and compute, and a practitioner can choose the number of sub-graphs their compute availablility. Along with this, in Appendix Sec. C.5 we showed that we can achieve improved performance compared to NegCLIP with as low as 0.3x data closing the gap between NegCLIP and MosaiCLIP compute even more. It is to be noted that MosaiCLIP is a drop in replacement for CLIP after training and requires the same inference cost as CLIP.

<table><tr><td rowspan="2">Batch Size (B)</td><td colspan="2">ARO</td><td>SVO</td></tr><tr><td>Rel.</td><td>Attr.</td><td>Avg.</td></tr><tr><td>512</td><td>68.9</td><td>65.6</td><td>88.68</td></tr><tr><td>1024</td><td>67.6</td><td>65.1</td><td>88.93</td></tr><tr><td>2048</td><td>65.7</td><td>64.2</td><td>88.72</td></tr><tr><td>4096</td><td>62.5</td><td>63.7</td><td>88.11</td></tr></table>

Table 12: Performance of NegCLIP with increasing batch size. A batch size of B corresponds to an effective batch size of 8\*B in NegCLIP after image and text negative mining. Fine-tuning dataset: CC-FT.

## D Additional Results and Experiments

## D.1 Comparison with recent baselines

We compare with recently published and contemporary works (Cascante-Bonilla et al., 2023; Doveh et al., 2023). Doveh et al. (2023) show that one can create rule-based hard negative sentences and Large Language Models (LLMs) based hard negative sentences and use them when training CLIP style models to obtain an improved model that is better at handling tasks that require compositional reasoning. We fine-tune on CC3M (Sharma et al., 2018) for a fair comparison with Doveh et al. (2023). Results are reported in Table 13. A fair comparison with Syn-CLIP Cascante-Bonilla et al. (2023) is not possible since their synthetic dataset is not released. However in Table 13 we find that performance difference is large between MosaiCLIP and Syn-CLIP showing that our general coarse-to-fine grained approach is better than using targeted synthetic datasets for inducing compositional understanding in VLMs. Comparisons with Doveh et al. (2023) in

Table show that our approach is competitve or better at attribute, relation and object understanding as measured by the VL-Checklist benchmark (Zhao et al., 2022). Zero Shot performance on 21 datasets suffers minimally using our approach, and is even better than (Zhao et al., 2022). It is to be noted that both approaches Syn-CLIP (Cascante-Bonilla et al., 2023) and Doveh et al. (2023) are orthogonal to our approach and combining them with our coarseto-fine understanding approach will likely result in much better performance overall, as compared to individual techniques. In particular, Syn-CLIP (Cascante-Bonilla et al., 2023) faces the issue of having long captions for images, and they average out embeddings of parts of the caption before matching it to the image. This issue can be eaily resolved using our framework which can easily handle multiple positive captions for an image. Performing this ablation would be future work for us, once synthetic datasets like that used by Cascante-Bonilla et al. (2023) are open-sourced and gain more popularity. Our approach can similarly also include captions generated from LLMs, as explored by Doveh et al. (2023).

## D.2 Standard deviations for fine-tuning results

Here we provide fine-tuning results on the CC-FT dataset with standard deviations over 3 random seeds where OpenAI CLIP-ViT-B-32 is fine-tuned on CC-FT using MosaiCLIP and baseline techniques. See Table 14 for the results. The main paper Table 1 have average results for CC-FT while for COCO and YFCC-FT fine-tuning datasets, the results are for one seed. We do-not run multiple pre-training experiments since they significantly more costly.

## D.3 Robustness to natural distribution shifts

We find that pre-trained MosaiCLIP shows robustness to natural distribution shifts as measured by ImageNet natural distribution shifts benchmark. Results are presented in Table 15. We believe that MosaiCLIP sees a larger variety of texts in the form of sub-graphs which can provide it with extra supervision for tackling natural distribution shifts. Intutively, sub-graphs can lead to diversity of texts being seen by the model during training and this might lead to broader coverage of concepts and concept combinations, resulting in improved robustness. Along with this a coarse to fine hierarchical understanding of texts and thereby, of images should intuitively help in improving performance on robustness benchmarks given that the model will now be able to recognise details in images and texts more accuractely.

<table><tr><td rowspan="2">Benchmark → Method</td><td colspan="3">VL-Checklist</td><td colspan="3">ARO</td><td rowspan="2">ZS(21)</td></tr><tr><td>Obj.</td><td>Attr.</td><td>Rel.</td><td>Rel.</td><td>Attr.</td><td>Ord.</td></tr><tr><td>CLIP</td><td>81.6</td><td>67.6</td><td>63.1</td><td>59.9</td><td>63.6</td><td>53.3</td><td>Avg. 56.4</td></tr><tr><td>CLIP-FT</td><td>79.0</td><td>64.7</td><td>54.3</td><td>41.7</td><td>59.3</td><td>25.2</td><td>56.9</td></tr><tr><td>Syn-CLIP† (Cascante-Bonilla et al., 2023)</td><td></td><td>70.4</td><td>69.4</td><td>71.4</td><td>66.9</td><td>65.1</td><td>55.3</td></tr><tr><td>Teaching SVLC (Doveh et al., 2023)</td><td>85.0</td><td>72.0</td><td>69.0</td><td></td><td></td><td></td><td>54.8</td></tr><tr><td>MosaiCLIPNoCurric</td><td>86.4</td><td>75.0</td><td>69.6</td><td>83.2</td><td>78.6</td><td>77.3</td><td>54.9</td></tr><tr><td>MosaiCLIP WiSE-FT</td><td>86.5</td><td>73.6</td><td>72.2</td><td>82.6</td><td>77.0</td><td>79.9</td><td>55.9</td></tr><tr><td>MosaiCLIP</td><td>86.4</td><td>73.7</td><td>71.9</td><td>83.7</td><td>78.0</td><td>79.4</td><td>53.5</td></tr></table>

Table 13: Comparison of MosaiCLIP with recently published and contemporary works Syn-CLIP (Cascante-Bonilla et al., 2023) and Teaching SVLC Doveh et al. (2023). Results are reported on VL-Checklist, ARO and Average Zero Shot results on 21 datasets from ELEVATER and Imagenet. Performance numbers of these models are reported from their respective papers (blank fields (—) are not reported in respective papers). †Uses million-scale synthetic data for fine-tuning. ‡Uses external Large Language Models (LLMs) like BLOOM (Mitchell et al., May 2021-May 2022) for text augmentation and hard negative text creation. See Sec. D.1 for more details.
<table><tr><td>Benchmark →</td><td colspan="3">ARO</td><td colspan="3">SVO-Probes</td></tr><tr><td>Method↓</td><td>Rel.</td><td>Attr.</td><td>Ord.</td><td>Obj.</td><td>Subj.</td><td>Verb.</td></tr><tr><td>CLIP-FT</td><td>58.1±0.63</td><td>63.3±0.28</td><td>42.7±0.18</td><td>93.17±0.11</td><td>88.64±0.17</td><td>83.87±0.03</td></tr><tr><td>NegCLIP</td><td>71.5±0.40</td><td>65.4±0.58</td><td>84.5±0.11</td><td>92.90±0.09</td><td>88.16±0.11</td><td>84.02±0.02</td></tr><tr><td>MosaiCLIPNoCurric MosaiCLIP</td><td>79.0±0.66 80.4±0.63</td><td>69.6±0.19 69.8±0.21</td><td>80.6±0.17 85.5±0.16</td><td>93.37±0.04 93.45±0.04</td><td>89.74±0.13 89.39±0.07</td><td>83.62±0.04 83.35±0.05</td></tr></table>

Table 14: Fine-Tuning Results on CC-FT dataset with standard deviations across 3 random seeds. These results correspond to the CC-FT fine-tuning results in main paper Table 1. Here the base model which is fine-tuned using different techniques is OpenAI-CLIP-ViT-B-32.

## E Dataset Details

Here we provide detailes about datasets used for fine-tuning, pre-training and evaluating models in this study. A summary is shown in Table 16

## E.1 Fine-tuning datasets

Following NegCLIP (Yuksekgonul et al., 2022) we use the COCO dataset released by (Yuksekgonul et al., 2022) having 109k samples that had hard negative sentences that (Yuksekgonul et al., 2022) create for training NegCLIP. As mentioned in the main paper, COCO dataset images are used for creating Visual Genome (Krishna et al., 2016), and this is further used to create datasets such as CREPE (Ma et al., 2022), ARO (Yuksekgonul et al., 2022) and a part of VL-Checklist (Zhao et al., 2022). This can lead to confounding and potentially misleading results, since it is unclear if the performance increase using any method comes from the finetuning dataset (COCO) being close to the domain of test datasets, or if it’s the fine-tuning methodology that leads to an increase in performance. Hence, for rigourous experimentation of the developed methods, one must use other datasets to finetune contrastively trained VLMs. We randomly sample similar sized (100k datapoints) from popular pre-training datasets CC-12M and YFCC-15M, and call these smaller datasets CC-FT and YFCC-FT. To train NegCLIP, hard negative sentences and images are required, for which we first use the code released by (Yuksekgonul et al., 2022)<sup>2</sup> to create hard negatives sentences as well as sample three hard negative images for each image based on OpenAI CLIP ViT-B/32 features, strictly following (Yuksekgonul et al., 2022). For comparing with contemporary works (Doveh et al., 2023), (Cascante-Bonilla et al., 2023) (as shown in Table

<table><tr><td></td><td></td><td></td><td colspan="2">ImageNet-A</td><td colspan="2">ImageNet-R</td><td colspan="2">ImageNet-S</td><td colspan="2">ImageNet-V2</td></tr><tr><td>Arch.</td><td>Data</td><td>Method</td><td>Top1</td><td>Top5</td><td>Top1</td><td>Top5</td><td>Top1</td><td>Top5</td><td>Top1</td><td>Top5</td></tr><tr><td rowspan="6">Swi-T</td><td rowspan="2">CC-12M</td><td>CLIP</td><td>6.4</td><td>24.5</td><td>42.6</td><td>68.8</td><td>22.2</td><td>45.5</td><td>28.2</td><td>54.1</td></tr><tr><td>NegCLIP</td><td>6.6</td><td>25.0</td><td>43.1</td><td>68.7</td><td>22.2</td><td>45.4</td><td>29.4</td><td>55.2</td></tr><tr><td rowspan="2"></td><td>MosaiCLIP</td><td>9.1</td><td>29.4</td><td>48.6</td><td>74.3</td><td>27.2</td><td>52.6</td><td>33.6</td><td>61.6</td></tr><tr><td>CLIP</td><td>10.9</td><td>34.2</td><td>20.6</td><td>42.0</td><td>6.4</td><td>16.7</td><td>26.1</td><td>49.9</td></tr><tr><td rowspan="2">YFCC-15M</td><td>NegCLIP</td><td>11.4</td><td>35.6</td><td>20.0</td><td>41.7</td><td>6.0</td><td>16.0</td><td>27.2</td><td>50.7</td></tr><tr><td>MosaiCLIP</td><td>14.6</td><td>40.2</td><td>22.3</td><td>44.9</td><td>6.8</td><td>17.7</td><td>32.0</td><td>57.2</td></tr><tr><td rowspan="6">RN-50</td><td rowspan="2">CC-12M</td><td>CLIP</td><td>7.3</td><td>27.4</td><td>41.4</td><td>67.8</td><td>21.7</td><td>44.3</td><td>29.8</td><td>56.4</td></tr><tr><td>NegCLIP</td><td>7.7</td><td>27.7</td><td>41.0</td><td>66.9</td><td>21.7</td><td>43.9</td><td>30.2</td><td>56.0</td></tr><tr><td rowspan="2"></td><td>MosaiCLIP</td><td>11.1</td><td>35.6</td><td>52.1</td><td>76.9</td><td>29.5</td><td>55.4</td><td>37.0</td><td>66.5</td></tr><tr><td>CLIP</td><td>13.4</td><td>37.3</td><td>17.2</td><td>37.2</td><td>4.9</td><td>13.6</td><td>25.8</td><td>49.4</td></tr><tr><td rowspan="2">YFCC-15M</td><td>NegCLIP</td><td>12.9</td><td>38.0</td><td>18.0</td><td>37.3</td><td>5.1</td><td>14.7</td><td>26.0</td><td>49.0</td></tr><tr><td>MosaiCLIP</td><td>17.4</td><td>46.6</td><td>21.0</td><td>42.7</td><td>6.5</td><td>16.9</td><td>32.2</td><td>57.9</td></tr></table>

Table 15: Results on ImageNet - Natural Distribution Shifts datasets. MosaiCLIP leads to improved robustness to natural distribution shifts. NegCLIP performs similarly as CLIP. Models are zero-shot tested on ImageNet-A (Hendrycks et al., 2021b), ImageNet-R (Hendrycks et al., 2021a), ImageNet-S(ketch) (Wang et al., 2019) and ImageNet-V2 (Recht et al., 2019).
<table><tr><td>Benchmark/Dataset</td><td>#Examples</td><td>#Subtasks Subtask Examples</td><td>Datasets Used for Creation</td></tr><tr><td colspan="4">Compositional Reasoning (Evaluation)</td></tr><tr><td>ARO</td><td>77K 3</td><td>Attribute, Relation, Order understanding</td><td>Visual Genome, COCO, Flickr</td></tr><tr><td>CREPE-Systematicity</td><td>642K 2</td><td>Systematic generalization generalization</td><td>Visual Genome</td></tr><tr><td>VL-Checklist</td><td>410K 3</td><td>Attribute, Relation Object understanding</td><td>Visual Genome HAKE, VAW, SWiG</td></tr><tr><td>SVO-Probes</td><td>48K 3</td><td>Verbs (Relations) understanding</td><td></td></tr><tr><td>CREPE-Productivity</td><td>183K 9</td><td>Productivity</td><td>Visual Genome</td></tr><tr><td colspan="4">Fine-Tuning datasets</td></tr><tr><td>COCO</td><td>109K</td><td></td><td></td></tr><tr><td>CC-FT</td><td>100K</td><td></td><td></td></tr><tr><td>YFCC-FT</td><td>100K</td><td></td><td></td></tr><tr><td>CC-3M</td><td>3.11M</td><td></td><td></td></tr><tr><td colspan="4">Pre-Training datasets</td></tr><tr><td>CC-12M</td><td>11.26M</td><td></td><td></td></tr><tr><td>YFCC-15M</td><td>14.20M</td><td></td><td></td></tr></table>

Citations: ARO(Yuksekgonul et al., 2022), CREPE(Ma et al., 2022), VL-Checklist(Zhao et al., 2022), SVO(Hendricks and Nematzadeh, 2021), Visual Genome(Krishna et al., 2016), COCO(Lin et al., 2014), Flickr(Young et al., 2014), HAKE(Li et al., 2019), VAW(Pham et al., 2021), SWiG(Pratt et al., 2020)

Table 16: Details of datasets used in this study for testing compositional reasoning, for fine-tuning and pre-training models. See Appendix Sec. E for more details.

2), we use CC3M (Sharma et al., 2018) since it’s used by these baselines, and makes a direct comparison possible with them.

## E.2 Pre-training datasets

We use popular and standard large scale pretraining datasets CC-12M (Changpinyo et al., 2021)

and YFCC-15M (Thomee et al., 2016) for pretraining all models in this study, including CLIP, NegCLIP and MosaiCLIP.

## E.3 Evaluation datasets

Here we list the evaluation detailes used in this study and also provide a short description for each CREPE-Systematicity (Ma et al., 2022): CREPE provides systematic generalization datasets to test models trained on popular pre-training datasets including CC-12M and YFCC-15M. While creating CREPE, Ma et al. (2022) make sure to split the dataset into seen and unseen parts, which correspond to weather the model has seen or not seen the combination of concepts, when pre-trained with popular pre-training datasets. We measure and report performance on both seen and unseen splits in our work.

ARO (Yuksekgonul et al., 2022): This benchmark consists of four datasets, including VG-Relation, VG-Attribution, COCO-Order, and Flickr-Order. The first two measure attribute and relation understanding of VL models, respectively, and the last two measure the word order understanding of VL models. VG-Relation and VG-Attribution consist of tuples having an image and two texts (one positive and one negative), and the model’s task is to match the image with the correct text. order datasets have four negative texts and one positive text for each image, and the task is again to match the image with the correct text.

SVO-Probes (Hendricks and Nematzadeh, 2021): This dataset consists of tuples having two images and one text. All texts and images have a subject, verb, and object, and the images differ in only one of subject, verb, or object. This dataset helps in understanding if VL models can compositionally understand combinations of objects having a relation between them. The original dataset contains 48K examples.<sup>3</sup>

CREPE-Productivity (Ma et al., 2022): Productivity dataset tests the model’s ability to generalize to longer and more complex sentences, with complexity ranging from 4 atoms to 12 atoms, where an atom can be an attribute, relation, or object. The CREPE-Productivity dataset has a number of test sets for each sentence complexity ranging from 4 atoms to 12 atoms.

VL-Checklist (Zhao et al., 2022): This benchmark is created by combining annotations from datasets like Visual Genome (Krishna et al., 2016), SWiG (Pratt et al., 2020), HAKE (Li et al., 2019), VAW (Pham et al., 2021). Each image in the resulting dataset has two captions, a positive and a negative. The positive caption is taken from the source dataset of the image, while the negative caption differs from the positive in only one word which makes it a hard negative and helps in testing compositional and fine-grained understanding of VLMs across various dimensions like attributes, relations, and size and locations of objects.

## F Baselines:

Here we list the baselines used in this study and also provide a short description for each. CLIP(Radford et al., 2021): Our first baseline is CLIP model released by OpenAI CLIP(Radford et al., 2021) and OpenCLIP (Ilharco et al., 2021). In particular we use the ViT-B/32 model for fine-tuning results Table 1 of the main paper, except for CREPE dataset, which requires using models pre-traoined on specific datasets, for which we use ResNet-50 (RN-50) models pre-trained on CC-12M and YFCC-15M released by OpenCLIP repository<sup>4</sup> (Ilharco et al., 2021).

CLIP-FT: For disentangling the effects of fine-tuning data, and fine-tuning methodology, we create a CLIP-FT baseline where we simply fine-tune the pre-trained CLIP model on the dataset at hand, by using the standard contrastive learning technique used by CLIP.

NegCLIP(Yuksekgonul et al., 2022) [ICLR 2023]: NegCLIP is trained using negative mining of texts and images. Yuksekgonul et al. (2022) create sentence level hard negatives by swapping different linguistic elements. They also additionally include hard-negative images and their corresponding texts in the batch by fetching K nearest neighbours (K=3) for each image in the feature space constructed using a pretrained CLIP model.

Teaching SVLC(Doveh et al., 2023) [CVPR 2023]: This method uses LLM’s like BLOOM (Mitchell et al., May 2021-May 2022) along with rules to create additional positive and negative sentences for each image while fine-tuning CLIP. Syn-CLIP(Cascante-Bonilla et al., 2023) [Arxiv

2023]: Syn-CLIP uses a million scale synthetic dataset to fine-tune CLIP and improve it’s performance on compositional reasoning tasks. The synthetic data is created using a 3D physics-based simulation platform built on Unity3D, called ThreeDWorld (Gan et al., 2021). This contemporary work is complementary to our data-centric approach and we believe our methods can help fine-tuning with synthetic datasets as well. Cascante-Bonilla et al. (2023) in their paper showed how dense and long captions can be obtained for synthetic images and which require splitting into sub-captons followed by averaging of features from all captions while fine-tuning CLIP. This is one avenue where we believe our method can be useful since our method inherently allows matching of images to multiple texts. This is part of future work, once such synthetic datasets are released and are easily available.

## G Detailed Experimental Results

In the main paper Table 1 and Table 3 we had provided concise results for some datasets, based on lack of space due to extensive experimental results. Here we provide detailed results on these datasets:

## G.1 VL-Checklist: detailed results

Detailed Fine-tuning results on VL-Checklist dataset are provided in Table 17. These are an extension to the VL-Checklist results provided in the main paper Table 1. Detailed Pre-training results for VL-Checklist dataset are provided in Table 18 which are an extension to the VL-Checklist results provided in the main paper Table 3.

## G.2 SVO-Probes: detailed results

Detailed Fine-tuning results on SVO-Probes dataset are provided in Table 19. These are an extension to the SVO-Probes results provided in the main paper Table 1. Detailed Pre-training results for SVO-Probes dataset are provided in Table 20 which are an extension to the SVO-Probes results provided in the main paper Table 3.

## G.3 CREPE-Systematicity: detailed results

Here we provide detailed results on CREPE-Systematicity dataset used for measuring systematic generalization. In the main paper we had only provided the results related to systematic generalization (i.e., the unseen split), but here we provide results on both the seen and unseen split, for both hard negative retrieval sets (Comp and Atom) that are used when evaluating performance on CREPE by Ma et al. (2022). Detailed Fine-tuning results on CREPE-Systematicity dataset on both the seen and unseen splits are provided in Table 21. These are an extension to the CREPE-Systematicity results provided in the main paper Table 1. Detailed Pretraining results for CREPE-Systematicity dataset are provided in Table 22 which are an extension to the CREPE-Systematicity results provided in the main paper Table 3.

## H Reproducibility

Here we provide necessary details to reproduce our work, that might not have been included in the main paper.

## H.1 Training and hyperparameter details

Fine-tuning: For all fine-tuning experiments, we follow Yuksekgonul et al. (2022) for hyperparameters. In particular, all models are fine-tuned for 5 epochs, with a batch size of 256, using a cosine learning rate schedule with 50 steps of warmup and random-crop augmentation during training. AdamW is used for optimization. 1e 5 is used as the initial learning rate. Training is performed using 4 NVIDIA A100 GPUs for all models. From the ARO dataset, 10% examples from attribute and relation splits are used as validation examples, and the rest are used as the test set for all models. On all other datasets, we evaluate zero-shot performance. For MosaiCLIP, we find that sampling a maximum of 3 positive and 6 negative sub-graphs per image during fine-tuning gives the best result on the ARO validation set and hence is used in all our experiments (including pre-training experiments). For MosaiCLIP, we keep sub-graph sampling probabilities as $p _ { 2 } = p _ { 3 }$ . We vary p<sub>1</sub> in 0, 0.08, 0.15 while fine-tuning on the randomly chosen YFCC dataset. We choose the best model according to the ARO val-set and keep the hyperparameters the same for all other fine-tuning datasets.

Pre-training: For pre-training experiments, we follow the training protocol used in Yang et al. (2022); Radford et al. (2021). In particular, all models are trained for 32 epochs, with a batch size of 4096, using a cosine learning rate schedule with 5000 steps of warmup and random-crop augmentation during training. AdamW is used for optimization. The initial learning rate is $1 e - 3 .$ , and weight decay is set to 0.1. Training is performed using 64 NVIDIA A100 GPUs. NegCLIP’s hard negative text creation method often results in no negative text for some texts in the pre-training dataset. Removing all such image-text pairs with no possible hard negative text results in poor performance for NegCLIP (due to fewer data to pre-train on). If we include these image-text pairs, the text batch size might differ for different GPUs since some image-text pairs are without hard negative texts and this causes instabilities. We hence keep a cache of sentences from previous batches and add it to the batch as negative examples so that all GPUs have the same text batch size during training. The same is done for MosaiCLIP since not all images might have the same number of unique positive and negative sub-graphs available. For NegCLIP we create hard negative sentences using code released by (Yuksekgonul et al., 2022). For MosaiCLIP training, for each image, we always use one hard negative text createdusing NegCLIP’s swapping technique, followed by positive and negative subgraphs created using our method. Sub-graph sampling probabilities are kept as $p _ { 2 } = p _ { 3 } , p _ { 1 } = 0 . 1 5$

<table><tr><td rowspan="2">Benchmark → Fine-tuning data →</td><td colspan="5">VL-Checklist</td></tr><tr><td>CC-100K</td><td></td><td>YFCC-100K</td><td>COCO</td><td></td></tr><tr><td>Method</td><td>Obj. Attr. Rel.</td><td>Obj.</td><td>Attr. Rel.</td><td>Obj.</td><td>Attr. Rel.</td></tr><tr><td>CLIP</td><td>81.6 67.6 63.1</td><td>81.6</td><td>67.6 63.1</td><td>81.6</td><td>67.6 63.1</td></tr><tr><td>CLIP-FT</td><td>81.9 69.3 60.9</td><td>80.7</td><td>68.1 60.2</td><td>83.7</td><td>66.7 59.3</td></tr><tr><td>NegCLIP</td><td>82.1 71.4 70.3</td><td>81.0</td><td>68.1 67.1</td><td>85.2</td><td>67.2 63.0</td></tr><tr><td>MosaiCLIPNoCurric</td><td>86.0 77.2 77.7</td><td>84.0</td><td>72.2 75.1</td><td>89.2</td><td>70.4 72.6</td></tr><tr><td>MosaiCLIP WiSE-FT</td><td>85.3 71.4 72.4</td><td>83.6</td><td>69.5 69.6</td><td>88.5</td><td>75.5 77.0</td></tr><tr><td>MosaiCLIP</td><td>86.0 76.8 78.4</td><td>84.1</td><td>72.1 74.8</td><td>89.0</td><td>70.1 71.3</td></tr></table>

Table 17: Fine-tuning results on the VL-Checklist benchmark, for testing compositionality in terms of attribute, relation and object understanding. OpenAI CLIP VIT-B-32 pre-trained model is used as the base model for finetuning. See Sec. G.1 for more details.

<table><tr><td rowspan="2">Benchmark → Pre-training data →</td><td rowspan="2"></td><td colspan="5">VL-Checklist</td></tr><tr><td colspan="2">CC-12M</td><td colspan="3">YFCC-15M</td></tr><tr><td rowspan="2"></td><td>Arch. Method</td><td>Obj.</td><td>Attr.</td><td>Rel.</td><td>Obj.</td><td>Attr. Rel.</td></tr><tr><td>CLIP NegCLIP</td><td>75.2 75.0</td><td>61.1 67.7</td><td>60.6 67.4</td><td>73.6 71.2</td><td>63.0 62.0 66.5 60.3</td></tr><tr><td rowspan="2">Swi-T</td><td>MosaiCLIP</td><td>80.0</td><td>72.9</td><td>64.4</td><td>79.3</td><td>71.3 64.8</td></tr><tr><td>CLIP</td><td>75.5</td><td>62.7</td><td>60.5</td><td>73.2</td><td>58.3</td></tr><tr><td rowspan="3">RN-50</td><td>NegCLIP</td><td>75.4</td><td>67.6</td><td>65.5</td><td>72.9</td><td>62.8 65.8</td></tr><tr><td>MosaiCLIP</td><td>79.2</td><td>73.2</td><td>65.3</td><td>71.6</td><td>59.7</td></tr><tr><td></td><td></td><td></td><td></td><td>80.1</td><td>65.1</td></tr></table>

Table 18: Pre-training results on VL-Checklist benchmark, for testing compositionality in terms of attribute, relation and object understanding. Results for both backbones Swin-Tiny and RN-50 are shown. See Sec. G.1 for more details.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Benchmark →</td><td colspan="4">SVO-Probes</td></tr><tr><td>Obj</td><td>Subj</td><td>Verb</td><td>Avg</td></tr><tr><td rowspan="3">C-0K</td><td>CLIP CLIP-FT</td><td>88.13 93.17</td><td>83.85 88.64</td><td>78.76 83.87</td><td>83.58 88.56</td></tr><tr><td>NegCLIP MosaiCLIPNoCurric</td><td>92.90 93.37</td><td>88.16 89.74</td><td>84.02 83.62</td><td>88.36 88.91</td></tr><tr><td>MosaiCLIP WiSE-FT MosaiCLIP</td><td>92.65 93.45</td><td>88.69 89.39</td><td>82.90 83.35</td><td>88.08 88.73</td></tr><tr><td>YCC0K</td><td>CLIP-FT NegCLIP MosaiCLIPNoCurric MosaiCLIP WiSE-FT</td><td>89.63 88.43 89.49 87.86</td><td>85.83 84.05 85.59 84.97</td><td>80.36 79.21 79.83 78.30</td><td>85.27 83.90 84.97 83.71</td></tr><tr><td rowspan="3">COCO</td><td>MosaiCLIP CLIP-FT</td><td>89.93 93.60</td><td>86.45 91.37</td><td>79.71 85.48</td><td>85.36 90.15</td></tr><tr><td>NegCLIP</td><td>93.59</td><td>91.43</td><td>85.58</td><td>90.20 90.20</td></tr><tr><td>MosaiCLIPNoCurric MosaiCLIPwisE-FT MosaiCLIP</td><td>94.14 93.13 94.16</td><td>92.22 92.07 93.04</td><td>84.23 83.75 84.82</td><td>89.65 90.67</td></tr></table>

Table 19: Detailed Fine-tuning results on the SVO-Probes dataset. See Sec. G.2 for more details.

<table><tr><td colspan="3">Benchmark →</td><td colspan="3">SVO-Probes</td></tr><tr><td>Arch.</td><td>Data Method</td><td></td><td>Obj</td><td>Subj</td><td>Verb Avg</td></tr><tr><td rowspan="3">Swi--</td><td>C-12M</td><td>CLIP NegCLIP</td><td>88.43 88.38</td><td>82.58 79.33 81.83 79.40</td><td>82.21 82.04</td></tr><tr><td></td><td>MosaiCLIP CLIP</td><td>91.89 87.11 83.38 77.09</td><td>82.20 72.80</td><td>85.62 76.27</td></tr><tr><td>YC15M</td><td>NegCLIP MosaiCLIP</td><td>84.07 76.87 86.20 79.24</td><td>72.28 73.61</td><td>76.10 77.87</td></tr><tr><td rowspan="5">RN-50</td><td></td><td>CLIP</td><td>87.86</td><td>82.54</td><td>79.45</td><td>82.13</td></tr><tr><td>C-12M</td><td>NegCLIP</td><td>87.58</td><td>82.47</td><td>79.42</td><td>82.03</td></tr><tr><td></td><td>MosaiCLIP</td><td>90.18</td><td>85.22</td><td>80.48</td><td>83.86</td></tr><tr><td></td><td>CLIP</td><td>82.61</td><td>76.21</td><td>72.27</td><td>75.60</td></tr><tr><td>YFCC115M</td><td>NegCLIP MosaiCLIP</td><td>81.40 84.25</td><td>76.05</td><td>72.06</td><td>75.18</td></tr></table>

Table 20: Detailed Pre-training results on the SVO-Probes dataset. See Sec. G.2 for more details.

<table><tr><td colspan="2">(Pre-training, Fine-tuning) data →</td><td colspan="4">CC-12M, CC-100K</td><td colspan="4">YFCC-15M, YFCC-100K</td></tr><tr><td>Retrieval Set →</td><td colspan="2">Comp</td><td colspan="2">Atom</td><td colspan="2">Comp</td><td colspan="2">Atom</td></tr><tr><td>Method Split →</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td></td><td>Unseen</td></tr><tr><td>CLIP</td><td>48.3</td><td></td><td>45.1</td><td>39.2</td><td>35.0</td><td>42.0</td><td>39.8</td><td>43.4</td><td>39.5</td></tr><tr><td>CLIP-FT</td><td>48.5</td><td></td><td>45.8</td><td>40.0</td><td>35.6</td><td>39.1</td><td>36.4</td><td>42.4</td><td>38.3</td></tr><tr><td>NegCLIP</td><td>55.1</td><td></td><td>53.1</td><td>41.5</td><td>37.5</td><td>41.9</td><td>38.8</td><td>42.8</td><td>39.0</td></tr><tr><td>MosaiCLIPNoCurric</td><td>71.4</td><td></td><td>71.1</td><td>45.3</td><td>40.2</td><td>50.1</td><td>46.6</td><td>44.9</td><td>40.5</td></tr><tr><td>MosaiCLIPwiSE-FT</td><td>68.4</td><td></td><td>67.5</td><td>46.1</td><td>41.2</td><td>48.9</td><td>48.1</td><td>46.2</td><td>43.6</td></tr><tr><td>MosaiCLIP</td><td>73.1</td><td></td><td>72.4</td><td>46.2</td><td>40.9</td><td>52.3</td><td>48.8</td><td>45.7</td><td>41.5</td></tr></table>

Table 21: Fine-tuning results on the CREPE - Systematicity datasets. We take OpenCLIP models pre-trained on CC-12M, and YFCC-15M, fine-tune them on CC-100K, and YFCC-100K, respectively, and test them on CC-12M, YFCC-15M split of CREPE dataset, respectively. See Sec. 4.1 for more details. We recalculate CLIP results since Ma et al. (2022) do not normalize CLIP embeddings before taking the dot product for text and image embeddings, resulting in an incorrect score.
<table><tr><td rowspan="3"></td><td rowspan="3" colspan="2">Pre-training data → Retrieval Set →</td><td colspan="4">CC-12M</td><td colspan="4">YFCC-15M</td></tr><tr><td colspan="2">Comp</td><td colspan="2">Atom</td><td colspan="2">Comp</td><td colspan="2">Atom</td></tr><tr><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td></tr><tr><td rowspan="4">Swi-T</td><td>CLIP</td><td></td><td></td><td>44.1</td><td>41.7</td><td>37.3</td><td>40.2</td><td>39.6</td><td>42.9</td><td>41.7</td></tr><tr><td>NegCLIP</td><td></td><td>45.9 76.4</td><td>80.3</td><td>45.1</td><td>39.6</td><td>47.3</td><td>47.1</td><td>43.2</td><td>41.5</td></tr><tr><td>MosaiCLIP</td><td></td><td>85.3</td><td>92.1</td><td>49.3</td><td>44.5</td><td>80.7</td><td>89.6</td><td>48.2</td><td>45.3</td></tr><tr><td>CLIP</td><td></td><td>44.9</td><td>42.9</td><td>40.9</td><td>36.7</td><td>38.7</td><td>38.9</td><td>40.6</td><td>38.9</td></tr><tr><td rowspan="3">RN-50</td><td>NegCLIP</td><td></td><td></td><td></td><td>46.8</td><td>41.4</td><td>61.5</td><td>67.2</td><td>43.5</td><td>41.5</td></tr><tr><td>MosaiCLIP</td><td></td><td>78.6 85.3</td><td>82.0 92.6</td><td>47.8</td><td>44.4</td><td>80.1</td><td>90.2</td><td>46.6</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>45.0</td></tr></table>

Table 22: Pre-training results on CREPE - Systematicity datasets. Models are pre-trained using CC-12M and YFCC-15M datasets and tested on the corresponding CC-12M and YFCC-15M split of the CREPE dataset. Results for both backbones Swin-Tiny and RN-50 are shown. See Sec. 4.1 for more details.

## H.2 Tree-Score details:

Murty et al. (2022) devised a method to calculate the tree-score of a transformer over a given dataset of sentences D. This tree-score measures the functional tree-structuredness of a given transformer encoder. See Murty et al. (2022) for exact details for the algorithm to calculate the tree-scores. We use the code released by the authors<sup>5</sup> for the purpose of calculating tree-scores for CLIP’s language encoder. In practice we use 5K sentences from the COCO-validation set as the held ouot test set D over which we calculate the tree-scores.

## H.3 Computing Infrastructure and Run-Time:

We use NVIDIA A100 GPUs for all our experiments. Pre-training experiments took about 1.5 days per model while using 64 GPUs. Fine-tuning experiments on CC-FT, YFCC-FT and COCO took about 45 mins each and experiments on CC3M took 5 hours per model, while using 4 GPUs.

## H.4 Model Parameters:

We use standrad CLIP models and as part of all models, is a transformer language encoder having 12 layers, 8 attention heads and 512 as it’s width. For vision encoders we use 1. ResNet-50 hvaing 23M trainable parameters and 2. Transformer vision encoders a) Swin-Tiny with patch-size 4 and window size 7 following (Yang et al., 2022) and b) ViT-B-32 which has patch size 32, 12 layers and 12 attention heads.

## H.5 Evaluation Metrics:

Strictly following the respective papers and released code<sup>6</sup>, for ARO, VL-Checklist, SVO we use accuracy as the metric as defined by the respecitve papers. And for CREPE-Productivty, and CREPE-Systematicity <sup>7</sup> we use Recall@1 as our metric of evaluation.

## H.6 Summary Statistics of results:

We provide standard deviation results using 3 random seeds in Appendix Section D.2 for Fine-tuning experiments on the CC-FT dataset. For all other datasets, including the expensive pre-training runs we use a single seed for our experiments.

![](images/1fb459d3d8798bd6a780294b38be2b7f2a7f567bde022c51754531da9426a1f7.jpg)  
(a) Finetuning data: CC-100k

![](images/42f96f27e178e79192bed815a4d853ebfd7f7ec7d7b29a17bf89c609def7f455.jpg)  
(b) Finetuning data: COCO

Figure 8: Fine-tuninig Results on CREPE - Productivity (generalization to longer and more complex sentences). Fine-tuning datasets are mentioned below each figure.  
![](images/aa57d316ee7abd9c23e986576b8fa4373b64e2d74ce4e0abb24d1253cda775b1.jpg)  
(a) Swin-Tiny, CC-12M

![](images/205ec46132bb0fb6e1ecfc4e1f92da6df6bbfaafe1b6e90c8dde90aba3947c1a.jpg)  
(b) Swin-Tiny, YFCC-15M

![](images/2b77fe56a0b12c644056355f5fa5eb2745a769f8ffb4aff0996289ab97f4244f.jpg)  
(c) RN-50, CC-12M

![](images/bb48c59e02a9f164ae2c5b6ada1e8de0fffffa430a60befe9116dd151f7a93ac.jpg)  
(d) RN-50, YFCC-15M

Figure 9: Pre-Training Results on CREPE - Productivity (generalization to longer and more complex sentences). Pre-Training model and datasets are mentioned below each figure.  
![](images/69f7c8b3346c4dac00e7be022f5fb416ef8f70e81cc58a68d60e65568ff068d8.jpg)  
(a) SVO  
Figure 10: Extension of Figure 4 c), d). Selectively finetuning of image, text encoders and measure performance on SVO-Probes dataset.

![](images/c5d4daa285272fe510b6b2924cd8243227922826dff1a58ebdbd36b00e75b50a.jpg)  
Figure 11: Comparing of CLIP, NegCLIP and MosaiCLIP on 20 datasets of from the ELEVATER (Li et al., 2022a) benchmark. Models in this graph are pretrained with CC-12M data and have Swin-Tiny as the vision backbone. See Sec. 4.1 for more details.

![](images/d92a8bc21a6bdb3f108c9c5fcf1c1a106ece469d67faa9ce97c17a613a6fe904.jpg)  
Figure 12: Comparing of CLIP, NegCLIP and MosaiCLIP on 20 datasets of from the ELEVATER (Li et al., 2022a) benchmark. Models in this graph are pretrained with YFCC-15M data and have Swin-Tiny as the vision backbone. See Sec. 4.1 for more details.

![](images/ed8ffe8a590764aec9061788fc58d37d97229816d9d54072b6d37c3b3ea1180c.jpg)  
Figure 13: Comparing of CLIP, NegCLIP and MosaiCLIP on 20 datasets of from the ELEVATER (Li et al., 2022a) benchmark. Models in this graph are pretrained with CC-12M data and have ResNet-50 as the vision backbone. See Sec. 4.1 for more details.

![](images/3e1fb7d7f9d35bbd12de8a85d1d60b87f33047be187a0345931de36bc3047ac1.jpg)  
Figure 14: Comparing of CLIP, NegCLIP and MosaiCLIP on 20 datasets of from the ELEVATER (Li et al., 2022a) benchmark. Models in this graph are pretrained with YFCC-15M data and have ResNet-50 as the vision backbone. See Sec. 4.1 for more details.