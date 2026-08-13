# GLEN: General-Purpose Event Detection for Thousands of Types

Qiusi Zhan<sup>1</sup>\*, Sha Li<sup>1</sup>\*, Kathryn Conger<sup>2</sup>, Martha Palmer<sup>2</sup>, Heng Ji<sup>1</sup>, Jiawei Han<sup>1</sup>

<sup>1</sup>University of Illinois Urbana-Champaign

<sup>2</sup>University of Colorado Boulder

{qiusiz2, shal2, hengji, hanj}@illinois.edu {kathryn.conger, martha.palmer}@colorado.edu

## Abstract

The progress of event extraction research has been hindered by the absence of wide-coverage, large-scale datasets. To make event extraction systems more accessible, we build a generalpurpose event detection dataset GLEN which covers 205K event mentions with 3,465 different types, making it more than 20x larger in ontology than today’s largest event dataset. GLEN is created by utilizing the DWD Overlay, which provides a mapping between Wikidata Qnodes and PropBank rolesets. This enables us to use the abundant existing annotation for PropBank as distant supervision. In addition, we also propose a new multi-stage event detection model CEDAR specifically designed to handle the large ontology size in GLEN. We show that our model exhibits superior performance compared to a range of baselines including InstructGPT. Finally, we perform error analysis and show that label noise is still the largest challenge for improving performance for this new dataset.<sup>1</sup>

## 1 Introduction

As one of the core IE tasks, event extraction involves event detection (identifying event trigger mentions and classifying them into event types), and argument extraction (extracting the participating arguments). Event extraction serves as the basis for the analysis of complex procedures and news stories which involve multiple entities and events scattered across a period of time, and can also be used to assist question-answering and dialog systems.

The development and application of event extraction techniques have long been stymied by the limited availability of datasets. Despite the fact that it is 18 years old, ACE 2005 <sup>2</sup> is still the de facto standard evaluation benchmark. Key limitations of ACE include its small event ontology of 33 types, small dataset size of around 600 documents and restricted domain (with a significant portion concentrated on military conflicts). The largest effort towards event extraction annotation is MAVEN (Wang et al., 2020), which expands the ontology to 168 types, but still exhibits limited domain diversity (32.5% of the documents are about military conflicts).

![](images/5214a305dca4e0137992b02afee6e9221f99827e775336d6d22e1f1bffabb297.jpg)  
Figure 1: The GLEN event detection dataset is constructed by using the DWD Overlay, which provides a mapping between PropBank rolesets and WikiData Qnodes. This allows us to create a large distantlysupervised training dataset with partial labels.

The focus of current benchmarks on restricted ontologies in limited domains is harmful to both developers and users of the system: it distracts researchers from building scalable general-purpose models for the sake of achieving higher scores on said benchmarks and it discourages users as they are left with the burden of defining the ontology and collecting data with little certainty as to how well the models will adapt to their domain. We believe that event extraction can, and should be made accessible to more users.

To develop a general-purpose event extraction system, we first seek to efficiently build a highquality open-domain event detection dataset. The difficulty of annotation has been a long-standing issue for event extraction as the task is defined with lengthy annotation guidelines, which require training expert annotators. Thus, instead of aiming for perfect annotation from scratch, we start with the curated DWD Overlay (Spaulding et al., 2023) which defines a mapping between Wikidata<sup>3</sup> and PropBank rolesets (Palmer et al., 2005)<sup>4</sup> as displayed in Figure 1. While Wikidata Qnodes can serve as our general-purpose event ontology, the PropBank roleset information allows us to build a large-scale distantly supervised dataset by linking our event types to existing expert-annotated resources. In this process, we reuse the span-level annotations from experts, while dramatically expanding the size of the target ontology. Our resulting dataset, GLEN(The GeneraL-purpose EveNt Benchmark), covers 3,465 event types over 208k sentences, which is a 20x increase in the number of event types and 4x increase in dataset size compared to the previous largest event extraction dataset MAVEN. We also show that our dataset has better type diversity and the label distribution is more natural (Figures 3 and 4).

![](images/6bd3dcb17a229adde1cfc04b90dc87a4e697637e66f813c39d3fa4282a81668f.jpg)  
Figure 2: Overview of our framework. We first identify the potential triggers from the sentence (trigger identification) and then find the best matching type for the triggers through a coarse-grained sentence-level type ranking followed by a fine-grained trigger-specific type classification.

We design a multi-stage cascaded event detection model CEDAR to address the challenges of large ontology size and distant-supervised data as shown in Figure 2. In the first stage, we perform trigger identification to find the possible trigger spans from each sentence. We can reuse the span-level annotations, circumventing the noise brought by our distant supervision. In the second stage, we perform type ranking between sentences and all event types. This model is based on Col-

BERT (Khattab and Zaharia, 2020), which is a very efficient ranking model based on separate encoding of the event type definition and the sentence. This stage allows us to reduce the number of type candidates for each sentence from thousands to dozens, retaining 90% recall@10. In the last stage, for each trigger detected, we perform type classification to connect it with one of the top-ranked event types from the first stage. We use a joint encoder over both the sentence and the event type definition for higher accuracy.

In summary, our paper makes contributions in (1) introducing a new event detection benchmark GLEN which covers 3,465 event types over 208k sentences that can serve as the basis for developing general-purpose event detection tools; (2) designing a multi-stage event detection model CEDAR for our large ontology and annotation via distant supervision which shows large improvement over a range of single-stage models and few-shot Instruct-GPT.

## 2 Related Work

As the major contribution of our work is introducing a new large-scale event detection dataset, we review the existing datasets available for event extraction and the various ways they were created.

Event Extraction Datasets ACE05 set the event extraction paradigm which consists of event detection and argument extraction. It is also the most widely used event extraction dataset to date. The more recent MAVEN (Wang et al., 2020) dataset has a larger ontology selected from a subset of FrameNet (Baker et al., 1998). FewEvent (Deng et al., 2020) is a compilation of ACE, KBP, and

<table><tr><td colspan="2">Dataset</td><td>#Documents</td><td>#Tokens</td><td>#Sentences</td><td>#Event Types</td><td>#Event Mentions</td></tr><tr><td colspan="2">ACE2005</td><td>599</td><td>303k</td><td>15,789</td><td>33</td><td>5,349</td></tr><tr><td colspan="2">MAVEN</td><td>4480</td><td>1276k</td><td>49,873</td><td>168</td><td>118,732</td></tr><tr><td rowspan="4">GLEN</td><td>Train</td><td>5607</td><td>3641k</td><td>187,468</td><td>3,464</td><td>184,806</td></tr><tr><td>Dev</td><td>311</td><td>204k</td><td>10,359</td><td>1720</td><td>9878</td></tr><tr><td>Test</td><td>306</td><td>201k</td><td>10,627</td><td>1334</td><td>8290</td></tr><tr><td>Total</td><td>6224</td><td>4045k</td><td>208,454</td><td>3,465</td><td>205,045</td></tr></table>

Table 1: Statistics of our GLEN dataset in comparison to the widely used ACE05 dataset and the previously largest event detection dataset MAVEN.

Wikipedia data designed for few-shot event detection. A number of datasets (DCFEE (Yang et al., 2018), ChFinAnn (Zheng et al., 2019), RAMS(Ebner et al., 2020), WikiEvents (Li et al., 2021), DocEE (Tong et al., 2022)) have also been proposed for document-level event extraction, with a focus on argument extraction, especially when arguments are scattered across multiple sentences. Within this spectrum, GLEN falls into the category of event detection dataset with a heavy focus on wider coverage of event types.

Weak Supervision for Event Extraction Due to the small size of existing event extraction datasets and the difficulty of annotation, prior work has attempted to leverage distant supervision from knowledge bases such as Freebase(Chen et al., 2017; Zeng et al., 2017) and WordNet (Araki and Mitamura, 2018; Tong et al., 2020). The former is limited by the number of compound value types (only 20 event types are used in both works) and the latter does not perform any typing. Weak supervision has also been used to augment training data for an existing ontology with the help of a masked language model (Yang et al., 2019) or adversarial training (Wang et al., 2019). Our dataset is constructed with the help of the DWD Overlay mapping which is defined between event types (Qnodes) and PropBank rolesets instead of on the level of concrete event instances (as in prior work that utilizes knowledge bases).

## 3 The GLEN Benchmark

To build the GLEN benchmark, we first build a general-purpose ontology based on the curated DWD Overlay and then create distantly-supervised training data based on the refined ontology (Section 3.1). In order to evaluate our model, we also create a labeled development set and test set through crowdsourcing (Section 3.2).

## 3.1 Event Ontology and Data

The DWD Overlay is an effort to align WikiData Qnodes to PropBank rolesets, their argument structures, and LDC tagsets. This mapping ensures that our ontology is a superset of the ontology used in ACE and ERE (Song et al., 2015). (See Section 3.3 for a detailed comparison of the ontology coverage.)

To make this ontology more suitable for the event extraction task, we remove the Qnodes related to cognitive events that do not involve any physical state change such as belief and doubt. 5 We also discovered that many rolesets such as ill.01 were heavily reused across the ontology (mainly due to the inclusion of very fine-grained types), therefore we manually cleaned up the event types that were associated with these rolesets. We show some examples of removed Qnodes in Appendix Table 8.

Since the DWD Overlay is aligned with Prop-Bank, we propose to reuse the existing PropBank annotations<sup>6</sup>. After the automatic mapping, each event mention in the dataset would be associated with one or more Qnodes, which leads to the partial label challenge when using this distantlysupervised data. We then perform another round of data filtering based on the frequency of rolesets (details in Appendix C). After these cleaning efforts, we used the annotation for 1,804 PropBank rolesets, which are mapped to a total of 3,465 event types.

To make our data split more realistic and to preserve the document-level context, we split the dataset into train, development, and test sets based on documents using a ratio of 90/5/5. Note that although our test set is only 5% of the full data, it is already similar in scale to the entire ACE05 dataset. For datasets such as OntoNotes and AMR that contain documents from multiple genres (newswire, broadcast, web blogs etc.), we perform stratified sampling to preserve the ratio between genres. The statistics of our dataset are listed in Table 1. Compared with ACE05 and MAVEN, GLEN utilizes a 20x larger ontology and 4x larger corpus.

![](images/c226806c48e193fc8370bfb8cb0ccfabe6c501460af09d4e35d2b652882bd9d2.jpg)  
(a)

![](images/98f7555f9882d11be1a818e96d469c7f32c591a59400a54100693f2d3524dc76.jpg)  
(b)  
Figure 3: The event type distribution of GLEN. In the training set, instances associated with N labels are weighted as $\textstyle { \frac { 1 } { N } }$ . Figure (a) illustrates the distribution of event types based on the number of instances. In figure (b), we show the top three most popular event types as well as randomly sampled types from the middle and tail of the distribution curve.

## 3.2 Data Annotation

Instead of performing annotation from scratch, we formulate the annotation task as a multiple-choice question: the annotators are presented with the trigger word in context and asked to choose from the Qnodes that are mapped to the roleset.

For the test set, we hired graduate students with linguistic knowledge to perform annotation. For the development set, we screened Mechanical Turk workers that had high agreement with our in-house annotators and asked those who passed the screening to participate in the annotation. The weighted average kappa value for exact match on the test set (27 annotators), is 0.60, while for the dev set (981 annotators) is 0.37 over 5.2 options. If we allow for a soft match for event types of different granularity, such as trade and international\_trade, the kappa value is 0.90 for the test set and 0.69 for the dev set. For more details on the annotation interface, see Appendix C.

## 3.3 Data Analysis

We first examine our event ontology by visualizing it as a hierarchy (as shown in Figure 4) with the parent-child relations taken from Wikidata (the overlay\_parent field in DWD Overlay). Our ontology offers a wider range of diverse events, including those related to military, disaster, sports, social phenomena, chemicals, and other topics, indicating its novelty and potential usefulness.

We show the event type distribution in our dataset in Figure 3. Our type distribution closely mirrors real-world event distributions. The distribution exhibits frequent events such as come and use, along with a long tail of rare events such as defect and impiety. In terms of type diversity, for ACE, the single most popular event type attack accounts for 28.8% of the instances and the top-10 event types account for 79.4% of the data. Our type distribution is much less skewed with the top-10 events composing 8.3% of the data.

Figure 5 illustrates the part-of-speech distribution of trigger words in our dataset<sup>7</sup>. Over 96% of trigger words are verbs or nouns, which is similar to that of 94% in MAVEN and 90% in ACE2005. In addition, 0.6% of the triggers are multi-word phrases.

## 4 Method

In the event detection task, the goal is to find the trigger word offset and the corresponding event type for every event. The main challenge for our dataset is the large ontology size and partial labels from distant supervision. To mitigate label noise, we first separate the trigger identification step from the event typing step, since trigger identification can be learned with clean data. Then, to handle the large ontology, we break the event typing task into two stages of type ranking and type classification to progressively narrow down the search space. Finally, in the type classification model, we adopt a self-labeling procedure to mitigate the effect of partial labels.

![](images/a586525e8a2e8bb88be0cae1fd590439e1df93fadad6fdbadbea0606006ee669.jpg)  
: the entire branch is not covered by ACE2005 : at least one event type in this branch is covered by ACE2005.

Figure 4: A comparison between our ontology and that of ACE. Our dataset offers broader coverage of events compared to ACE05, with diverse branches ranging from sports\_competition to military\_operation.  
![](images/4d3e25dd6ed3e06d910e5277ef43145f687f2194f1623cc5bbd44417e2a6ef0b.jpg)  
Figure 5: The Part-of-Speech distribution of trigger words for events in GLEN. Multiple POS tags mean that the trigger has multiple words.

## 4.1 Trigger Identification

In the trigger identification stage, the goal is to identify the location of event trigger words in the sentence. This step only involves the sentence and not the event types. We formulate the problem as span-level classification: to compute the probability of each span in sentence s being an event trigger, we first obtain sentence token representations based on a pre-trained language model:

$$
[ \mathbf { s } _ { 1 } \cdot \cdot \cdot \mathbf { s } _ { n } ] ^ { T } = \operatorname { P L M } ( [ \mathbf { C L S } ] s _ { 1 } \cdot \cdot \cdot s _ { n } [ \mathbf { S E P } ] ) \in \mathbb { R } ^ { n \times h } ,
$$

where $\mathbf { s } _ { i }$ is a $h \cdot$ -dimensional representation vector of token $s _ { i }$

Then we compute the scores for each token being the start, end, and part of an event trigger individually as:

$$
f _ { \sharp } ( s _ { i } ) = { \mathbf w } _ { \sqcup } ^ { T } { \mathbf s } _ { i } , \quad \bigsqcup \in \{ \mathrm { s t a r t , e n d , p a r t } \} ,
$$

where $\mathbf { w } _ { \perp } \in \mathbb { R } ^ { h }$ is a learnable vector. We then compute the probability of span $[ s _ { i } \cdot \cdot . s _ { j } ]$ to be an event trigger as the sum of its parts:

$$
\begin{array} { r l } & { p ( \bigl [ s _ { i } \cdot \cdot \cdot s _ { j } \bigr ] ) } \\ & { = \sigma ( f _ { \mathrm { s t a r t } } ( s _ { i } ) + f _ { \mathrm { e n d } } ( s _ { j } ) + \displaystyle \sum _ { k = i } ^ { j } f _ { \mathrm { p a r t } } ( s _ { k } ) ) . } \end{array}
$$

The model is trained using a binary cross entropy loss on all of the candidate spans in the sentence.

## 4.2 Event Type Ranking

In the next stage, we perform event type ranking over the entire event ontology for each sentence. Since our ontology is quite large, to improve efficiency we make two design decisions: (1) ranking is done for the whole sentence and not for every single trigger; (2) the sentence and the event type definitions are encoded separately.

We use the same model architecture as Col-BERT (Khattab and Zaharia, 2020), which is an efficient ranking model that matches the performance of joint encoders while being an order of magnitude faster.

We first encode the sentence as:

$$
\begin{array} { r l } & { \vec { \mathbf { s } } = \mathrm { P L M } \big ( [ \mathrm { C L S } ] [ \mathrm { S E N T } ] s _ { 1 } \cdot \cdot \cdot s _ { n } [ \mathrm { S E P } ] \big ) \in \mathbb { R } ^ { ( n + 1 ) \times h } } \\ & { [ \mathbf { h } _ { s } ^ { 1 } , \cdot \cdot \cdot , \mathbf { h } _ { s } ^ { m } ] = \mathrm { N o r m } \big ( 1 \mathrm { d C o n v } ( \vec { \mathbf { s } } ) \big ) \in \mathbb { R } ^ { m \times h } } \end{array}\tag{1}
$$

[SENT] is a special token to indicate the input type. The one-dimensional convolution operator serves as a pooling operation and the normalization operation ensures that each vector in the bag of embeddings $\mathbf { h } _ { s }$ has an L2 norm of 1.

The event type definition is encoded similarly, only using a different special token [EVENT].

Then the similarity score between a sentence and an event type is computed by the sum of the maximum similarity between the sentence embeddings and event embeddings:

$$
\rho _ { ( e , s ) } = \sum _ { h _ { s } } \operatorname* { m a x } _ { h _ { e } } ( { \bf h } _ { e } ^ { T } { \bf h } _ { s } )\tag{2}
$$

Our event type ranking model is trained using the distant supervision data using a margin loss since each instance has multiple candidate labels. This margin loss ensures that the best candidate is scored higher than all negative samples.

$$
\mathcal { L } = \frac { 1 } { N } \sum _ { s } \sum _ { e ^ { - } } \operatorname* { m a x } \bigl \{ 0 , \big ( \tau - \operatorname* { m a x } _ { e \in C _ { y } } \rho _ { ( e , s ) } + \rho _ { ( e ^ { - } , s ) } \big ) \bigr \}\tag{3}
$$

e− denotes negative samples, $C _ { y }$ is the set of candidate labels and τ is a hyperparameter representing the margin size.

## 4.3 Event Type Classification

Given the top-ranked event types from the previous stage and the detected event triggers, our final step is to classify each event trigger into one of the event types. Similar to (Lyu et al., 2021), we formulate this task as a Yes/No QA task to take advantage of pre-trained model knowledge. The input to the model is formatted as “ type is defined as definition . sentence . Does trigger indicate a type event? [MASK]”. We directly use a pretrained masked language model and take the probability of the model predicting “yes” or $\ " \mathrm { n o } ^ { \prime \prime }$ for the [MASK] token, denoted as $P _ { \mathrm { M L M } } ( w )$ , where $w \in \{ { } ^ {  } \mathrm { y e s } ^ { \prime \prime } , { } ^ {  } \mathrm { n o } ^ { \prime \prime } \}$ . From these probabilities, we calculate the probability of the event type e as follows:

$$
P ( e ) = \frac { \exp ( P _ { \mathrm { M L M } } ( ^ { \mathrm { * } } \mathrm { y e s } ^ { \mathrm { , 9 } } ) ) } { \exp ( P _ { \mathrm { M L M } } ( ^ { \mathrm { * } } \mathrm { y e s } ^ { \mathrm { , 9 } } ) ) + \exp ( P _ { \mathrm { M L M } } ( ^ { \mathrm { * } } \mathrm { n o } ^ { \mathrm { , 9 } } ) ) }\tag{4}
$$

To train the model, we employ binary cross-entropy loss across pairs of event triggers and event types.

As mentioned in Section 3.1, our data contains label noise due to the many-to-one mapping from Qnodes to PropBank rolesets. We adopt an incremental self-labeling procedure to handle the partial labels. We start by training a base classifier on clean data labeled with PropBank rolesets that map to only one candidate event type. Despite being trained on only a subset of event types, the base model exhibits good generalization to rolesets with multiple candidate event types. We then use the base classifier to predict pseudo-labels for the noisy portion of the training data, selecting data with high confidence to train another classifier in conjunction with the clean data.

## 5 Experiments

## 5.1 Experiment Setting

Evaluation Metrics Previous work mainly uses trigger identification F1 and trigger classification F1 to evaluate event detection performance. An event trigger is correctly identified if the span matches one of the ground truth trigger spans and it is correctly classified if the event type is also correct. In addition to these two metrics, due to the size of our ontology, we also report Hit@K which measures if the ground truth event type is within the top-K ranked event types for the event trigger (the event trigger span needs to be correct). This can be seen as a more lenient version of trigger classification F1.

Baselines We compare with three categories of models: (1) classification models including DM-BERT (Wang et al., 2019), token-level classification and span-level classification; (2) a definitionbased model ZED (Zhang et al., 2022) and (3) InstructGPT(Ouyang et al., 2022) with few-shot prompting. We attempted to compare with CRF models such as OneIE model (Lin et al., 2020) but were unable to do so due to the memory cost of training CRF with the large label space. For detailed baseline descriptions and hyperparameters, see Appendix B.

## 5.2 Results

We show the evaluation results in Table 2 and some example predictions in Table 3. The first group of baselines (DMBERT/TokCls/SpanCls) are all classification-based, which means that they treat event types as indexes and do not utilize any semantic representation of the labels. We observe that DMBERT will have a substantially lower trigger identification score if we allow spans of more than 1 token to be predicted due to its max pooling mechanism (it will produce overlapping predictions such as “served”, “served in”, “served in congress”). TokCls’s performance hints on the limit for learning only with partial labels. As shown in the example, TokCls usually predicts event types that are within the candidate set for the roleset, but since it has no extra information to tell the candidates apart, the prediction is often wrong.

Although ZED utilizes the event type definitions, it only achieves a minor improvement in performance compared to TokCls. ZED employs mean pooling to compress the definition embedding into a single vector, which is more restricted compared to the joint encoding used by our event type classification module.

We observe that InstructGPT with in-context learning does not perform well on our task. The low trigger identification scores might be attributed to the lack of fine-tuning and under-utilization of the training set. The low classification scores are mainly caused by the restriction in input length <sup>8</sup> which makes it impossible to let the model have full knowledge of the ontology. As a result, only 57.8% of the event type names generated by InstructGPT could be matched in the ontology. With a larger input window and possibly fine-tuning, we believe that large LMs could achieve better performance.

<table><tr><td>Model</td><td colspan="3">Trigger Identification</td><td colspan="3">Trigger Classification</td><td colspan="3">Hit@k</td></tr><tr><td></td><td>Prec</td><td>Recall</td><td>F1</td><td>Prec</td><td>Recall</td><td>F1</td><td>Hit@1</td><td>Hit@2</td><td>Hit@5</td></tr><tr><td>DMBERT (Wang et al., 2019)</td><td>56.87</td><td>84.32</td><td>67.93</td><td>32.93</td><td>48.84</td><td>39.34</td><td>一</td><td>-</td><td></td></tr><tr><td>Token Classification</td><td>68.04</td><td>82.19</td><td>74.46</td><td>41.48</td><td>50.10</td><td>45.38</td><td></td><td>一</td><td></td></tr><tr><td>Span Classification</td><td>62.36</td><td>78.71</td><td>69.58</td><td>37.36</td><td>47.16</td><td>41.69</td><td>-</td><td>-</td><td></td></tr><tr><td>TokCls + ZED (Zhang et al., 2022)</td><td>-</td><td>-</td><td>-</td><td>40.01</td><td>56.25</td><td>46.76</td><td>56.84</td><td>60.35</td><td>60.80</td></tr><tr><td>InstructGPT (32 shot)</td><td>28.41</td><td>42.30</td><td>33.99</td><td>11.76</td><td>17.52</td><td>14.08</td><td>一</td><td>-</td><td>一</td></tr><tr><td>CEDAR w/o self-labeling</td><td></td><td></td><td></td><td>49.21</td><td>61.62</td><td>54.72</td><td>68.21</td><td>80.21</td><td>89.18</td></tr><tr><td>CEDAR</td><td>71.05</td><td>88.96</td><td>79.00</td><td>50.91</td><td>63.74</td><td>56.60</td><td>71.30</td><td>80.84</td><td>88.63</td></tr></table>

Table 2: Quantitative evaluation results (%) for event detection on GLEN.
<table><tr><td rowspan=1 colspan=2>Context                                            Predictions</td></tr><tr><td rowspan=1 colspan=1>You do pay income tax on your paycheck and thesales tax on consumable goods?</td><td rowspan=1 colspan=1>CEDAR: payment(transfer of an item of value)TokCls: disbursement (payment from a public fund)ZED: income (consumption and savings opportunity)GPT3: None</td></tr><tr><td rowspan=1 colspan=1>... a situation (brought on mostly by the police )where information is discovered in a way thatperpetuates the story</td><td rowspan=1 colspan=1>CEDAR: discovery (detecting something new)TokCls: medical_findingZED: physical_finding (from a physical examination of patient)GPT3: discovery (detecting something new)</td></tr><tr><td rowspan=1 colspan=1>40% of the female students at Georgetown lawreported to us that they struggle financially as aresult of this policy.</td><td rowspan=1 colspan=1>CEDAR reporting (producing a oral or written report)TokCls: scoop (journalism term for a new interesting story)ZED: reporting (producing a oral or written report)GPT3: None</td></tr></table>

Table 3: Comparison across different systems. Correct predictions are shown in green. Predictions that map to the same PropBank roleset are shown in orange. For ZED, we show the TokCls+ZED variant which has better performance.

For our own model, we show that decoupling trigger identification from classification improves TI performance, and performing joint encoding of the definition and the context improves TC performance. Furthermore, using self-labeling can help improve top-1 classification performance by converting partial labels into clean labels.

## 6 Analysis

In this section, we investigate the following questions: (1) Which component is the main bottleneck for performance? (Section 6.1) and (2) Does our model suffer from label imbalance between types? (Section 6.2)

## 6.1 Per-Stage Performance

Our model CEDAR comprises three components: trigger identification, event type ranking, and event type classification. Table 2 shows precision, recall, and F1 scores for trigger identification, while Table 4 presents the performance of event type ranking and classification. We assess per-stage performance using Hit@k metrics for event type ranking on ground truth trigger spans and event type classification on event mentions where the ground truth event type appears in the ranked results by the type ranker. The scores indicate that the primary bottleneck exists in the precision of trigger identification and the Hit@1 score of type classification.

<table><tr><td>Component</td><td colspan="3">Hit@k</td></tr><tr><td rowspan="2">Type Ranking</td><td>Hit@10</td><td>Hit@20</td><td>Hit@50</td></tr><tr><td>89.86</td><td>93.52</td><td>95.44</td></tr><tr><td>Type Classification</td><td>Hit@1</td><td>Hit@2</td><td>Hit@5</td></tr><tr><td>(ground truth in top-10)</td><td>78.70</td><td>89.37</td><td>97.76</td></tr></table>

Table 4: Evaluation for type ranking and classification separately. The scores for type classification component are computed over the subset of data where the ground truth event is among the top-10 ranked results.

![](images/af9a5831079e3124c68b83442b8e5c40099dc90142ece4c7939ddf1ddbb2ff77.jpg)  
Figure 6: Relationship between the number of instances of different event types in the dataset and our model’s performance. The event types are divided into four groups based on their frequencies.

## 6.2 Type Imbalance

Figure 3 illustrates the long-tailed label distribution in our dataset. To investigate whether our model is affected by this imbalance, we divided the event types into four groups separated at the quartiles based on their frequency and calculated the performance per group. The resulting figure is shown in Figure 6. While we do see that the most popular group has the highest F1 score, the remaining groups have comparable scores. In fact, we see two factors at play in defining the dataset difficulty: the ambiguity of the event type and the frequency of the event type. Event types that are more popular are also often associated with rolesets that have a high level of ambiguity, which balances out the gains from frequency.

## 6.3 Error Categories

![](images/040b66f14bdd67b3576f1fc4c4ab3022cf877f0ca0efec59b157bf45dcbd9030.jpg)  
Figure 7: Categorization of errors based on the relation between the predicted event type and the candidate set produced by the mapping (left) and the predicted event type and the ground truth event type on the event ontology hierarchy (right).

We categorize the type classification errors from the CEDAR model based on the relationship between our predicted event type and the ground truth event type as shown in Figure 7 and Table 5.

Most of our errors come from the noisy annotation (Candidate Set): our model can predict an event type that falls within the set of candidate types associated with the ground truth PropBank roleset but fails to find the correct one. Extended Roleset refers to the predicted event being associated with a roleset that shares the same predicate as the ground truth. The uncategorized errors are often due to the imperfect recall of our event ranking module (as in the second example in Table 5 where the context is long and “fund” fails to be included in the top-10 ranked event types), or cases where our model prediction is related semantically to the ground truth but the event types have no connections in the hierarchy (as in the first example in Table 5 where we predicted “quantification” instead of “measurement”). On the other hand, in another 22.9% of the cases, we predict an event that is close to the ground truth on the XPO hierarchy, with the ground truth either being the child (Child), parent (Parent), or sibling node (Sibling) of our predicted type. This suggests that better modeling of the hierarchical relations within the ontology might be useful for performance improvement.

## 7 Conclusions and Future Work

We introduce a new general-purpose event detection dataset GLEN that covers over 3k event types for all domains. GLEN can be seen as our first attempt towards making event extraction systems more accessible to the general audience, removing the burden on users of designing their own ontology and collecting annotation. We also introduce a multi-stage model designed to handle the large ontology size and partial labels in GLEN and show that our tailored model achieves much better performance than a range of baselines.

Our model CEDAR can be used as an off-theshelf event detection tool for various downstream tasks, and we also encourage members of the community to improve upon our results, (e.g. tackle the noise brought by partial labels) or extend upon our benchmark (e.g. include other languages).

## 8 Limitations

We perceive the following limitations for our work:

• Lack of argument annotation. Event argument extraction is a critical component of event extraction. At the time of publication, GLEN is only an event detection dataset without argument annotation. This is an issue that we are actively working towards for future releases of the dataset.

<table><tr><td>Error Category</td><td>Context</td><td>Predicted</td><td>Gold</td></tr><tr><td rowspan="2">Candidate</td><td>3 workers set off a critical reaction in 990000 when they poured too much uranium into a precipitation tank.</td><td>reaction (response to stimulus)</td><td>chemical_reaction</td></tr><tr><td>France is urged to increase research funding and support for innovations as a way to deal with the problem.</td><td>research_method</td><td>research (systematic study)</td></tr><tr><td>Extended Can- didate</td><td>Regulators and research firms promised that the $1.5 bil- lion settlement would be finalized two months ago.</td><td>settlement(distortion of a building)</td><td>settlement(operations relating to the pay- ment)</td></tr><tr><td>Child</td><td>People working for minimum wage are producing a large number of products or services.</td><td>work (activities per- formed as a means of support)</td><td>work (activity done by a person for eco- nomic gain)</td></tr><tr><td>Sibling</td><td>In the middle east conflict, do you think the United States should take Israel&#x27;s side, take the Palestinians&#x27; side, or not take either side?</td><td>social_conflict (struggle for agency or power in society)</td><td>armed_conflict (con- flict including vio- lence)</td></tr><tr><td>Parent</td><td>Doctors will examine him for signs that the cancer may have come back while he awaiting trial in a Russian jail.</td><td>inspection</td><td>physical_examination (process by medical professional)</td></tr><tr><td rowspan="3">Other</td><td>Ironically, in the 90&#x27;s, “character matters&quot; became a well- worn slogan on the right.</td><td>wears(clothing or accessory)</td><td>wear (damaging, gradual removal or deformation)</td></tr><tr><td>In theory, one could argue that the computer models are accurate and that the real measurements have some prob- lems.</td><td>quantification</td><td>measurement</td></tr><tr><td>Though funds have already been allocated and voted on for the project, Blair himself insists that things are still “very much open&quot;..</td><td>voting</td><td>fund</td></tr></table>

Table 5: Examples of erroneous type predictions. The trigger word (phrase) is shown in bold. In some cases, the error falls into multiple categories. We prioritize the XPO hierarchy-related categories since they are rarer.

• Timeliness of documents. The source documents of the PropBank annotated datasets are not very new. In particular, the OntoNotes dataset contains news articles from 2000-2006 <sup>9</sup>. Hence, there is the possibility of a data distribution shift between our training set and any document that our model is being applied, which might cause performance degrade, as shown in other tasks (Rijhwani and Preotiuc-Pietro, 2020).

• Completeness of ontology. Although our dataset is the most comprehensive event detection dataset of date, we acknowledge that there might be event types that we have overlooked. We plan to keep GLEN as a living project and update our ontology and dataset over time.

• Multilingual support. Currently, our documents are from the English PropBank annotation dataset so our system only supports English. One idea would be to utilize resources such as the Universal PropBank (Jindal et al., 2022)which can help us align corpora in other languages to PropBank and then further to our ontology.

## Acknowledgements

This research is based on work supported by U.S. DARPA KAIROS Program No. FA8750-19-2- 1004. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of DARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

## References

Jun Araki and Teruko Mitamura. 2018. Open-domain event detection using distant supervision. In Proceedings of the 27th International Conference on Computational Linguistics, pages 878–891, Santa Fe, New Mexico, USA. Association for Computational Linguistics.

Collin F. Baker, Charles J. Fillmore, and John B. Lowe. 1998. The Berkeley FrameNet project. In 36th Annual Meeting ofthe Associationfor Computational Linguistics and 17th International Conference on Computational Linguistics, Volume 1, pages 86–90, Montreal, Quebec, Canada. Association for Computational Linguistics.

Yubo Chen, Shulin Liu, Xiang Zhang, Kang Liu, and Jun Zhao. 2017. Automatically labeled data generation for large scale event extraction. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 409–419, Vancouver, Canada. Association for Computational Linguistics.

Yubo Chen, Liheng Xu, Kang Liu, Daojian Zeng, and Jun Zhao. 2015. Event extraction via dynamic multipooling convolutional neural networks. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 167–176, Beijing, China. Association for Computational Linguistics.

Shumin Deng, Ningyu Zhang, Jiaojian Kang, Yichi Zhang, Wei Zhang, and Huajun Chen. 2020. Metalearning with dynamic-memory-based prototypical network for few-shot event detection. In Proceedings ofthe 13th International Conference on Web Search and Data Mining. ACM.

Seth Ebner, Patrick Xia, Ryan Culkin, Kyle Rawlins, and Benjamin Van Durme. 2020. Multi-sentence argument linking. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 8057–8077, Online. Association for Computational Linguistics.

Ishan Jindal, Alexandre Rademaker, Michał Ulewicz, Ha Linh, Huyen Nguyen, Khoi-Nguyen Tran, Huaiyu Zhu, and Yunyao Li. 2022. Universal proposition bank 2.0. In Proceedings ofthe Language Resources and Evaluation Conference, pages 1700–1711, Marseille, France. European Language Resources Association.

O. Khattab and Matei A. Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval.

Sha Li, Heng Ji, and Jiawei Han. 2021. Document-level event argument extraction by conditional generation. In Proceedings ofthe 2021 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 894–908, Online. Association for Computational Linguistics.

Ying Lin, Heng Ji, Fei Huang, and Lingfei Wu. 2020. A joint neural model for information extraction with global features. In Proceedings of the 58th Annual

Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 7999–8009. Association for Computational Linguistics.

Qing Lyu, Hongming Zhang, Elior Sulem, and Dan Roth. 2021. Zero-shot event extraction via transfer learning: Challenges and insights. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 322–332, Online. Association for Computational Linguistics.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke E. Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Francis Christiano, Jan Leike, and Ryan J. Lowe. 2022. Training language models to follow instructions with human feedback. ArXiv, abs/2203.02155.

Martha Palmer, Paul R. Kingsbury, and Daniel Gildea. 2005. The proposition bank: An annotated corpus of semantic roles. Computational Linguistics, 31:71– 106.

Shruti Rijhwani and Daniel Preotiuc-Pietro. 2020. Temporally-informed analysis of named entity recognition. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7605–7617, Online. Association for Computational Linguistics.

Zhiyi Song, Ann Bies, Stephanie Strassel, Tom Riese, Justin Mott, Joe Ellis, Jonathan Wright, Seth Kulick, Neville Ryant, and Xiaoyi Ma. 2015. From light to rich ERE: Annotation of entities, relations, and events. In Proceedings ofthe The 3rd Workshop on EVENTS: Definition, Detection, Coreference, and Representation, pages 89–98, Denver, Colorado. Association for Computational Linguistics.

Elizabeth Spaulding, Kathryn Conger, Anatole Gershman, Rosario Uceda-Sosa, Susan Windisch Brown, James Pustejovsky, Peter Anick, and Martha Palmer. 2023. The DARPA Wikidata Overlay: Wikidata as an ontology for natural language processing. In Proceedings of the 19th Joint ACL - ISO Workshop on Interoperable Semantic Annotation (ISA-19), pages 1–10, Nancy, France.

Meihan Tong, Bin Xu, Shuai Wang, Yixin Cao, Lei Hou, Juanzi Li, and Jun Xie. 2020. Improving event detection via open-domain trigger knowledge. In Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics, pages 5887–5897, Online. Association for Computational Linguistics.

MeiHan Tong, Bin Xu, Shuai Wang, Meihuan Han, Yixin Cao, Jiangqi Zhu, Siyu Chen, Lei Hou, and Juanzi Li. 2022. DocEE: A large-scale and finegrained benchmark for document-level event extraction. In Proceedings of the 2022 Conference of the

North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 3970–3982, Seattle, United States. Association for Computational Linguistics.

Xiaozhi Wang, Xu Han, Zhiyuan Liu, Maosong Sun, and Peng Li. 2019. Adversarial training for weakly supervised event detection. In Proceedings of the 2019 Conference ofthe North American Chapter of the Associationfor Computational Linguistics: Hu man Language Technologies, Volume 1 (Long and Short Papers), pages 998–1008, Minneapolis, Minnesota. Association for Computational Linguistics.

Xiaozhi Wang, Ziqi Wang, Xu Han, Wangyi Jiang, Rong Han, Zhiyuan Liu, Juanzi Li, Peng Li, Yankai Lin, and Jie Zhou. 2020. MAVEN: A Massive General Domain Event Detection Dataset. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1652– 1671, Online. Association for Computational Linguistics.

Hang Yang, Yubo Chen, Kang Liu, Yang Xiao, and Jun Zhao. 2018. DCFEE: A document-level Chinese financial event extraction system based on automatically labeled training data. In Proceedings ofACL 2018, System Demonstrations, pages 50–55, Melbourne, Australia. Association for Computational Linguistics.

Sen Yang, Dawei Feng, Linbo Qiao, Zhigang Kan, and Dongsheng Li. 2019. Exploring pre-trained language models for event extraction and generation. In Proceedings of the 57th Annual Meeting of the Associationfor Computational Linguistics, pages 5284– 5294, Florence, Italy. Association for Computational Linguistics.

Ying Zeng, Yansong Feng, Rong Ma, Zheng Wang, Rui Yan, Chongde Shi, and Dongyan Zhao. 2017. Scale up event extraction learning via automatic training data generation. In AAAI Conference on Artificial Intelligence.

Hongming Zhang, Wenlin Yao, and Dong Yu. 2022. Efficient zero-shot event extraction with contextdefinition alignment. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 7169–7179, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Shun Zheng, Wei Cao, Wei Xu, and Jiang Bian. 2019. Doc2EDAG: An end-to-end document-level framework for Chinese financial event extraction. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 337–346, Hong Kong, China. Association for Computational Linguistics.

## A Implementation Details

Table 6 presents the hyperparameters for three components in our model. During trigger identification, we restrict token spans to a maximum length of 10 tokens. For event type ranking, we use a designed loss function with $\tau = 1 . 0$ . The top 10 ranked events are selected as candidates for event type classification. In event type classification, we do one round of self-labeling. Our model is trained on a single Tesla P100 GPU with 16GB DRAM.

<table><tr><td>Component</td><td>TI</td><td>ETR</td><td>ETC</td></tr><tr><td>Training epochs</td><td>5</td><td>5</td><td>2</td></tr><tr><td>Batch size</td><td>128</td><td>64</td><td>32</td></tr><tr><td>Max sequence length</td><td>128</td><td>128</td><td>512</td></tr><tr><td>Base Model</td><td></td><td>Bert-base-uncased</td><td></td></tr><tr><td>learning rate</td><td></td><td>1e-5</td><td></td></tr><tr><td>Weight decay</td><td></td><td>0.01</td><td></td></tr><tr><td>Scheduler</td><td></td><td>Linear (with 50 warmup steps)</td><td></td></tr></table>

Table 6: The training hyper-parameters of our model. TI: Trigger Identification. ETR: Event Type Ranking. ETC: Event Type Classification.

## B Baseline Implementation Details

We list the details of our baselines as below:

1. Token Classification: We use the IO tagging scheme to classify tokens.

2. Span Classification: We use the embedding of the first and last token to represent the span for classification.

3. DMBERT (Wang et al., 2019) is a BERTbased model that applies dynamic pooling (Chen et al., 2015) according to the candidate trigger’s location. We consider single tokens to be candidate triggers in our dataset during testing.

4. ZED (Zhang et al., 2022) is an event detection model that utilizes definitions. Instead of pre-training on WordNet, we train the model on our noisy training data. As ZED only performs classification, we report the results with the trigger spans predicted by the token classification model (TokCls+ZED).

5. InstructGPT(Ouyang et al., 2022), also referred to as GPT-3.5, is the improved version of GPT3 trained with instruction-tuning. We use the text-davinci-003 model through the OpenAI API. We provide the model with an instruction for the task and 32 training examples from our training set for in-context learning.

We list the hyperparameters for our baseline models in Table 7. For ZED, we follow the original paper and set the margin to 0.2. We set the threshold for predicting an event type to 0.3 (if the cosine similarity between the event type representation and the trigger representation is smaller than this value, we will refrain from predicting any event type).

For the InstructGPT baseline, we use the text-davinci-003 model with a temperature of 0.2 and top\_p set to 0.95 for decoding. We show our detailed prompt in Figure 8. The first part of the prompt is the task instruction and then we include 32 input-output examples. Due to the current input length limit of InstructGPT, we were unable to feed the ontology into the model as part of the input.

## C Data Filtering and Annotation Details

Table 8 shows some examples of removed Qnodes from DWD Overlay in our ontology with different kinds of reasons.

To improve dataset quality, we perform sentence de-duplication, remove sentences with less than 3 tokens and omit special tokens (marked by \* or brackets). We ensure that every trigger is a continuous token span and we remove events with overlapping triggers. For the AMR dataset, we additionally remove triggers with a part-of-speech tag of MD (modal verbs) or TO (the word “to”) (such cases do not appear in other datasets).

Based on this distant supervision dataset, we make further adjustments to the ontology. We manually inspected the most popular rolesets that have more than 1000 event mentions and removed rolesets that are too general or ambiguous (for instance, cause.01 and see.01). Finally, we remove the rolesets that have less than 3 event mentions across all datasets.

Figure 9 displays our annotation interface. The left box features the context, highlighting a single trigger, while the right box enumerates candidate event types, expressed as a combination of name and description, with an extra choice labeled "None of the above options is correct." Each Qnode is represented by its name and description. The number of options varies from 2 to 9 based on the ontology.

<table><tr><td>Model</td><td>DMBERT</td><td>SpanCls</td><td>TokCls</td><td>ZED</td></tr><tr><td rowspan="4">Training epochs Batch size Negative samples Learning rate</td><td>10</td><td>10</td><td>10</td><td>10</td></tr><tr><td>64</td><td>64</td><td>16</td><td>16</td></tr><tr><td>5</td><td>5</td><td></td><td>5</td></tr><tr><td>5e-5</td><td>5e-5 64</td><td>2e-5</td><td>2e-5</td></tr><tr><td>Max sequence length Base Model</td><td colspan="4">Bert-base-uncased</td></tr><tr><td>Weight decay</td><td colspan="4">0.0</td></tr><tr><td>Scheduler</td><td colspan="4">Linear</td></tr><tr><td></td><td colspan="4"></td></tr></table>

Table 7: Hyperparameter settings for baseline models.  
![](images/12d2c33e927ab6beba0e7436429a3d7c5743943b23ece0a181a4c2fce3e612bf.jpg)  
Figure 8: Truncated version of our prompt to InstructGPT.

The annotator’s task is to select the option that most accurately represents the trigger word.

Each instance was annotated by two annotators separately. For PropBank rolesets that were frequently labeled as “None of the above”, we performed manual inspection to determine if the mapping should be removed or revised. Finally, for the affected instances and the instances with disagreement, we asked our in-house annotators to perform a third pass as adjudication.

<table><tr><td>Qnode</td><td>Name</td><td>Roleset</td><td>Description</td></tr><tr><td colspan="4">Removed during cleaning the heavily used rolesets</td></tr><tr><td>Q2536390</td><td>abdominal_distention</td><td>ill.01</td><td>Physical symptom</td></tr><tr><td>Q192989</td><td>acculturation</td><td>change.01</td><td>process of cultural and psychological change</td></tr><tr><td>Q422268</td><td>actinomycosis</td><td>ill.01</td><td>Human disease</td></tr><tr><td>Q1319035</td><td>adult_education</td><td>educate.01</td><td>form of learning adults engage in beyond traditional schooling</td></tr><tr><td>Q9363879</td><td>stamping</td><td>make.01</td><td>metalworking</td></tr><tr><td>Q615857</td><td>stapedectomy</td><td>surgery.01</td><td>surgical procedure of the middle ear performed to improve hearing</td></tr><tr><td>Q366774</td><td>adrenalectomy</td><td>remove.01</td><td>surgical removal of the adrenal gland</td></tr><tr><td>Q2035485</td><td>subcutaneous_injection</td><td>inject.01</td><td>Medical procedure</td></tr><tr><td colspan="4">Removing reason: cognitive events that do not involve any physical state change</td></tr><tr><td>Q241625</td><td>wish</td><td>wish.01</td><td>desire for a specific item or event</td></tr><tr><td>Q26256512</td><td>want</td><td>want.01</td><td>economic term for something that is desired</td></tr><tr><td>Q26253999</td><td>yearning</td><td>yearn.01</td><td>deep and aching desire for someone or something</td></tr><tr><td>Q706622</td><td>intention</td><td>intend.01</td><td>mental state representing commitment to perform an action</td></tr><tr><td>Q3027692</td><td>differentiation</td><td>differentiate.01</td><td>process by which two closely related linguistic varieties diverge from one another during their evolution</td></tr><tr><td>Q104776298</td><td>crosspatch</td><td>grouch.01</td><td>a person who is easily annoyed</td></tr><tr><td>Q516519</td><td>suspicion</td><td>suspect.01</td><td>emotion</td></tr><tr><td>Q659974</td><td>trust</td><td>trust.02</td><td>assumption of and reliance on the honesty of another party</td></tr><tr><td colspan="4">Removing reason: mapped to too general or ambiguous rolesets</td></tr><tr><td>Q105606485</td><td>intellectual_activity</td><td>think.01</td><td>human activity comprising of mental actions</td></tr><tr><td>Q2944236</td><td>photosensitivity</td><td>see.01</td><td>Light sensitivity in homo sapiens</td></tr><tr><td>Q9174</td><td>religion</td><td>believe.01</td><td>set of beliefs, practices and traditions for a group or community</td></tr><tr><td>Q16513426</td><td>decision</td><td>decide.01</td><td>result of deliberation</td></tr><tr><td>Q2827815</td><td>international_aid</td><td>give.01</td><td>voluntary transfer of resources from one country to another</td></tr><tr><td>Q9081</td><td>knowledge</td><td>know.01</td><td>experience or education by perceiving, discovering, or learning agreement between employer and employee on terms of work and</td></tr><tr><td>Q1221208</td><td>employment_contract</td><td>agree.01</td><td>compensation</td></tr><tr><td>Q56274009</td><td>looking</td><td>look.01</td><td>act of intentionally focusing visual perception on someone or something</td></tr><tr><td colspan="4">Removing reason: low frequency</td></tr><tr><td>Q379788</td><td>advection</td><td>advect.01</td><td>transport of a substance by bulk motion</td></tr><tr><td>Q381105</td><td>aeration</td><td>aerate.01</td><td>process of circulating or mixing air with water colloid of fine solid particles or liquid droplets, in air or another</td></tr><tr><td>Q104541</td><td>aerosol</td><td>aerosolize.01</td><td>gas</td></tr><tr><td>Q623179</td><td>state_terrorism</td><td>terrorism.03</td><td>acts of terrorism against individuals conducted by organs of a state</td></tr><tr><td>Q98394474</td><td>stenciling</td><td>stencil.01</td><td>artistic technique for transferring images using stencils</td></tr><tr><td>Q844613</td><td>sintering</td><td>sinter.01</td><td>process of forming material by heat or pressure</td></tr><tr><td>Q249697</td><td>eulogy</td><td>eulogize.01</td><td>speech in praise of a person, usually recently deceased</td></tr><tr><td>Q901882</td><td>interface</td><td>interface.01</td><td>boundary between different phases of matter</td></tr></table>

Table 8: Examples of removed Qnodes in XPO Overlay. Note that one node can be removed due to multiple reasons.

## D Impact of Self-labeling

<table><tr><td>Roleset Category</td><td>Clean</td><td>Covered</td><td>Other</td><td>Total</td></tr><tr><td>#Instances</td><td>3642</td><td>3308</td><td>1223</td><td>8173</td></tr><tr><td>Hit@1 before</td><td>95.74</td><td>48.52</td><td>37.37</td><td>67.89</td></tr><tr><td>Hit@1 after</td><td>95.47</td><td>54.59</td><td>39.17</td><td>70.50</td></tr></table>

Table 9: Hit@1 scores before and after self-labeling on different categories of PropBank rolesets. Clean: rolesets with only one candidate labels. Covered: rolesets covered in the self-labeled training data.

Figure 10 indicates that with a threshold<sup>10</sup> of 0.9, the accuracy of selecting the correct label from a candidate set reaches 57.8% on the dev set. To investigate how self-labeling contributes to the improvements, we categorize test instances into three groups based on their PropBank rolesets, as shown in Table 9. The ‘Clean’ rolesets map to only one event type in DWD Overlay. We train the base classifier on 71,834 training instances corresponding to these ‘Clean’ rolesets, which naturally performs significantly well on this portion of data. The model after self-labeling is trained with an additional 25,549 self-labeled data. The rolesets corresponding to these data are categorized as “Covered”. Table 9 indicates that the main performance gain comes from the “Covered” data, which is boosted directly by including corresponding training data. The “Other” category also sees some improvement, at the cost of a slight drop in the “Clean” category.

![](images/986da413474b492b5fb5183c21d841030eb0743f66b9a3584f4c268904baaa47.jpg)  
Figure 9: Annotation interface built with Amazon Mechanical Turk for labeling the development and test set.

![](images/26610eb1df31b5748256f97accce42a6a7dcfe9f0a94904e781fedfd3b3604e2.jpg)  
Figure 10: Relationship between the threshold and two metrics: the accuracy of label selection on dev set and the number of selected instances. The x-axis represents the threshold.