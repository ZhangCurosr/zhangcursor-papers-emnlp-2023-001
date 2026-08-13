# Hierarchical Pretraining on Multimodal Electronic Health Records

Xiaochen Wang<sup>1</sup>, Junyu Luo<sup>1</sup>, Jiaqi Wang<sup>1</sup>, Ziyi Yin<sup>1</sup>, Suhan Cui<sup>1</sup>, Yuan Zhong<sup>1</sup>, Yaqing Wang<sup>2</sup>, Fenglong Ma<sup>1</sup>

<sup>1</sup>The Pennsylvania State University, <sup>2</sup>Google Research

<sup>1</sup>{xcwang, junyu, jqwang, ziyiyin, suhan, yuanzhong, fenglong}@psu.edu <sup>2</sup>yaqingwang@google.com

## Abstract

Pretraining has proven to be a powerful technique in natural language processing (NLP), exhibiting remarkable success in various NLP downstream tasks. However, in the medical domain, existing pretrained models on electronic health records (EHR) fail to capture the hierarchical nature of EHR data, limiting their generalization capability across diverse downstream tasks using a single pretrained model. To tackle this challenge, this paper introduces a novel, general, and unified pretraining framework called MEDHMP<sup>1</sup>, specifically designed for hierarchically multimodal EHR data. The effectiveness of the proposed MEDHMP is demonstrated through experimental results on eight downstream tasks spanning three levels. Comparisons against eighteen baselines further highlight the efficacy of our approach.

## 1 Introduction

Pretraining is a widely adopted technique in natural language processing (NLP). It entails training a model on a large dataset using unsupervised learning before fine-tuning it on a specific downstream task using a smaller labeled dataset. Pretrained models like BERT (Devlin et al., 2018) and GPT (Radford et al., 2018) have demonstrated remarkable success across a range of NLP tasks, contributing to significant advancements in various NLP benchmarks.

In the medical domain, with the increasing availability of electronic health records (EHR), researchers have attempted to pre-train domainspecific models to improve the performance of various predictive tasks further (Qiu et al., 2023). For instance, ClinicalBERT (Huang et al., 2019) and ClinicalT5 (Lehman and Johnson, 2023) are pretrained on clinical notes, and Med2Vec (Choi et al., 2016a) and MIME (Choi et al., 2018) on medical codes. These models pretrained on a single type of data are too specific, significantly limiting their transferability. Although some pretraining models (Li et al., 2022a, 2020; Meng et al., 2021) are proposed to use multimodal EHR data<sup>2</sup>, they ignore the heterogeneous and hierarchical characteristics of such data.

![](images/e5037b7fa0f94de4ad16ae5f3fd5c9e124a5facfad02a072207528d26d7dd802.jpg)  
Figure 1: An illustration of EHR hierarchy.

The EHR data, as depicted in Figure 1, exhibit a hierarchical structure. At the patient level, the EHR systems record demographic information and capture multiple admissions/visits in a timeordered manner. Each admission represents a specific hospitalization period and contains multiple stay records, International Classification of Diseases (ICD) codes for billing, drug codes, and a corresponding clinical note. Each stay record includes hourly clinical monitoring readings like heart rate, arterial blood pressure, and respiratory rate.

In addition to the intricate hierarchy of EHR data, the prediction tasks vary across levels. As we move from the top to the bottom levels, the prediction tasks become more time-sensitive. Patientlevel data are usually used to predict the risk of a patient suffering from potential diseases after six months or one year, i.e., the health risk prediction task. Admission-level data are employed for relatively shorter-term predictions, such as readmission within 30 days. Stay-level data are typically utilized for hourly predictions, such as forecasting acute respiratory failure (ARF) within a few hours.

Designing an ideal “one-in-all” medical pretraining model that can effectively incorporate multimodal, heterogeneous, and hierarchical EHR data as inputs, while performing self-supervised learning across different levels, is a complex undertaking. This complexity arises due to the varying data types encountered at different levels. At the stay level, the data primarily consist of time-ordered numerical clinical variables. However, at the admission level, the data not only encompass sequential numerical features from stays but also include sets of discrete ICD and drug codes, as well as unstructured clinical notes. As a result, it becomes challenging to devise appropriate pretraining tasks capable of effectively extracting knowledge from the intricate EHR data.

In this paper, we present a novel Hierarchical Multimodal Pretraining framework (called MEDHMP) to tackle the aforementioned challenges in the Medical domain. MEDHMP simultaneously incorporates five modalities as inputs, including patient demographics, temporal clinical features for stays, ICD codes, drug codes, and clinical notes. To effectively pretrain MEDHMP, we adopt a “bottom-to-up” approach and introduce level-specific self-supervised learning tasks. At the stay level, we propose reconstructing the numerical time-ordered clinical features. We devise two pretraining strategies for the admission level. The first focuses on modeling intra-modality relations by predicting a set of masked ICD and drug codes. The second involves modeling inter-modality relations through modality-level contrastive learning. To train the complete MEDHMP model, we utilize a two-stage training strategy from stay to admission levels<sup>3</sup>.

We utilize two publicly available medical datasets for pretraining the proposed MEDHMP and evaluate its performance on three levels of downstream tasks. These tasks include ARF, shock and mortality predictions at the stay level, readmission prediction at the admission level, and health risk prediction at the patient level. Through our experiments, we validate the effectiveness of the proposed MEDHMP by comparing it with stateof-the-art baselines. The results obtained clearly indicate the valuable contribution of MEDHMP in the medical domain and highlight its superior performance enhancements in these predictive downstream tasks.

## 2 Methodology

As highlighted in Section 1, EHR data exhibit considerable complexity and heterogeneity. To tackle this issue, we introduce MEDHMP as a solution that leverages pretraining strategies across multiple modalities and different levels within the EHR hierarchy to achieve unification. In the following sections, we present the design details of the proposed MEDHMP.

## 2.1 Model Input

As shown in Figure 1, each patient data consist of multiple time-ordered hospital admissions, i.e., $P = [ A _ { 1 } , A _ { 2 } , \cdots , A _ { N } ]$ , where $\mathcal { A } _ { i } \left( i \in \left[ 1 , n \right] \right)$ is the i-th admission, and N is the number of admissions. Note that for different patients, N may be different. Each patient also has a set of demographic features denoted as $\mathcal { D } .$ . Each admission $\mathbf { \mathcal { A } } _ { i }$ consists of multiple time-ordered staylevel data denoted as $\textstyle S _ { i } .$ , a set of ICD codes denoted as $\mathcal { C } _ { i }$ , a piece of clinical notes denoted as $\mathcal { L } _ { i }$ , and a set of drug codes $\mathcal { G } _ { i }$ , i.e., $\mathbf { \mathcal { A } } _ { i } ~ =$ $\{ S _ { i } , \mathcal { C } _ { i } , \mathcal { L } _ { i } , \mathcal { G } _ { i } \}$ The stay-level data $S _ { i }$ contains a sequence of hourly-recorded monitoring stays, $\mathrm { i . e . , } \bar { \mathbf { \mathcal { S } } } _ { i } = [ \mathbf { S } _ { i } ^ { 1 } , \mathbf { S } _ { i } ^ { 2 } , \cdots , \mathbf { S } _ { i } ^ { M _ { i } } ]$ , where $\mathbf { S } _ { i } ^ { j }$ represents the feature matrix of the j-th stay, and $M _ { i }$ denotes the number of stays within each admission.

## 2.2 Stay-level Self-supervised Pretraining

We conduct the self-supervised pretraining in a bottom-to-top way and start from the stay level. When pretraining the stay-level data, we only use $S _ { i }$ and $\mathcal { D }$ since the diagnosis codes $\mathcal { C } _ { i }$ , drug codes $\mathcal { G } _ { i }$ and clinical notes $\mathcal { L } _ { i }$ are recorded at the end of the i-th admission. However, demographic information is highly related to a patient’s clinical monitoring features in general. Due to the monitoring features being recorded with numerical values, we propose to use a reconstruction strategy as the stay-level pretraining task, as illustrated in Figure 2.

![](images/87a814e566fba33194a2378c2453feaba6385281901d3fadbb363b7746e67343.jpg)  
Figure 2: Stay-level self-supervised pretraining.

## 2.2.1 Stay-level Feature Encoding

Each stay $\mathbf { S } _ { i } ^ { j } ~ \in ~ S _ { i }$ consists of a set of timeordered hourly clinical features, $\begin{array} { r l } { \mathrm { i . e . , } ~ \mathbf { S } _ { i } ^ { j } } & { { } = } \end{array}$ $[ \mathbf { m } _ { i , 1 } ^ { j } , \mathbf { m } _ { i , 2 } ^ { j } , \cdot \cdot \cdot , \mathbf { m } _ { i , T } ^ { j } ]$ , where $\mathbf { m } _ { i , t } ^ { j } \in \mathbb { R } ^ { d _ { f } }$ is the recorded feature vector at the t-th hour, $T$ is the number of monitoring hours, and $d _ { f }$ denotes the number of time-series clinical features. To model the temporal characteristic of $\mathbf { S } _ { i } ^ { j }$ , we directly apply long-short term memory (LSTM) network (Hochreiter and Schmidhuber, 1997) and treat the output cell state $\mathbf { h } _ { i } ^ { j }$ as the representation of the j-th stay, i.e.,

$$
\mathbf { h } _ { i } ^ { j } = \mathbf { L S T M } _ { e n c } ( [ \mathbf { m } _ { i , 1 } ^ { j } , \mathbf { m } _ { i , 2 } ^ { j } , \cdot \cdot \cdot , \mathbf { m } _ { i , T } ^ { j } ] ) ,\tag{1}
$$

where $\mathrm { L S T M } _ { e n c }$ is the encoding LSTM network.

## 2.2.2 Clinical Feature Reconstruction

A naive approach to reconstructing the input staylevel feature $\mathbf { S } _ { i } ^ { j }$ is simply applying an LSTM decoder as (Srivastava et al., 2015) does. However, this straightforward approach may not work for the clinical data. The reason is that the clinical feature vector $\mathbf { m } _ { i , k } ^ { j } \in \mathbf { S } _ { i } ^ { j }$ is extremely sparse due to the impossibility of monitoring all the vital signs and conducting all examinations for a patient. To accurately reconstruct such a sparse matrix, we need to use the demographic information  as the guidance because some examinations are highly related to age or gender, which also makes us achieve the goal of multi-modal pretraining.

Specifically, we first embed the demographic information into a dense vector representation, i.e., $\textbf { d } = \mathbf { M L P } _ { d } ( \mathcal { D } )$ , where M $. \mathrm { P } _ { d }$ denotes the multilayer perceptron activated by the ReLU function. To fuse the demographic representation and the stay representation, we propose to use a transformer block in which self-attention is performed for modality fusion, followed by residual calculation, normalization, and a pooling operation compressing the latent representation to the unified dimension size. We obtain the bimodal representation $\mathbf { b } _ { i } ^ { j }$ as follows:

$$
\begin{array} { r l r } & { } & { \hat { \mathbf { b } } _ { i } ^ { j } = \mathrm { S o f t m a x } ( \frac { \mathbf { W } _ { h } ^ { Q } \langle \mathbf { h } _ { i } ^ { j } , \mathbf { d } \rangle \cdot \mathbf { W } _ { h } ^ { K } \langle \mathbf { h } _ { i } ^ { j } , \mathbf { d } \rangle } { \sqrt { d _ { r } } } ) \cdot \mathbf { W } _ { h } ^ { V } \langle \mathbf { h } _ { i } ^ { j } , \mathbf { d } \rangle , } \\ & { } & { \mathbf { b } _ { i } ^ { j } = \mathbf { M a x P o o l i n g } ( \mathrm { L a y e r N o r m } ( \langle \mathbf { h } _ { i } ^ { j } , \mathbf { d } \rangle + \hat { \mathbf { b } } _ { i } ^ { j } ) ) , \quad } \end{array}\tag{2}
$$

where $\langle \cdot , \cdot \rangle$ means the operation of stacking, $\mathbf { W } _ { h } ^ { Q } ;$ $\mathbf { W } _ { h } ^ { K } , \dot { \mathbf { W } } _ { h } ^ { V } \in \mathbb { R } ^ { d _ { r } \times d _ { r } }$ are trainable parameters, and $d _ { r }$ is the unified size of representation.

Using the fused representation $\mathbf { b } _ { i } ^ { j }$ , MEDHMP then reconstructs the input clinical feature matrix $\mathbf { S } _ { i } ^ { j }$ . Since the clinical features are time-series data, we take $\mathbf { b } _ { i } ^ { j }$ as the initial hidden state of the LSTM decoder $\mathrm { L S T M } _ { d e c }$ to sequentially reconstruct the corresponding clinical feature $\tilde { \mathbf { m } } _ { i , k } ^ { j } =$ $\mathrm { L S T M } _ { d e c } ( \mathbf { b } _ { i } ^ { j } )$

## 2.2.3 Stay-level Pretraining Loss

After obtaining the reconstructed clinical features $[ \tilde { \mathbf { m } } _ { i , 1 } ^ { j } , \tilde { \mathbf { m } } _ { i , 2 } ^ { j } , \cdot \cdot \cdot , \tilde { \mathbf { m } } _ { i , T } ^ { j } ]$ , we then apply the mean squared error (MSE) as the pretraining loss to train the parameters in the stay-level as follows:

$$
\mathcal { L } _ { \mathrm { s t a y } } = \frac { 1 } { N * M * T } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } \sum _ { t = 1 } ^ { T } | | \mathbf { m } _ { i , t } ^ { j } - \tilde { \mathbf { m } } _ { i , t } ^ { j } | | _ { 2 } ^ { 2 } .\tag{3}
$$

## 2.3 Admission-level Pretraining

The stay-level pretraining allows MEDHMP to acquire the sufficient capability of representing stays, laying the groundwork for the pretraining at the admission level. Next, we introduce the details of pretraining at this level.

## 2.3.1 Admission-level Feature Encoding

As introduced in Section 2.1, each admission $\mathcal { A } _ { i } =$ $\{ S _ { i } , \mathcal { C } _ { i } , \mathcal { L } _ { i } , \mathcal { G } _ { i } \}$ . To conduct the self-supervised pretraining, the first step is to encode each input to a latent representation.

In Section 2.2, we can obtain the representation of each hourly feature $\mathbf { b } _ { i } ^ { j }$ using Eq. (2). Thus, we can further have the stay-level overall representation $\mathbf { s } _ { i }$ by aggregating all hourly representations of $s _ { i }$ via a linear transformation as follows:

$$
\mathbf { s } _ { i } = \mathbf { W } _ { s } ^ { \top } \langle \mathbf { b } _ { i } ^ { 1 } ; \mathbf { b } _ { i } ^ { 2 } ; \cdot \cdot \cdot ; \mathbf { b } _ { i } ^ { M } \rangle + \mathbf { b } _ { s } ,\tag{4}
$$

where $\langle \cdot ; \cdot \rangle$ is the concatenation operation. ${ \bf W } _ { s } \in  \qquad $   
$\mathbb { R } ^ { d _ { r } \times M * d _ { r } }$ and $\mathbf { b } _ { s } \in \mathbb { R } ^ { d _ { r } }$ are parameters.

For ICD codes $\mathcal { C } _ { i }$ and drug codes $\mathcal { G } _ { i }$ , they will be converted to binary vectors and then map them to

![](images/5e9b44ed042b7fe4a2554c032296e43e869f0cf5d50a74295605010c67866eb8.jpg)  
Figure 3: Admission-level self-supervised pretraining.

latent representations via MLP layers, which is similar to the mapping of the demographic information, as follows:

$$
\mathbf { c } _ { i } = \mathbf { M } \mathbf { L } \mathbf { P } _ { c } ( \mathcal { C } _ { i } ) , \mathbf { g } _ { i } = \mathbf { M } \mathbf { L } \mathbf { P } _ { g } ( \mathcal { G } _ { i } ) .\tag{5}
$$

For the unstructured clinical notes $\mathcal { L } _ { i } .$ , we directly use a pretrained domain-specific encoder (Lehman and Johnson, 2023) to generate its representation $\mathbf { l } _ { i }$

Using the learned representations, we can conduct admission-level pretraining. Due to the unique characteristics of multimodal EHR data, we will focus on two kinds of pretraining tasks: mask code prediction for intra-modalities and contrastive learning for inter-modalities, as shown in Figure 3.

## 2.3.2 Intra-modality Mask Code Prediction

In the natural language processing (NLP) domain, mask language modeling (MLM) (Devlin et al., 2018) is a prevalent pretraining task encouraging the model to capture correlations between tokens. However, the EHR data within an admission $\mathbf { \mathcal { A } } _ { i }$ are significantly different from text data, where the ICD and drug codes are sets instead of sequences. Moreover, the codes are distinct. In other words, no identical codes appear in $\mathcal { C } _ { i }$ and $\mathcal { G } _ { i }$ . Thus, it is essential to design a new loss function to predict the masked codes.

Let $\mathbf { c } _ { i } ^ { m } \in \mathbb { R } ^ { | \mathcal { C } | }$ and $\mathbf { g } _ { i } ^ { m } \in \mathbb { R } ^ { | \mathcal { G } | }$ denote the mask indicator vectors, where $| { \mathcal { C } } |$ and  denote the distinct number of ICD codes and drug codes, respectively. If the $j \cdot$ th ICD code is masked, then $\mathbf { c } _ { i } ^ { m } [ j ] = 1$ ; otherwise, $\mathbf { c } _ { i } ^ { m } [ j ] = 0$ . Let $\mathbf { c } _ { i } ^ { \prime }$ and $\mathbf { g } _ { i } ^ { \prime }$ denote the embeddings learned for the remaining codes. To predict the masked codes, we need to obtain the admission representation. Toward this end, we first stack all the learned embeddings as follows:

$$
\mathbf { f } _ { i } = \langle \mathbf { s } _ { i } , \mathbf { c } _ { i } ^ { \prime } , \mathbf { g } _ { i } ^ { \prime } , \mathbf { l } _ { i } \rangle .\tag{6}
$$

Then another transformer encoder block is used to obtain the cross-modal admission representation as follows:

$$
\begin{array} { r } { \hat { \mathbf { a } } _ { i } = \mathrm { S o f t m a x } ( \frac { \mathbf { W } _ { a } ^ { Q } \mathbf { f } _ { i } \cdot \mathbf { W } _ { a } ^ { K } \mathbf { f } _ { i } } { \sqrt { d _ { r } } } ) \cdot \mathbf { W } _ { a } ^ { V } \mathbf { f } _ { i } , } \\ { \mathbf { a } _ { i } = \mathbf { M a x P o o l i n g } ( \mathrm { L a y e r N o r m } ( \mathbf { f } _ { i } + \hat { \mathbf { a } } _ { i } ) ) , } \end{array}\tag{7}
$$

where $\mathbf { W } _ { a } ^ { Q } , \mathbf { W } _ { a } ^ { K }$ , and $\mathbf { W } _ { a } ^ { V } \in \mathbb { R } ^ { d _ { r } \times d _ { r } }$ are trainable parameters.

We can predict the masked codes using the learned admission representation ${ \bf a } _ { i }$ using Eq. (7) as follows:

$$
\begin{array} { r } { \mathbf { p } _ { i } ^ { c } = \mathrm { S i g m o i d } ( \mathrm { M L P } _ { m c } ( \mathbf { a } _ { i } ) ) , } \\ { \mathbf { p } _ { i } ^ { g } = \mathrm { S i g m o i d } ( \mathrm { M L P } _ { m g } ( \mathbf { a } _ { i } ) ) , } \end{array}\tag{8}
$$

where the predicted probability vectors $\mathbf { p } _ { i } ^ { c } \in \mathbb { R } ^ { | c | }$ and $\mathbf { p } _ { i } ^ { g } \in \bar { \mathbb { R } } ^ { | \mathcal { G } | }$

Finally, the MSE loss serves as the objective function of the masked code prediction (MCP) task for the intra-modality modeling as follows:

$$
\mathcal { L } _ { \mathrm { M C P } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( | | \mathbf { p } _ { i } ^ { c } - \mathbf { c } _ { i } ^ { m } | | _ { 2 } ^ { 2 }\tag{9}
$$

where  is the element-wise multiplication.

## 2.3.3 Inter-modality Contrastive Learning

The intra-modality modeling aims to learn feature relations within a single modality using other modalities’ information. On top of it, we also consider inter-modality relations. Intuitively, the four representations $\left\{ \mathbf { s } _ { i } , \mathbf { c } _ { i } , \mathbf { g } _ { i } , \mathbf { l } _ { i } \right\}$ within $\mathbf { \mathcal { A } } _ { i }$ share similar information. If a certain modality $\mathbf { r } _ { i } ~ \in$ $\left\{ \mathbf { s } _ { i } , \mathbf { c } _ { i } , \mathbf { g } _ { i } , \mathbf { l } _ { i } \right\}$ is masked, the similarity between $\mathbf { r } _ { i }$ and the aggregated representation $\mathbf { a } _ { i } \backslash \mathbf { r } _ { i }$ learned from the remaining ones should be still larger than that between $\mathbf { r } _ { i }$ and another admission’s representation $\mathbf { a } _ { j } \backslash \mathbf { r } _ { j }$ within the same batch, where $j \neq i .$

Based on this intuition, we propose to use the noise contrastive estimation (NCE) loss as the intermodality modeling objective as follows:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { C L } } = \displaystyle \frac { 1 } { 3 N } \sum _ { i = 1 } ^ { N } \sum _ { \mathbf { r } _ { i } \in \{ \mathbf { c } _ { i } , \mathbf { g } _ { i } , \mathbf { l } _ { i } \} } u ( \mathbf { r } _ { i } ) , } \\ & { u ( \mathbf { r } _ { i } ) = - \log \frac { e ^ { \mathrm { s i m } ( \mathbf { r } _ { i } , \mathbf { a } _ { i } \setminus \mathbf { r } _ { i } ) / \tau } } { \sum _ { j = 1 , j \neq i } ^ { B } e ^ { \mathrm { s i m } ( \mathbf { r } _ { i } , \mathbf { a } _ { j } \setminus \mathbf { r } _ { j } ) / \tau } } , } \end{array}\tag{10}
$$

where sim $( \cdot , \cdot )$ denotes the cosine similarity, $B$ is the batch size, and $\tau$ is the temperature hyperparameter. $\mathbf { a } _ { i } \backslash \mathbf { r } _ { i }$ is obtained using Eqs. (6) and (7) by removing the masked modality $\mathbf { r } _ { i }$ . Note that in our design, s<sub>i</sub> is a trained representation by optimizing the stay-level objective via Eq. (3). However, the other three modality representations are learned from scratch or the pretrained initialization. To avoid overfitting $\mathbf { s } _ { i } ,$ , we do not mask the stay-level representation $\mathbf { s } _ { i }$ in Eq. (10).

## 2.3.4 Admission-level Pretraining Loss

The final loss function in the admission-level pretraining is represented as follows:

$$
\mathcal { L } _ { \mathrm { a d m i s s i o n } } = \mathcal { L } _ { \mathrm { M C P } } + \lambda \mathcal { L } _ { \mathrm { C L } } ,\tag{11}
$$

where λ is a hyperparameter to balance the losses between the intra-modality mask code prediction task and the inter-modality contrastive learning.

## 2.4 Training of MEDHMP

We use a two-stage training strategy to train the proposed MEDHMP. In the first stage, we pre-train the stay-level task via Eq. (3) by convergence. In the second stage, we use the learned parameters in the first stage as initialization and then train the admission-level task via Eq. (11).

## 3 Experiments

In this section, we first introduce the data for pretraining and downstream tasks and then exhibit experimental results (mean values of five runs).

## 3.1 Data Extraction

We utilize two publicly available multimodal EHR datasets – MIMIC-III (Johnson et al., 2016) and MIMIC-IV (Johnson et al., 2020) – to pretrain the proposed MEDHMP. We adopt FIDDLE (Tang et al., 2020) to extract the pretraining data and use different levels’ downstream tasks to evaluate the effectiveness of the proposed MEDHMP. For the stay-level evaluation, we predict whether the patient will suffer from acute respiratory failure (ARF)/shock/mortality within 48 hours by extracting data from the MIMIC-III dataset<sup>4</sup>. For the admission-level evaluation, we rely on the same pipeline for extracting data from the MIMIC-III dataset to predict the 30-day readmission rate. For the patient-level evaluation, we conduct four health risk prediction tasks by extracting the heart failure data from MIMIC-III following (Choi et al., 2016b) and the data of chronic obstructive pulmonary disease (COPD), amnesia, and heart failure from TriNetX<sup>5</sup>. The details of data extraction and statistics can be found in Appendix A. The implementation details of MEDHMP are in Appendix B.

## 3.2 Stay-level Evaluation

We conduct two experiments to validate the usefulness of the proposed MEDHMP at the stay level.

## 3.2.1 Stay-level Multimodal Evaluation

In this experiment, we take two modalities, i.e., demographics and clinical features, as the model inputs. The bimodal representation $\mathbf { b } _ { i } ^ { j }$ learned by Eq. (2) is then fed into a fully connected layer followed by the sigmoid activation function to calculate the prediction. We use the cross entropy as the loss function to finetune MEDHMP.

We use F-LSTM (Tang et al., 2020), F-CNN (Tang et al., 2020), RAIM (Xu et al., 2018), and DCMN (Feng et al., 2019) as the baselines. The details of each baseline can be found in Appendix C. We utilize the Area Under the Receiver Operating Characteristic curve (AUROC) and the Area Under the Precision-Recall curve (AUPR) as evaluation metrics.

The experimental results are presented in Table 1, showcasing the superior performance of MEDHMP compared to the bimodal baselines in all three stay-level tasks. This indicates the proficiency of MEDHMP in effectively utilizing both clinical and demographic features. Remarkably, MEDHMP demonstrates a particularly strong advantage when handling tasks with smaller-sized datasets (See Table 8 for data scale). This observation suggests that MEDHMP greatly benefits from our effective pre-training procedure, enabling it to deliver impressive performance, especially in low-resource conditions.

<table><tr><td>Task</td><td colspan="2">ARF</td><td colspan="2">Shock</td><td colspan="2">Mortality</td></tr><tr><td>Metric</td><td>AUROC</td><td>AUPR</td><td>AUROC</td><td>AUPR</td><td>AUROC</td><td>AUPR</td></tr><tr><td>F-LSTM</td><td>69.67</td><td>10.57</td><td>70.28</td><td>23.09</td><td>81.55</td><td>48.62</td></tr><tr><td>F-CNN</td><td>69.61</td><td>10.68</td><td>69.27</td><td>23.51</td><td>80.71</td><td>42.29</td></tr><tr><td>RAIM</td><td>59.38</td><td>8.42</td><td>66.20</td><td>20.02</td><td>77.17</td><td>39.96</td></tr><tr><td>DCMN</td><td>68.98</td><td>10.07</td><td>68.68</td><td>21.72</td><td>80.05</td><td>42.93</td></tr><tr><td>MEDHMP</td><td>71.66</td><td>14.34</td><td>71.04</td><td>24.19</td><td>82.17</td><td>47.52</td></tr></table>

Table 1: Results (%) on stay-level tasks.

Note that in the previous work (Yang and Wu, 2021), except for the demographics and clinical features, clinical notes are used to make predictions on the ARF task. We also conducted such experiments on the three tasks, and the results are listed in Appendix D. The experimental results still demonstrate the effectiveness of the proposed pretraining framework.

## 3.2.2 Stay-level Unimodal Evaluation

To validate the transferability of the proposed MEDHMP, we also conduct the following experiment by initializing the encoders of baselines using the pretrained MEDHMP. In this experiment, we only take the clinical features as models’ inputs. Two baselines are used: LSTM (Hochreiter and Schmidhuber, 1997) and Transformer (Vaswani et al., 2017). We use the pretrained LSTM encoder $\mathrm { L S T M } _ { e n c }$ in Section 2.2.1 to replace the original linear encoders in LSTM and Transformer. Our encoder will be finetuned with the training of LSTM and Transformer.

The experimental results on the ARF task are shown in Figure 4. As mentioned in Section 2.4, we train the LSTM encoder $\mathbf { L S T M _ { e n c } }$ twice. “w. MEDHMP<sub>a</sub>” means that the baselines use a well-trained admission-level $\mathrm { L S T M } _ { e n c }$ $\mathbf { \tilde { \Sigma } } ^ {  } \mathbf { w } . \mathbf { M E D H M P } _ { s }$ ” indicates that the baselines use a stay-level trained $\mathrm { L S T M } _ { e n c } .$ “Original” denotes the original baselines. We can observe that using partially- or well-trained encoders helps improve performance. These results also confirm the necessity of the proposed two-stage training strategy.

## 3.3 Admission-level Evaluation

We also adopt the readmission prediction task within 30 days to evaluate MEDHMP at the admission level. In this task, the model will task all modalities as the input, including demographics, clinical features, ICD codes, drug codes, and a corresponding clinical note for admission. In this experiment, we first learn the representations, i.e., s<sub>i</sub> using Eq. (4), $\mathbf { c } _ { i }$ and g<sub>i</sub> via Eq. (5), and $\mathbf { l } _ { i } .$ to obtain the stacked embedding $\mathbf { f } _ { i } .$ . We then apply Eq. (7) to obtain the admission embedding a<sub>i</sub>. Finally, a fully connected layer with the sigmoid function is used for prediction. We still use the cross-entropy loss as the optimization function.

![](images/afbedc39e30685326b540217c6a7b40ba5bf50774b90cc852311fe6e8aa859ad.jpg)  
Figure 4: Unimodal evaluation on the ARF task.

<table><tr><td>Model</td><td>AUROC</td><td>AUPR</td></tr><tr><td>BertLstm</td><td>63.35</td><td>7.24</td></tr><tr><td>LstmBert</td><td>60.67</td><td>6.84</td></tr><tr><td>BertCnn</td><td>63.07</td><td>7.19</td></tr><tr><td>CnnBert</td><td>61.59</td><td>7.04</td></tr><tr><td>BertStar</td><td>61.28</td><td>6.84</td></tr><tr><td>StarBert</td><td>60.67</td><td>6.84</td></tr><tr><td>BertEncoder</td><td>61.94</td><td>6.82</td></tr><tr><td>EncoderBert</td><td>60.57</td><td>7.00</td></tr><tr><td>MEDHMP</td><td>67.77</td><td>9.34</td></tr></table>

Table 2: Results (%) on the readmission task.

We follow the existing work (Yang and Wu, 2021) and use its eight multimodal approaches as baselines, which adopt modality-specific encoders and perform modality aggregation via a gating mechanism. Different from the original model design, we perform a pooling operation on the latent representation of multiple clinical time series belonging to a specific admission, such that baselines can also take advantage of multiple stays. Details of these models can be found in Appendix D. We still use AUROC and AUPR as evaluation metrics.

Admission-level results are listed in Table 2, and we can observe that the proposed MEDHMP outperforms all baselines. Compared to the best baseline performance, the AUROC and AUPR scores of MEDHMP increase 7% and 29%, respectively. These results once again prove the effectiveness of the proposed pretraining model.

<table><tr><td>Database</td><td colspan="2">MIMIC-III</td><td colspan="4">TriNetX</td></tr><tr><td>Task</td><td colspan="2">Heart Failure</td><td>Heart Failure</td><td>COPD</td><td>Amnesia</td><td></td></tr><tr><td>Metric</td><td>AUPR F1</td><td>KAPPA AUPR</td><td>F1 KAPPA</td><td>AUPR F1 KAPPA</td><td>AUPR F1</td><td>KAPPA</td></tr><tr><td>LSTMa</td><td>57.83 59.40 35.86</td><td>50.16</td><td>46.08 29.26</td><td>50.16 49.34 34.64</td><td>48.68 49.64</td><td>34.46</td></tr><tr><td>LSTM</td><td>57.83 56.70 33.03</td><td>48.20 44.44</td><td>26.64 49.52</td><td>47.76 33.44</td><td>47.92 48.80</td><td>32.98</td></tr><tr><td>Dipolea</td><td>59.71 60.50 37.68</td><td>47.70 41.86</td><td>25.52</td><td>48.92 41.06 28.30</td><td>48.74 45.78</td><td>30.78</td></tr><tr><td>Dipole</td><td>59.43 58.63 36.03</td><td>47.16 40.16</td><td>24.28</td><td>49.44 39.48 27.86</td><td>48.36 45.63</td><td>30.40</td></tr><tr><td>RETAINa</td><td>68.71 66.20 47.12</td><td>58.16 52.18</td><td>35.64</td><td>57.62 50.66 38.36</td><td>62.70 56.50</td><td>43.90</td></tr><tr><td>RETAIN</td><td>67.76 65.56</td><td>45.63 57.50</td><td>50.88 34.52</td><td>57.40 49.85 37.36</td><td>62.52 56.32</td><td>43.66</td></tr><tr><td>AdaCarea</td><td>58.40 59.47</td><td>35.77 57.63</td><td>47.98 32.03</td><td>54.06 47.10 34.70</td><td>62.62 52.56</td><td>41.54</td></tr><tr><td>AdaCare</td><td>59.40 57.58</td><td>35.84 55.43</td><td>45.13 31.43</td><td>56.63 46.60 34.53</td><td>61.62 50.54</td><td>39.22</td></tr><tr><td>HiTANeta</td><td>69.42 68.44 50.01</td><td>60.12 50.48</td><td>36.08</td><td>64.04 54.46 43.38</td><td>67.54 58.18</td><td>47.78</td></tr><tr><td>HiTANet</td><td>70.36 66.60</td><td>46.60 54.76 47.92</td><td>32.04</td><td>60.10 52.40 39.93</td><td>63.08 54.60</td><td>43.44</td></tr></table>

Table 3: Performance (%) of baselines with/without pretraining for the health risk prediction task.

## 3.4 Patient-level Evaluation

Even though MEDHMP has not been pretrained on patient-level tasks, it is still capable of handling tasks at this level since its unimodal encoders acquire the ability to generate a high-quality representation of each admission, thus become feasible to be utilized to boost existing time series-targeting models. Health risk prediction, which utilizes a sequence of hospital admissions for illness forecasting, is applied as the task at the patient level.

In this experiment, the model will take a sequence of admission-level ICD codes as the input, which is still a unimodal evaluation. We use the following approaches as baselines: LSTM (Hochreiter and Schmidhuber, 1997), Dipole (Ma et al., 2017), RETAIN (Choi et al., 2016b), AdaCare (Ma et al., 2020), and HiTANet (Luo et al., 2020). Details of these approaches can be found in Appendix E. Following previous health risk prediction work (Chen et al., 2021; Cui et al., 2022a), we use AUPR, F1, and Cohen’s Kappa as the evaluation metrics.

## 3.4.1 Performance Comparison

The experimental results are shown in Table 3, where the approach with the subscript “a” denotes the baseline using the pretrained MEDHMP to initialize the ICD code embedding $\mathbf { c } _ { i }$ via Eq. (5). We can find that introducing the pretrained unimodal encoder from MEDHMP achieves stable improvement across most of the baselines and tasks. These results demonstrate the flexibility and effectiveness of our proposed MEDHMP in diverse medical scenarios. The knowledge from our pretrained model can be easily adapted to any sub-modality setting.

## 3.4.2 Influence of Training Size

Intuitively, pretraining could lead to improved initialization performance compared to models trained from scratch, thereby enhancing its suitability in low-resource settings such as zero-shot learning and few-shot learning. Inspired by these characteristics, we explore low-resource settings that simulate common real-world health-related scenarios. We replicate the experiments introduced in the previous section but vary the size of the training set from 1% to 100%.

![](images/b05c0b5f2842e68c67b59c8c9e6cdacae8b383ab61997b2f1d52dcdb2c9df38f.jpg)  
Figure 5: Performance change with different training data sizes using HiTANet on the TriNetX amnesia prediction task.

![](images/1212b97c0cb624e268e033621916d88bead11e91ceacd1caa8c910032de18bca.jpg)  
Figure 6: Performance changes w.r.t. the number of epochs on the TriNetX heart failure prediction task.

Figure 5 shows the experimental results using the HiTANet model. We can observe that using the pretraining initialization, $\mathrm { H i T A N e t } _ { a }$ always achieves better performance. Even with 10% training data, it can achieve comparable performance with the plain HiTANet using 100% data. This promising result confirms that the proposed pretraining framework MEDHMP is useful and meaningful for medical tasks, especially when the training data are insufficient.

<table><tr><td>Database</td><td colspan="2">MIMIC-III</td><td colspan="8">TriNetX</td></tr><tr><td>Task</td><td colspan="2">Heart Failure</td><td colspan="2">Heart Failure</td><td colspan="2">COPD</td><td colspan="2"></td><td colspan="2">Amnesia</td></tr><tr><td>Metric</td><td>AUPR F1</td><td>KAPPA</td><td>AUPR F1</td><td>KAPPA</td><td>AUPR</td><td>F1</td><td>KAPPA</td><td>AUPR</td><td>F1</td><td>KAPPA</td></tr><tr><td> $\overline { { \mathrm { I } \mathrm { S } \mathrm { T } \mathbf { M } _ { a + s } } }$ </td><td>57.83</td><td>59.40 35.86</td><td>50.16 46.08</td><td>29.26</td><td></td><td>50.16 49.34</td><td>34.64</td><td>48.68</td><td>49.64</td><td>34.46</td></tr><tr><td>LSTMa</td><td>57.57</td><td>58.27 35.67</td><td>49.88 44.86</td><td>28.58</td><td>49.90</td><td>47.65</td><td>33.77</td><td>48.48</td><td>48.70</td><td>33.52</td></tr><tr><td>LSTM</td><td>57.83</td><td>56.70 33.03</td><td>48.20 44.44</td><td>26.64</td><td>49.52</td><td>47.76</td><td>33.44</td><td>47.92</td><td>48.80</td><td>32.98</td></tr></table>

Table 4: Performance (%) of LSTM on the health risk prediction task, which is initialized with parameters from MEDHMP, admission-level pertaining only, and without pretraining.

## 3.4.3 Convergence Analysis with Pretraining

In this experiment, we aim to explore whether using pretraining can speed up the convergence of model training. We use the basic LSTM model as the baseline and output the testing performance at each epoch. Figure 6 shows the results. We can observe that at each epoch, the F1 score of $\mathrm { L S T M } _ { a }$ is higher than that of LSTM, indicating the benefit of using pretraining. Besides, LSTM<sub>a</sub> achieves the best performance at the 5-th epoch, but the F1 score of the plain LSTM still vibrates. Thus, these results clearly demonstrate that using pretraining techniques can make the model converge faster with less time and achieve better performance.

## 4 Ablation Study

## 4.1 Hierarchical Pretraining

For the comprehensive analysis of the effect of stayand admission-level pretraining, we perform ablation studies spanning downstream tasks at all three levels. Results of patient-level, admission-level, and stay-level tasks are listed in Table 4, 5 and 6, respectively. The subscripts “a” (admission) and “s” (stay) in these tables indicate which pretrained model is used as the initialization of MEDHMP.

From the results of all three tables, we can observe that the combination of both stay- and admission-level pretraining manifests superior performance, further underlining the necessity of adopting hierarchical pretraining strategies. Besides, compared with the model without any pretraining techniques, merely using a part of the proposed pretraining strategy for initialization can improve the performance. These observations imply the correct rationale behind our design of hierarchical pretraining strategies.

## 4.2 Multimodal Modeling

To investigate how intra- and inter-modality modeling techniques benefit our admission-level pretraining, we perform an ablation study on three tasks at the stay-level to examine the effectiveness of Mask

<table><tr><td>Model</td><td>AUROC</td><td>AUPR</td></tr><tr><td>MEDHMPa+s</td><td>67.77</td><td>9.34</td></tr><tr><td>MEDHMPa</td><td>65.75</td><td>9.08</td></tr><tr><td>MEDHMPs</td><td>64.87</td><td>8.60</td></tr><tr><td>MEDHMP</td><td>64.74</td><td>8.61</td></tr></table>

Table 5: Results (%) on the readmission task, where MEDHMP is initialized with bi-level pretraining, admission-level pretraining, stay-level pertaining, and without pretraining.

<table><tr><td>Task</td><td colspan="2">ARF</td><td colspan="2">Shock</td><td colspan="2">Mortality</td></tr><tr><td>Metric</td><td>AUROC</td><td>AUPR</td><td>AUROC</td><td>AUPR</td><td>AUROC</td><td>AUPR</td></tr><tr><td>MEDHMPa+s</td><td>71.66</td><td>14.34</td><td>71.04</td><td>24.19</td><td>82.17</td><td>47.52</td></tr><tr><td>MEDHMPs</td><td>64.65</td><td>10.59</td><td>67.94</td><td>22.50</td><td>79.67</td><td>42.66</td></tr><tr><td>MEDHMP</td><td>64.06</td><td>10.80</td><td>67.71</td><td>23.19</td><td>79.04</td><td>40.12</td></tr></table>

Table 6: Results (%) on the stay-level task, where MEDHMP is initialized with bi-level pretraining, staylevel pertaining, and without pretraining.

Code Prediction (MCP) and Contrastive Learning (CL) losses. We compare MEDHMP pretrained with all loss terms, with MCP and stay-level loss, with CL and stay-level loss, and stay-level loss only, respectively. Results presented in Table 7 clearly demonstrate the efficacy of each proposed loss term as well as the designed pretraining strategy. Besides, lacking each of them results in performance reduction, highlighting that combining intra- and inter-modality modeling is indispensable for boosting the model comprehensively.

## 5 Related Work

Predictive modeling using EHR data has attracted significant attention in recent years (Cui et al., 2022b; Ma et al., 2021; Xiao et al., 2018; Wang et al., 2022). To enhance predictive performance, pretraining techniques have been explored. In this section, we provide a concise overview of studies conducted on pretraining with both single-modal and multimodal EHR data.

## 5.1 Unimodal Pretraining with EHR Data

Several pretrained models have been proposed by utilizing single-modal EHR data. Building upon the success of Large Language Models (LLMs) (Devlin et al., 2018; Radford et al., 2018) in NLP, researchers have endeavored to train medical-specific language models using clinical notes (Li et al., 2022b; Lehman and Johnson, 2023; Alsentzer et al., 2019; Peng et al., 2019) and PubMed data (Luo et al., 2022; Lee et al., 2020; Yuan et al., 2022; Jin et al., 2019; Warikoo et al., 2021). However, these models primarily rely on mask language modeling techniques for pretraining, thereby overlooking the distinctive characteristics of medical data.

<table><tr><td>Task</td><td colspan="2">ARF</td><td colspan="2">Shock</td><td colspan="2">Mortality</td></tr><tr><td>Metric</td><td>AUROC</td><td>AUPR</td><td>AUROC</td><td>AUPR</td><td>AUROC</td><td>AUPR</td></tr><tr><td>MEDHMPa+s</td><td>71.66</td><td>14.34</td><td>71.04</td><td>24.19</td><td>82.17</td><td>47.52</td></tr><tr><td>MEDHMPMCP+s</td><td>64.91</td><td>12.35</td><td>68.61</td><td>25.29</td><td>81.32</td><td>47.50</td></tr><tr><td>MEDHMPCL+s</td><td>62.99</td><td>13.88</td><td>70.05</td><td>22.81</td><td>80.58</td><td>44.40</td></tr><tr><td>MEDHMPs</td><td>64.65</td><td>10.59</td><td>67.94</td><td>22.50</td><td>79.67</td><td>42.66</td></tr></table>

Table 7: Ablation results (%) regarding MCP and CL on the readmission task.

Given the time-ordered nature of admissions, medical codes can be treated as a sequence. Some pertaining models have proposed to establish representations of medical codes (Rasmy et al., 2021; Li et al., 2020; Shang et al., 2019; Choi et al., 2016a, 2018). Nevertheless, these studies still adhere to the commonly used pretraining techniques in the NLP domain. Another line of work (Tipirneni and Reddy, 2022; Wickstrøm et al., 2022) is to conduct self-supervised learning on clinical features. However, these pretrained models can only be used for the downstream tasks at the stay level, limiting their transferability in many clinical application scenarios.

## 5.2 Multimodal Pretraining with EHR data

Most of the multimodal pretraining models in the medical domain are mainly using medical images (Qiu et al., 2023) with other types of modalities, such as text (Hervella et al., 2021, 2022a,b; Khare et al., 2021) and tabular information (Hager et al., 2023). Only a few studies focus on pretraining on multimodal EHR data without leveraging medical images. The work (Li et al., 2022a, 2020) claims their success on multimodal pretraining utilizing numerical clinical features and diagnosis codes. In (Liu et al., 2022), the authors aim to model the interactions between clinical language and clinical codes. Besides, the authors in (Meng et al., 2021) use ICD codes, demographics, and topics learned from text data as the input and utilize the mask language modeling technique to pretrain the model. However, all existing pretrained work on EHR data still follows the routine of NLP pretraining but ignores the hierarchical nature of EHRs in their pretraining, resulting in the disadvantage that the pretrained models cannot tackle diverse downstream tasks at different levels.

## 6 Conclusion

In this paper, we present a novel pretraining model called MEDHMP designed to address the hierarchical nature of multimodal electronic health record (EHR) data. Our approach involves pretraining MEDHMP at two levels: the stay level and the admission level. At the stay level, MEDHMP uses a reconstruction loss applied to the clinical features as the objective. At the admission level, we propose two losses. The first loss aims to model intra-modality relations by predicting masked medical codes. The second loss focuses on capturing inter-modality relations through modality-level contrastive learning. Through extensive multimodal evaluation on diverse downstream tasks at different levels, we demonstrate the significant effectiveness of MEDHMP. Furthermore, experimental results on unimodal evaluation highlight its applicability in low-resource clinical settings and its ability to accelerate convergence.

## 7 Limitations

Despite the advantages outlined in the preceding sections, it is important to note that MEDHMP does have its limitations. Owing to the adoption of a large batch size to enhance contrastive learning (see Appendix B for more details), it becomes computationally unfeasible to fine-tune the language model acting as the encoder for clinical notes during our admission-level pretraining. As a result, ClinicalT5 is held static to generate a fixed representation of the clinical note, which may circumscribe potential advancements. Additionally, as described in Appendix A, we only select admissions with ICD-9 diagnosis codes while excluding those with ICD-10 to prevent conflicts arising from differing coding standards. This selection process, however, implies that MEDHMP currently lacks the capacity to be applied in clinical scenarios where ICD-10 is the standard for diagnosis code.

## Acknowledgement

This work is partially supported by the US National Science Foundation under Grants #2238275 and #2212323, and the US National Institutes of Health under Grant R01AG077016.

## References

Emily Alsentzer, John R Murphy, Willie Boag, Wei-Hung Weng, Di Jin, Tristan Naumann, and Matthew McDermott. 2019. Publicly available clinical bert embeddings. arXiv preprint arXiv:1904.03323.

Chacha Chen, Junjie Liang, Fenglong Ma, Lucas Glass, Jimeng Sun, and Cao Xiao. 2021. Unite: Uncertaintybased health risk prediction leveraging multi-sourced data. In Proceedings of the Web Conference 2021, pages 217–226.

Changyou Chen, Jianyi Zhang, Yi Xu, Liqun Chen, Jiali Duan, Yiran Chen, Son Tran, Belinda Zeng, and Trishul Chilimbi. 2022. Why do we need large batch sizes in contrastive learning? a gradient-bias perspective.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PMLR.

Edward Choi, Mohammad Taha Bahadori, Elizabeth Searles, Catherine Coffey, Michael Thompson, James Bost, Javier Tejedor-Sojo, and Jimeng Sun. 2016a. Multi-layer representation learning for medical concepts. In proceedings ofthe 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pages 1495–1504.

Edward Choi, Mohammad Taha Bahadori, Jimeng Sun, Joshua Kulas, Andy Schuetz, and Walter Stewart. 2016b. Retain: An interpretable predictive model for healthcare using reverse time attention mechanism. Advances in neural information processing systems, 29.

Edward Choi, Cao Xiao, Jimeng Sun, and Walter F Stewart. 2018. Mime: Multilevel medical embedding of electronic health records for predictive healthcare. Advances in Neural Information Processing Systems, 2018:4547–4557.

Suhan Cui, Junyu Luo, Muchao Ye, Jiaqi Wang, Ting Wang, and Fenglong Ma. 2022a. Medskim: Denoised health risk prediction via skimming medical claims data. In 2022 IEEE International Conference on Data Mining (ICDM), pages 81–90. IEEE.

Suhan Cui, Jiaqi Wang, Xinning Gui, Ting Wang, and Fenglong Ma. 2022b. Automed: Automated medical risk predictive modeling on electronic health records. In 2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pages 948–953. IEEE.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Yujuan Feng, Zhenxing Xu, Lin Gan, Ning Chen, Bin Yu, Ting Chen, and Fei Wang. 2019. Dcmn: Double

core memory network for patient outcome prediction with multimodal data. In 2019 IEEE International Conference on Data Mining (ICDM), pages 200–209. IEEE.

Paul Hager, Martin J Menten, and Daniel Rueckert. 2023. Best of both worlds: Multimodal contrastive learning with tabular and imaging data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23924–23935.

Álvaro S Hervella, José Rouco, Jorge Novo, and Marcos Ortega. 2021. Self-supervised multimodal reconstruction pre-training for retinal computeraided diagnosis. Expert Systems with Applications, 185:115598.

Alvaro S Hervella, José Rouco, Jorge Novo, and Marcos Ortega. 2022a. Multimodal image encoding pretraining for diabetic retinopathy grading. Computers in Biology and Medicine, 143:105302.

Álvaro S Hervella, José Rouco, Jorge Novo, and Marcos Ortega. 2022b. Retinal microaneurysms detection using adversarial pre-training with unlabeled multimodal images. Information Fusion, 79:146–161.

Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation, 9(8):1735– 1780.

Kexin Huang, Jaan Altosaar, and Rajesh Ranganath. 2019. Clinicalbert: Modeling clinical notes and predicting hospital readmission. arXiv preprint arXiv:1904.05342.

Qiao Jin, Bhuwan Dhingra, William Cohen, and Xinghua Lu. 2019. Probing biomedical embeddings from language models. In Proceedings of the 3rd Workshop on Evaluating Vector Space Representationsfor NLP, pages 82–89.

Alistair Johnson, Lucas Bulgarelli, Tom Pollard, Steven Horng, Leo Anthony Celi, and Roger Mark. 2020. Mimic-iv. PhysioNet. Available online at: https://physionet. org/content/mimiciv/1.0/(accessed August 23, 2021).

Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. 2016. Mimic-iii, a freely accessible critical care database. Scientific data, 3(1):1–9.

Yash Khare, Viraj Bagal, Minesh Mathew, Adithi Devi, U Deva Priyakumar, and CV Jawahar. 2021. Mmbert: Multimodal bert pretraining for improved medical vqa. In 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI), pages 1033–1036. IEEE.

Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. 2020. Biobert: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics, 36(4):1234–1240.

Eric Lehman and Alistair Johnson. 2023. Clinical-t5: Large language models built using mimic clinical text.

Yikuan Li, Mohammad Mamouei, Gholamreza Salimi-Khorshidi, Shishir Rao, Abdelaali Hassaine, Dexter Canoy, Thomas Lukasiewicz, and Kazem Rahimi. 2022a. Hi-behrt: Hierarchical transformer-based model for accurate prediction of clinical events using multimodal longitudinal electronic health records. IEEE Journal ofBiomedical and Health Informatics.

Yikuan Li, Shishir Rao, José Roberto Ayala Solares, Abdelaali Hassaine, Rema Ramakrishnan, Dexter Canoy, Yajie Zhu, Kazem Rahimi, and Gholamreza Salimi-Khorshidi. 2020. Behrt: transformer for electronic health records. Scientific reports, 10(1):1–12.

Yikuan Li, Ramsey M Wehbe, Faraz S Ahmad, Hanyin Wang, and Yuan Luo. 2022b. Clinical-longformer and clinical-bigbird: Transformers for long clinical sequences. arXiv preprint arXiv:2201.11838.

Sicen Liu, Xiaolong Wang, Yongshuai Hou, Ge Li, Hui Wang, Hui Xu, Yang Xiang, and Buzhou Tang. 2022. Multimodal data matters: Language model pre-training over structured and unstructured electronic health records. IEEE Journal of Biomedical and Health Informatics.

Junyu Luo, Muchao Ye, Cao Xiao, and Fenglong Ma. 2020. Hitanet: Hierarchical time-aware attention networks for risk prediction on electronic health records. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 647–656.

Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon, and Tie-Yan Liu. 2022. Biogpt: generative pre-trained transformer for biomedical text generation and mining. Briefings in Bioinformatics, 23(6).

Fenglong Ma, Radha Chitta, Jing Zhou, Quanzeng You, Tong Sun, and Jing Gao. 2017. Dipole: Diagnosis prediction in healthcare via attention-based bidirectional recurrent neural networks. In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, pages 1903– 1911.

Fenglong Ma, Muchao Ye, Junyu Luo, Cao Xiao, and Jimeng Sun. 2021. Advances in mining heterogeneous healthcare data. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pages 4050–4051.

Liantao Ma, Junyi Gao, Yasha Wang, Chaohe Zhang, Jiangtao Wang, Wenjie Ruan, Wen Tang, Xin Gao, and Xinyu Ma. 2020. Adacare: Explainable clinical health status representation learning via scaleadaptive feature extraction and recalibration. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 825–832.

Yiwen Meng, William Speier, Michael K Ong, and Corey W Arnold. 2021. Bidirectional representation learning from transformers using multimodal electronic health record data to predict depression. IEEEjournal ofbiomedical and health informatics, 25(8):3121–3129.

Yifan Peng, Shankai Yan, and Zhiyong Lu. 2019. Transfer learning in biomedical natural language processing: an evaluation of bert and elmo on ten benchmarking datasets. arXiv preprint arXiv:1906.05474.

Yixuan Qiu, Feng Lin, Weitong Chen, and Miao Xu. 2023. Pre-training in medical data: A survey. Machine Intelligence Research, 20(2):147–179.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training.

Laila Rasmy, Yang Xiang, Ziqian Xie, Cui Tao, and Degui Zhi. 2021. Med-bert: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction. NPJ digital medicine, 4(1):86.

Junyuan Shang, Tengfei Ma, Cao Xiao, and Jimeng Sun. 2019. Pre-training of graph augmented transformers for medication recommendation. arXiv preprint arXiv:1906.00346.

Nitish Srivastava, Elman Mansimov, and Ruslan Salakhudinov. 2015. Unsupervised learning of video representations using lstms. In International conference on machine learning, pages 843–852. PMLR.

Shengpu Tang, Parmida Davarmanesh, Yanmeng Song, Danai Koutra, Michael W Sjoding, and Jenna Wiens. 2020. Democratizing ehr analyses with fiddle: a flexible data-driven preprocessing pipeline for structured clinical data. Journal ofthe American Medical Informatics Association, 27(12):1921–1934.

Sindhu Tipirneni and Chandan K Reddy. 2022. Selfsupervised transformer for sparse and irregularly sampled multivariate clinical time-series. ACM Transactions on Knowledge Discoveryfrom Data (TKDD), 16(6):1–17.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Jiaqi Wang, Cheng Qian, Suhan Cui, Lucas Glass, and Fenglong Ma. 2022. Towards federated covid-19 vaccine side effect prediction. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 437–452. Springer.

Neha Warikoo, Yung-Chun Chang, and Wen-Lian Hsu. 2021. Lbert: Lexically aware transformer-based bidirectional encoder representation model for learning universal bio-entity relations. Bioinformatics, 37(3):404–412.

Kristoffer Wickstrøm, Michael Kampffmeyer, Karl Øyvind Mikalsen, and Robert Jenssen. 2022. Mixing up contrastive learning: Self-supervised representation learning for time series. Pattern Recognition Letters, 155:54–61.

Cao Xiao, Edward Choi, and Jimeng Sun. 2018. Opportunities and challenges in developing deep learning models using electronic health records data: a systematic review. Journal of the American Medical Informatics Association, 25(10):1419–1428.

Yanbo Xu, Siddharth Biswal, Shriprasad R Deshpande, Kevin O Maher, and Jimeng Sun. 2018. Raim: Recurrent attentive and intensive model of multimodal patient monitoring data. In Proceedings of the 24th ACM SIGKDD international conference on Knowledge Discovery & Data Mining, pages 2565–2573.

Bo Yang and Lijun Wu. 2021. How to leverage the multimodal EHR data for better medical prediction? In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 4029–4038, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Hongyi Yuan, Zheng Yuan, Ruyi Gan, Jiaxing Zhang, Yutao Xie, and Sheng Yu. 2022. Biobart: pretraining and evaluation of a biomedical generative language model. arXiv preprint arXiv:2204.03905.

## A Data Processing

We utilize two publicly available multimodal EHR datasets – MIMIC-III (Johnson et al., 2016) and MIMIC-IV (Johnson et al., 2020) – to pretrain the proposed MEDHMP. Considering that MIMIC-III uses ICD-9 codes while MIMIC-IV incorporates both ICD-9 and ICD-10 codes, we only select admissions with ICD-9 diagnosis codes to avoid potential conflicts between different coding standards. To prevent the label leakage issue during the testing stage, we remove all signals related to downstream tasks from the original data.

We adopt the EHR-oriented preprocessing pipeline, FIDDLE (Tang et al., 2020), for feature and label extraction at the stay level. We standardize the length of the clinical monitoring feature to T = 48 hours, which is the upper bound for clinical feature-related tasks mentioned in (Tang et al., 2020). We filter out the features with a frequency lower than 5% since extremely sparse features can significantly harm computing efficiency and be memory burdensome. After data preprocessing, each hourly clinical feature $\mathbf { m } _ { i , t } ^ { j }$ is represented as a 1,318-dimensional sparse vector, i.e., $d _ { f } = 1$ , 318. The demographics for each patient are represented as a 73-dimensional sparse vector, i.e., the length of is 73. The number of unique ICD codes is 7,686, and the number of unique drug codes is 1,701. Finally, we get 99,000 admissions with 100,563 stays for pretraining MEDHMP.

The three datasets extracted from TriNetX are supervised by clinicians. We employ the extraction method described in (Choi et al., 2016b) to identify case patients. Specifically, we identify the initial diagnosis date and utilize the patient’s historical data leading up to a six-month window, where the diagnosis date marks its end. This approach ensures that we prevent label leakage and successfully accomplish the objective of early prediction. Three control cases are chosen for each positive case based on matching criteria such as gender, age, race, and underlying diseases. For control patients, we use the last 50 visits in the database.

The statistics of data used for both pretraining and downstream tasks can be found in Table 8.

## B Implementation and Configuration

All models were implemented using PyTorch 2.0.0 and Python 3.9.12. Preprocessing and experiments were conducted in the Ubuntu 20.04 system with 376 GB of RAM and two NVIDIA A100 GPUs.

Each experiment was repeated five times to eliminate randomness, and the mean of the evaluation metrics was reported in all experimental results.

For unimodal evaluations, we used the same set of hyperparameters, regardless of whether a pretrained encoder was used, to ensure a fair comparison. For multimodal evaluations, we either used the hyperparameters reported by the authors of the baselines or suggested in their release codes. For detailed hyperparameters not provided by these authors, we used the same hyperparameters as in our model for a fair comparison.

$d _ { r }$ was set to 256 for our pretraining and evaluation in downstream tasks. For stay-level pretraining, our model was pretrained for 200 epochs with a learning rate of 5e-4, a batch size of 128, and a weight decay of 1e-8. At the admission level, our model was pretrained for 300 epochs, with a learning rate of 2e-5 and a weight decay of 1e-8. Following previous works (Chen et al., 2022, 2020), which emphasized the necessity of adopting a large batch size in contrastive learning, we set the batch size to 4096 to enhance our inter-modality modeling. τ in contrastive learning loss was set to 0.1. The hyperparameter λ mentioned in Eq. (11) was set to 0.1 to balance the masked code prediction (MCP) and contrastive learning (CL) losses in the stay-level pretraining. The masking rate in the MCP task was set to 15%, following the design of (Devlin et al., 2018). The optimizer used throughout the pretraining stage was AdamW.

<table><tr><td rowspan="2">Pretraining</td><td colspan="3">Number of of stays</td><td colspan="3">100,563</td></tr><tr><td colspan="3">Number of admissions</td><td colspan="3">99,000</td></tr><tr><td rowspan="9">Downstream</td><td>Level</td><td>Dataset</td><td>Predictive Task</td><td>Total</td><td>Positive</td><td>Negative</td></tr><tr><td rowspan="3">Stay</td><td rowspan="3">MIMIC-III</td><td>ARF within 48 hours</td><td>5,038</td><td>402</td><td>4,636</td></tr><tr><td>Shock within 48 hours</td><td>7,182</td><td>693</td><td>6,489</td></tr><tr><td>Mortality within 48 hours</td><td>11,695</td><td>1,581</td><td>10,114</td></tr><tr><td>Admission</td><td>MIMIC-III</td><td>Readmission within 30 days</td><td>33,179</td><td>1,444</td><td>31,735</td></tr><tr><td rowspan="4">Patient</td><td>MIMIC-III</td><td>Heart Failure after six months</td><td>7,522</td><td>2,820</td><td>4,702</td></tr><tr><td rowspan="3">TriNetX</td><td>COPD after six months</td><td>29,256</td><td>7,314</td><td>21,942</td></tr><tr><td>Amnesia after six months</td><td>11,928</td><td>2,982</td><td>8,946</td></tr><tr><td>Heart Failure after six months</td><td>12,320</td><td>3,080</td><td>9,240</td></tr></table>

Table 8: Data statistics.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Main Modality</td><td rowspan=1 colspan=1>Auxiliary Modalities</td><td rowspan=1 colspan=1>Clinical Feature Encoder</td></tr><tr><td rowspan=1 colspan=1>BertLstm</td><td rowspan=1 colspan=1>Clinical Notes</td><td rowspan=1 colspan=1>Clinical Features and Demographics</td><td rowspan=1 colspan=1>LSTM</td></tr><tr><td rowspan=1 colspan=1>LstmBert</td><td rowspan=1 colspan=1>Clinical Features</td><td rowspan=1 colspan=1>Clinical Notes and Demographics</td><td rowspan=1 colspan=1>LSTM</td></tr><tr><td rowspan=1 colspan=1>BertCnn</td><td rowspan=1 colspan=1>Clinical Notes</td><td rowspan=1 colspan=1>Clinical Features and Demographics</td><td rowspan=1 colspan=1>CNN</td></tr><tr><td rowspan=1 colspan=1>CnnBert</td><td rowspan=1 colspan=1>Clinical Features</td><td rowspan=1 colspan=1>Clinical Notes and Demographics</td><td rowspan=1 colspan=1>CNN</td></tr><tr><td rowspan=1 colspan=1>BertStar</td><td rowspan=1 colspan=1>Clinical Notes</td><td rowspan=1 colspan=1>Clinical Features and Demographics</td><td rowspan=1 colspan=1>StarTransformer</td></tr><tr><td rowspan=1 colspan=1>StarBert</td><td rowspan=1 colspan=1>Clinical Features</td><td rowspan=1 colspan=1>Clinical Notes and Demographics</td><td rowspan=1 colspan=1>StarTransformer</td></tr><tr><td rowspan=1 colspan=1>BertEncoder</td><td rowspan=1 colspan=1>Clinical Notes</td><td rowspan=1 colspan=1>Clinical Features and Demographics</td><td rowspan=1 colspan=1>Transformer</td></tr><tr><td rowspan=1 colspan=1>EncoderBert</td><td rowspan=1 colspan=1>Clinical Features</td><td rowspan=1 colspan=1>Clinical Notes and Demographics</td><td rowspan=1 colspan=1>Transformer</td></tr></table>

Table 9: Baselines for the admission-level task.

For downstream tasks, we selected the hyperparameters of our model using Grid Search. The batch size was chosen from the set [16, 32, 64], and the learning rate was searched in the range from 2e-5 to 5e-3. The maximum number of epochs was set to 30, and the patience for early stopping and weight decay were configured to 5 and 1e-2, respectively, to avoid overfitting. We found that the SGD optimizer performed better during the fine-tuning procedure.

## C Stay-level Experiments

Besides unimodal baselines mentioned in Section 3.2.2, the following approaches serving as baselines in the multimodal evaluation at the stay level are listed below: (1) F-LSTM (Tang et al., 2020) is a classic Long Short-Term Memory (LSTM) model taking concatenation of clinical features and demographic information as input. (2) F-CNN (Tang et al., 2020) is a typical Convolutional Neural Network (CNN) architecture using the concatenation of clinical features and demographic information for prediction. (3) Raim (Xu et al., 2018) is an attention-based model specially designed for analyzing ICU monitoring data, which uses a combination of attention mechanisms and multimodal data integration. (4) DCMN (Feng et al., 2019) combines two separate memory networks, one for processing clinical time series and one for processing static tables. Its dual-attention mechanism design allows the model to aggregate features effectively.

## D Stay-level Experiments with Clinical Notes

All the baselines utilized in the readmission prediction task are based on the previous work (Yang and Wu, 2021). In their study, the authors investigate various combinations of unimodal encoders and employ a gating mechanism for modality aggregation. In this approach, one modality is considered the main modality, and the embeddings from the other modalities are added as auxiliary modalities. Specific details regarding the composition of these baselines, including how the unimodal encoders are combined, can be found in Table 9.

The experimental results are presented in Table 10. It is evident that relying solely on a single modality, such as clinical notes, is inadequate for achieving accurate predictions when compared to multimodal baselines. Among all the multimodal models, our proposed MEDHMP consistently outperforms the others in the majority of scenarios. These results highlight two key findings: (1) the significance of integrating multimodal information in health predictive modeling tasks and (2) the efficacy of the proposed pretraining technique.

<table><tr><td rowspan="2">Modalities</td><td rowspan="2">Models</td><td colspan="2">ARF</td><td colspan="2">Shock</td><td colspan="2">Mortality</td></tr><tr><td>AUROC</td><td>AUPR</td><td>AUROC</td><td>AUPR</td><td>AUROC</td><td>AUPR</td></tr><tr><td>Clinical Notes</td><td>ClinicalT5 (Lehman and Johnson, 2023) ClinicalBERT (Huang et al., 2019)</td><td>50.06 51.75</td><td>5.92 7.60</td><td>56.76</td><td>13.01</td><td>72.18</td><td>26.55</td></tr><tr><td>Demographics</td><td>BertLstm (Yang and Wu, 2021)</td><td>64.90</td><td>9.03</td><td>44.28 70.22</td><td>9.89 23.32</td><td>58.17 80.80</td><td>16.42 45.59</td></tr><tr><td rowspan="8">+ Clinical Features + Clincial Notes</td><td>LstmBert (Yang and Wu, 2021)</td><td>61.74</td><td>9.79</td><td>66.31</td><td>22.56</td><td>78.31</td><td>42.31</td></tr><tr><td>BertCnn (Yang and Wu, 2021)</td><td>67.04</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CnnBert (Yang and Wu, 2021)</td><td></td><td>9.49</td><td>64.34</td><td>21.13</td><td>81.12</td><td>44.01</td></tr><tr><td>BertStar (Yang and Wu, 2021)</td><td>63.38 58.11</td><td>10.31</td><td>66.40</td><td>22.46</td><td>77.25</td><td>36.43</td></tr><tr><td></td><td></td><td>6.86</td><td>59.24</td><td>19.58</td><td>76.24</td><td>36.12</td></tr><tr><td>StarBert (Yang and Wu, 2021)</td><td>51.96</td><td>5.73</td><td>58.89</td><td>16.92</td><td>76.19</td><td>34.87</td></tr><tr><td>BertEncoder (Yang and Wu, 2021)</td><td>61.30</td><td>9.06</td><td>52.88</td><td>14.66</td><td>74.95</td><td>35.02</td></tr><tr><td>EncoderBert (Yang and Wu, 2021)</td><td>62.65</td><td>7.44</td><td>61.39</td><td>18.66</td><td>74.35</td><td>33.68</td></tr><tr><td>MEDHMP (ours)</td><td>71.67</td><td>11.05</td><td>70.57</td><td>24.30</td><td>82.06</td><td>42.18</td></tr></table>

Table 10: Comparison results (%) of stay-level tasks using clinical notes.

## E Patient-level Experiments

Baselines regarding the patient-level task are listed below. (1) LSTM(Hochreiter and Schmidhuber, 1997) is a typical backbone model appearing in time series forecasting tasks. (2) HiTANet(Luo et al., 2020) adopts the time-aware attention mechanism design that enables itself to capture the dynamic disease progression pattern. (3) Dipole(Ma et al., 2017) relies on the combination of bidirectional GRU and attention mechanism to analyze sequential visits of a patient. (4) AdaCare(Ma et al., 2020) applies the Convolutional Neural Network for feature extraction, followed by a GRU block for prediction. (5) Retain (Choi et al., 2016b) utilizes the reverse time attention mechanism to capture dependency between various visits of a patient.

## F Evaluation Metrics

Evaluation metrics used in our experiments are listed below:

• AUROC (Area Under the Receiver Operating Characteristic Curve) represents the likelihood that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative instance. It provides an aggregate measure of performance across all possible classification thresholds.

<table><tr><td>Task</td><td colspan="2">Mortality</td></tr><tr><td>Metric</td><td>AUROC</td><td>AUPR</td></tr><tr><td>F-LSTM F-CNN RAIM</td><td>82.30 76.37 83.64</td><td>45.01 35.11 46.40</td></tr><tr><td>DCMN</td><td>83.57</td><td>46.96</td></tr><tr><td>MEDHMP</td><td>84.43</td><td>49.00</td></tr></table>

Table 11: Results (%) on stay-level tasks.

• AUPRC (Area Under the Precision-Recall Curve) measures the area beneath the Precision-Recall curve, a plot of the precision against recall for different threshold values.

• F1 Score is the harmonic mean of precision and recall, offering a balance between the two when their values diverge.

• Cohen’s Kappa is a statistic that measures inter-rater agreement for categorical items, accounting for the possibility of the agreement occurring by chance.

## G Experiments on EICU Database

To further validate the transferability of our proposed MEDHMP, we conduct experiments using data from additional medical databases, i.e., eICU<sup>6</sup>. Results can be found in Table 11. Our proposed MEDHMP shows superior performance consistent with experiments on the MIMIC-III database, implying its excellent capability of learning general medical features.