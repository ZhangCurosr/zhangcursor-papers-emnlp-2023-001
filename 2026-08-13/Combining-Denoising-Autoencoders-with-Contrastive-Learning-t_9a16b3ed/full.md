# Combining Denoising Autoencoders with Contrastive Learning to fine-tune Transformer Models

Alejo López-Ávila Huawei London Research Centre London, UK alejo.lopez.avila@huawei.com

Víctor Suárez-Paniagua Huawei Ireland Research Center Dublin, Ireland victor.suarez.paniagua@huawei-partners.com

## Abstract

Recently, using large pre-trained Transformer models for transfer learning tasks has evolved to the point where they have become one of the flagship trends in the Natural Language Processing (NLP) community, giving rise to various outlooks such as prompt-based, adapters, or combinations with unsupervised approaches, among many others. In this work, we propose a 3-Phase technique to adjust a base model for a classification task. First, we adapt the model’s signal to the data distribution by performing further training with a Denoising Autoencoder (DAE). Second, we adjust the representation space of the output to the corresponding classes by clustering through a Contrastive Learning (CL) method. In addition, we introduce a new data augmentation approach for Supervised Contrastive Learning to correct the unbalanced datasets. Third, we apply fine-tuning to delimit the predefined categories. These different phases provide relevant and complementary knowledge to the model to learn the final task. We supply extensive experimental results on several datasets to demonstrate these claims. Moreover, we include an ablation study and compare the proposed method against other ways of combining these techniques.

## 1 Introduction

The never-ending starvation of pre-trained Transformer models has led to fine-tuning these models becoming the most common way to solve target tasks. A standard methodology uses a pre-trained model with self-supervised learning on large data as a base model. Then, replace/increase the deep layers to learn the task in a supervised way by leveraging knowledge from the initial base model. Even though specialised Transformers, such as Pegasus (Zhang et al., 2020) for summarisation, have appeared in recent years, the complexity and resources needed to train Transformer models of these scales make fine-tuning methods the most reasonable choice. This paradigm has raised interest in improving these techniques, either from the point of the architecture (Qin et al., 2019), by altering the definition of the fine-tuning task (Wang et al., 2021b) or the input itself (Brown et al., 2020), but also by revisiting and proposing different selfsupervised training practices (Gao et al., 2021). We decided to explore the latter and offer a novel approach that could be utilised broadly for NLP classification tasks. We combine some self-supervised methods with fine-tuning to prepare the base model for the data and the task. Thus, we will have a model adjusted to the data even before we start fine-tuning without needing to train a model from scratch, thus producing better results, as shown in the experiments.

A typical way to adapt a neural network to the input distribution is based on Autoencoders. These systems introduce a bottleneck for the input data distribution through a reduced layer, a sparse layer activation, or a restrictive loss, forcing the model to reduce a sample’s representation at the narrowing. A more robust variant of this architecture is DAE, which corrupts the input data to prevent it from learning the identity function. The first phase of the proposed method is a DAE (Fig. 1a), replacing the final layer of the encoder with one more adapted to the data distribution.

Contrastive Learning has attracted the attention of the NLP community in recent years. This family of approaches is based on comparing an anchor sample to negative and positive samples. There are several losses in CL, like the triplet loss (Schroff et al., 2015), the contrastive loss (Chopra et al., 2005), or the cosine similarity loss. The second phase proposed in this work consists of a Contrastive Learning using the cosine similarity (as shown in Fig. 1b). Since the data is labelled, we use a supervised approach similar to the one presented in (Khosla et al., 2020) but through Siamese Neural Networks. In contrast, we consider Contrastive

Learning and fine-tuning the classifier (FT) as two distinct stages. We also add a new imbalance correction during data augmentation that avoids overfitting. This CL stage has a clustering impact since the vector representation belonging to the same class will tend to get closer during the training. We chose some benchmark datasets in classification tasks to support our claims. For the hierarchical ones, we can adjust labels based on the number of similar levels among samples.

Finally, once we have adapted the model to the data distribution in the first phase and clustered the representations in the second, we apply fine-tuning at the very last. Among the different variants from fine-tuning, we use the traditional one for Natural language understanding (NLU), i.e., we add a small Feedforward Neural Network (FNN) as the classifier on top of the encoder with two layers. We use only the target task data without any auxiliary dataset, making our outlook self-contained. The source code is publicly available at GitHub<sup>1</sup>.

To summarise, our contribution is fourfold:

1. We propose a 3-P hase fine-tuning approach to adapt a pre-trained base model to a supervised classification task, yielding more favourable results than classical fine-tuning.

2. We propose an imbalance correction method by sampling noised examples during the augmentation, which supports the Contrastive Learning approach and produces better vector representations.

3. We analyze possible ways of applying the described phases, including ablation and joint loss studies.

4. We perform experiments on several wellknown datasets with different classification tasks to prove the effectiveness of our proposed methodology.

## 2 Related Work

One of the first implementations was presented in (Reimers and Gurevych, 2019), an application of Siamese Neural Networks using BERT (Devlin et al., 2018) to learn the similarity between sentences.

Autoencoders were introduced in (Kramer, 1991) and have been a standard for self-supervised learning in Neural Networks since then. However, new modifications were created with the explosion of Deep Learning architectures, such as DAE (Vincent et al., 2010) and masked Autoencoders (Germain et al., 2015). The Variational Autoencoder (VAE) has been applied for NLP in (Miao et al., 2015) or (Li et al., 2019) with RNN networks or a DAE with Transformers in (Wang et al., 2021a). Transformers-based Sequential Denoising Auto-Encoder (Wang et al., 2021a) is an unsupervised method for encoding the sentence into a vector representation with limited or no labelled data, creating noise through an MLM task. The authors of that work evaluated their approach on the following three tasks: Information Retrieval, Re-Ranking, and Paraphrase Identification, showing an increase of up to 6.4 points compared with previous state-of-the-art approaches. In (Savinov et al., 2021), the authors employed a new Transformer architecture called Step-unrolled Denoising Autoencoders. In the present work, we will apply a DAE approach to some Transformer models and extend its application to sentence-based and more general classification tasks.

The first work published on Contrastive Learning was (Chopra et al., 2005). After that, several versions have been created, like the triplet net in Facenet (Schroff et al., 2015) for Computer Vision. The triplet-loss compares a given sample and randomly selected negative and positive samples making the distances larger and shorter, respectively. One alternative approach for creating positive pairs is slightly altering the original sample. This method was followed by improved losses such as N-pairLoss (Sohn, 2016) and the Noise Contrastive Estimation (NCE) (Gutmann and Hyvärinen, 2010), extending the family of CL techniques. In recent years, further research has been done on applying these losses, e.g. by supervised methods such as (Khosla et al., 2020), which is the one that most closely resembles one in our second phase.

Sentence-BERT (Reimers and Gurevych, 2019) employs a Siamese Neural Network using BERT with a pooling layer to encode two sentences into a sentence embedding and measure their similarity score. Sentence-BERT was evaluated on Semantic Textual Similarity (STS) tasks and the SentEval toolkit (Conneau and Kiela, 2018), outperforming other embedding strategies in most tasks. In our particular case, we also use Siamese Networks within the CL options. Similar to this approach, the Simple Contrastive Sentence Embedding (Gao et al., 2021) is used to produce better embedding representations. This unlabelled data outlook uses two different representations from the same sample, simply adding the noise through the standard dropout. In addition, they tested it using entailment sentences as positive examples and contradiction sentences as negative examples and obtained better results than SBERT in the STS tasks.

![](images/846df451e4904b327cfb53d2d62630abb5036d16d82dd351becbc2c5fca532b5.jpg)  
Figure 1: (a) First stage: Denoising Autoencoder architecture where the encoder and the decoder are based on the pre-trained Transformers. The middle layer following the pooling together with the encoder model will be the resulting model of this stage. (b) Second stage: Supervised Contrastive Learning phase. We add a pooling layer to the previous one to learn the new clustered representation. The set of blocks denoted as DAECL will be employed and adapted as a base model in the fine-tuning phase. (c) Third stage: Classification phase through fine-tuning

Whereas until a few years ago, models were trained for a target task, the emergence of pretrained Transformers has changed the picture. Most implementations apply transfer learning on a Transformer model previously pre-trained on general NLU, on one or more languages, to the particular datasets and for a predefined task.

This new paradigm has guided the search for the best practice in each of the trending areas such as prompting strategies (Brown et al., 2020), Few Shot Learning (Wang et al., 2021b), meta-learning (Finn et al., 2017), also some concrete tasks like Intent Detection (Qin et al., 2019), or noisy data (Siddhant Garg, 2021). Here, we present a taskagnostic approach, outperforming some competitive methods for their corresponding tasks. It should also be noted that our method benefits from using mostly unlabelled data despite some little labelled data. In many state-of-the-art procedures, like (Sun et al., 2020a), other datasets than the target dataset are used during training. In our case, we use only the target dataset.

## 3 Model

In this section, we describe the different proposed phases: a Denoising Autoencoder that makes the inputs robust against noise, a Contrastive Learning approach to identify the similarities between different samples in the same class and the dissimilarities with the other class examples together with a novel imbalance correction, and finally, the traditional fine-tuning of the classification model. In the last part of the section, we describe another approach combining the first two phases into one loss.

## 3.1 DAE: Denoising Autoencoder phase

The Denoising Autoencoder is the first of the proposed 3-P hase approach, shown in Fig. 1a. Like any other Autoencoder, this model consists of an encoder and a decoder, connected through a bottleneck. We use two Transformer models as encoders and decoders simultaneously: RoBERTa (Liu et al., 2019), and all-M iniLM-L12-v2 (Wang et al., 2020). The underlying idea is that the bottleneck represents the input according to the general distribution of the whole dataset. A balance needs to be found between preventing the information from being memorised and having sufficient sensitivity to be reconstructed by the decoder, forcing the bottleneck to learn the general distribution. We add noise to the Autoencoder to prevent the bottleneck from memorising the data. We apply a Dropout on the input to represent the noise. Formally, for a sequence of tokens $X = \{ x _ { 0 } , \cdots , x _ { n } \}$ coming from a data distribution D, we define the loss as

<table><tr><td>Dataset</td><td>Train DAE</td><td>Train SCL</td><td>Test</td></tr><tr><td>SST2</td><td>69170</td><td>67349</td><td>872</td></tr><tr><td>SNIPS</td><td>262</td><td>262</td><td>65</td></tr><tr><td>SNIPS2</td><td>13084</td><td>13084</td><td>700</td></tr><tr><td>SST5</td><td>8544</td><td>8544</td><td>2210</td></tr><tr><td>AGNews</td><td>120K</td><td>120K</td><td>7600</td></tr><tr><td>IMDB</td><td>25K</td><td>25K</td><td>25K</td></tr></table>

Table 1: Statistics for Train, Validation and Test dataset splits.
<table><tr><td>Dataset</td><td>Avg. Length</td><td>Max. Length</td></tr><tr><td>SST2</td><td>9</td><td>52</td></tr><tr><td>SNIPS</td><td>9</td><td>20</td></tr><tr><td>SNIPS2</td><td>9</td><td>35</td></tr><tr><td>SST5</td><td>19</td><td>56</td></tr><tr><td>AGNews</td><td>37</td><td>177</td></tr><tr><td>IMDB</td><td>229</td><td>2450</td></tr></table>

Table 2: Average and max lengths for each of the datasets mentioned in the paper.

$$
\mathcal { L } _ { D A E } = \mathbb { E } _ { D } [ \log P _ { \theta } ( X | \bar { X } ) ]\tag{1}
$$

where $\bar { X }$ is the sequence X after adding the noise. To support with an example, masking a token at the position i would produce $\bar { X } \ =$ $\left\{ x _ { 0 } , \cdot \cdot \cdot , x _ { i - 1 } , 0 , x _ { i + 1 } , \cdot \cdot \cdot , x _ { n } \right\}$ . The distribution of $P _ { \theta }$ in this Cross-Entropy loss corresponds to the composition of the decoder with the encoder. We consider the ratio of noise like another hyperparameter to tune. More details can be found in Appendix A.1, and the results section 5. Instead of applying the noise to the dataset and running a few epochs over it, we apply the noise on the fly, getting a very low probability of a repeated input.

Once the model has been trained, we extract the encoder and the bottleneck, resulting in DAE (Fig. 1a), which will be the encoder for the next step. Each model’s hidden size has been chosen as its bottleneck size (768 in the case of RoBERTa). The key point of the first phase is to adapt the embedding representation of the encoder to the target dataset distribution, i.e., this step shifts the distribution from the general distribution learned by the pre-trained Transformers encoder into one of the target datasets. It should be noted here that this stage is in no way related to the categories for qualification. The first phase was implemented using the SBERT library.<sup>2</sup>

## 3.2 CL: Contrastive Learning phase

The second stage will employ Contrastive Learning, more precisely, a Siamese architecture with cosine similarity loss. The contrastive techniques are based on comparing pairs of examples or anchorpositive-negative triplets. Usually, these methods have been applied from a semi-supervised point of view. We decided on a supervised outlook where the labels to train in a supervised manner are the categories for classification, i.e., we pick a random example. Then, we can get a negative input by sampling from the other categories or a positive one by sampling from the same category. We combine this process of creating pairs with the imbalance correction explained below to get pairs of vector outputs $( u , v )$ Given two inputs v and u and a label $l a b e l _ { u , v }$ based on their class similarity.

$$
l a b e l _ { u , v } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } u \mathrm { ~ a n d ~ } v \mathrm { ~ a r e ~ i n ~ t h e ~ s a m e ~ c l a s s . } } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{2}
$$

We use a straightforward extension for the hierarchical dataset (AGNews) by normalising these weights along the number of levels. In the case of two levels, we assign 1 for the case where all the labels match, 0.5 when only the first one matches, and 0 if none. After applying the encoder, we obtain u¯ and v¯. We define the loss over these outputs as the Mean Squared Error (MSE):

$$
\mathcal { L } _ { C L } = | | l a b e l _ { u , v } - C o s i n e S i m ( u , v ) ) | | _ { 2 }\tag{3}
$$

We apply this Contrastive Learning to the encoder DAE, i.e., the encoder and the bottleneck from the previous step. Again, we add an extra layer to this encoder model, producing our following final embedding. We denote this encoder after applying CL over DAE as DAECL (Fig. 1b). Similarly, we chose the hidden size per model as the embedding size. CL plays the role of soft clustering for the embedding representation of the text based on the different classes. This will make fine-tuning much easier as it will be effortless to distinguish representations from distinct classes. The second phase was also implemented through the SBERT library.

## 3.2.1 Imbalance correction

As mentioned in the previous section, we are using a Siamese network and creating the pairs in a supervised way. It is a common practice to augment the data before applying CL. In theory, we can make as many example combinations as possible. In the case of the Siamese models, we have a total of n! unique pair combinations that correspond to the potential number of pairs based on symmetry. Typically, data can be increased as much as one considers suitable. The problem is that one may fall short or overdo it and produce overfitting in the smaller categories. Our augmentation is based on two pillars. We start with correcting the imbalance in the dataset by selecting underrepresented classes in the dataset more times but without repeating pairs. Secondly, on the classes that have been increased, we apply noise on them to provide variants that are close but not the same to define more clearly the cluster to which they belong.

We balance the dataset by defining the number of examples that we are going to use per class based on the most significant class (where $m a x _ { k }$ is its size) and a range of ratios, from $\operatorname* { m i n } _ { r a t i o }$ the minimum and $m a x _ { r a t i o }$ the maximum. These upper and lower bounds for the ratios are hyper-parameters, and we chose 4 and 1.5 as default values. Formally, the new ratio is defined by the function:

$$
f ( x ) = \log \left( { \frac { m a x _ { k } } { x } } \right) \times { \frac { m a x _ { r a t i o } } { \log m a x _ { k } } }\tag{4}
$$

where x refers to the initial size of a given class. After adding a lower bound for the ratio, we get the final amount

$$
n e w r a t i o _ { k } = \operatorname* { m i n } \bigg ( \operatorname* { m i n } _ { r a t i o } , f ( c l a s s _ { k } ) \bigg )\tag{5a}
$$

$$
n e w c l a s s _ { k } = n e w r a t i o _ { k } \times c l a s s _ { k }\tag{5b}
$$

for $k = 1 , 2 , \cdots , K$ in a classification problem with K classes, where class<sub>k</sub> and newclass<sub>k</sub> are the cardinalities of the class k, before and after the resizing, respectively. As we can observe, the function 4 gives $f ( 1 ) = m a x _ { r a t i o }$ for one example, so we will never get something bigger than the maximum. The shape of this function is similar to a negative log-likelihood, given a high ratio to the small classes and around the minimum for medium or bigger. A simple computation shows that the ratio 1 in function 4 is obtained at

$$
x = m a x _ { k } ^ { ( m a x _ { r a t i o } - 1 ) / m a x _ { r a t i o } }\tag{6}
$$

or $m a x _ { k } ^ { 0 . 7 5 }$ in our default case.

We duplicate this balanced dataset and shuffle both to create the pairs. Since many combinations exist, we only take the unique ones without repeating pairs, broadly preserving the proportions. Even so, if just a few examples represent one class, the clustering process from CL can be affected by the augmentation because the border between them would be defined just for a few examples. To avoid this problem, we add a slight noise to the tokens when the text length is long enough. This noise consists of deleting some of the stop-words from the example. Usually, this noise was added to create positive samples and produced some distortion to the vector representation. Adding it twice, in the augmentation and the $C L ,$ would produce too much distortion. However, since we are using a supervised approach, this does not negatively affect the model, as shown in the ablation section 5.

## 3.3 FT: Fine-tuning phase

Our final stage is fine-tuning, obtained by employing DAECL as our base model (as indicated in Fig 1b). We add a two-layer $M L P$ on top as a classifier. We tried both to freeze and not to freeze the previous neurons from DAECL.

As the final activation, we use Softmax, which is a sigmoid function for the binary cases. More formally, for K classes, softmax corresponds to

$$
\sigma ( z _ { i } ) = \frac { e ^ { z _ { i } } } { \sum _ { j = 1 } ^ { K } e ^ { z _ { j } } } f o r i = 1 , 2 , \ldots , K\tag{7}
$$

As a loss function for the classification, we minimize the use of Cross-Entropy:

$$
\mathcal { L } _ { F T } = - \sum _ { k = 1 } ^ { K } y _ { k } \log ( p _ { k } )\tag{8}
$$

where $p _ { k }$ is the predicted probability for the class k and $y _ { k }$ for the target. For binary classification datasets this can be further expanded as

$$
\mathcal { L } _ { F T } = - ( y \log ( p ) + ( 1 - y ) \log ( 1 - p ) )\tag{9}
$$

## 3.3.1 Joint

We wanted to check if we could benefit more when combining losses, i.e. by creating a joint loss based on the first and the second loss, (1 and 3), respectively.

$$
\mathcal { L } _ { J o i n t } = \mathcal { L } _ { D A E } + \mathcal { L } _ { C L }\tag{10}
$$

This training was obtained as a joint training of stages one and two. By adding the classification head, like in the previous section, for fine-tuning, we got the version we denote as Joint (see Table 3).

## 4 Experimental Setup

The datasets for the experiments, the two base models used and the metrics employed are detailed below.

## 4.1 Datasets

We have chosen several well-known datasets to carry out the experiments:

• Intent recognition on SNIPS (SNIPS Natural Language Understanding benchmark) (Coucke et al., 2018). For this dataset, we found different versions, the first one with just 327 examples and 10 categories. This one was obtained from the huggingface library<sup>3</sup>, which shows how this procedure performs on small datasets.

• The second version for SNIPS from $K a g g l e ^ { 4 }$ containing more samples and split into 7 classes that we call SNIPS2 is the most common one.

• The third one is commonly used for slot prediction (Qin et al., 2019), although here we only consider task intent recognition. We used SST2 and SST5 from (Socher et al., 2013) for classification containing many short text examples.

• We add AGNews<sup>5</sup> (Zhang et al., 2015) to our list, a medium size dataset that shows our method over long text.

• We complement the experiments with IMDB<sup>6</sup> (Maas et al., 2011), a big dataset with long inputs for binary classification in sentiment analysis. The length and size of this data made us consider only the RoBERT a as the base model.

We used Hugging Face API to download all the datasets apart from the second version of SNIPS. In some cases, there was no validation dataset, so we used 20% of the training set to create the validation dataset. There was an unlabelled test set in SNIPS, so we extracted another 10% for testing. The unlabelled data was used for the first training phase, not for the other ones. The first and second phases did not use the test set. We selected the best model for each of them based on the results of the validation dataset. The test set was used at the end of phase 3.

## 4.2 Models and Training

We carried out experiments with two different models, a small model for short text all-MiniLM-L12-v2 (Wang et al., 2020)<sup>7</sup>, more concretely, a version fine-tuned for sentence similarity and a medium size model RoBERTa-base (Liu et al., 2019)<sup>8</sup> which we abbreviate as RoBERTa. The ranges for the hyper-parameters below, as well as the values of the best accuracy, can be found in Appendix A.1.

We use Adam as the optimiser. We test different combinations of hyper-parameters, subject to the model size. We tried batch sizes from 2 to 128, whenever possible, based on the dataset and the base model. We tested the encoder with both frozen and notfrozen weights - almost always getting better results when no freezing is in place. We tested two different truncations’ lengths based either on the maximum length in the training dataset plus a 20% or the default maximum length for the inputs in the base model. We tried to give more importance to one phase over the other by applying data augmentation in CL or extending the number of epochs for the Autoencoder (Appendix A.1). We increased the relevance of the first phase by augmenting the data before applying random noise instead of changing the number of epochs. We tune the value of ratio, getting 0.6 as the most common best value. To increase the relevance of the second phase, we increased the number of inputs by creating more pair combinations. Since we increased or decreased the relevance of one or other phases based on the amount of data, we used the same learning rate for the first two phases.

<table><tr><td>Dataset</td><td>3-Phase</td><td>Joint</td><td>FT</td><td>S-P</td><td>EFL</td><td>CAE</td><td>Self-E</td><td>STC-DeBERTa</td><td>FTBERT</td></tr><tr><td colspan="10">RoBERTa</td></tr><tr><td>SNIPS</td><td>99.81</td><td>94.92</td><td>91.01</td><td>99.0</td><td></td><td>98.3</td><td></td><td></td><td></td></tr><tr><td>SNIPS2</td><td>98.29</td><td>98.0</td><td>97.57</td><td>99.0</td><td>1</td><td>98.3</td><td>一</td><td></td><td></td></tr><tr><td>SST2</td><td>95.07</td><td>93.12</td><td>90.28</td><td></td><td>96.9</td><td></td><td></td><td>94.78</td><td></td></tr><tr><td>SST5</td><td>56.79</td><td>53.88</td><td>52.27</td><td></td><td></td><td></td><td>56.2</td><td></td><td></td></tr><tr><td>AGNews</td><td>95.08</td><td>94.82</td><td>92.47</td><td></td><td>86.1</td><td></td><td>一</td><td></td><td>95.20</td></tr><tr><td>IMDB</td><td>99.0</td><td>95.07</td><td>91.0</td><td></td><td>96.1</td><td></td><td>一</td><td></td><td></td></tr><tr><td colspan="10">all-MiniLM-L12-v2</td></tr><tr><td>SNIPS</td><td>100.00</td><td>91.98</td><td>92.89</td><td>99.0</td><td></td><td>98.3</td><td></td><td></td><td></td></tr><tr><td>SNIPS2</td><td>98.57</td><td>98.57</td><td>93.86</td><td>99.0</td><td></td><td>98.3</td><td>1</td><td></td><td></td></tr><tr><td>SST2</td><td>93.89</td><td>90.04</td><td>88.21</td><td></td><td>96.9</td><td></td><td></td><td>94.78</td><td></td></tr><tr><td>SST5</td><td>54.77</td><td>52.25</td><td>49.24</td><td></td><td></td><td></td><td>56.2</td><td>一</td><td></td></tr><tr><td>AGNews</td><td>94.83</td><td>94.28</td><td>89.57</td><td></td><td>86.1</td><td>一</td><td></td><td>1</td><td>95.20</td></tr></table>

Table 3: Performance accuracy on different datasets using RoBERTa and all-MiniLM-L12-v2 models in %. 3-Phase refers to our main 3 stages approach, while Joint denotes one whose loss is based on the combination of the first two losses, and FT corresponds to the fine-tuning. We also add some SOTA results from other papers: S-P denotes Stack-Propagation (Qin et al., 2019), Self-E is used to denote (Sun et al., 2020b) (in this case we chose the value from RoBERTa-base for a fair comparison), CAE is used to detonate (Phuong et al., 2022), STC-DeBERTa refers to (Karl and Scherp, 2022), EFL points to (Wang et al., 2021b) (this one uses RoBERTa-Large), and FTBERT is (Sun et al., 2020a)

## 4.3 Metrics

We conducted experiments using the datasets previously presented in Section 4.1. We used the standard macro metrics for classification tasks, i.e., Accuracy, F1-score, Precision, Recall, and the confusion matrix. We present only the results from Accuracy in the main table to compare against other works. The results for other metrics can be found in Appendix A.2.

## 5 Results

We assess the performance of the three methods against the several prominent publicly available datasets. Thus, here we evaluate the 3-P hase procedure compared to Joint and FT approaches. We report the performance of these approaches in terms of accuracy in Table 3

We observe a noticeable improvement of at least 1% with 3-P hase as compared to the second best performing approach, Joint, in almost all the datasets. In SNIPS2 with all-MiniLM-L12-v2 we get the same values, while in IMDB with RoBERTa we get a gap of 4 point and 8 for the SNIPS dataset with all-MiniLM-L12-v2. We apply these three implementations to both base models, except IMDB, as the input length is too long for all-MiniLM-L12-v2. FT method on its own performs the worst concerning the other two counterparts for both the models tested. Eventually, we may conclude that the 3-P hase approach is generally the best one, followed by Joint, and as expected, FT provides the worst. We can also observe that Joint and FT tend to be close in datasets with short input, while AGNews gets closer results for 3-Phase and Joint.

We did not have a bigger model for those datasets with long input, so we tried to compare against approaches with similar base models. We truncated the output for the datasets with the longest inputs, which may reflect smaller values in our case. Since the advent of (Sun et al., 2020a), several current techniques are based on combining different datasets in the first phase through multitask learning and then fine-tuning each task in detail. Apart from (Sun et al., 2020a), this is the case for the prompt-base procedure from EF L (Wang et al., 2021b) as well. Our method focuses on obtaining the best results for a single task and dataset. Several datasets to pre-train the model could be used as a phase before all the others. However, we doubt the possible advantages of this as the first and second phases would negatively affect this learning, and those techniques focused on training the classifier with several models would negatively affect the first two phases.

<table><tr><td>Dataset</td><td>Base Model</td><td>3-Phase</td><td>Joint</td><td> $D A E { + } F T$ </td><td> $C L { + } F T$ </td><td>Extra Imb.</td><td>No Imb.</td><td>FT</td></tr><tr><td>SNIPS</td><td> $a l l - M i n i L M { - } L 1 2 { - } v 2$ </td><td>100.00</td><td>91.98</td><td>98.05</td><td>99.81</td><td>99.92</td><td>99.88</td><td>92.89</td></tr><tr><td>SNIPS2</td><td> $a l l - M i n i L M { - } L 1 2 { - } v 2$ </td><td>98.57</td><td>98.57</td><td>94.14</td><td>97.86</td><td>98.00</td><td>97.85</td><td>93.86</td></tr><tr><td>SST2</td><td>RoBERTa</td><td>95.07</td><td>93.12</td><td>83.72</td><td>94.72</td><td>94.50</td><td>94.04</td><td>90.28</td></tr><tr><td>SST5</td><td>RoBERTa</td><td>56.79</td><td>53.88</td><td>46.06</td><td>56.52</td><td>56.24</td><td>55.84</td><td>52.27</td></tr><tr><td>AGNews</td><td>RoBERTa</td><td>95.08</td><td>94.82</td><td>91.53</td><td>95.24</td><td>95.01</td><td>94.26</td><td>92.47</td></tr><tr><td>IMDB</td><td>RoBERTa</td><td>99.00</td><td>95.07</td><td>93.00</td><td>94.86</td><td>97.00</td><td>94.76</td><td>91.10</td></tr></table>

Table 4: Ablation results. As before, 3-P hase, Joint, and FT correspond to the 3 stages approach, joint losses, and Fine-tuning, respectively. Here, DAE+FT denotes the denoising autoencoder together with fine-tuning, CL+FT denotes the contrastive Siamese training together with fine-tuning, No Imb. means 3-Phase but skipping the imbalance correction, and Extra Imb. refers to an increase of the imbalance correction to a $m i n _ { r a t i o } { = } 1 . 5$ and $m a x _ { r a t i o } { = } 2 0$

To complete the picture, CAE (Phuong et al., 2022) is an architecture method, which is also independent of the pre-training practice. A selfexplaining framework is proposed in (Sun et al., 2020b) as an architecture method that could be implemented on top of our approach.

## 5.1 Ablation study

We conducted ablation experiments on all the datasets, choosing the same hyper-parameters and base model as the best result for each one. The results can be seen in Table 4.

We start the table with the approaches mentioned: 3-Phaseand Joint. We wanted to see if combining the first two phases could produce better results as those would be learned simultaneously. The results show that merging the two losses always leads to worse results, except for one of the datasets where they give the same value.

We start the ablations by considering only DAE right before fine-tuning, denoting it as DAE+FT. In this case, we assume that the fine-tuning will carry all the class information. One advantage of this outlook is that it still applies to models that employ a small labelled fraction of all the data (i.e., the unlabelled data represents the majority). The next column, CL+FT, replaces DAE with Contrastive Learning, concentrating the attention on the classes and not the data distribution. Considering only the classes and fine-tuning in CL+FT, we get better results than in DAE+FT, but still lower than the 3-Phasein almost all the datasets. Right after, we add two extreme cases of the imbalance correction, where Extra Imb. increases the upper bound for the ratio and No Imb. excludes the imbalance method. Both cases generally produce lower accuracies than 3-Phase, being No Imb. slightly lower. The last column corresponds to fine-tuning FT.

All these experiments proved that the proposed 3-P hase approach outperformed all the steps independently, on its own, or combined the Denoising Autoencoder and the Contrastive Learning as one loss.

## 6 Conclusion

The work presented here shows the advantages of fine-tuning a model in different phases with an imbalance correction, where each stage considers certain aspects, either as an adaptation to the text characteristics, the class differentiation, the imbalance, or the classification itself. We have shown that the proposed method can be equally or more effective than other methods explicitly created for a particular task, even if we do not use auxiliary datasets. Moreover, in all cases, it outperforms classical fine-tuning, thus proving that classical fine-tuning only partially exploits the potential of the datasets. Squeezing out all the juice from the data requires adapting to the data distribution and grouping the vector representations according to the task before the fine-tuning, which in our case is targeted towards classification.

## 7 Future work

The contrastive training phase benefits of data augmentation, i.e., we can increase the number of examples simply through combinatorics. However, this can lead to space deformation for small datasets, even with the imbalance correction, as fewer points are considered. Therefore, overfitting occurs despite the combinatoric strategy. Another advantage of this phase is balancing the number of pairs with specific values. This practice allows us, for example, to increase the occurrence of the underrepresented class to make its cluster as well defined as those of the more represented categories (i.e. ones with more examples). This is a partial solution for the imbalance problem.

In the future, we want to address these problems. For the unbalanced class in the datasets, seek a solution to avoid overfitting to the under-represented classes and extend our work to support a few shot learning settings (F SL). To do so, we are going to analyze different data augmentation techniques. Among others, Variational Autoencoders. Recent approaches for text generation showed that hierarchical VAE models, such as stable diffusion models, may produce very accurate augmentation models. One way to investigate this is to convert the first phase into a VAE model, allowing us to generate more examples from underrepresented classes and generally employ them all in the F SL setting.

Finally, we would like to combine our approach with other fine-tuning procedures, like prompt methods. Adding a prompt may help the model gain previous knowledge about the prompt structure instead of learning the prompt pattern simultaneously during the classification while fine-tuning.

## Limitations

This approach is hard to combine with other finetuning procedures, mainly those which combine different datasets and use the correlation between those datasets, since this one tries to extract and get as close as possible to the target dataset and task. The imbalance correction could be improved, restricting to cases where the text is short because it could be too noisy or choosing the tokens more elaborately and not just stop words. It would be necessary to do more experiments combined with other approaches, like the prompt base, to know if they benefit from each other or if they could have negative repercussions in the long term.

## Acknowledgments

The authors would like to thank the members of the AIApps Research Group in the Huawei Ireland and London Research Centers for their valuable discussion and comments. We especially want to thank Milan Redzic, Tri Kurniawan Wijaya and Jinhua Du for their help.

## References

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, and Dario Amodei. 2020. Language models are few-shot learners.

S. Chopra, R. Hadsell, and Y. LeCun. 2005. Learning a similarity metric discriminatively, with application to face verification. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), volume 1, pages 539–546 vol. 1.

Alexis Conneau and Douwe Kiela. 2018. SentEval: An evaluation toolkit for universal sentence representations. In Proceedings ofthe Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA).

Alice Coucke, Alaa Saade, Adrien Ball, Théodore Bluche, Alexandre Caulier, David Leroy, Clément Doumouro, Thibault Gisselbrecht, Francesco Caltagirone, Thibaut Lavril, Maël Primet, and Joseph Dureau. 2018. Snips voice platform: an embedded spoken language understanding system for privateby-design voice interfaces.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. 2017. Model-agnostic meta-learning for fast adaptation of deep networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1126–1135. PMLR.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. 2015. Made: Masked autoencoder for distribution estimation. In Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37, ICML’15, page 881–889. JMLR.org.

Michael Gutmann and Aapo Hyvärinen. 2010. Noisecontrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings ofthe Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, pages 297–304, Chia Laguna Resort, Sardinia, Italy. PMLR.

Fabian Karl and Ansgar Scherp. 2022. Transformers are short text classifiers: A study of inductive short text classifiers on benchmarks and real-world datasets.

Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. 2020. Supervised Contrastive Learning. arXiv e-prints, page arXiv:2004.11362.

Mark A. Kramer. 1991. Nonlinear principal component analysis using autoassociative neural networks. Aiche Journal, 37:233–243.

Ruizhe Li, Xiao Li, Chenghua Lin, Matthew Collinson, and Rui Mao. 2019. A stable variational autoencoder for text modelling. In Proceedings ofthe 12th International Conference on Natural Language Generation, pages 594–599, Tokyo, Japan. Association for Computational Linguistics.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. ArXiv, abs/1907.11692.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Associationfor Computational Linguistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA. Association for Computational Linguistics.

Yishu Miao, Lei Yu, and Phil Blunsom. 2015. Neural variational inference for text processing.

Nguyen Minh Phuong, Tung Le, and Nguyen Le Minh. 2022. Cae: Mechanism to diminish the class imbalanced in slu slot filling task. In Advances in Computational Collective Intelligence, pages 150–163, Cham. Springer International Publishing.

Libo Qin, Wanxiang Che, Yangming Li, Haoyang Wen, and Ting Liu. 2019. A stack-propagation framework with token-level intent detection for spoken language understanding. pages 2078–2087.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks.

Nikolay Savinov, Junyoung Chung, Mikolaj Binkowski, Erich Elsen, and Aaron van den Oord. 2021. Stepunrolled denoising autoencoders for text generation.

Florian Schroff, Dmitry Kalenichenko, and James Philbin. 2015. Facenet: A unified embedding for face recognition and clustering. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 815–823.

Varun Thumbe Siddhant Garg, Goutham Ramakrishnan. 2021. Towards robustness to label noise in text classification via noise modeling. arXiv preprint arXiv:2101.11214v3.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington, USA. Association for Computational Linguistics.

Kihyuk Sohn. 2016. Improved deep metric learning with multi-class n-pair loss objective. In Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc.

Chi Sun, Xipeng Qiu, Yige Xu, and Xuanjing Huang. 2020a. How to fine-tune bert for text classification?

Zijun Sun, Chun Fan, Qinghong Han, Xiaofei Sun, Yuxian Meng, Fei Wu, and Jiwei Li. 2020b. Selfexplaining structures improve nlp models.

Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. 2010. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. J. Mach. Learn. Res., 11:3371–3408.

Kexin Wang, Nils Reimers, and Iryna Gurevych. 2021a. Tsdae: Using transformer-based sequential denoising auto-encoder for unsupervised sentence embedding learning.

Sinong Wang, Han Fang, Madian Khabsa, Hanzi Mao, and Hao Ma. 2021b. Entailment as few-shot learner.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020. Minilm: Deep selfattention distillation for task-agnostic compression of pre-trained transformers. In NeurIPS 2020. ACM.

Jingqing Zhang, Yao Zhao, Mohammad Saleh, and Peter J. Liu. 2020. Pegasus: Pre-training with extracted gap-sentences for abstractive summarization.

Xiang Zhang, Junbo Jake Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In NIPS.

## A Appendix

## A.1 Hyper-parameters

This appendix section shows the final hyper-parameters from the best results in Table 3. The column at the end on the right contains the search space used for training. Some of the values were not used for training, either because of computational limitations or because they were not realistic for some datasets.
<table><tr><td colspan="8">Dataset Best Hyper-parameters</td></tr><tr><td>Parameter</td><td>SNIPS</td><td>SNIPS2</td><td>SST2</td><td>SST5</td><td>AGNews</td><td>IMDB</td><td>SearchSpace</td></tr><tr><td colspan="8">RoBERTa</td></tr><tr><td>Learning rate DAE</td><td>5e-5</td><td>5e-5</td><td>1e-5</td><td>1e-5</td><td>2e-4</td><td>1e-5</td><td>{1e-4, 2e-4, 5e-5, 1e-5, 1e-6}</td></tr><tr><td>Learning rate CL</td><td>5e-5</td><td>5e-5</td><td>1e-5</td><td> $_ { 1 \mathrm { e } - 5 }$ </td><td>2e-5</td><td>1e-5</td><td>{1e-4, 2e-4, 5e-5, 1e-5, 1e-6}</td></tr><tr><td>Learning rate FT</td><td>1e-5</td><td>1e-5</td><td>1e-5</td><td>1e-5</td><td>2e-5</td><td>1e-5</td><td>{1e-4, 1e-5, 2e-5, 1e-6}</td></tr><tr><td>Epochs  $D A E ^ { * 1 }$ </td><td>12</td><td>4</td><td>12</td><td>12</td><td>4</td><td>2</td><td>{2, 3, 4, 9, 12}</td></tr><tr><td>Epochs CL</td><td>12</td><td>12</td><td>12</td><td>12</td><td>4</td><td>4</td><td>{2, 3, 4, 9, 12}</td></tr><tr><td>Max Epochs  $F T ^ { * 2 }$ </td><td>70</td><td>70</td><td>70</td><td>70</td><td>70</td><td>50</td><td>{20, 50, 70}</td></tr><tr><td>Batch size DAE</td><td>32</td><td>32</td><td>64</td><td>64</td><td>16</td><td>6</td><td>{2,6,8,12,16,32,64,128}</td></tr><tr><td>Batch size CL</td><td>32</td><td>32</td><td>64</td><td>64</td><td>16</td><td>6</td><td>{2,6,8,12,16,32,64,128}</td></tr><tr><td>Batch size FT</td><td>32</td><td>32</td><td>64</td><td>64</td><td>16</td><td>6</td><td>{2,6,8,12,16,32,64,128}</td></tr><tr><td>Eps in  $F T ^ { * 3 }$ </td><td> $2 \mathrm { e } { \cdot } 5$ </td><td> $2 \mathrm { e } { \cdot } 5$ </td><td>2e-5</td><td> $2 \mathrm { e } { \cdot } 5$ </td><td>2e-5</td><td> $2 \mathrm { e } { \cdot } 5$ </td><td>{2e-05}</td></tr><tr><td>Use Length*4</td><td>48*4</td><td> $1 0 8 ^ { * 4 }$ </td><td>512</td><td>512</td><td>512</td><td> $5 0 0 ^ { * 4 }$ </td><td> $\{ ^ { * ^ { 5 } } , { } ^ { * ^ { 4 } } , 5 1 2 \}$ </td></tr><tr><td>Freezing Encoder</td><td>False</td><td>False</td><td>False</td><td>False</td><td>True</td><td>False</td><td>{True, False}</td></tr><tr><td>Deleting  $\mathrm { r a t i o } ^ { * 5 }$ </td><td>0.6</td><td>0.7</td><td>0.6</td><td>0.5</td><td>0.6</td><td>0.6</td><td>{0.3, 0.4, 0.5, 0.6, 0.7}</td></tr><tr><td colspan="8">all-MiniLM-L12-v2</td></tr><tr><td>Learning rate DAE</td><td>5e-5</td><td>1e-5</td><td>1e-5</td><td>5e-5</td><td>5e-5</td><td></td><td>{1e-4, 2e-4, 5e-5, 1e-5, 1e-6}</td></tr><tr><td>Learning rate CL</td><td>5e-5</td><td>1e-5</td><td>1e-5</td><td>5e-5</td><td>5e-5</td><td></td><td>{1e-4, 2e-4, 5e-5, 1e-5, 1e-6}</td></tr><tr><td>Learning rate  $F T$ </td><td>5e-5</td><td>1e-5</td><td>1e-5</td><td>1e-5</td><td>1e-5</td><td></td><td>{1e-4, 1e-5, 2e-5, 1e-6}</td></tr><tr><td>Epochs  $D A E ^ { * 1 }$ </td><td>4</td><td>4</td><td>4</td><td>12</td><td>3</td><td></td><td>{2, 3, 4, 9, 12}</td></tr><tr><td>Epochs CL</td><td>12</td><td>12</td><td>4</td><td>12</td><td>3</td><td></td><td>{2, 3, 4, 9, 12}</td></tr><tr><td>Max Epochs  $F T ^ { * 2 }$ </td><td>70</td><td>70</td><td>70</td><td>70</td><td>70</td><td></td><td>{20, 50, 70}</td></tr><tr><td>Batch size  $D A E$ </td><td>32</td><td>32</td><td>32</td><td>128</td><td>32</td><td></td><td>{2,6,8,12,16,32,64,128}</td></tr><tr><td>Batch size CL</td><td>32</td><td>32</td><td>32</td><td>128</td><td>32</td><td></td><td>{2,6,8,12,16,32,64,128}</td></tr><tr><td>Batch size  $F T$ </td><td>32</td><td>32</td><td>32</td><td>128</td><td>32</td><td></td><td>{2,6,8,12,16,32,64,128}</td></tr><tr><td>Eps in  $F T ^ { * 3 }$ </td><td>2e-5</td><td>2e-5</td><td>2e-5</td><td>2e-5</td><td>2e-5</td><td></td><td>{2e-05}</td></tr><tr><td>Use Length</td><td>*5</td><td>108*4</td><td>512</td><td>108*4</td><td>512</td><td></td><td> $\{ ^ { * ^ { 5 } } , { } ^ { * ^ { 4 } } , 5 1 2 \}$ </td></tr><tr><td>Freezing Encoder</td><td>True</td><td>False</td><td>False</td><td>False</td><td>True</td><td></td><td>{True, False}</td></tr><tr><td> $\mathrm { D e l e t i n g ~ r a t i o ^ { * } } ^ { 5 }$ </td><td>0.6</td><td>0.3</td><td>0.6</td><td>0.7</td><td>0.6</td><td></td><td>{0.3, 0.4, 0.5, 0.6, 0.7}</td></tr></table>

Table 5: Hyper-parameters configurations and search space of the experiments.\*<sup>1</sup> means these are not real epochs since the input data is not always the same. The data was masked on the fly; therefore, each epoch differs. \*<sup>2</sup> We used a an early stopping approach for the FT phase. \*<sup>3</sup> We only consider the epsilon hyperparameter in the AdamW optimizer for FT. The other two phases use the default value from the Transformers library (1e-06). \*<sup>4</sup> This hyper-parameter was estimated initially with the training dataset with a large margin. This was applied for datasets with very short sentences, like SNIPS. \*<sup>5</sup> This hyper-parameter estimates the max length of the sequences using the 10% of the examples. This estimation is multiplied by 1.2 and is added as the maximum size of the sequences for the embedding layers. The difference is that it was done on the fly and not preserved in this case.

## A.2 Other metrics

In this section of the appendix, we present the results obtained for metrics other than accuracy. More specifically, we present three tables: Precision and Recall in Table 6, and F1 (Table 7). These metrics show better the role played by the imbalance correction. The notation follows the Table 3.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Precision                         Recall</td></tr><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=2>3-Phase   Joint   FT</td><td rowspan=1 colspan=1>3-Phase  Joint   FT</td></tr><tr><td rowspan=1 colspan=4>RoBERTa</td></tr><tr><td rowspan=1 colspan=1>SNIPS</td><td rowspan=1 colspan=2>99.81     94.94 91.04</td><td rowspan=6 colspan=1>99.81    94.99  91.0398.30    98.09  97.7295.05    92.32  89.2153.71    51.18  49.0295.08    94.82  92.47100.0    96.74  91.03</td></tr><tr><td rowspan=1 colspan=1>SNIPS2</td><td rowspan=1 colspan=1>98.2697.969</td><td rowspan=5 colspan=2>97.5095.62     94.29  92.5355.51     53.23  49.5095.09     94.82  92.4498.08     95.56  91.83</td></tr><tr><td rowspan=1 colspan=1>SST2</td><td></td></tr><tr><td rowspan=1 colspan=1>SST5</td><td></td></tr><tr><td rowspan=1 colspan=1>AGNews</td><td></td></tr><tr><td rowspan=1 colspan=1>IMDB</td><td></td></tr><tr><td rowspan=1 colspan=4>all-MiniLM-L12-v2</td></tr><tr><td rowspan=1 colspan=1>SNIPS</td><td rowspan=1 colspan=2>100.00    92.05  93.12</td><td rowspan=1 colspan=1>100.00   92.06  92.98</td></tr><tr><td rowspan=1 colspan=1>SNIPS2SST2SST5</td><td rowspan=2 colspan=2>98.61     98.64  93.8993.92     92.32  88.7959.57     61.62  53.1594.84    94.28  89.57</td><td rowspan=2 colspan=1>98.68    98.60 94.1494.89    91.24  88.1349.02    46.02  40.0294.83    94.28 89.57</td></tr><tr><td rowspan=1 colspan=1>AGNews</td></tr></table>

Table 6: Precision and Recall values. The best values are shown in bold

<table><tr><td>Dataset</td><td>3-Phase</td><td>Joint</td><td>FT</td><td>S-P</td><td>EFL</td><td>CAE</td><td>FTBERT</td></tr><tr><td colspan="8">RoBERTa</td></tr><tr><td>SNIPS</td><td>99.81</td><td>94.95</td><td>91.03</td><td>97.0</td><td>=</td><td>97.0</td><td>一</td></tr><tr><td>SNIPS2</td><td>98.28</td><td>98.01</td><td>97.57</td><td>97.0</td><td>一</td><td>97.0</td><td>一</td></tr><tr><td>SST2</td><td>95.14</td><td>93.29</td><td>90.82</td><td>-</td><td>1</td><td>-</td><td></td></tr><tr><td>SST5</td><td>54.59</td><td>52.18</td><td>49.24</td><td>-</td><td></td><td>1</td><td></td></tr><tr><td>AGNews</td><td>95.08</td><td>94.81</td><td>92.45</td><td>一</td><td>79.5</td><td>=</td><td>95.20</td></tr><tr><td>IMDB</td><td>99.03</td><td>95.10</td><td>91.43</td><td></td><td>一</td><td></td><td></td></tr><tr><td colspan="8">all-MiniLM-L12-v2</td></tr><tr><td>SNIPS</td><td>100.00</td><td>92.0</td><td>92.81</td><td>97.0</td><td></td><td>97.0</td><td>-</td></tr><tr><td>SNIPS2</td><td>98.63</td><td>98.60</td><td>93.90</td><td>97.0</td><td>一</td><td>97.0</td><td>-</td></tr><tr><td>SST2</td><td>94.39</td><td>91.78</td><td>88.46</td><td>-</td><td>一</td><td>一</td><td>一</td></tr><tr><td>SST5</td><td>53.78</td><td>52.69</td><td>45.66</td><td>-</td><td></td><td>一</td><td></td></tr><tr><td>AGNews</td><td>94.83</td><td>94.27</td><td>89.56</td><td>一</td><td>79.5</td><td>一</td><td>95.20</td></tr></table>

Table 7: F1 values for the best results.The best values are shown in bold