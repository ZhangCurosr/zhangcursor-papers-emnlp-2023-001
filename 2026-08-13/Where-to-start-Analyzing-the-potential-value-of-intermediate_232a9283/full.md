# Where to start? Analyzing the potential value of intermediate models

Leshem Choshen∗†, Elad Venezian∗, Shachar Don-Yehiya‡, Noam Slonim, Yoav Katz

IBM Research, †MIT, ‡The Hebrew University

{leshem.choshen@, eladv@il.,shachar.don-yehiya@ noams@il., katz@il.}ibm.com

## Abstract

Previous studies observed that finetuned models may be better base models than the vanilla pretrained model. Such a model, finetuned on some source dataset, may provide a better starting point for a new finetuning process on a desired target dataset. Here, we perform a systematic analysis of this intertraining scheme, over a wide range of English classification tasks. Surprisingly, our analysis suggests that the potential intertraining gain can be analyzed independently for the target dataset under consideration, and for a base model being considered as a starting point. Hence, a performant model is generally strong, even if its training data was not aligned with the target dataset. Furthermore, we leverage our analysis to propose a practical and efficient approach to determine if and how to select a base model in real-world settings. Last, we release an updating ranking of best models in the HuggingFace hub per architecture.<sup>1</sup>

## 1 Introduction

Finetuning pretrained models (Devlin et al., 2019), is currently the standard and best approach for adjusting such models to perform a downstream task (Chen et al., 2022). The resulting finetuned models are typically used for inferring the labels of new examples that are reminiscent of the data used for finetuning. However, it was shown (Phang et al., 2018a) that finetuned models, trained on some source dataset, may represent better base models, namely a better starting point for a new finetuning process on a desired target dataset. This scheme, often referred to as intertraining, is the focus of the present work.

Given a target dataset, one may wonder what could be the intertraining gain, to determine whether it is worthwhile spending resources on selecting a base model. Assuming the potential gain is high, the following natural question is which base models are most promising, out of countless options available through hubs HuggingFace (e.g. Wolf et al., 2020). We propose pragmatic methods to answer both questions, supported by extensive experiments.

We begin with two observations: (i) some target datasets are intertraining-sensitive, i.e., have the potential to gain significantly from intertraining, while others are not, and are typically indifferent to the base model selection. Furthermore, revealing this property of the target dataset can be done efficiently, by examining the gains obtained when using a single representative base model as a starting point; (ii) some base models are of high quality, i.e. finetuning on them provides consistent improvements on target datasets, but most base models are inferior and degrade performance. Furthermore, ranking base models by quality can be done on one target task – and efficiently, via linear probing, namely training only the base model classification head, over a single representative dataset.

Thus, we argue that a preferable base model can be selected independently of the target dataset. This is in contrast to the common perception (c.f. §7) that the alignment of the target dataset and the source dataset – used to generate the base model – is a major factor in determining intertraining success. We substantiate our observation of independence by conducting experiments on a comprehensive set of target datasets and base models, comprising models obtained under controlled conditions as well as models from HuggingFace. In addition to these findings, we analyze attributes of the source and target datasets that affect gains (§6).

As some models are just better, not due to the choice of a current dataset, it makes sense to rank the models once and pick the best ones. But even ranking a thousand models is costly. In §8, we rely on our analysis to propose a practical approach to efficiently select models in a real-world setting. Moreover, instead of expecting others to rank the models, we share an updating site site featuring the best models currently found. So far, we tested over 2.5K models.

## 2 Preliminaries

In this paper, we use the following terminology. A dataset is a set of examples and labels. Our goal is to maximize accuracy on the test of the target dataset, "target" for short. We discuss the difference between domain, task, and dataset in App. A.

A pretrained (PT) model is a self-supervised model, e.g., RoBERTa (Liu et al., 2019). A finetuned model is a PT model that was further trained over some source dataset, denoted henceforth as "source". We assume access to many such models, e.g., through HuggingFace. A base model can be either a PT model or a finetuned model. When finetuning over the target train data, one can start from any base model. Intertraining refers to starting from a finetuned model as a base model, and in this case, we refer to this base model as an intermediate model. We denote by $S _ { m } ^ { t }$ , the accuracy score obtained over the target test set, $t ,$ after finetuning some base model m over the target train set. The intertraining gain of model m w.r.t. using the PT model, is thus defined via gain $( m , t ) = s _ { m } ^ { t } - s _ { P T } ^ { t }$ . Note that the gain may be negative. Given a set of intermediate models, $M \ = \ m _ { 1 } \dots m _ { n }$ , the intertraining max-gain is defined as $\mathrm { m a x } _ { m \in M } \left( s _ { m } ^ { t } - s _ { P T } ^ { t } \right)$ . Thus, theoretically, max-gain is achieved by finetuning all the available intermediate models and picking the one best performing on the target test set. To avoid overfitting and reduce costs, our goal is to find an intermediate model with a gain that is as close as possible to the max-gain, without explicitly finetuning all the intermediate models.

## 3 Experimental Setup

Our experimental setup is described next. The parameters for reproducibility are detailed in App. B.

## 3.1 Dataset Groups

Our experiments are based on 3 groups of English text classification datasets described next (App. C).

We focus on text classification for ease of evaluation, but assume the tasks are diverse enough for our conclusions to extend to other settings.

General. Containing GLUE (Wang et al., 2018) and SuperGLUE classification datasets (Wang et al., 2019a), excluding test only and regression datasets. The datasets cover a wide range of tasks, from sentiment analysis through linguistic acceptability to natural language inference. It is the most commonly used benchmark in related work (§7).

NLI. Containing Natural Language Inference and Entailment tasks. Datasets of this group all share the same task. There is some overlap between NLI and General; in Fig. 1 and mean comparison we subtract the overlapping datasets from General.

Twitter. Containing 6 Twitter datasets collected by TweetEval (Barbieri et al., 2020). The tasks range from irony detection to emoji prediction. Datasets of this group all share the same domain.

## 3.2 Models

Unless stated otherwise, our PT model of choice is RoBERTA (Liu et al., 2019). We acquire intermediate models in two ways:

In-house. Obtained by finetuning the PT model over General, NLI, and Twitter dataset groups as the source datasets, with 5 seeds. Since we control the datasets and have knowledge about their features, this enables us to find relations between the dataset properties and the intermediate models generated from these datasets.

Off-the-shelf. 66 RoBERTa models downloaded from HuggingFace (see App. §E for more details). Since these models do not carry information excluding their names, this set allows us to validate our claims on a "real-world" model distribution.

## 3.3 Models/Targets experiments

We test many intermediate models on various target datasets. We finetune each intermediate model and the PT on the target train, and report the intertraining gain over the target test. In the In-house models/targets experiment, all 22 datasets from the General, NLI, and Twitter groups act as both source and target and gains are average of 5 seeds. In the

![](images/96e2be0b1a621570c04d61a65cb00c87fe47094172dbd49cbf439ab3c9787497.jpg)  
Figure 1: Results of in-house models/targets experiment. Columns correspond to target datasets and Rows correspond to intermediate models generated based on same datasets as source. The 22 datasets come from the General, NLI and Twitter groups. Each value indicates intertraining gain w.r.t. the PT model, averaged over 5 seeds. Sorted by group and source average gain (bottom row). Positive significant cells (>2 STD) are italicized.

Off-the-shelfmodels/targets experiment, we download the 66 source models from Huggingface and test on the 14 General datasets as targets.

## 4 Results

Most models are worse than PT and about 1 in 6 are better, providing positive intertraining gain. The in-house models/targets results are depicted in Fig. 1 and STDs and reference results in App. §D. App. §E reports results with off-the-shelf RoBERTa and T5 intermediate models.

The rows and columns in Fig. 1 are similarly ordered – first the General datasets, then the NLI datasets, and last the Twitter datasets. Loosely speaking, we do not recognize an approximate green block structure across the diagonal; namely, we do not observe clear intertraining gain for similar tasks (NLI); nor for similar domains (Twitter). However, some columns and some rows depict higher intertraining gains, while for others, the impact is minor. Taken together, these observations suggest little dependence between the source used to generate the intermediate model and the performance over the target. This is in contrast to the common assumption (§7) that the source and target need to be similar for intertraining to work. Next, we delve deeper into these observations.

![](images/3d806babf37deae6fc2430f0c3dd0434699c3b5507c17a8c2dfe088541a6f5d8.jpg)  
Figure 2: Linear probing MNLI (x) is enough to predict finetuning gains (y) averaged over 14 General datasets. Each point corresponds to one off-the-shelf base model.

## 4.1 Target Sensitivity to Intertraining

Considering Fig. 1 columns, we notice that for some target datasets (e.g., ESNLI) intertraining makes little difference, while for others (e.g., COPA) the impact is quite significant. We argue that this target property can be predicted via an efficient and straightforward method. Specifically, the gains of one strong intermediate model should resemble the max-gains of a group of models. Indeed, MNLI highly correlates both with the max-gain of in-house models tested on the 22 targets in Fig. 1 (Spearman: 0.89, Pearson 0.99) and off-the-shelf models tested on the 14 General targets (Spearman: 0.90, Pearson: 0.94, $p < 0 . 0 1$ for all). The replication on off-the-shelf models shows that this is a general result and not a reflection of MNLI being top of the in-house group. Overall, we find sensitivity is a characteristic of the target dataset separated from the source factor.

## 4.2 Ranking Intermediate Models

Considering Fig. 1 rows, we notice that some intermediate models – e.g., MNLI – provide relatively high gains for many targets. We observe that this is a general phenomenon – stronger models are typically stronger for many datasets.

Identifying such models in advance could be practically valuable, since for a new target, one would consider only the highly ranked intermediate models (see §8). In the following, we propose a simple yet efficient method to obtain such a static ranking - which is made once, without accounting for the target. A more comprehensive ranking alternative is described in App. §F.

Given an intermediate model m, we train a linear probe (LP) – i.e., train only the classification head – over MNLI, and consider the gain, denoted $L P ( m , M N L I ) ^ { 2 }$ . Evidently, this gain is a good proxy for the quality of $m$ . Specifically, let $g _ { m } ^ { a v g }$ be the average gain of m over a set of target datasets. As depicted in Fig. 2, we observe that $L P ( m , M N L I )$ and $g _ { m } ^ { a v g }$ are highly correlated in the in-house models/targets experiment (Spearman: 0.46, Pearson: 0.78, $p < 0 . 0 1 )$ and the off-the-shelf models/targets experiment (Spearman: 0.51, Pearson: $0 . 6 6 , p < 0 . 0 1 \AA$ ). In other words, if available intermediate models are ranked by LP on one dataset $L P ( m , M N L I )$ , highly ranked models represent the most promising starting points on average. The high correlation means this connection holds not only for the top-ranked models, but throughout. Moreover, the replication on off-theshelf models shows this is robust not only across targets but across sources.

To further support this claim, we use this ranking to find the most promising intermediate models. For each target t, we consider the gain obtained by the top-ranked model and the max-gain obtained by one of the three top-ranked models, denoted $g _ { ( 1 ) } ^ { t }$ and $g _ { ( 3 ) } ^ { t }$ , respectively. In comparison, we consider the max-gain obtained while considering all available models, denoted $g _ { ( m a x ) } ^ { t }$ . We further denote los $s _ { 1 } ^ { t } \equiv g _ { ( m a x ) } ^ { t } - g _ { ( 1 ) } ^ { t }$ and $\begin{array} { r } { l o s s _ { 3 } ^ { t } \equiv g _ { m a x } ^ { t } - g _ { ( 3 ) } ^ { t } } \end{array}$ In other words, loss<sup>t</sup> represents the potential gain loss when using the top statically ranked model versus using the best available model for the target under consideration. Similarly, loss<sup>t</sup> represents the lost gain when using the best model out of the top 3 ranked models versus using the best available model. Note, that even in this latter case, finding the best model involves finetuning only 3 models over the target train data, which is far less demanding compared to finetuning all available models.

In Table 1, we report statistics for loss<sup>t</sup> and $\boldsymbol { l o s s _ { 3 } ^ { t } }$ over the in-house and off-the-shelf models/targets experiments. Although the ranking is target-independent, the top 3 ranked models typically provide most of the potential intertraining gain. For example, instead of checking all 66 available models, using this simple strategy of checking the top 3 ranked models, each of the 14 targets lost at most 1.62 points.

<table><tr><td>Models</td><td>@Top</td><td> $\operatorname { A v g } .$ </td><td>Max</td><td># targets  $\mathrm { s . t . ~ } l o s s _ { n } ^ { t } > 1$ </td></tr><tr><td rowspan="2">In-house</td><td>loss1</td><td>0.37</td><td>2.11</td><td>3/22</td></tr><tr><td> $l o s s _ { 3 } ^ { t }$ </td><td>0.2</td><td>1.15</td><td>1/22</td></tr><tr><td rowspan="2">Off-the-shelf</td><td>losst</td><td>2.33</td><td>12.0</td><td>8/14</td></tr><tr><td> $l o s s _ { 3 } ^ { t }$ </td><td>0.34</td><td>1.62</td><td>2/14</td></tr></table>

Table 1: Lost Gain $( l o s s _ { n } )$ is small when choosing the n top-ranked models. Columns present the aggregation across target datasets of the lost gain: average, max and the number of targets that lose at least one point. Rows present either in-house (22 models and 22 targets) or off-the-shelf (66 models and 14 targets) experiments.

## 5 Source and Target Interaction Analysis

Next, we analyse the interaction between the source dataset and the target dataset. Obviously, such interaction may impact the gain. On one extreme, since finetuning on the same data twice does not improve performance, intertraining is not valuable when the source and target data are identical. On another extreme, consider partitions of the same data. Obviously, training over half the data as the source would be beneficial for the other half, the target. Thus, we do not question that interaction may exist, but rather investigate how common and strong it is.

Interaction between dataset groups. Our General dataset group consists of diverse datasets, while the other groups share a domain (Twitter) or a task (NLI). We analyze whether datasets that share a domain or task with the target represent better source datasets than others.

Table 2 depicts the average gain of each source group vs. each target group. Comparing targets (table columns), we find the models trained on similar tasks, as a group, have a distinct behavior (ANOVA $p < 0 . 0 1 )$ . On average, using NLI intermediate models on NLI targets, yields a gain of 0.63, compared to a strong negative gain when using intermediate models from other groups. Similarly, there is a same-group gain of 0.5 on Twitter.

Comparing sources (table rows), while NLI is best improved by NLI models, NLI models im-

<table><tr><td></td><td>General</td><td>NLI</td><td>Twitter</td></tr><tr><td>General</td><td>-0.37</td><td>-2.68</td><td>-0.54</td></tr><tr><td>NLI</td><td>1.26</td><td>0.63</td><td>-0.03</td></tr><tr><td>Twitter</td><td>-0.4</td><td>-2.39</td><td>0.53</td></tr></table>

Table 2: Intermediate models trained on sources from the same domain (Twitter) or task (NLI) as the target, yield greater gain. Numbers represent the average gain of intermediate models of a source group (rows) on a given target group (columns) .

prove General datasets even more than NLI ones.   
Possibly, NLIs are just good intermediate models.   
Twitter models, however, do seem to improve Twitter targets more (ANOVA $p < 0 . 0 1 )$ , by 1 point.   
Hence, it seems the effects are mixed.

In summary, as a group, a similar source tends to yield greater gains than an unrelated source. However, in the rest of this section, we find this effect is of secondary importance to predicting model gains. A similar source may be more beneficial than a random one, but a well chosen source produces larger benefits regardless of similarity.

Symmetry as a similarity bound. We consider dataset similarity from another perspective. Similarity is a symmetric relation. Hence, if sourcetarget similarity was a main reason for gains, we would expect that when the source A helps the target B, the source B would help A as well. We assess the symmetry of the in-house models/targets experiment. We find that gains are far from symmetric (details in App. §G). Thus, (symmetric) similarity seems like a weak explanation of which source data would help which target.<sup>3</sup>

Regression. As additional support for the relatively small impact of the source-target interaction, we show that the interaction is hardly needed to predict the scores. Specifically, a simple regressor can approximate the gain without considering such interactions. The regression fits 3 sets of coefficients. For each target two coefficients $t _ { i } , t _ { i } ^ { \prime } - \mathrm { w h i c h o n e }$ may think of as capturing average gain and sensitivity to inter-training, per target; and for each base model $b _ { j } - \mathrm { w h i c h }$ one may think of as capturing average inter-training gain, per base model. We then define $\hat { g a i n } ( b a s e _ { i } , t a r g e t _ { j } ) = \left( b _ { i } + t _ { j } \right) t _ { j } ^ { \prime }$ where i and j are the base model and target indices, respectively. Note that by construction, this regressor has $2 \dot { N } + n$ parameters, while trying to model Nn˙ interactions; thus, it does not have enough degrees of freedom to explicitly model all source/target interactions. Still, it obtains satisfactory performance, as shown next. As a baseline, we fit the same regressor after randomly shuffling all the gains in the experiment. This shuffling ensures no information comes from the specific source and target, while maintaining the gain value distribution. We minimize Mean Squared Error (MSE) and fit with SGD until convergence.

Before we present the results, it would be beneficial to give some intuition on MSE. As MSE is the square of the error, an average MSE of 4, means that the average prediction was accurate up to 2 $( { \sqrt { 4 } } )$ points on average or more accurate than that (as outliers have more effect on MSE than on the average error rate).

We obtain an MSE of 4.2 on the in-house models/targets experiment (3.3), versus $9 . 6 , \sigma =$ 0.9 when the gains are shuffled. Thus, knowing the source id and the target id provides significant information about the expected gain, with no need of keeping a designated parameter for each source-target combination. We further compare this regressor with two other regressors, one that considers only base model information $( \overrightharpoon { g a i n } ( b a s e _ { i } , t a r g e t _ { j } ) = b _ { i } )$ and one that considers only target related information $( \tilde { g a i n } ( b a s e _ { i } , t a r g e t _ { j } ) = t _ { j } )$ . The MSE fit is 10.4 and 8.2, respectively, compared to $1 0 . 8 , \sigma = 0 . 4$ on shuffled gains. This suggests both the source and the target impact the intertraining gain, with potentially somewhat stronger influence by the target.

In conclusion, our observations suggest that when considering intertraining potential, loosely speaking it is enough to consider two separate issues – (i) choosing a base model; and (ii) determining the expected effect on the target.

## 6 Factors Contributing to Gains

So far, we showed the effects of the intermediate models are largely separate from those of the target. We proceed to examine how specific factors in each contribute to intertraining gains.

## 6.1 Source and target sizes

Following the above observations, we ask what makes a base model good, or a target sensitive?

![](images/3b0be32b10aa3b4a430fbea5ea8dd0a61a3b8dab062b27d5072b954490b2aabe.jpg)  
Figure 3: For ’good’ sources the average gain increase as the source training size increases, while for ’bad sources it decreases.

![](images/028fda5b04ae41155e22a8047d5fb658ae5cf085719e5bba900113750e0922c6.jpg)  
Figure 4: The average gain across targets decreases as the target training size increases.

We examine the effects of architecture (§6.3) and source score (§6.2), but start by examining the data sizes effect on the gain: First, we identify effects of the datasets sizes in controlled experiments. Next, we assess the extent to which the effect of dataset size is evident in previous experiments. For more related analysis we refer to Phang et al. (2018a); Pruksachatkun et al. (2020).

Effect of dataset size. We first control the source dataset train size. We create intermediate models on 4 source datasets – the top 2 and lowest 2 in-house models, according to the static ranking (§4.2). For each, we limit the training to 50 3200 samples and use the General group datasets as targets. Evidently, for good sources (ANLI, MNLI), more training data yields greater gains (Fig. 3). However, additional data decreases performance for bad sources (MultiRC, QQP). We conclude that the amount of source data amplifies the effect determined by the type of data.

We experiment on the effect of target size, using General sources and General targets with train data of more than 1600 examples. We limit the target train sizes to between 50 – namely, few-shot setting – and 1600. As depicted in Fig. 4, the gain decreases as the target training size increases, implying larger potential for intertraining when the target training data is limited. Interestingly, for 3 targets we see positive gains on small datasets, which drops to negative gain as training data increases, and then seem to rise again towards zero. This ’U-shape’ effect hints at two competing effects that should be better examined in future work (c.f. App. H).

Training size effects in practice. We examine whether the effect above is strong in comparison to other unknown factors. Considering the in-house models/targets experiment, the Pearson Correlation between source training size and source average gain is 0.75. Out of the 5 largest sources (ESNLI, MNLI, QQP, ANLI, and QNLI), 3 are also the source tasks with the highest average gain (MNLI, ANLI and ESNLI) and QQP is the source dataset with the second-lowest gain (negative). This is in line with our previous observation that additional source training data magnifies the positive or the negative intertraining effect of the source data.

We observe no significant correlation between target training size and target average gain, where the average is taken across sources. Still, targets with small training data seem to be more sensitive to intertraining. Specifically, the 5 targets with the smallest training data (CB, CoPA, WSC, WNLI, RTE) are also those for which we observe the maximal gain difference across all sources.

## 6.2 Similar Source Score, Different Gain

One can expect that two models with similar performance on a source dataset would also have similar intertraining gains. Our results suggest otherwise. We finetune 20 models over MNLI source and use them as intermediate models on the General target group. We compare the scores on the source test data to the average score obtained on the target datasets. Source task scores vary between 86.5 and 87.5 while General target average varies between 74.5 and 79, without correlation (c.f. App. I).

McCoy et al. (2019) found that finetuned models that show similar performance on their test data, still have different performance on out-of-domain challenge sets, suggesting the models learnt different generalizations schemes. Juneja et al. (2022a) suggested that those models converged to different basins in the loss space. They tagged models from one basin that tend to generalize better as good, and the rest as bad. We check whether good models are also better intermediate models for other tasks. We took their BERT models finetuned on MNLI as intermediate models – 12 good and 12 bad models – and used the General datasets as targets, finding that good models are indeed better for intertraining (3.65 avg. gain for good vs. 2.16 for bad).

The differences discussed above are due to using different seeds. In App. I.1 we show that hyperparameter selection can also impact the quality of an intermediate model, regardless of the score observed on the source test data.

In summary, similar training and/or similar performance on the source test, do not translate to similar intertraining gains on new target tasks.

## 6.3 Effect of different architectures

We validate our main conclusions across different architectures. To that end, we repeat the inhouse models/targets experiment with BERT (Devlin et al., 2019) and T5 (Raffel et al., 2020) architectures. (see full tables in App. J).

We start by showing that the loose source-target coupling holds across architectures. We then show that different source datasets rank differently across architecture, but that target sensitivity is similar.

To show the source-target independence, we repeat the regression fit (§5). As before, the fit explains each architecture’s gains much better than when data is shuffled (BERT MSE 10.5, random 30.1, $\sigma = 4 . 1 7 ;$ T5 MSE 8.11, random 13.51, σ = 1.5). Neither the average gain of intermediate models - trained over various sources, nor the average gain for target tasks, correlate across different architectures. However, the target sensitivity, measured by max-gain, is correlated across all architectures (pairwise Pearson $0 . 6 - 0 . 9 4 , p < 0 . 0 5 )$ . Thus, although the source-target decoupling and the target sensitivity are shared across architectures, a source dataset that produces high gains in one architecture might not do so in another.

A notable exception is the MNLI source dataset which achieves the highest gain in all three architectures. Possibly, some data sources provide a strong intermediate model regardless of PT, with MNLI as a prominent example.

## 7 Related Work

Various works use intertraining, often following the assumption of task alignment necessity (Ein-Dor et al., 2022; Don-Yehiya et al., 2022a; Awasthy et al., 2020), namely, that the source acts as weak labeled data (Shnarch et al., 2018; Yu et al., 2021). While we consider intertraining through finetuning, adaptation to the target (Shnarch et al., 2022) or the domain (Gururangan et al., 2020) was also suggested. Such adaptation may be applied to any base model, and is complementary to the choice among base models. The need for alignment was also previously debated in the context of pretraining tasks (Krishna et al., 2021; Rothe et al., 2021; Zhang et al., 2020; Ram et al., 2021).

The properties of intertraining were studied in other contexts. Phang et al. (2018a) suggested the intertraining scheme. Pruksachatkun et al. (2020) probed the linguistic knowledge changes after intertraining, noting correlations to some target tasks, and hypothesized why some source tasks are good for intertraining. Mosbach et al. (2020); Chang and Lu (2021) replicated the existence of good sources, but rejected the hypothesis. Others tried to find which tasks have an affinity to each other (Poth et al., 2021; Bassignana et al., 2022a,b; Vu et al., 2020), or when to prefer multitask (Weller et al., 2022). We study a much larger number of sources and targets, aiming to describe their natural distribution (c.f. §9) and also find that while specific choice may help, given enough models, it is safe to identify the models that just excel generally.

Recent work focuses on fusing multiple base models rather than picking just one (Choshen et al., 2022; Matena and Raffel, 2021; Wortsman et al., 2022; Yadav et al., 2023). We expect our understanding to aid in choosing a subset of models to fuse as well.

Multi-task learning is another related field. It studies finetuning on different tasks at once (Aribandi et al., 2021; Aghajanyan et al., 2021) and recently also a way to recycle models to do so was proposed (Don-Yehiya et al., 2022b). In contrast to our analysis, similarity between tasks aids multi-task learning (Abnar et al., 2021).

Deciding which information should accompany publications is an active research field, covering datasets (Holland et al., 2018; Gebru et al., 2021), human annotation sheets (Shimorina and Belz, 2022), and models (McMillan-Major et al., 2021; Mitchell et al., 2019). Our work proposes to report upon sharing a model the aspects shown to affect its quality, such as $L P \left( m , M N L I \right)$ . For datasets, we propose to report intertraining sensitivity.

## 8 Practical recommendations

Based on the observations (§4) and analysis (§5), we propose a methodology for efficiently utilizing intertraining in real-world settings. We suggest to collaboratively rank all available base models for intertraining and to utilize this list whenever intertraining is applied to a new target.

New model. When a new model becomes available, we encourage practitioners to assess and share its quality. This can be done efficiently by linear probing on MNLI (§4.2) or comprehensively (App. §F) by finetuning on various datasets.

We created statically ranked lists for RoBERTabase and T5-small in App. §E. Moreover, we apply our methods to many architectures and 36 test sets in an updating site , reporting the best models.

New target. When considering intertraining on a new task, we recommend checking the target dataset sensitivity, and then choosing the base model. Since the model’s rank hardly depends on the target dataset, we suggest using the static ranking. Specifically, we propose to finetune the top-ranked model, and compare the results to those obtained when finetuning the respective PT model. Assuming the gain justifies it, one should consider a few additional intermediate models, in descending order, according to the allocated resources.

## 9 Discussion

In §8, we highlighted our practical recommendations for intertraining. Those, together with a systematic analysis of what affects intertraining, cover our main contributions. We hope this analysis would also aid future work on interactions between datasets; intertraining practices; and reusing finetuned models.

Our experiments mainly characterize what is probable rather than what is possible. We do not create specific models or aim to improve a specific task. Instead, we investigate what is likely to be found in typical practical settings. Accordingly, our findings are probabilistic in nature: Most models are not beneficial as intermediate models, but there are enough that are. Mostly, beneficial models are beneficial for many targets.

As a side effect, we do identify specific strong source models. MNLI was already considered a beneficial source dataset (Phang et al., 2018a), a finding which we corroborate in a comprehensive manner. Furthermore, when considering off-theshelf models we find models that outperform it (e.g., STS-B based for RoBERTa, and Quora for T5). To facilitate finding additional and better base models we will continuously report in the website website the best models per architecture.

## 10 Limitations

One obvious limitation is that our work is empirical in nature. As such, we report observations, sampled or common, with no theoretical guarantees, and one should recognize the existence of exceptions. Specifically, even though we have not observed it – there might exist target tasks that benefit greatly from a certain type of intermediate models; or intermediate models that help greatly in many targets while degrading performance in others.

Moreover, while testing 22 source datasets, many of which previously untested, we did not find a new top source for intertraining. The best one we found for RoBERTa was already known to be good (MNLI; Phang et al., 2018b; Pruksachatkun et al., 2020). With that, by checking dozens of offthe-shelf models, we did uncover new intermediate models that seem to outperform MNLI (e.g., STS-B for RoBERTa and QQP for T5 – c.f. App. §E). More work is needed to assess the potential intertraining gain of the available datasets and models.

We ran thousands of finetuning experiments, spanning a vast number of tasks and base models. Thus, while we believe it is unlikely that reproducing our experiments will result in different outcomes, the large scale of our experiments places a heavy burden on trying to replicate our results. Moreover, the off-the-shelf models used in our experiments might not be hosted publicly in the future.

Another limitation is that we could not upload all of the models to a shared location. This project was very computationally demanding, but more than that, it was demanding in terms of disk space, hence we had to delete many models along the way.

Finally, for practical reasons, our results are limited to classification tasks in English. We hope that future work will aim to test our conclusions beyond this scope. Overall, in the space of classification, we see our results as robust, testing on 22 datasets (about double the amount of previous works (Pruksachatkun et al., 2020)). We hope the diversity of targets brings large enough differences between datasets that the results would hold in other scopes.

## References

Samira Abnar, Mostafa Dehghani, Behnam Neyshabur, and Hanie Sedghi. 2021. Exploring the limits of large scale pre-training. ArXiv, abs/2110.02095.

Armen Aghajanyan, Anchit Gupta, Akshat Shrivastava, Xilun Chen, Luke Zettlemoyer, and Sonal Gupta. 2021. Muppet: Massive multi-task representations with pre-finetuning. ArXiv, abs/2101.11038.

Vamsi Aribandi, Yi Tay, Tal Schuster, Jinfeng Rao, Huaixiu Steven Zheng, Sanket Vaibhav Mehta, Honglei Zhuang, Vinh Q Tran, Dara Bahri, Jianmo Ni, et al. 2021. Ext5: Towards extreme multitask scaling for transfer learning. arXiv preprint arXiv:2111.10952.

Parul Awasthy, Bishwaranjan Bhattacharjee, John Kender, and Radu Florian. 2020. Predictive model selection for transfer learning in sequence labeling tasks. In Proceedings of SustaiNLP: Workshop on Simple and Efficient Natural Language Processing, pages 113–118, Online. Association for Computational Linguistics.

Roy Bar-Haim, Ido Dagan, Bill Dolan, Lisa Ferro, Danilo Giampiccolo, and Bernardo Magnini. 2006. The second pascal recognising textual entailment challenge.

Francesco Barbieri, Jose Camacho-Collados, Luis Espinosa Anke, and Leonardo Neves. 2020. TweetEval: Unified benchmark and comparative evaluation for tweet classification. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1644–1650, Online. Association for Computational Linguistics.

Francesco Barbieri, Jose Camacho-Collados, Francesco Ronzano, Luis Espinosa-Anke, Miguel Ballesteros, Valerio Basile, Viviana Patti, and Horacio Saggion. 2018. SemEval-2018 Task 2: Multilingual Emoji Prediction. In Proceedings of the 12th International Workshop on Semantic Evaluation (SemEval-2018), New Orleans, LA, United States. Association for Computational Linguistics.

Valerio Basile, Cristina Bosco, Elisabetta Fersini, Debora Nozza, Viviana Patti, Francisco Manuel Rangel Pardo, Paolo Rosso, and Manuela Sanguinetti. 2019. SemEval-2019 task 5: Multilingual detection of hate speech against immigrants and women in Twitter. In Proceedings of the 13th International Workshop on Semantic Evaluation, pages 54–63, Minneapolis, Minnesota, USA. Association for Computational Linguistics.

Elisa Bassignana, Max Müller-Eberstein, Mike Zhang, and Barbara Plank. 2022a. Evidence > intuition: Transferability estimation for encoder selection. ArXiv, abs/2210.11255.

Elisa Bassignana, Max Müller-Eberstein, Mike Zhang, and Barbara Plank. 2022b. Evidence > intuition:

Transferability estimation for encoder selection. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 4218–4227, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Luisa Bentivogli, Peter Clark, Ido Dagan, and Danilo Giampiccolo. 2009. The sixth pascal recognizing textual entailment challenge. In TAC.

Oana-Maria Camburu, Tim Rocktäschel, Thomas Lukasiewicz, and Phil Blunsom. 2018. e-snli: Natural language inference with natural language explanations. In NeurIPS.

Ting-Yun Chang and Chi-Jen Lu. 2021. Rethinking why intermediate-task fine-tuning works. In EMNLP.

Guanzheng Chen, Fangyu Liu, Zaiqiao Meng, and Shangsong Liang. 2022. Revisiting parameterefficient tuning: Are we really there yet? ArXiv, abs/2202.07962.

Sanyuan Chen, Yutai Hou, Yiming Cui, Wanxiang Che, Ting Liu, and Xiangzhan Yu. 2020. Recall and learn: Fine-tuning deep pretrained language models with less forgetting. arXiv preprint arXiv:2004.12651.

Leshem Choshen, Elad Venezian, Noam Slonim, and Yoav Katz. 2022. Fusing finetuned models for better pretraining. ArXiv, abs/2204.03044.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936, Minneapolis, Minnesota. Association for Computational Linguistics.

Ido Dagan, Oren Glickman, and Bernardo Magnini. 2005. The pascal recognising textual entailment challenge. In MLCW.

Marie-Catherine de Marneffe, Mandy Simons, and Judith Tonhauser. 2019. The CommitmentBank: Investigating projection in naturally occurring discourse. To appear in Proceedings of Sinn und Bedeutung 23. Data can be found at https://github.com/mcdm/ CommitmentBank/.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (NAACL-HLT). Association for Computational Linguistics.

William B. Dolan and Chris Brockett. 2005. Automatically constructing a corpus of sentential paraphrases. In Proceedings of the Third International Workshop on Paraphrasing (IWP2005).

Shachar Don-Yehiya, Leshem Choshen, and Omri Abend. 2022a. PreQuEL: Quality estimation of machine translation outputs in advance. pages 11170– 11183.

Shachar Don-Yehiya, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, and Leshem Choshen. 2022b. Cold fusion: Collaborative descent for distributed multitask finetuning. arXiv preprint arXiv:2212.01378.

Liat Ein-Dor, Ilya Shnayderman, Artem Spector, Lena Dankin, Ranit Aharonov, and Noam Slonim. 2022. Fortunately, discourse markers can enhance language models for sentiment analysis.

Timnit Gebru, Jamie H. Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna M. Wallach, Hal Daumé, and Kate Crawford. 2021. Datasheets for datasets. Communications ofthe ACM, 64:86 – 92.

Danilo Giampiccolo, Bernardo Magnini, Ido Dagan, and William B. Dolan. 2007. The third pascal recognizing textual entailment challenge. In ACL-PASCAL@ACL.

Suchin Gururangan, Ana Marasovic, Swabha´ Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. 2020. Don’t stop pretraining: Adapt language models to domains and tasks. ArXiv, abs/2004.10964.

Stephen Hanson and Lorien Pratt. 1988. Comparing biases for minimal network construction with backpropagation. Advances in neural information processing systems, 1.

Sarah Holland, Ahmed Hosny, Sarah Newman, Joshua Joseph, and Kasia Chmielinski. 2018. The dataset nutrition label: A framework to drive higher data quality standards. ArXiv, abs/1805.03677.

Jeevesh Juneja, Rachit Bansal, Kyunghyun Cho, João Sedoc, and Naomi Saphra. 2022a. Linear connectivity reveals generalization strategies. arXiv preprint arXiv:2205.12411.

Jeevesh Juneja, Rachit Bansal, Kyunghyun Cho, João Sedoc, and Naomi Saphra. 2022b. Linear connectivity reveals generalization strategies. ArXiv, abs/2205.12411.

Daniel Khashabi, Snigdha Chaturvedi, Michael Roth, Shyam Upadhyay, and Dan Roth. 2018. Looking beyond the surface: A challenge set for reading comprehension over multiple sentences. In Proceedings of the Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (NAACL-HLT). Association for Computational Linguistics.

Diederik P. Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint 1412.6980.

Kundan Krishna, Jeffrey Bigham, and Zachary C Lipton. 2021. Does pretraining for summarization require knowledge transfer? arXiv preprint arXiv:2109.04953.

Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, and Percy Liang. 2022. Fine-tuning can distort pretrained features and underperform outof-distribution. ArXiv, abs/2202.10054.

Hector Levesque, Ernest Davis, and Leora Morgenstern. 2012. The Winograd schema challenge. In Thirteenth International Conference on the Principles of Knowledge Representation and Reasoning.

Hector J. Levesque, Ernest Davis, and L. Morgenstern. 2011. The winograd schema challenge. In KR.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. ArXiv, abs/1907.11692.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In ICLR.

Michael Matena and Colin Raffel. 2021. Merging models with fisher-weighted averaging. arXiv preprint arXiv:2111.09832.

R Thomas McCoy, Junghyun Min, and Tal Linzen. 2019. Berts of a feather do not generalize together: Large variability in generalization across models with similar test set performance. arXiv preprint arXiv:1911.02969.

Angelina McMillan-Major, Salomey Osei, Juan Diego Rodriguez, Pawan Sasanka Ammanamanchi, Sebastian Gehrmann, and Yacine Jernite. 2021. Reusable templates and guides for documenting datasets and models for natural language processing and generation: A case study of the HuggingFace and GEM data and model cards. In Proceedings ofthe 1st Workshop on Natural Language Generation, Evaluation, and Metrics (GEM 2021), pages 121–135, Online. Association for Computational Linguistics.

Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. 2019. Model cards for model reporting. Proceedings ofthe Conference on Fairness, Accountability, and Transparency.

Saif M. Mohammad and Felipe Bravo-Marquez. 2017. Emotion intensities in tweets. In Proceedings ofthe sixth joint conference on lexical and computational semantics (\*Sem), Vancouver, Canada.

Saif M. Mohammad, Svetlana Kiritchenko, Parinaz Sobhani, Xiaodan Zhu, and Colin Cherry. 2016. Semeval-2016 task 6: Detecting stance in tweets. In Proceedings of the International Workshop on Semantic Evaluation, SemEval ’16, San Diego, California.

Marius Mosbach, Anna Khokhlova, Michael A. Hedderich, and Dietrich Klakow. 2020. On the interplay between fine-tuning and sentence-level probing for linguistic knowledge in pre-trained transformers. In Proceedings ofthe Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networksfor NLP, pages 68–82, Online. Association for Computational Linguistics.

Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. 2020. Adversarial NLI: A new benchmark for natural language understanding. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4885–4901, Online. Association for Computational Linguistics.

Jason Phang, Thibault Févry, and Samuel R Bowman. 2018a. Sentence encoders on STILTs: Supplementary training on intermediate labeled-data tasks. arXiv preprint 1811.01088.

Jason Phang, Thibault Févry, and Samuel R. Bowman. 2018b. Sentence encoders on stilts: Supplementary training on intermediate labeled-data tasks. ArXiv, abs/1811.01088.

Mohammad Taher Pilehvar and Jose Camacho-Collados. 2019. WiC: The word-in-context dataset for evaluating context-sensitive meaning representations. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT). Association for Computational Linguistics.

Clifton Poth, Jonas Pfeiffer, Andreas Rücklé, and Iryna Gurevych. 2021. What to pre-train on? Efficient intermediate task selection. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 10585–10605, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Yada Pruksachatkun, Jason Phang, Haokun Liu, Phu Mon Htut, Xiaoyi Zhang, Richard Yuanzhe Pang, Clara Vania, Katharina Kann, and Samuel R. Bowman. 2020. Intermediate-task transfer learning with pretrained language models: When and why does it work? In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5231–5247, Online. Association for Computational Linguistics.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofMachine Learning Research, 21:1– 67.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas. Association for Computational Linguistics.

Ori Ram, Yuval Kirstain, Jonathan Berant, Amir Globerson, and Omer Levy. 2021. Few-shot question answering by pretraining span selection. arXiv preprint arXiv:2101.00438.

Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S. Gordon. 2011. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In 2011 AAAI Spring Symposium Series.

Sara Rosenthal, Noura Farra, and Preslav Nakov. 2017. SemEval-2017 task 4: Sentiment analysis in Twitter. In Proceedings ofthe 11th International Workshop on Semantic Evaluation (SemEval-2017), pages 502– 518, Vancouver, Canada. Association for Computational Linguistics.

Sascha Rothe, Joshua Maynez, and Shashi Narayan. 2021. A thorough evaluation of task-specific pretraining for summarization. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 140–145, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Anastasia Shimorina and Anya Belz. 2022. The human evaluation datasheet: A template for recording details of human evaluation experiments in NLP. In Proceedings of the 2nd Workshop on Human Evaluation of NLP Systems (HumEval), pages 54–75, Dublin, Ireland. Association for Computational Linguistics.

Eyal Shnarch, Carlos Alzate, Lena Dankin, Martin Gleize, Yufang Hou, Leshem Choshen, Ranit Aharonov, and Noam Slonim. 2018. Will it blend? blending weak and strong labeled data in a neural network for argumentation mining. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 599–605, Melbourne, Australia. Association for Computational Linguistics.

Eyal Shnarch, Ariel Gera, Alon Halfon, Lena Dankin, Leshem Choshen, Ranit Aharonov, and Noam Slonim. 2022. Cluster & tune: Boost cold start performance in text classification. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7639–7653, Dublin, Ireland. Association for Computational Linguistics.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington, USA. Association for Computational Linguistics.

Cynthia Van Hee, Els Lefever, and Véronique Hoste. 2018. SemEval-2018 task 3: Irony detection in English tweets. In Proceedings of The 12th International Workshop on Semantic Evaluation, pages 39– 50, New Orleans, Louisiana. Association for Computational Linguistics.

Tu Vu, Tong Wang, Tsendsuren Munkhdalai, Alessandro Sordoni, Adam Trischler, Andrew Mattarella-Micke, Subhransu Maji, and Mohit Iyyer. 2020. Exploring and predicting transferability across NLP tasks. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7882–7926, Online. Association for Computational Linguistics.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019a. Superglue: A stickier benchmark for general-purpose language understanding systems. In NeurIPS.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings ofthe 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353–355, Brussels, Belgium. Association for Computational Linguistics.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019b. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations.

Alex Warstadt, Amanpreet Singh, and Samuel R. Bowman. 2019. Neural network acceptability judgments. Transactions of the Association for Computational Linguistics, 7:625–641.

Orion Weller, Kevin Seppi, and Matt Gardner. 2022. When to use multi-task learning vs intermediate finetuning for pre-trained encoder transfer learning. In Proceedings of the 60th Annual Meeting of the Associationfor Computational Linguistics (Volume 2: Short Papers), pages 272–282, Dublin, Ireland. Association for Computational Linguistics.

Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings ofthe 2018 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1112–1122, New Orleans, Louisiana. Association for Computational Linguistics.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. 2022. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. 2023. Resolving interference when merging models. ArXiv, abs/2306.01708.

Yue Yu, Simiao Zuo, Haoming Jiang, Wendi Ren, Tuo Zhao, and Chao Zhang. 2021. Fine-tuning pretrained language model with weak supervision: A contrastive-regularized self-training approach. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1063–1077, Online. Association for Computational Linguistics.

Marcos Zampieri, Shervin Malmasi, Preslav Nakov, Sara Rosenthal, Noura Farra, and Ritesh Kumar. 2019. Predicting the Type and Target of Offensive Posts in Social Media. In Proceedings of NAACL.

Jingqing Zhang, Yao Zhao, Mohammad Saleh, and Peter Liu. 2020. Pegasus: Pre-training with extracted gap-sentences for abstractive summarization. In International Conference on Machine Learning, pages 11328–11339. PMLR.

## A Task, Domain and Dataset

A task is defined by the input and the output. The input in our context is a text instance. The output could be, e.g., positive/negative/neutral for a sentiment analysis task, entailed/not-entailed for a textual entailment task, etc.

A domain is the type of text found in the examples, regardless of the labels. For example, a domain may be financial or comments for twitter.

A dataset for our purpose is a set of examples and their labels, divided into train, dev, and test folds. Being such, each dataset has a domain (characterizing its examples) and a task (for which its labels are annotated). Hence, we consider a subset of a dataset as an another dataset. Note that in the literature those terms are often not well defined and may even be interchangeable (Wang et al., 2019b).

## B Hyperparameters

For RoBERTa, we train for 10 epochs with linear learning rate 5e-5 with warm-up of 0.6% of training, batch size of 256, early stop epsilon 0.001 accuracy points, patience of 20 50 256 examples, validate every 50  256 examples, optimizer ADAMW(Loshchilov and Hutter, 2019), with weight decay 0.01 or 0. For BERT-baseuncased we use 2e-5 learning rate and never use decay. For T5 we use 1e-4 learning rate and train until early stopping occurs. We used A100 and V100 GPUs. Finetuning times vary, but all end within a couple of hours, most in less than an hour, some up to 8 hours.

## C Datasets used

All datasets could be downloaded from huggingface datasets. As we used groups of datasets we report here the full list of datasets they contain.

General GLUE: CoLA (Warstadt et al., 2019), SST2 (Socher et al., 2013), MRPC (Dolan and Brockett, 2005), QQP (data.quora.com/First-Quora-Dataset-Release-Question-Pairs), MNLI (Williams et al., 2018), QNLI Rajpurkar et al. 2016, RTE (Dagan et al., 2005; Bar-Haim et al., 2006; Giampiccolo et al., 2007; Bentivogli et al., 2009), WNLI (Levesque et al., 2011)

SuperGLUE: BoolQ (Clark et al., 2019), CB (de Marneffe et al., 2019), CoPA (Roemmele et al.,

2011), MULTIRC (Khashabi et al., 2018), WIC (Pilehvar and Camacho-Collados, 2019), WSC (Levesque et al., 2012)

NLI: MNLI (Williams et al., 2018), QNLI Rajpurkar et al. 2016, RTE (Dagan et al., 2005; Bar-Haim et al., 2006; Giampiccolo et al., 2007; Bentivogli et al., 2009), WNLI (Levesque et al., 2011), ESNLI (Camburu et al., 2018), adversarial NLI (Nie et al., 2020).

Twitter: EmoInt (Mohammad and Bravo-Marquez, 2017), Emoji (Barbieri et al., 2018), Irony (Van Hee et al., 2018), OffenseEval (Zampieri et al., 2019), HatEval (Basile et al., 2019) , Sentiment Analysis (Rosenthal et al., 2017)

Whenever the test set is held out (such as is GLUE and SuperGLUE), we extracted 1K or 10% of the training examples as test set, the smaller. We did not experiment with the small Stance (Mohammad et al., 2016) Twitter dataset originally found in TweetEval(Barbieri et al., 2020) to reduce noise. In the T5 experiment (§E) we used stance datasets as well, to have a large group. For MNLI we use the mismatched validation set as a test and the matched as a validation set.

## D In-house models/targets additional information

We report in Table 3 the score of finetuning RoBERTa without intertraining.

We also report the standard deviation for each cell in the experiment, i.e., taking into account differences due to finetuning the intermediate model and target dataset in figure 5. For each seed, we finetune the PT over the source dataset to produce the base model, and use it to finetune the target task. It means that each seed utilises a different base model. Note that §6.2 suggest different seeds may create base models with different quality. Note that to assess the variability of the averages reported in the main text (§4) the Standard Error of the Mean is required, this is the STD divided by the square of the number of seeds $S E = S T D / \sqrt { 5 }$

## E Models in the wild

## E.1 Models used

We collected manually 66 RoBERTa-base models. From their names most were finetuned from vanila

![](images/c2d88c68628c741b67fa3b93467a60b64a446b4ed2a1a256b18315a811ae8973.jpg)  
Figure 5: Standard deviation of in-house models/targets experiment. Rows correspond to intermediate models, generated based on 22 source datasets from the General, NLI and Twitter groups. Columns correspond to the same datasets, now being used as target datasets. Each value indicates standard deviation over 5 seeds.

<table><tr><td>Dataset</td><td>Mean</td><td>Std</td></tr><tr><td>MultiRC</td><td>61.07</td><td>2.01</td></tr><tr><td>QQP</td><td>90.92</td><td>0.29</td></tr><tr><td>WSC</td><td>63.46</td><td>0.00</td></tr><tr><td>MRPC</td><td>87.70</td><td>0.95</td></tr><tr><td>CoLA</td><td>83.11</td><td>1.34</td></tr><tr><td>WIC</td><td>65.55</td><td>2.32</td></tr><tr><td>BoolQ</td><td>77.09</td><td>3.19</td></tr><tr><td>COPA</td><td>49.00</td><td>4.90</td></tr><tr><td>SST2</td><td>93.81</td><td>0.26</td></tr><tr><td>CB</td><td>70.36</td><td>3.11</td></tr><tr><td>QNLI</td><td>92.28</td><td>0.48</td></tr><tr><td>WNLI</td><td>56.34</td><td>0.00</td></tr><tr><td>RTE</td><td>72.42</td><td>0.93</td></tr><tr><td>ESNLI</td><td>91.05</td><td>0.18</td></tr><tr><td>ANLI</td><td>51.67</td><td>0.36</td></tr><tr><td>MNLI</td><td>87.07</td><td>0.23</td></tr><tr><td>Twitter Hate</td><td>52.30</td><td>1.03</td></tr><tr><td>Twitter Offensive</td><td>84.67</td><td>0.41</td></tr><tr><td>Twitter Irony</td><td>70.84</td><td>2.53</td></tr><tr><td>Twitter Sentiment</td><td>70.59</td><td>0.34</td></tr><tr><td>Twitter Emoji</td><td>46.32</td><td>0.56</td></tr><tr><td>Twitter Emotion</td><td>82.08</td><td>0.58</td></tr></table>

Table 3: Scores of finetuning RoBERTa without intertraining.

RoBERTa, but a few were trained from scratch. They vary in hyperparameters and training details, just as one can expect training approaches to vary between different researchers. We supply the full list of models in Tables 4,5 for RoBERTa and T5 respectively.

## E.2 Results and discussion

We report the full results in Fig. 6. T5 models are reported in §7 (off-the-sheld are on Twitter datasets only, as most of General and NLI were part of T5’s pretraining, in-house use t5.1.1). Seemingly, some traits of the training that we did not account for are important. It is exemplified by models that are associated with the same datasets but differ in their gains. For example cross-encoder implementations outperform other models and sentencetransformers underperform them.

Notably, most models are not useful for intertraining. Still, many models do.

The best model on average is MUPPET (Aghajanyan et al., 2021) which is a massive multitask learning approach. However, our results show that finetuning only on STS-B (rather than on 40 datasets) yielded similar results. We note that unlike MNLI which is known to be a good source (Phang et al., 2018a), STS-B was previously only considered as a target task only. The results suggest, it might be a good source.

We gathered the models by manually searching for ’RoBERTA-base’ models, ignoring ones that were working on languages other than English. It is possible we have missed models that did not clearly state their architecture as part of the model name. We are already aware of such models, for which the PT is not deducible from their title, for example those lately released by Juneja et al. (2022b).

## F Rank by Average over Targets

In §4.2 we show training one Linear Probe is enough to rank base models. Although tested on a large number of target datasets, presumably, this method does not always achieve accurate predictions. For example, the target domain might be so different that MNLI would not be relevant. As a more accurate alternative, one can use several datasets to provide a more reliable picture. We show that an average of finetuning gains over different datasets is a reliable way for choosing a base model. As in the simpler case of LP, this supports the decoupling.

We report in Table 6 the lost gain when choosing the highest models, ranked by average gain over the General group. This ranking method generalize well; The 1 or 3 best-ranked models are close to the best possible model overall for each target. For example, only 2 targets lose more than 1 point by choosing the top 3 models.

Practically, we suggest to rank either in this method or by LP. If some use one method and others choose another it might be hard to compare the two rankings. Thus, we report that in our experiments the best predictor of the average score by the LP score is $g _ { m } ^ { \bar { a } v g } = L P ( m , M N L I ) \cdot 0 . 0 8 2 2 -$ 0.940

## G Symmetry metric

To measure symmetry of a matrix M, we decompose it to it its symmetrical and skew symmetrical parts: $M = S + V$ where $S = { \textstyle { \frac { 1 } { 2 } } } ( M + M ^ { T } )$ and $\begin{array} { r } { V = \frac { 1 } { 2 } ( M - M ^ { T } ) } \end{array}$ . S is symmetrical: $S = S ^ { T }$ and V is skew-symmetrical: $\dot { V } = - V ^ { T }$ . The measure

<table><tr><td colspan="2">set</td><td>name</td><td>avg. gain over General</td><td>LP gain over MNLI</td></tr><tr><td>0</td><td>imdb _1</td><td>aychang/roberta-base-imdb</td><td>-5.91</td><td>-12.62</td></tr><tr><td>1</td><td>sentence_4</td><td>sentence-transformers/stsb-roberta-base</td><td>-5.84</td><td>2.59</td></tr><tr><td>2</td><td>models_1</td><td>textattack/roberta-base-ag-news</td><td>-5.73</td><td>-17.09</td></tr><tr><td>3</td><td>twitter_10</td><td>lucaordronneau/twitter-roberta-base-sentiment-...</td><td>-5.71</td><td>-9.76</td></tr><tr><td>4</td><td>sentence_5</td><td>sentence-transformers/roberta-base-nli-stsb-me...</td><td>-5.58</td><td>2.59</td></tr><tr><td>5</td><td>finance_0</td><td>zhayunduo/roberta-base-stocktwits-finetuned</td><td>-4.49</td><td>-12.60</td></tr><tr><td>6</td><td>sentence 2</td><td>sentence-transformers/msmarco-roberta-base-v3</td><td>-4.17</td><td>-8.82</td></tr><tr><td>7</td><td>twitter_8</td><td>cardiffnlp/twitter-roberta-base-emotion</td><td>-4.17</td><td>1.00</td></tr><tr><td>8</td><td>sentence_1</td><td>sentence-transformers/roberta-base-nli-mean-to...</td><td>-4.07</td><td>5.47</td></tr><tr><td>9</td><td>qa_3</td><td>navteca/roberta-base-squad2</td><td>-4.04</td><td>4.03</td></tr><tr><td>10</td><td>quora_0</td><td>cross-encoder/quora-roberta-base</td><td>-3.61</td><td>8.35</td></tr><tr><td>11</td><td>scratch_0</td><td>neoyipeng/twitter-roberta-base-sentiment-mlm-c...</td><td>-3.51</td><td>-3.13</td></tr><tr><td>12</td><td>models_5</td><td>neoyipeng/twitter-roberta-base-sentiment-mlm-c...</td><td>-3.51</td><td>-3.13</td></tr><tr><td>13</td><td>sentence_0</td><td>sentence-transformers/nli-roberta-base</td><td>-3.50</td><td>5.47</td></tr><tr><td>14</td><td>models_8</td><td>cointegrated/roberta-base-formality</td><td>-3.29</td><td>-4.96</td></tr><tr><td>15</td><td>legal_1</td><td>saibo/legal-roberta-base</td><td>-3.20</td><td>-1.76</td></tr><tr><td>16</td><td>twitter_12</td><td>cardiffnlp/twitter-roberta-base-stance-abortion</td><td>-2.95</td><td>-8.89</td></tr><tr><td>17</td><td>sst2 _0</td><td>Bhumika/roberta-base-finetuned-sst2</td><td>-2.77</td><td>-3.00</td></tr><tr><td>18</td><td>models_14</td><td>cestwc/roberta-base-unigram-ternary</td><td>-2.73</td><td>-8.39</td></tr><tr><td>19</td><td>models_11</td><td>mariagrandury/roberta-base-finetuned-sms-spam-...</td><td>-2.67</td><td>-5.90</td></tr><tr><td>20</td><td>qa_1</td><td>nlpconnect/roberta-base-squad2-nq</td><td>-2.57</td><td>3.67</td></tr><tr><td>21</td><td>twitter_13</td><td>bdotloh/twitter-roberta-base-finetuned-twitter...</td><td>-2.47</td><td>-3.12</td></tr><tr><td>22</td><td>models_0</td><td>textattack/roberta-base-CoLA</td><td>-2.38</td><td>-2.47</td></tr><tr><td>23</td><td>twitter_6</td><td>cardiffnlp/twitter-roberta-base-dec2021</td><td>-2.31</td><td>-11.45</td></tr><tr><td>24</td><td>twitter_5</td><td>cardiffnlp/twitter-roberta-base-mar2022</td><td>-2.21</td><td>-5.38</td></tr><tr><td>25</td><td>quora_1</td><td>navteca/quora-roberta-base</td><td>-2.13</td><td>8.35</td></tr><tr><td>26</td><td>models_16</td><td>hoanhkhoa/roberta-base-finetuned-ner</td><td>-2.12</td><td>-6.82</td></tr><tr><td>27</td><td>twitter_3</td><td>cardiffnlp/twitter-roberta-base-2021-124m</td><td>-2.05</td><td>-5.30</td></tr><tr><td>28</td><td>twitter_11</td><td>cardiffnlp/twitter-roberta-base-stance-climate</td><td>-1.98</td><td>-6.29</td></tr><tr><td>29</td><td>legal_0</td><td>akdeniz27/roberta-base-cuad</td><td>-1.90</td><td>-8.57</td></tr><tr><td>30</td><td>models_13</td><td>cardiffnlp/twitter-roberta-base-stance-feminist</td><td>-1.86</td><td>-4.25</td></tr><tr><td>31</td><td>imdb_2</td><td>aypan17/roberta-base-imdb</td><td>-1.86</td><td>-3.50</td></tr><tr><td>32</td><td>models_2</td><td>ghanashyamvtatti/roberta-fake-news</td><td>-1.81</td><td>-12.47</td></tr><tr><td>33</td><td>scratch_1</td><td>neoyipeng/twitter-roberta-base-sentiment-mlm-skep</td><td>-1.71</td><td>-8.73</td></tr><tr><td>34</td><td>models_12</td><td>surrey-nlp/roberta-base-finetuned-abbr</td><td>-1.65</td><td>-0.66</td></tr><tr><td>35</td><td>twitter_4</td><td>cardiffnlp/twitter-roberta-base-emoji</td><td>-1.63</td><td>-2.76</td></tr><tr><td>36</td><td>models_4</td><td>textattack/roberta-base-rotten-tomatoes</td><td>-1.61</td><td>3.73</td></tr><tr><td>37</td><td>legal_2</td><td>marshmellow77/roberta-base-cuad</td><td>-1.53</td><td>-8.57</td></tr><tr><td>38</td><td>legal_3</td><td>Rakib/roberta-base-on-cuad</td><td>-1.33</td><td>-2.48</td></tr><tr><td>39</td><td>twitter_7</td><td>cardiffnlp/twitter-roberta-base-sentiment</td><td>-1.24</td><td>3.03</td></tr><tr><td>40</td><td>twitter_9</td><td>cardiffnlp/twitter-roberta-base</td><td>-1.16</td><td>-1.49</td></tr><tr><td>41</td><td>models_15</td><td>thatdramebaazguy/roberta-base-wikimovies</td><td>-1.11</td><td>-1.76</td></tr><tr><td>42</td><td>finance_1</td><td>vanadhi/roberta-base-fiqa-flm-sq-flit</td><td>-1.09</td><td>-1.00</td></tr><tr><td>43</td><td>models_3</td><td>allenai/reviews</td><td>-1.01</td><td>-1.17</td></tr><tr><td>44</td><td>models_7</td><td>princeton-nlp/sup-simcse-roberta-base</td><td>-0.99</td><td>4.26</td></tr><tr><td>45</td><td>mrpc _1</td><td>ji-xin/roberta</td><td>-0.94</td><td>-1.67</td></tr><tr><td>46</td><td>twitter_1</td><td>cardiffnlp/twitter-roberta-base-offensive</td><td>-0.91</td><td>-2.79</td></tr><tr><td>47</td><td>sst2_1</td><td>textattack/roberta-base-SST-2</td><td>-0.84</td><td>3.83</td></tr><tr><td>48</td><td>sentence_3</td><td>sentence-transformers/stsb-roberta-base-v2</td><td>-0.80</td><td>1.68</td></tr><tr><td>49</td><td>twitter_2</td><td>cardiffnlp/twitter-roberta-base-irony</td><td>-0.69</td><td>0.46</td></tr><tr><td>50</td><td>imdb _0</td><td>textattack/roberta-base-imdb</td><td>-0.40</td><td>-3.64</td></tr><tr><td>51</td><td>models_6</td><td>VictorSanh/roberta-base-finetuned-yelp-polarity</td><td>-0.25</td><td>-0.66</td></tr><tr><td>52</td><td>models_10</td><td>gargam/roberta-base-crest</td><td>-0.12</td><td>-4.21</td></tr><tr><td>53</td><td>twitter_0</td><td>bhadresh-savani/roberta-base-emotion</td><td>-0.05</td><td>-4.61</td></tr><tr><td>54</td><td>qa_4</td><td>shahrukhx01/roberta-base-boolq</td><td>0.25</td><td>10.42</td></tr><tr><td>55</td><td>mrpc _0</td><td>textattack/roberta-base-MRPC</td><td>0.39</td><td>7.27</td></tr><tr><td>56</td><td>nli_3</td><td>textattack/roberta-base-RTE</td><td>0.42</td><td>14.82</td></tr><tr><td>57</td><td>nli_2</td><td>mujeensung/roberta-base_mnli_bc</td><td>0.48</td><td>23.43</td></tr><tr><td>58</td><td> $\mathtt { q a \_ 0 }$ </td><td>deepset/roberta-base-squad2-covid</td><td>0.50</td><td>-1.11</td></tr><tr><td>59</td><td> $\mathtt { q a \ 2 }$ </td><td>csarron/roberta-base-squad-v1</td><td>0.99</td><td>3.38</td></tr><tr><td>60</td><td> $\mathrm { s t s b \_ 0 }$ </td><td>textattack/roberta-base-STS-B</td><td>1.05</td><td>14.66</td></tr><tr><td>61 62</td><td> $\mathrm { m o d e l s \_ 9 }$ </td><td>textattack/roberta-base-QNLI</td><td>1.10</td><td>6.49</td></tr><tr><td>63</td><td> $\mathrm { n l i } \_ { 0 }$   $\mathrm { n l i } \_ { 1 }$ </td><td>textattack/roberta-base-MNLI</td><td>2.09 2.77</td><td>34.39 34.09</td></tr><tr><td>64</td><td></td><td>cross-encoder/nli-roberta-base</td><td>2.82</td><td>30.19</td></tr><tr><td>65</td><td> $\mathrm { s t s b } _ { - 1 }$  multitask_0</td><td>cross-encoder/stsb-roberta-base facebook/muppet-roberta-base</td><td>3.00</td><td>31.58</td></tr></table>

Table 4: RoBERTa models we used, collected from Hugging Face models hub. Models sorted by average gain over the General targets. 462

<table><tr><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>-1.79</td><td>-2.01</td><td>-2.00</td><td>-2.72</td><td>-4.66</td><td>2.15</td><td>-0.02</td><td>-2.03 -3.45</td><td>-20.94 -20.22</td><td>-0.92</td><td>-5.17</td><td>0.00</td><td>-25.96</td><td>-5.71 -5.58</td></tr><tr><td>Sentence 5 Finance 0</td><td>-3.67 -8.29</td><td>-3.57</td><td>-4.22</td><td>2.00</td><td>-2.66</td><td>-14.22</td><td>-2.00</td><td>-1.46</td><td>-0.48</td><td>-21.30</td><td>-0.34 -0.11</td><td>-5.64 -6.43</td><td>-33.80 2.82</td><td>0.00 -2.88</td><td>-4.49</td></tr><tr><td>Sentence 2</td><td>-2.57</td><td>-1.79</td><td>-1.92</td><td>-4.00</td><td>-2.23</td><td>-2.70</td><td>0.74</td><td>-0.48</td><td>0.18</td><td>-14.44</td><td>1.15</td><td>-5.02</td><td>-25.35</td><td>0.00</td><td>4.17</td></tr><tr><td>Twitter 8 Sentence 1</td><td>-5.41 -2.57</td><td>-1.79 5.36</td><td>-2.01 -2.97</td><td>-10.00 4.00</td><td>-0.34 -3.10</td><td>0.49 -3.19</td><td>-3.80 -3.16</td><td>-0.51 -0.16</td><td>0.13</td><td>-20.58 -15.52</td><td>-0.80</td><td>-1.10</td><td>-12.68</td><td>0.00</td><td>-4.17</td></tr><tr><td>QA3</td><td>-15.11</td><td>0.00</td><td>-6.14</td><td>4.00 -7.11</td><td></td><td>-1.96</td><td>-3.16</td><td>0.07</td><td>-1.18 -1.27</td><td>-25.63</td><td>-0.11 -0.57</td><td>-3.45 0.31</td><td>-30.99 0.00</td><td>0.00 0.00</td><td>-4.07 -4.04</td></tr><tr><td>Quora 0</td><td>-11.44</td><td>-21.43</td><td>-4.12 -8.00</td><td>0.66</td><td>1.47</td><td></td><td>2.81</td><td>-0.20</td><td>1.56</td><td>1.08</td><td>-1.61</td><td>1.10</td><td>-2.82</td><td>-9.62</td><td>-3.61</td></tr><tr><td>Models 5 Scratch 0</td><td>-5.87</td><td>3.57</td><td>-1.25</td><td>5.00</td><td>-10.75</td><td>0.74</td><td>-3.16</td><td>-5.42</td><td>-5.45</td><td>-17.33</td><td>-0.23</td><td>-3.76</td><td>-4.23</td><td>-0.96</td><td>-3.51</td></tr><tr><td>Sentence 0</td><td>-5.87 -1.80</td><td>3.57 7.14</td><td>-1.25 -2.30</td><td>5.00 4.00</td><td>-10.75 -3.82</td><td>0.74 -1.47</td><td>-3.16 -3.16</td><td>-5.42 0.09</td><td>-5.45 -3.37</td><td>-17.33 -19.86</td><td>-0.23 0.11</td><td>-3.76</td><td>-4.23 -21.13</td><td>-0.96 0.00</td><td>-3.51</td></tr><tr><td>Models 8</td><td>-0.86</td><td>-1.79</td><td>-0.86</td><td>4.00</td><td>0.27</td><td>0.00</td><td>2.06</td><td>-0.51</td><td>-0.59</td><td>-18.77</td><td>-0.23</td><td>-3.45 -2.82</td><td>0.00</td><td>-25.96</td><td>-3.50 -3.29</td></tr><tr><td>Legal 1</td><td>-4.74</td><td>0.00</td><td>-1.05</td><td>-8.00</td><td>-10.55</td><td>1.23</td><td>-0.21</td><td>-4.56</td><td>-5.76</td><td>-7.58</td><td>-1.26</td><td>-2.35</td><td>0.00</td><td>0.00</td><td>-3.20</td></tr><tr><td>Twitter 12 SST2 0</td><td>-6.33</td><td>-1.79</td><td>-1.25</td><td>0.00</td><td>-1.75</td><td>0.74</td><td>-3.16</td><td>-0.73</td><td>-1.66</td><td>-15.52</td><td>-0.23</td><td>-2.51</td><td>-7.04</td><td>0.00</td><td>-2.95</td></tr><tr><td>Models 15</td><td>-2.66</td><td>0.00</td><td>-1.73</td><td>2.00</td><td>-0.24</td><td>-0.49</td><td>2.02</td><td>-0.40</td><td>-0.15</td><td>-15.88</td><td>-0.46</td><td>-5.33</td><td>-15.49</td><td>0.00</td><td>-2.77</td></tr><tr><td>Models 12</td><td>-5.69 -0.55</td><td>-3.57 -1.79</td><td>-8.15 4.51</td><td>4.00 4.00</td><td>-3.51 -1.12</td><td>-0.25 -0.49</td><td>0.68 2.33</td><td>-1.46 0.33</td><td>-2.43 -3.21</td><td>-13.36 -3.25</td><td>-0.92 0.57</td><td>-3.61 -2.82</td><td>0.00 0.00</td><td>0.00</td><td>-2.73</td></tr><tr><td>QA1 Twitter 13</td><td>1.90</td><td>0.00</td><td>-4.89</td><td>4.00</td><td>0.36</td><td>-0.98</td><td>-0.72</td><td>0.18</td><td>-9.96</td><td>-27.08</td><td>-0.46</td><td>1.72</td><td>0.00</td><td>-26.92 0.00</td><td>-2.67 -2.57</td></tr><tr><td>Models 0 2.66</td><td>-4.98</td><td>0.00 -1.79</td><td>-2.01 0.96</td><td>-2.00 4.00</td><td>-2.23 -1.37</td><td>-0.49 3.19</td><td>-3.16 3.05</td><td>-1.24 -0.15</td><td>-2.71 -2.13</td><td>-9.39 -15.16</td><td>-0.46 -0.23</td><td>-1.72</td><td>4.23</td><td>0.00</td><td>-2.47</td></tr><tr><td>Twitter 6 -0.15 Twitter 5 -2.05</td><td></td><td>-3.57 -1.05</td><td></td><td>5.00</td><td>-1.00</td><td>0.00</td><td>0.02</td><td>-1.04</td><td>-0.17</td><td>-10.83</td><td>0.11</td><td>-3.76 -1.41</td><td>-19.72 -18.31</td><td>-2.88 0.00</td><td>-2.38 -2.31</td></tr><tr><td>Quora 1 -10.28</td><td></td><td>-3.57 -8.93</td><td>-2.21 -4.12</td><td>5.00 -5.00</td><td>-0.54 0.67</td><td>1.72 0.49</td><td>-0.66 1.79</td><td>-1.43 -0.20</td><td>-1.27 1.09</td><td>-7.94 2.89</td><td>0.57 -1.26</td><td>-0.31 0.31</td><td>-18.31 1.41</td><td>0.00</td><td>-2.21</td></tr><tr><td>Models 17 Twitter 3</td><td>-9.57</td><td>-1.79</td><td>-1.15</td><td>-4.00</td><td>0.03</td><td>1.96</td><td>-3.16</td><td>0.13</td><td>-0.83</td><td>-6.50</td><td>-0.46</td><td>-2.98</td><td>-1.41</td><td>-8.65 0.00</td><td>-2.13 -2.12</td></tr><tr><td>Twitter 11 -5.66</td><td>-0.03</td><td>-3.57 -1.15 -1.79 -2.11</td><td></td><td>3.00</td><td>-0.27</td><td>1.23 1.23</td><td>-0.87 -1.24</td><td>-1.21 -0.82</td><td>-0.03</td><td>-9.39</td><td>-0.11</td><td>-0.78</td><td>-15.49</td><td>0.00</td><td>-2.05</td></tr><tr><td>Legal 0 -5.20</td><td></td><td>1.79 -3.45</td><td>4.00 4.00</td><td>-1.57</td><td>-0.02 -1.72</td><td></td><td>-3.16</td><td>-1.23</td><td>-1.63 0.04</td><td>-16.25 -7.94</td><td>0.11 -0.92</td><td>-2.04</td><td>0.00</td><td>0.00</td><td>-1.98</td></tr><tr><td>Models 14 -7.03</td><td></td><td>-1.79 -1.15</td><td>4.00</td><td></td><td>-1.67</td><td>0.98</td><td>1.82</td><td>-1.03</td><td>-1.67</td><td>-16.97</td><td>-0.80</td><td>-8.78 -0.78</td><td>0.00 0.00</td><td>0.00</td><td>-1.90</td></tr><tr><td>Imdb 2 Models 2 -2.84</td><td>-0.83</td><td>3.57</td><td>-3.26</td><td>4.00</td><td>0.21</td><td>1.47</td><td>2.50</td><td>0.04</td><td>-0.20</td><td>-5.42</td><td>0.11</td><td>-1.25</td><td>0.00</td><td>0.00 -26.92</td><td>-1.86 -1.86</td></table>

Figure 6: Results of the off-the-shelf models/targets experiment. Rows correspond to off-the-shelf RoBERTa models obtained by downloading from HuggingFace model hub. Columns correspond to the General datasets group. Each value indicates intertraining gain w.r.t. using the PT model,

![](images/f77c99285fd35bfe67c56d32e4faa17a5522701e3d4e14476517673ad3a2319b.jpg)  
Figure 7: T5 off-the-shelf base models and targets. The intertrain gain over the pretrained model for each source (row) and target (column) datasets.

<table><tr><td></td><td>set</td><td>name</td><td>avg. gain across the General targets</td></tr><tr><td>0</td><td>QA_3</td><td>allenai/t5-small-squad2-next-word-generator-squad</td><td>-1.28</td></tr><tr><td>1</td><td>QA_5</td><td>allenai/t5-small-squad11</td><td>-0.77</td></tr><tr><td>2</td><td>Summarization_9</td><td>jazzisfuture/new_summary_t5_small</td><td>-0.25</td></tr><tr><td>3</td><td>Classification_2</td><td>mrm8488/t5-small-finetuned-imdb-sentiment</td><td>-0.22</td></tr><tr><td>4</td><td>Sentiment_1</td><td>mrm8488/t5-small-finetuned-imdb-sentiment</td><td>-0.22</td></tr><tr><td>5</td><td>Summarization_0</td><td>furyhawk/t5-small-finetuned-bbc-headline</td><td>-0.06</td></tr><tr><td>6</td><td>Classification_0</td><td>mrm8488/t5-small-finetuned-boolq</td><td>-0.03</td></tr><tr><td>7</td><td>Summarization_10</td><td>aseda/t5-small-finetuned-xsum</td><td>0.14</td></tr><tr><td>8</td><td> $\mathrm { S u m m a r i z a t i o n } \_ 3$ </td><td>mengsay/t5-small-finetuned-gigaword</td><td>0.23</td></tr><tr><td>9</td><td>Paraphrasing_1</td><td>hetpandya/t5-small-tapaco</td><td>0.34</td></tr><tr><td>10</td><td> $\mathrm { S u m m a r i z a t i o n \_ 7 }$ </td><td>bhuvaneswari/t5-small-text_summarization</td><td>0.36</td></tr><tr><td>11</td><td>Summarization_5</td><td>bochaowei/t5-small-finetuned-cnn-wei1</td><td>0.40</td></tr><tr><td>12</td><td>QA_1</td><td>allenai/unifiedqa-t5-small</td><td>0.43</td></tr><tr><td>13</td><td>QA_2</td><td>allenai/t5-small-squad2-question-generation</td><td>0.62</td></tr><tr><td>14</td><td>QG_0</td><td>allenai/t5-small-squad2-question-generation</td><td>0.62</td></tr><tr><td>15</td><td>QA_6</td><td>mrm8488/t5-small-finetuned-squadv2</td><td>0.90</td></tr><tr><td>16</td><td>Summarization_6</td><td>stevhliu/t5-small-finetuned-billsum-ca_test</td><td>0.94</td></tr><tr><td>17</td><td>Summarization_2</td><td>furyhawk/t5-small-finetuned-bbc</td><td>1.17</td></tr><tr><td>18</td><td>Summarization_4</td><td>bochaowei/t5-small-finetuned-cnn-wei0</td><td>1.36</td></tr><tr><td>19</td><td>Summarization_1</td><td>Frederick0291/t5-small-finetuned-billsum</td><td>1.51</td></tr><tr><td>20</td><td>Classification_1</td><td>mrm8488/t5-small-finetuned-emotion</td><td>1.54</td></tr><tr><td>21</td><td>Sentiment_0</td><td>mrm8488/t5-small-finetuned-emotion</td><td>1.54</td></tr><tr><td>22</td><td>Summarization_8</td><td>airKlizz/t5-small-multi-combine-wiki-news</td><td>1.79</td></tr><tr><td>23</td><td> $\mathrm { P a r a p h r a s i n g / Q A \_ 0 }$ </td><td>mrm8488/t5-small-finetuned-quora-for-paraphrasing</td><td>2.03</td></tr><tr><td>24</td><td> $\mathrm { P a r a p h r a s i n g / Q A \_ 4 }$ </td><td>hetpandya/t5-small-quora</td><td>2.05</td></tr></table>

Table 5: T5 models we used, collected from Hugging Face models hub. Models sorted by average gain over the General targets.

<table><tr><td>Models</td><td>@Top</td><td> $\operatorname { A v g } .$ </td><td>Max</td><td>number of datasets</td></tr><tr><td rowspan="3">In-house</td><td></td><td></td><td></td><td> $\mathrm { w i t h } \ : l o s s _ { n } > 1$ </td></tr><tr><td> $\_ l o s s _ { 1 }$ </td><td>0.37</td><td>2.11</td><td>3/22</td></tr><tr><td> $\it { l o s s } _ { 3 }$ </td><td>0.2</td><td>1.15</td><td>1/22</td></tr><tr><td rowspan="2">Off-the-shelf</td><td>loss1</td><td>1.41</td><td>12.0</td><td>3/14</td></tr><tr><td> $\it { l o s s } _ { 3 }$ </td><td>0.29</td><td>1.44</td><td>2/14</td></tr></table>

Table 6: Lost Gain per target is minimal when choosing the highest models, ranked by average intertraining gain on General datasets. Results are reported when selecting top rank model or best of 3 top rank models (@Top column). Columns represent the aggregation of the lost gain: average, max and the number of target datasets that lose at least one point. Rows represent two sets of experiments, in-house (with 22 models and 22 target datasets) or off-the-shelf (with 66 base modes and 14 target datasets).

of symmetry s of the matrix M, considers the relations between S and $V , s = ( | S | { - } | V | ) / ( | S | { + } | V | )$ $s \in [ - 1 , 1 ]$ if s is close to -1 it means that M is almost skew-symmetric (or anti-symmetric), if it is almost 1, it means that M is almost symmetric. If it around zero, it means that it neither symmetrical or skew-symmetric.

## H U-shape

We analyse how the intertraining gains change when more target data is available. We find that while intertraining often improves results for small data size, the effect is decreasing with the size. Surprisingly, the decrease drops below zero and at some size increases again. This suggests an unexplained underlying behaviour, presumably of two competing effects, one that decreases gains with size and one that improves them (in general or towards 0). We produce three examples of the U-shape in Figures 8, 9, 10.

![](images/cab4b3ca6f7a91ccde1e6121d1a11ebfdbe53646cce1bc6520ee41183f4fc5a0.jpg)  
Figure 8: Gains of QNLI from intertraining with different amount of training data (X-axis) and different base models (lines).

![](images/457a0c4d549bfc3725965cb1990a8859769fcf21d7ad0fd9ebc19203639cc82c.jpg)  
Figure 9: Gains of SST2 from intertraining with different amount of training data (X-axis) and different base models (lines).

![](images/f78e78b712a26607b57e706d29cc28ad6f534e80b7d4f5ee0af3269653b079a4.jpg)  
Figure 10: Gains of WIC from intertraining with different amount of training data (X-axis) and different base models (lines).

## I Scores

We report the source and target scores of training on MNLI datasets with 20 seeds. Target scores are the average over the General datasets. In Fig. 11, we present the results. Evidently the two are not correlated.

## I.1 Forgetting

If PT’s success comes from honing the parameters, shifting from them and forgetting the knowledge gained during pretraining is inadvisable in general (Chen et al., 2020) and possibly for intertraining (Pruksachatkun et al., 2020). With more training data, comes also more forgetting. This may also explain why most source models have a negative gain and intertraining hurts. Despite that, we observed in §6.1 that more source data empowers intertraining and improves gains. Following that observation, we analyze the importance of forgetting to the choice of a source model.

One common practice that causes forgetting is weight decay (Hanson and Pratt, 1988; Loshchilov and Hutter, 2019) – a regularization term added to the model updates. The decay term penalizes large weights, shrinking $\mathrm { P T } { \bf \ ' } _ { \bf S }$ large weights that are not necessary for the target objective.

For this experiment we use the following experimental setup: ADAMW (Loshchilov and Hutter, 2019) optimizer with weight decay 1 for decay and 0 (ADAM; Kingma and Ba, 2014) otherwise. L2 regularization is 0.1, results with other rates showed similar tendencies with effect size corresponding to the rate.

We find intermediate models trained with weight decay to be worse, but only if the pretraining did not include weight decay. Specifically, RoBERTa had weight decay during pretraining and T5 had not. We consider $g _ { M N L I } ^ { a v g }$ the average gain when source is MNLI and targets are General with and without weight decay. With RoBERTa as PT, the gain with decay was slightly better than without, by 0.02 points, while with T5 decay lost 3.3 points. These changes were not reflected on the source task performance.

Second, we limit the forgetting by adding a regularization forcing the model not to be far from the pretrained model. This can be seen as the complement of the previous method, rather than update toward zero update toward the pretrained model. We find this method did not change much for MNLI (- 0.28), but for models that hurt overall performance they prevented some performance loss, e.g. QQP (4.3). We note that while this regularization did not improve the top base model’s gain, it did hurt the original finetuning on the source task (-5.2 points on MNLI). We further address the effect of source task score on base model quality in §6.2.

![](images/10fa9577a78cf6dbe34cbc013a8bb206df9a7cb0e6c83fcf92d8fcf35826f501.jpg)  
Figure 11: Source (MNLI) score against average score on General datasets after intertraining. Each point represents a different MNLI intermediate nodel trained on a different seed.

We also followed Kumar et al. (2022) method that should reduce forgetting with LP, but it had little effect.

All of the above findings imply that what determines a base model’s effectiveness may be hidden in training hyperparameters. For example training on the same data and achieving similar results on the source, may still get quite different results on the target, depending on whether weight decay was used.

## J Architectures

Figs. 12, 13 depict the gains of In-house models trained on General datasets over the General datasets. In Fig. 7 we report the gains from training on off-the-shelf T5 models. Interestingly, QQP which is known as a bad source for RoBERTa, is among the best intermediate models for T5. Presumably, this is due to different training, where T5 generates the paraphrases rather than picks between several ones.

On a similar note, many of the top base models train on non-classification tasks, such as paraphrasing and question answering. This implies that the model weights converge to something quite general, learning linguistic traits that are not all discarded during finetuning. We say those are linguistic, in the sense that the language, and perhaps common knowledge are what makes the datasets similar, the tasks are quite different.

![](images/a11828c0ddd9ca14a97b32406b1590bc2a0676c22b01bc449b9c34f4113b7f01.jpg)  
Figure 12: BERT General sources and targets. The intertrain gain over the pretrained model for each source (row) and target (column) datasets.

![](images/9cf4cd1fdf3760b161c5cc79dd12c90b3d036cdfe82f74bd8794e4b4510f6f16.jpg)  
Figure 13: T5 General sources and targets. The intertrain gain over the pretrained model for each source (row) and target (column) datasets.