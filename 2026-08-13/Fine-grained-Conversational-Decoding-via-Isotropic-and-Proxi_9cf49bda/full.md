# Fine-grained Conversational Decoding via Isotropic and Proximal Search

Yuxuan Yao Han Wu Qiling Xu Linqi Song

City University of Hong Kong

yuxuanyao3-c@my.cityu.edu.hk hanwu32-c@my.cityu.edu.hk qilingxu2-c@my.cityu.edu.hk linqi.song@cityu.edu.hk

## Abstract

General-purpose text decoding approaches are usually adopted for dialogue response generation. Although the quality of the generated responses can be improved with dialogue-specific encoding methods, conversational decoding methods are still under-explored. Inspired by Wu et al. (2023) that a good dialogue feature space should follow the rules of locality and isotropy, we present a fine-grained conversational decoding method, termed isotropic and proximal search (IPS). Our method is designed to generate the semantic-concentrated response, while still maintaining informativeness and discrimination against the context. Experiments show that our approach outperforms existing decoding strategies in the dialogue field across both automatic and human evaluation metrics. More in-depth analyses further confirm the effectiveness of our approach.

## 1 Introduction

Dialogue response generation (Li et al., 2017; Wang et al., 2020) aims to generate the utterance that forms a coherent and fluent continuation given a dialogue context. Generic text decoding strategies (Rieser et al., 2014; Ritter et al., 2011; Chen et al., 2017) are usually adopted to produce grammatical and contextual responses. As an independent technique, decoding strategy can also enhance the generation quality of large language models.

Existing text decoding methods have been explored in various generic text generation tasks, but lack tailoring for dialogue generation, e.g., capturing dialogue-specific features and generating an informative and discriminative dialogue response (Su et al., 2021; Wu et al., 2023). Early maximizationbased methods, e.g., greedy search (Li et al., 2016b) and beam search (Wiseman et al., 2017), may lead to dullness and degeneration (Fan et al., 2018; Holtzman et al., 2018). Later sampling-based improvements are proposed to tackle these problems, including top-k sampling (Fan et al., 2018) and nucleus search (Holtzman et al., 2018). While alleviating degeneration, these sampling methods introduce critical semantic inconsistency and are not aligned with human-written prefix (Basu et al., 2021). Specifically, a bunch of studies (Ethayarajh, 2019; Su and Collier, 2023) have asserted that the problem of anisotropy, i.e., a distribution pattern in the latent space with features occupying a narrow cone in the space, leads to inconsistency and degradation of the generation. Although contrastive search (Su et al., 2022) has been proposed correspondingly to mitigate the issue, as a generalized text decoding strategy, it still ignores dialoguespecific features, such as utterance dependencies and conversational structure information. Therefore, research on conversational decoding methods is warmly needed.

In this work, we propose a fine-grained conversational decoding method, namely isotropic and proximal search (IPS). Different from traditional approaches, we consider the previous tokens and contexts separately from a granular perspective. Acknowledging that locality and isotropy are two important properties for refining the dialogue feature space, we design our IPS following these rules: (i) the generated output should be selected from the most probable candidate set predicted by the dialogue model; (ii) the generated tokens in the same utterance should be proximal to each other for expressing a concentrated idea; and (iii) the newly generated utterance should be discriminative enough with respect to the context utterances. In this way, our method encourages informativeness and discrimination among different utterances as well as maintains a concentrated idea within an utterance. We evaluate our approach on two commonly-used dialogue datasets, DailyDialog (Li et al., 2017) in English and LCCC (Wang et al., 2020) in Chinese. Both human and automatic evaluation results, i.e., indicators based on GPT3.5, consistently show that IPS can generate more fluent, coherent, and human-like responses than existing decoding methods.

## 2 Methodology

## 2.1 Preliminary

Dialogue response generation Given a dialogue context $D = \{ u _ { 1 } , u _ { 2 } , . . . , u _ { N } \}$ composed of N utterances, where $u _ { i } = \left\{ x _ { i , 1 } , x _ { i , 2 } , . . . , x _ { i , | u _ { i } | } \right\}$ is a sequence of consecutive words, the task of dialogue response generation is to produce the continuation utterance $u _ { r } = \{ w _ { 1 } , w _ { 2 } , . . . , w _ { | u _ { r } | } \} , ( r = N + 1 )$

There are generally two key steps to finish the task, including context encoding and response decoding. For the first step, we obtain the context representations H from the language model by concatenating the utterances into a sequence.

$$
\mathbf { H } = \mathrm { P r L M } ( u _ { 1 } \mathrm { [ E O U ] } u _ { 2 } \mathrm { [ E O U ] } \dots u _ { N } \mathrm { [ E O U ] } ) ,
$$

where [EOU] is the special token inserted as the last token of each utterance.

For the decoding step, the response is generally produced in an auto-regressive manner as follows

$$
p ( w _ { 1 : | u _ { r } | } ) = \prod _ { i = 1 } ^ { | u _ { r } | } p ( w _ { i } | w _ { < i } , D )\tag{1}
$$

Dialogue modeling Wu et al. (2023) has demonstrated that locality and isotropy are two key properties for building a good conversational feature space. Specifically, locality encourages the model to aggregate the representations of tokens within an utterance while isotropy pushes away the representations of distinct utterances.

## 2.2 Isotropic and Proximal Search

We present a fine-grained conversational decoding method, i.e., isotropic and proximal search (IPS). Specifically, we expect the generated response to satisfy two requirements: 1) representations of the response tokens are nearby to convey a concentrated idea, saying proximity; 2) the response representation is discriminative to the context utterance representations, saying isotropy.

During the decoding stage, for proximal search, we try to select the candidate token having the shortest average distance to the existing generated tokens. For isotropic search, we try to choose the token that enables the response representation most discriminative to representations of context utterances. As the response representation cannot be determined during the decoding stage, we calculate it in an approximate way, i.e., averaging the representations of the already generated tokens, as follows:

$$
{ \bf h } _ { R T } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } { \bf h } _ { w _ { i } }\tag{2}
$$

where h $. R T$ is the response representation which will be dynamically updated along with the generation process, and $T$ is the number of already generated tokens.

Up to now, the problem changes to how to generate the first token for starting the isotropic and proximal search since the method is heavily dependent on the previous tokens. To address this problem, we attempt to finish the first n-steps generation by traditional decoding methods, such as beam search, top-k sampling or nucleus sampling. On the other hand, as IPS is essentially a deterministic decoding strategy, this solution also enables it to produce diverse responses by using different decoding strategies in the first n steps. Therefore, in each step t after the first sampling stage, we calculate the proximal and isotropic values as follows:

$$
\mathsf { p \_ v a l u e } _ { t } = \frac { 1 } { t - 1 } \sum _ { i = 1 } ^ { t - 1 } s ( \mathbf h _ { w _ { t } } , \mathbf h _ { w _ { i } } )\tag{3}
$$

$$
\mathrm { i } \mathrm { \_ v a l u e } _ { t } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s ( \mathbf { h } _ { R T } , \mathbf { h } _ { u _ { i } } )\tag{4}
$$

where s is the cosine similarity. $\mathbf { h } _ { u _ { i } }$ are the utterance representations obtained from the special token [EOU]. The proximal value measures the average distance between the candidate token and the already generated tokens while the isotropic value stands for the average similarity between the undergoing response representation and all utterance representations. Next, the selection of the candidate token $w _ { t }$ is formulated as,

$$
\begin{array} { r l } & { w _ { t } = \underset { w _ { t } \in V ^ { ( m ) } } { \mathrm { a r g m a x } } \{ \alpha \times \underbrace { p ( w _ { t } \mid w _ { < t } , D ) } _ { \mathrm { m o d e l c o n f d e n c e } } } \\ & { ~ + \left( 1 - \alpha \right) \times \underbrace { \left( \mathbf { p } _ { - } \mathrm { v a l u e } _ { t } - \mathrm { i } _ { - } \mathrm { v a l u e } _ { t } \right) } _ { \mathrm { i s o t r o p i c a n d p r o x i m a l p e n a l t y } } \} } \end{array}\tag{5}
$$

where $V ^ { ( m ) }$ is the set of top-m predictions from the model’s probability distribution $p ( w _ { t } \mid w _ { < t } , D )$ and $m ,$ is typically set as $4 \sim 8$ . In Eq. (5), the first term, model confidence, is the probability of the candidate $w _ { t }$ predicted by the model. The second term, isotropic and proximal penalty, aims to maximize the discrimination between the undergoing response and previous utterances and minimize the token difference within the response. The hyperparameter $\alpha \in [ 0 , 1 ]$ regulates the importance of these two components. When $\alpha = 1$ , our method degenerates to the greedy search method.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Strategy</td><td colspan="5">DailyDialog</td><td colspan="5">LCCC</td></tr><tr><td>BS↑</td><td>MV↑</td><td>GE↑</td><td>Distinct</td><td>Dis4 ↑</td><td>BS↑</td><td>MV↑</td><td>GE↑</td><td>Distinct</td><td>Dis4 ↑</td></tr><tr><td rowspan="6">BART</td><td>greedy</td><td>0.1275</td><td>0.569</td><td>2.17</td><td>Dis2 ↑ 0.344</td><td>0.776</td><td>0.0636</td><td>0.062</td><td>1.88</td><td>Dis2 ↑ 0.126</td><td>0.437</td></tr><tr><td>beam</td><td>0.1317</td><td>0.599</td><td>2.29</td><td>0.341</td><td>0.755</td><td>0.0639</td><td>0.145</td><td>1.91</td><td>0.155</td><td>0.466</td></tr><tr><td>top-k</td><td>0.1312</td><td>0.623</td><td>2.20</td><td>0.350</td><td>0.780</td><td>0.0648</td><td>0.154</td><td>1.94</td><td>0.152</td><td>0.487</td></tr><tr><td>nucleus</td><td>0.1298</td><td>0.642</td><td>2.34</td><td>0.352</td><td>0.791</td><td>0.0626</td><td>0.178</td><td>1.91</td><td>0.156</td><td>0.534</td></tr><tr><td>contrastive</td><td>0.1147</td><td>0.622</td><td>2.07</td><td>0.396</td><td>0.810</td><td>0.0538</td><td>0.205</td><td>1.90</td><td>0.190</td><td>0.583</td></tr><tr><td>IPS</td><td>0.1335</td><td>0.647</td><td>2.43</td><td>0.355</td><td>0.798</td><td>0.0653</td><td>0.212</td><td>1.98</td><td>0.176</td><td>0.540</td></tr><tr><td rowspan="6">SimCTG (ρ = 0.5)</td><td>greedy</td><td>0.1099</td><td>0.447</td><td>2.21</td><td>0.306</td><td>0.709</td><td>0.0678</td><td>0.088</td><td>1.82</td><td>0.137</td><td>0.470</td></tr><tr><td>beam</td><td>0.1196</td><td>0.556</td><td>2.27</td><td>0.314</td><td>0.713</td><td>0.0692</td><td>0.206</td><td>2.02</td><td>0.179</td><td>0.539</td></tr><tr><td>top-k</td><td>0.1169</td><td>0.544</td><td>2.06</td><td>0.322</td><td>0.733</td><td>0.0695</td><td>0.195</td><td>2.11</td><td>0.168</td><td>0.534</td></tr><tr><td>nucleus</td><td>0.1169</td><td>0.571</td><td>2.32</td><td>0.327</td><td>0.753</td><td>0.0680</td><td>0.223</td><td>2.10</td><td>0.169</td><td>0.575</td></tr><tr><td>contrastive</td><td>0.1123</td><td>0.608</td><td>2.17</td><td>0.395</td><td>0.807</td><td>0.0607</td><td>0.278</td><td>1.98</td><td>0.197</td><td>0.618</td></tr><tr><td>IPS</td><td>0.1293</td><td>0.628</td><td>2.36</td><td>0.359</td><td>0.787</td><td>0.0704</td><td>0.294</td><td>2.31</td><td>0.196</td><td>0.580</td></tr><tr><td rowspan="6">SimDRC  $( \delta = 0 . 7 ,$  α = 0.3)</td><td>greedy</td><td>0.1255</td><td>0.560</td><td>2.06</td><td>0.345</td><td>0.774</td><td>0.0699</td><td>0.090</td><td>2.21</td><td>0.136</td><td>0.471</td></tr><tr><td>beam</td><td>0.1315</td><td>0.632</td><td>2.18</td><td>0.338</td><td>0.745</td><td>0.0715</td><td>0.196</td><td>2.11</td><td>0.180</td><td>0.543</td></tr><tr><td>top-k</td><td>0.1068</td><td>0.648</td><td>2.20</td><td>0.345</td><td>0.773</td><td>0.0720</td><td>0.203</td><td>2.19</td><td>0.166</td><td>0.540</td></tr><tr><td>nucleus</td><td>0.1284</td><td>0.632</td><td>2.16</td><td>0.353</td><td>0.793</td><td>0.0697</td><td>0.226</td><td>1.88</td><td>0.166</td><td>0.569</td></tr><tr><td>contrastive</td><td>0.1174</td><td>0.653</td><td>2.16</td><td>0.397</td><td>0.819</td><td>0.0613</td><td>0.271</td><td>2.21</td><td>0.197</td><td>0.614</td></tr><tr><td>IPS</td><td>0.1336</td><td>0.665</td><td>2.46</td><td>0.366</td><td>0.800</td><td>0.0722</td><td>0.272</td><td>2.32</td><td>0.192</td><td>0.569</td></tr></table>

Table 1: Automatic evaluation results on DailyDialog and LCCC, where BS means F1 value of BERTScore (Zhang\* et al., 2020), MV represents MAUVE (Pillutla et al., 2021), and GE represents G-Eval (Liu et al., 2023).

We claim our method is fine-grained because the generic auto-regressive generation predicts the next token by jointly considering the already generated tokens $w _ { < t }$ and the context D, formulated as $p ( w _ { t } | w _ { < t } , D )$ while IPS splits these two factors. Specifically, proximity value only focuses on the effects of the already generated tokens, i.e., p\_value $\sim \ p ( w _ { t } | w _ { < t } )$ , and isotropy value pays more attention to the context, i.e., i\_value<sub>t</sub> $p ( w _ { t } | D , ( w _ { < t } ) )$ wherein $w _ { < t }$ is just used to obtain the undergoing response representation $\mathbf { h } _ { R T }$

## 3 Experiments

Dataset We evaluate our method on two commonly-used datasets, DailyDialog (Li et al., 2017) in English and LCCC (Wang et al., 2020) in Chinese. Both of them are open-domain multi-turn dialogue datasets, collected from social media. For LCCC, owing to the academic-level computing resource, we follow previous work (Su et al., 2022), and sample a subset of the dataset, consisting of 100,000 dialogue examples.

Baselines Following Wu et al. (2023), we use BART (Lewis et al., 2020) as our backbone. We evaluate the performance of decoding strategies with different models, including vanilla BART, BART with SimCTG (Su et al., 2022), and BART with SimDRC (Wu et al., 2023). We compare IPS to greedy search, beam search, top-k sampling (Fan et al., 2018), nucleus sampling (Holtzman et al., 2018) and contrastive search (Su et al., 2022).

Settings We fine-tune the models on DailyDialog and LCCC datasets for 6k steps and 7k steps, respectively. We use a batch size of 64 and truncate the training samples to a maximum length of 256. The parameters of the models are initialized from HuggingFace libraries and updated by Adam optimizer (Kingma and Ba, 2017) with a learning rate of 3e-5. We adopt the margin values of SimCTG and SimDRC suggested in their work, i.e., $\rho = 0 . 5$ for SimCTG and $\delta = 0 . 7 , \alpha = 0 . 3$ for SimDRC. We conduct the isotropic and proximal search with the first n = 2 steps adopting top-k sampling (k = 7). The weight α is 0.6. We run all experiments with five different seeds and report the average score.

Evaluation Metrics Traditional n-gram overlap and text matching metrics such as BLEU (Papineni et al., 2002) and ROUGE (Lin, 2004) are not proper to evaluate plausible output diversity for open-domain dialog systems. Therefore, for automatic evaluation, we choose the following metrics, including BERTScore (Zhang\* et al., 2020),

![](images/1717793a1432dc6c14cd5d6090e7c1e23ef1f6f8ad6808c84ee5c162264e888d.jpg)  
(a) Ablation on the first n steps

![](images/afb446e0d1f438ce06e7e643300e7a395305eba71d51b79bb695db5ad4544648.jpg)  
(b) Ablation on top-k sampling

![](images/f431411014c1b2df7cafd2db3d19dd2fd5dbbacdc4d2b2f56d1aff2502553ec7.jpg)  
(c) Ablation on the hyperparameter α  
Figure 1: Ablation study on the DailyDialog dataset.

MAUVE (Pillutla et al., 2021), Distinct2/4 (Li et al., 2016a), and G-Eval, an automatic evaluation metric based on GPT3.5 (Liu et al., 2023).

We also conduct a human evaluation with the help of recruited proficient English/Chinese speakers. We randomly sample 100 dialogue examples from DailyDialog and LCCC test sets. For each dialogue context, we generate responses using the aforementioned backbone models (BART, BART+SimCTG, BART+SimDRC) with six different inference strategies. Five annotators are hired independently to measure these samples. Annotators are instructed to give a score ranging from 1 to 5 over the following aspects, including fluency, informativeness, coherence, and semantic coverage<sup>1</sup>.

Results and Discussion Table 1 lists the automatic evaluation results of the different methods with different decoding strategies. Similar results can be also found in human evaluation, as shown in Table 2. We can see that the models, collaborating with IPS, can produce more semantically consistent(high BERTScores and MAUVE scores) and human-like (high G-Eval scores) responses. Although contrastive search can generate more novel and diverse tokens (high Distinct scores), it usually suffers from the problem of prediction deviation, i.e., the predicted token being weakly related to the main idea of the response. This is also in line with the worse performance of contrastive search on other metrics, such as BERTScore, and G-Eval, indicating that the diverse responses produced by contrastive search are not accurate and human-like enough. Different from contrastive search, IPS tries to concentrate on the core meaning of the response and express it clearly, thus a slightly lower Distinct score is acceptable and expected. Note that IPS still has better distinct scores than other traditional decoding methods since it encourages discrimination and isotropy among utterances.

Although IPS can be directly used with different models and achieve good performance, the models trained with SimDRC are the best testbed for IPS. We can see that SimDRC+IPS can mostly achieve the best performance across the board on both automatic and human evaluation. This is reasonable because the training process of SimDRC is greatly consistent with the search criterion of IPS, and they both push away the inter-utterance features and pull close the intra-utterance features.

Ablation Study Figure 1 shows the ablation studies on different components of the method, including the first n steps, the sampling strategy for the first n-step decoding, and the weight α. As shown in Figure 1(a), our method consistently outperforms the contrastive search no matter the number of first steps. We find some performance drops with the increase of the first-stage sampling steps. We think this is because more generic tokens are selected by traditional search methods, thus weakening the proximity and isotropy of the response. For strategies in the first n steps, we attempt beam search, top-k sampling, and nucleus sampling. We finally select top-k sampling as our first stage’s strategy owing to its better performance in the comparisons. Figure 1(b) shows the results of different k values adopted in top-k sampling. We can see that our method exceeds the baseline by a large margin when $k > 5$ . The effect of weight α is also studied, as shown in Figure 1(c). Our method consistently outperforms the baseline with the different weights, suggesting the robustness of our method.

Hyperparameter Analysis To explore the effects of isotropy and proximity, in our experiments, we introduced a hyperparameter $\beta$ to balance the $p _ { - }$ value and i\_value as:

$$
( 1 - \beta ) \times p _ { - } \mathrm { { v a l u e \ } } - \beta \times i \_ \mathrm { v a l u e }\tag{6}
$$

We tried the effects of $\beta$ ranging from 0.2 to 0.8. We surprisingly found that the balance of proximal value and isotropy value leads to the best performance, saying $\beta$ equals 0.5. This finding is a bit different from the observations in SimDRC(Wu et al., 2023) which suggests that larger isotropy loss weight is needed to balance the two properties in the training stage. We think this is because our method is a decoding strategy, rather than the training optimization process. The sparse isotropy values would not cause the model bias in the decoding stage. So, the harmonious balance of proximity and isotropy can be simply achieved by giving a moderate value of $\beta .$

## 4 Conclusion

In this work, we present a fine-grained conversational decoding strategy, namely isotropic and proximal search (IPS) to encourage the generation of isotropic and conversational tokens. Superior to existing decoding methods, IPS decouples the previous tokens and the context. Experiments show that our method achieves impressive performance on both automatic and human evaluation.

## Ackonwledgements

This work was supported in part by the InnoHK initiative, the Government of the HKSAR, Laboratory for AI-Powered Financial Technologies.

## Limitations

During the experiments, we found that for a single piece of data in the DailyDialog test set, traditional text decoding methods such as beam search, top-k sampling and beam search take less than 1 second, the contrastive search takes about 5.07s, and the decoding time required by our proposed IPS is about 2.16s. Although our approach takes longer than the traditional text decoding method, our calculation speed is obviously faster than contrastive search. How to further improve the computing speed is still the direction we need to work on.

## Ethics Statement

In this work, we use publicly released datasets to auxiliary our dialogue response generation. Generally, these previous works have considered ethical issues when creating the datasets. We have manually checked some samples for the datasets we used in this work, and do not find any obvious ethical concerns, such as violent or offensive content. We will also release the source decoding code with friendly instructions to support its correct use. However, we still need to emphasize that text generation is not as controllable as we think. It still would generate some novel or unexpected words occasionally. We may take actions to decrease generation diversity to alleviate this problem.

## References

Sourya Basu, Govardana Sachitanandam Ramachandran, Nitish Shirish Keskar, and Lav R. Varshney. 2021. Mirostat: A neural text decoding algorithm that directly controls perplexity. In International Conference on Learning Representations.

Hongshen Chen, Xiaorui Liu, Dawei Yin, and Jiliang Tang. 2017. A survey on dialogue systems. ACM SIGKDD Explorations Newsletter, 19(2):25–35.

Kawin Ethayarajh. 2019. How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 55–65, Hong Kong, China. Association for Computational Linguistics.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889–898, Melbourne, Australia. Association for Computational Linguistics.

Ari Holtzman, Jan Buys, Maxwell Forbes, Antoine Bosselut, David Golub, and Yejin Choi. 2018. Learning to write with cooperative discriminators. In Proceedings ofthe 56th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1638–1649, Melbourne, Australia. Association for Computational Linguistics.

Diederik P. Kingma and Jimmy Ba. 2017. Adam: A method for stochastic optimization.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 7871–7880, Online. Association for Computational Linguistics.

Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. 2016a. A diversity-promoting objective function for neural conversation models. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies,

pages 110–119, San Diego, California. Association for Computational Linguistics.

Jiwei Li, Will Monroe, Alan Ritter, Dan Jurafsky, Michel Galley, and Jianfeng Gao. 2016b. Deep reinforcement learning for dialogue generation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1192– 1202, Austin, Texas. Association for Computational Linguistics.

Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. 2017. DailyDialog: A manually labelled multi-turn dialogue dataset. In Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 986–995, Taipei, Taiwan. Asian Federation of Natural Language Processing.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: Nlg evaluation using gpt-4 with better human alignment.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th Annual Meeting ofthe Associationfor Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaid Harchaoui. 2021. MAUVE: Measuring the gap between neural text and human text using divergence frontiers. In Advances in Neural Information Processing Systems.

Verena Rieser, Oliver Lemon, and Simon Keizer. 2014. Natural language generation as incremental planning under uncertainty: Adaptive information presentation for statistical dialogue systems. IEEE/ACM Trans. Audio, Speech and Lang. Proc., 22(5):979994.

Alan Ritter, Colin Cherry, and William B. Dolan. 2011. Data-driven response generation in social media. In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing, pages 583– 593, Edinburgh, Scotland, UK. Association for Computational Linguistics.

Yixuan Su and Nigel Collier. 2023. Contrastive search is what you need for neural text generation. Transactions on Machine Learning Research.

Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier. 2022. A contrastive framework for neural text generation. In Advances in Neural Information Processing Systems.

Yixuan Su, Yan Wang, Deng Cai, Simon Baker, Anna Korhonen, and Nigel Collier. 2021. Prototype-tostyle: Dialogue generation with style-aware editing on retrieval memory. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:2152– 2161.

Yida Wang, Pei Ke, Yinhe Zheng, Kaili Huang, Yong Jiang, Xiaoyan Zhu, and Minlie Huang. 2020. A large-scale chinese short-text conversation dataset. In Natural Language Processing and Chinese Computing: 9th CCF International Conference, NLPCC 2020, Zhengzhou, China, October 1418, 2020, Proceedings, Part I, page 91103, Berlin, Heidelberg. Springer-Verlag.

Sam Wiseman, Stuart Shieber, and Alexander Rush. 2017. Challenges in data-to-document generation. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 2253–2263, Copenhagen, Denmark. Association for Computational Linguistics.

Han Wu, Haochen Tan, Mingjie Zhan, Gangming Zhao, Shaoqing Lu, Ding Liang, and Linqi Song. 2023. Learning locality and isotropy in dialogue modeling. In The Eleventh International Conference on Learning Representations.

Tianyi Zhang\*, Varsha Kishore\*, Felix Wu\*, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations.

## A Appendix

## A.1 Human Evaluation Instructions

Please rate the quality of the generated response based on the given dialogue context and the target response over the following aspects: (1) Fluency; (2) Informativeness; (3) Coherence; (4) Semantic Coverage. We provide some instructions for your rating.

## A.1.1 Fluency

This measures whether the generated text has no formatting problems, capitalization errors, or obviously ungrammatical sentences (e.g., fragments, missing components) that make the text difficult to read. The definitions of different scores are:

• 5: The text is fluent, grammatically correct, and has no errors. It is easy to read.

• 4: The text is grammatically correct but has a few spelling or capitalization errors, which does not affect your understanding.

• 3: The text has minor errors in both grammar and spelling. The errors slightly affect your understanding.

• 2: The text has major errors in both grammar and spelling. The errors make the text hard to read.

• 1: The text does not make sense and it is unreadable.

## A.1.2 Informativeness

This measures whether the generated text has diverse, informative, novel, or logically related content. The definitions of different scores are:

• 5: The text contains very diverse, informative, and novel content. It is enjoyable to read the text.

• 4: The text contains many informative and novel contents. (Choose this score when you hesitate between 3 and 5.)

• 3: The text contains some new information but also contains a few repetitions of the context.

• 2: The text only contains a few informative and new terms. (Choose this score when you hesitate between 1 and 3.)

• 1: The text is dull, repetitive, and has no new information. All contents are from the dialogue context.

## A.1.3 Coherence

This measures whether the generated text is semantically and factually consistent with the dialogue context. The definitions of different scores are:

• 5: The text is semantically, factually, and topically consistent with the dialogue context. All contents of the text are related to the source text or can be inferred from the source.

• 4: The text is very related to the context but has minor inconsistencies or contradictions that do not affect its overall relevance.

• 3: The text is related to the context but has some obvious inconsistencies and contradictions.

• 2: The text is slightly consistent with the context. Many inconsistencies and contradictions in the context can be found.

• 1: The text is totally inconsistent with the context. It semantically or factually contradicted the context.

## A.1.4 Semantic Coverage

This measures how many semantic content units from the target response are covered by the generated text. The definitions of different scores are:

• 5: All semantic content units of the target text can be found in the generated text. They are semantically consistent.

• 4: Most of the content units of the target text can be found from the generated text while a few missing units do not affect the overall coverage.

• 3: Some semantic content units can be found in the generated text but also miss some important units.

• 2: Most of the semantic content units are not covered. Only a few insignificant units can be found in the generated text.

• 1: The text does not have any overlapping semantic content units with the target text.

We recruit five human workers to annotate 3,600 samples. To make sure the workers are fairly paid, we pay 0.1 dollars for each sample. Therefore, the total amount spent on participant compensation is 360 dollars. The annotators take 24 hours to finish the task, suggesting the hourly wage for each worker is 15 dollars.

## A.2 More Details of the Task

## A.2.1 Evaluation of G-EVAL Score

The API we used to test G-EVAl is gpt-3.5-turbo, and the following is the prompt (Liu et al., 2023):

You will be given a conversation between two individuals. You will then be given one potential response for the next turn in the conversation. Your task is to give a final score for utterance. Please make sure you read and understand these instructions carefully.

The evaluation aspects are:

1. Engagingness: Is the response dull or interesting?

2. Naturalness: This measures whether the generated text has no formatting problems, capitalization errors, or obviously ungrammatical sentences to read.

3. Informativeness: This measures whether the generated text has diverse, informative, novel, or logically related content.

4. Coherence: This measures whether the generated text is semantically and factually consistent with the dialogue context.

The evaluation steps are:

1. Read the conversation, the corresponding label, and the response carefully.

2. Considering the above evaluation aspects, return a comprehensive final score ranging from 1 to 5 for each conversation.

3. Please only return 1 overall score, without any extra text descriptions. The return format should be like Score:1.

Now please read the following conversation, and return the score.

## A.2.2 More Experimental Results

Table 2 lists the results of human evaluation.

## A.3 Surface-level Analysis

## A.3.1 Score Distribution According to the Length of the Previous Context

Table 3 and Table 4 illustrate the relations between the context length and the human evaluation metrics while using the IPS (the above one) and beam search (the below one) decoding strategies. Observing the table, when the context length is particularly short (<10), we speculate that the context may consist of simple greetings or introductions, resulting in lower difficulty of generation and thus higher scores. When the context length varies in the range of approximately 10 to 40, due to differences in the complexity of context content and semantics, the scores exhibit a fluctuating trend. As the length continues to increase, the information provided by the previous context becomes richer, leading to improved effectiveness of both decoding methods. We also note that when faced with exceptionally long contexts, the generation quality of IPS is superior to the baselines.

## A.3.2 Utterance Length Analysis

Table 5 shows that both IPS and contrastive search tend to produce shorter sentences than traditional methods. We explain in the main text that by incorporating isotropy, achieved through contrastive search and IPS, redundancy is minimized, resulting in more concise generated text compared to previous methods. Considering the nature of the conversation, our IPS strategy expects proximity and does not enlarge the token distance in the same utterance, thus responses of IPS are slightly longer than that of contrastive search.

## A.4 Qualitative Analysis

## A.4.1 Instances Illustration

Some examples are presented to illustrate the effect of our IPS search.

In summation, according to Table 6 and Table 7, some qualitative observations are as follows:

• Replies generated by IPS are more natural and accurate.

• IPS tends to generate relatively concise responses.

• With more complex previous contexts, we observed that IPS does not prioritize shortening the length of response. IPS can generate responses that are more in line with the situation based on the characteristics of the conversation.

## A.5 Cosine Similarity Heatmap

To ensure utterances generated by our IPS are isotropic and proximal, and observe the representations produced by different decoding methods, we showcase the cosine similarity matrix of token representations correspondingly.

The larger color difference between different sentences represents greater isotropy, indicating discrimination among utterances; while the darker the color within the same sentence, the greater the proximity, conveying a more concentrated thought.

Choosing SimDRC as the backbone model, cosine similarity heatmaps of different inference methods are shown as follows. Tokens generated by IPS exhibit brighter colors in the heatmap, indicating increased proximity within the same sentence, while tokens from IPS showcase darker colors for different sentences, signifying greater isotropy. Contrastingly, traditional methods like beam search showed anisotropy(i.e. features occupy a narrow cone in the vector space, thus leading to the problem of degeneration.) in the figures.

![](images/51a906f08ad700f32132827e06a53d38fb2ece376a80b98aa998ae16af10c4f1.jpg)  
Figure 2: An image of a cosine similarity heatmap

<table><tr><td rowspan="2">Model</td><td rowspan="2">Strategy</td><td colspan="4">DailyDialog</td><td colspan="4">LCCC</td></tr><tr><td>Fluency</td><td>Info.</td><td>Coherence</td><td>SC</td><td>Fluency</td><td>Info.</td><td>Coherence</td><td>SC</td></tr><tr><td rowspan="6">BART</td><td>greedy</td><td>4.636</td><td>3.302</td><td>3.362</td><td>2.386</td><td>4.626</td><td>2.810</td><td>3.414</td><td>1.996</td></tr><tr><td>beam</td><td>4.584</td><td>3.362</td><td>3.508</td><td>2.390</td><td>4.452</td><td>2.950</td><td>3.278</td><td>2.054</td></tr><tr><td>top-k</td><td>4.634</td><td>3.416</td><td>3.554</td><td>2.432</td><td>4.554</td><td>2.954</td><td>3.384</td><td>2.072</td></tr><tr><td>nucleus</td><td>4.666</td><td>3.478</td><td>3.578</td><td>2.420</td><td>4.602</td><td>3.042</td><td>3.434</td><td>3.358</td></tr><tr><td>contrastive</td><td>4.678</td><td>3.406</td><td>3.476</td><td>2.416</td><td>4.560</td><td>2.888</td><td>3.358</td><td>2.118</td></tr><tr><td>IPS</td><td>4.710</td><td>3.562</td><td>3.768</td><td>2.566</td><td>4.718</td><td>3.152</td><td>3.622</td><td>2.184</td></tr><tr><td rowspan="6">SimCTG (ρ = 0.5)</td><td>greedy</td><td>4.652</td><td>3.288</td><td>3.362</td><td>2.394</td><td>4.622</td><td>2.884</td><td>3.580</td><td>2.026</td></tr><tr><td>beam</td><td>4.696</td><td>3.404</td><td>3.446</td><td>2.390</td><td>4.602</td><td>2.918</td><td>3.224</td><td>2.040</td></tr><tr><td>top-k</td><td>4.718</td><td>3.398</td><td>3.406</td><td>2.414</td><td>4.554</td><td>2.970</td><td>3.464</td><td>2.040</td></tr><tr><td>nucleus</td><td>4.734</td><td>3.372</td><td>3.348</td><td>2.386</td><td>4.582</td><td>2.938</td><td>3.548</td><td>2.068</td></tr><tr><td>contrastive</td><td>4.712</td><td>3.304</td><td>3.31</td><td>2.332</td><td>4.582</td><td>2.882</td><td>3.456</td><td>2.076</td></tr><tr><td>IPS</td><td>4.758</td><td>3.586</td><td>3.578</td><td>2.584</td><td>4.708</td><td>3.084</td><td>3.688</td><td>2.176</td></tr><tr><td rowspan="6">SimDRC (δ = 0.7, α = 0.3)</td><td>greedy</td><td>4.774</td><td>3.632</td><td>3.484</td><td>2.684</td><td>4.634</td><td>2.920</td><td>3.390</td><td>1.990</td></tr><tr><td>beam</td><td>4.820</td><td>3.580</td><td>3.404</td><td>2.586</td><td>4.620</td><td>2.920</td><td>3.632</td><td>2.034</td></tr><tr><td>top-k</td><td>4.854</td><td>3.614</td><td>3.396</td><td>2.618</td><td>4.590</td><td>2.950</td><td>3.572</td><td>2.048</td></tr><tr><td>nucleus</td><td>4.864</td><td>3.622</td><td>3.450</td><td>2.600</td><td>4.588</td><td>2.930</td><td>3.628</td><td>2.088</td></tr><tr><td>contrastive</td><td>4.872</td><td>3.692</td><td>3.798</td><td>2.694</td><td>4.594</td><td>2.890</td><td>3.582</td><td>1.984</td></tr><tr><td>IPS</td><td>4.892</td><td>3.768</td><td>3.942</td><td>2.826</td><td>4.712</td><td>3.120</td><td>3.734</td><td>2.332</td></tr></table>

Table 2: Results of human evaluation on DailyDialog and LCCC datasets, where SC means the semantic coverage, info. means informativeness.
<table><tr><td>Length</td><td>Fluency</td><td>Informativeness</td><td>Coherence</td><td>Semantic Coverage</td><td>num</td></tr><tr><td>[0,10)</td><td>4.94</td><td>4.56</td><td>4.67</td><td>3.06</td><td>9</td></tr><tr><td>[10,20)</td><td>4.93</td><td>3.5</td><td>3.62</td><td>2.77</td><td>13</td></tr><tr><td>[20,30)</td><td>4.73</td><td>4</td><td>4</td><td>3.09</td><td>11</td></tr><tr><td>[30,40)</td><td>4.75</td><td>3.56</td><td>4</td><td>2.67</td><td>12</td></tr><tr><td>[40,50)</td><td>4.85</td><td>3.67</td><td>3.56</td><td>2.28</td><td>9</td></tr><tr><td>[50,75)</td><td>4.95</td><td>3.52</td><td>3.61</td><td>2.45</td><td>17</td></tr><tr><td>[75,100)</td><td>4.8</td><td>3.52</td><td>3.79</td><td>2.84</td><td>15</td></tr><tr><td>over 100</td><td>4.93</td><td>3.79</td><td>4.21</td><td>2.96</td><td>14</td></tr></table>

Table 3: Relations between the context length and the human evaluation metrics while using the IPS.

## A.6 Examples of Generated Texts

For non-native Chinese speakers, translations of Table 9 are presented in Table 10. The quality of the LCCC dataset still requires optimization, as it contains numerous colloquial and slang expressions. We are not professional translators, and in our attempts, we noticed that the translated meanings sometimes diverged from the original Chinese. We apologize for the inconvenience.

<table><tr><td>Length</td><td>Fluency</td><td>Infomrativeness</td><td>Coherence</td><td>Semantic Coverage</td><td>num</td></tr><tr><td>[0,10)</td><td>4.89</td><td>4.44</td><td>4.33</td><td>2.78</td><td>9</td></tr><tr><td>[10,20)</td><td>4.77</td><td>3.31</td><td>3.15</td><td>2.46</td><td>13</td></tr><tr><td>[20,30)</td><td>4.55</td><td>3.86</td><td>3.41</td><td>3</td><td>11</td></tr><tr><td>[30,40)</td><td>4.88</td><td>3.29</td><td>3.13</td><td>2.42</td><td>12</td></tr><tr><td>[40,50)</td><td>4.88</td><td>3.11</td><td>2.89</td><td>1.83</td><td>9</td></tr><tr><td>[50,75)</td><td>4.82</td><td>3.43</td><td>3.17</td><td>2.43</td><td>17</td></tr><tr><td>[75,100)</td><td>4.78</td><td>3.48</td><td>3.45</td><td>2.5</td><td>15</td></tr><tr><td>over 100</td><td>4.93</td><td>3.5</td><td>3.64</td><td>2.43</td><td>14</td></tr></table>

Table 4: Relations between the context length and the human evaluation metrics while using the beam search.

<table><tr><td>Sentence_Len</td><td>dd-drc</td><td>dd-ctg</td><td>dd-bart</td><td>lccc-drc</td><td>lccc-ctg</td><td>lccc-bart</td></tr><tr><td>Greedy</td><td>11.33</td><td>11.39</td><td>10.98</td><td>14.05</td><td>13.76</td><td>14.60</td></tr><tr><td>Beam</td><td>12.56</td><td>12.65</td><td>12.24</td><td>11.89</td><td>11.69</td><td>11.88</td></tr><tr><td>Top_k</td><td>13.10</td><td>13.21</td><td>12.71</td><td>12.39</td><td>12.12</td><td>12.40</td></tr><tr><td>Top_p</td><td>12.76</td><td>13.60</td><td>13.17</td><td>12.55</td><td>12.33</td><td>12.56</td></tr><tr><td>Simctg</td><td>9.02</td><td>9.05</td><td>8.72</td><td>9.50</td><td>9.17</td><td>8.88</td></tr><tr><td>IPS</td><td>9.59</td><td>10.31</td><td>9.13</td><td>9.95</td><td>9.46</td><td>9.14</td></tr></table>

Table 5: Sentence-level Analysis

<table><tr><td rowspan="4">context</td><td>A: I have a cell phone in my car. Now it&#x27;s probably on the floor on the passenger side.</td></tr><tr><td>Why don&#x27;t you get it for me, and then I can call the police?</td></tr><tr><td>B: Alright A: It doesn&#x27;t work. It looks like it&#x27;s broken. I need to get to a hospital. You should drive me there.</td></tr><tr><td>B: Oh, yeah? It&#x27;s better if we make a police report first. Then you can go to the hospital.</td></tr><tr><td>label</td><td>&quot;Damn it! I&#x27;m injured here. We could wait all day for the police.&quot;</td></tr><tr><td>greedy</td><td>&quot;Alright. I&#x27;ll go get it.&quot;</td></tr><tr><td>beam/top-p/top-k</td><td>&quot;I don&#x27;t feel like making a police report. I need to get to the hospital.&quot;</td></tr><tr><td>contrastive</td><td>&quot;Alright. I&#x27;ll call the police.&quot;</td></tr><tr><td>IPS</td><td>&quot;Oh, I don&#x27;t think I can afford to wait for the police. I need to get to the hospital.&quot;</td></tr></table>

Table 6: Qualitative Instance A

<table><tr><td>context</td><td>A: Good afternoon! Did you call for a plumber? B: Yes, yes I did. Please come in! I&#x27; m so glad you came! This old house is falling apart!</td></tr><tr><td>greedy</td><td>Come on into the bathroom. See, here, there&#x27; s water leaking everywhere! &quot;Well, it&#x27; s leaking because of the leaking faucet in the shower.&quot;</td></tr><tr><td>beam/top-p</td><td>&quot;Well, it&#x27; s leaking because of the leaking faucet in the shower.&quot;</td></tr><tr><td>top-k</td><td>&quot;I&#x27; m sorry, madam, but I don&#x27;t think that water is leaking. I&#x27;m going to have to check it out.&quot;</td></tr><tr><td>contrastive</td><td>&quot;Well, it&#x27; s leaking all over! I don&#x27; t know what the plumber can do about it!&quot;</td></tr><tr><td>IPS</td><td>&quot;Well, it&#x27; s leaking because of the leaking faucet in the shower. I guess the water is backed up.&quot;</td></tr></table>

Table 7: Qualitative Instance B

<table><tr><td rowspan=1 colspan=2>A: I didn&#x27;t receive an unemployment check this week and need to know why.B: Did you mail us your last Continued Claim Form?content            A: I am afraid that I forgot to send it in.B: We cannot pay you unless you turn that form in every two weeks.A: Is it too late to mail the Continued Claim Form in now?</td></tr><tr><td rowspan=3 colspan=2>greedy            I will mail it in right now.beam search       It&#x27; s too late to mail it in now. You will receive your check this week.top-k sampling     I will mail it in right now if you can&#x27; t wait.It&#x27; s too late to mail it in now. You will receive your check this week.Yes, you can do it now.IPS               Yes, it is.</td></tr><tr><td rowspan=1 colspan=1>nucleus search</td></tr><tr><td rowspan=1 colspan=1>contrastive search</td><td rowspan=1 colspan=1>chYes</td></tr><tr><td rowspan=1 colspan=2>A: Hello, this is Mike, Kara.B: Mike! Good to hear from you. How are you?A: Everything is fine, and how are you?contentB: Things are going well with me.A: Kara, I had fun the other night at the movies and was wondering if you would like to go out again this Friday.</td></tr><tr><td rowspan=6 colspan=2>greedy            I was thinking of it, but I am not feeling too well. I have to go to the hospital for a couple of days.That sounds great. I would love to go out with youI would love to, but I have to work late on Friday.That sounds great. I would love to go out with you.Mike, I was thinking of it.IPS               That sounds great. I would love to.content            A: What do you think I should do, doctor?</td></tr><tr><td rowspan=1 colspan=1>beam searchThat sounds</td></tr><tr><td rowspan=2 colspan=1>top-k sampling I would love to, b</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>nucleus searchThat sound</td><td rowspan=1 colspan=1>nds great. I wou</td></tr><tr><td rowspan=1 colspan=1>contrastive search Mike, I</td></tr><tr><td rowspan=1 colspan=2>greedy            I think you should have a good rest for three days.beam search       I think you&#x27;d better have a good rest for three daystop-k sampling     I think you should go to see the head of the family.nucleus search     I think you&#x27;d better have a good rest for three days.contrastive search  Well, it&#x27;s hard to say. You look a little pale, don&#x27;t you?IPS               Well, you should have a good rest for three days. Then you&#x27;ll feel better.</td></tr></table>

Table 8: Results of generation texts of DailyDialog.

<table><tr><td>content</td><td>A: 姐做的好喝 B: 这句姐喜欢 A:如果做给俺喝，那绝对是美味啊，人间极品，佳肴中滴佳肴 B:家里有蕃茄但现在姐不想做哩</td></tr><tr><td>greedy beam search top-k sampling</td><td>我也喜欢喝，但是我不会做给你喝 我也不想做，我喜欢吃番茄 我也喜欢吃这种美味</td></tr><tr><td>nucleus search contrastive search IPS</td><td>我也不想做了，就是喜欢吃&quot; 我也不想做，就是喜欢吃这种东西而已 我也不想做，但是我喜欢吃 A:魔都都不冷了？</td></tr><tr><td>content greedy</td><td>B:这两天有点反常吧…估计过两天会冷下去… 我们这里也是</td></tr><tr><td>beam search top-k sampling</td><td>我们这里也是这样的天气</td></tr><tr><td>nucleus search</td><td>我们这里也是这样的天气 这两天有点冷，但是也不是特别冷</td></tr><tr><td>contrastive search IPS</td><td>这两天好冷啊，估计明天开始就冷</td></tr><tr><td>content</td><td>这两天天气不错，不过今天有点冷</td></tr><tr><td></td><td>A: 我饿了。</td></tr><tr><td>greedy</td><td>我也饿了。</td></tr><tr><td>beam search</td><td>我也想吃，但是没有时间。</td></tr><tr><td>top-k sampling</td><td>我饿了，我要去吃饭了。</td></tr><tr><td>nucleus search</td><td>你怎么这么早就睡了。</td></tr><tr><td></td><td></td></tr><tr><td>contrastive search</td><td>我也饿了。</td></tr><tr><td>IPS</td><td>我也饿了，你吃了吗？</td></tr></table>

Table 9: Results of generation texts of LCCC.

<table><tr><td>content</td><td>A: You are good at beverage B: I like your comment. A: If made for me to drink, it would definitely be delicious, an exquisite delicacy among delicacies.</td></tr><tr><td>greedy</td><td>B: There are tomatoes at home, but right now, I just don&#x27;t feel like dealing with them. I also like to drink, but I won&#x27;t make it for you to drink. I don&#x27;t want to cook either, I enjoy eating tomatoes.</td></tr><tr><td>beam search top-k sampling nucleus search</td><td>I also enjoy eating this delicious treat. I don&#x27;t want to make it anymore, I just like to eat it</td></tr><tr><td>contrastive search IPS</td><td>I don&#x27;t want to cook either, I simply enjoy eating this kind of thing. I don&#x27;t want to do it either, but I&#x27;d love to eat.</td></tr><tr><td>content</td><td>A: Is Shanghai not cold anymore? B: These two days have been a bit abnormal... I estimate it will get colder in the next two days</td></tr><tr><td>greedy beam search top-k sampling</td><td>We here the same We have the same weather here</td></tr><tr><td>nucleus search</td><td>We have the same weather here It&#x27;s been a bit cold these two days, but it&#x27;s not particularly cold either</td></tr><tr><td>contrastive search IPS</td><td>It&#x27;s been so cold these two days. It&#x27;s expected to start cold tomorrow The weather has been good these two days, but it&#x27;s a bit cold today</td></tr><tr><td>content</td><td></td></tr><tr><td></td><td>A: I&#x27;m hungry</td></tr><tr><td></td><td></td></tr><tr><td>greedy</td><td>I&#x27;m also hungry</td></tr><tr><td></td><td></td></tr><tr><td>beam search</td><td>I want to eat too, but I don&#x27;t have time</td></tr><tr><td></td><td></td></tr><tr><td>top-k sampling</td><td>I&#x27;m hungry, I&#x27;m going to eat now</td></tr><tr><td>nucleus search</td><td></td></tr><tr><td></td><td>Why did you go to bed so early</td></tr><tr><td>contrastive search</td><td>I&#x27;m also hungry</td></tr><tr><td>IPS</td><td>I&#x27;m also hungry, have you eaten?</td></tr><tr><td></td><td></td></tr></table>

Table 10: Translation of generation texts of LCCC.