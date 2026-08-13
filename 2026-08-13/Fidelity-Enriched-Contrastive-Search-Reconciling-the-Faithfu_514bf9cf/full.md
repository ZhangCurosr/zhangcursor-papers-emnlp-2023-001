# Fidelity-Enriched Contrastive Search: Reconciling the Faithfulness-Diversity Trade-Off in Text Generation

Wei-Lin Chen<sup>12\*</sup> Cheng-Kuang Wu<sup>1</sup> Hsin-Hsi Chen<sup>1</sup> Chung-Chi Chen<sup>2</sup>

<sup>1</sup>National Taiwan University, Taiwan <sup>2</sup>Artificial Intelligence Research Center, AIST, Japan wlchen@nlg.csie.ntu.edu.tw hhchen@ntu.edu.tw c.c.chen@acm.org

## Abstract

In this paper, we address the hallucination problem commonly found in natural language generation tasks. Language models often generate fluent and convincing content but lack consistency with the provided source, resulting in potential inaccuracies. We propose a new decoding method called Fidelity-Enriched Contrastive Search (FECS), which augments the Contrastive Search framework with contextaware regularization terms. FECS promotes tokens that are semantically similar to the provided source while penalizing repetitiveness in the generated text. We demonstrate its effectiveness across two tasks prone to hallucination: abstractive summarization and dialogue generation. Results show that FECS consistently enhances faithfulness across various language model sizes while maintaining output diversity comparable to well-performing decoding algorithms.<sup>1</sup>

## 1 Introduction

Language models (LMs) have achieved remarkable success in generating human-like text, fostering advancements across numerous Natural Language Processing (NLP) applications. Despite the fluent and seemingly convincing outputs produced by LMs, these models can occasionally generate content that is factually inconsistent with the provided source (Koehn and Knowles, 2017; Rohrbach et al., 2018; Raunak et al., 2021), an issue known as the hallucination problem (Maynez et al., 2020; Ji et al., 2023). Methods to mitigate hallucination have been explored from various facets, including data perspectives (Wang, 2019; Filippova, 2020; Shuster et al., 2021), model architectures (Cao et al., 2018; Aralikatte et al., 2021; Xiao and Wang, 2021), and training strategies (Huang et al., 2020; Chen et al., 2021; Li et al., 2021). In this work, we turn to a less investigated lens—decoding—to improve faithfulness,<sup>2</sup> and introduces a novel decoding method named Fidelity-Enriched Contrastive Search (FECS).

![](images/a51b74dcefb19b18a268c66fb60554c8f88cbee5430541cc2dd0a9218decf56a.jpg)  
Figure 1: Results on CNN-DailyMail show our proposed FECS mitigates hallucination (i.e., improves factuality) while maintaining diversity of the generated summarization.

Decoding algorithms can be categorized into deterministic and stochastic groups. Deterministic methods such as beam search and greedy decoding aim to generate the most probable text continuations. While these methods might appear to be less unfaithful, they are often degenerated. That is, the outputs are uninformative, monotonous, or repetitive (Li et al., 2016; Holtzman et al., 2019; Welleck et al., 2019). Conversely, stochastic methods such as top-k (Fan et al., 2018) and nucleus sampling (Holtzman et al., 2019) inject randomness into the generation process, thereby promoting the diversity. Yet, these sampling-based approaches often come at the cost of coherency and semantic consistency (Basu et al., 2020; Su et al., 2022; Su and Collier, 2023), where increasing the output diversity positively correlates with hallucinating (Dziri et al., 2021). To reconcile this faithfulness-diversity trade-off, we proposed FECS—a simple yet effective decoding strategy which extends the Contrastive Search framework (Su et al., 2022) and introduces context-aware regularization terms to enhance faithfulness and penalize degeneration. Specifically, a candidate token which exhibits (1) a great semantic similarity with tokens from the provided source and (2) a low semantic similarity with previously generated tokens is rewarded with a higher score to promote its selection. Importantly, FECS can be readily applied to existing LMs offthe-shelf, without requiring further training.

We evaluate FECS on two tasks particularly prone to text hallucination: abstractive summarization and dialogue generation (Ji et al., 2023). Experimental results show that FECS consistently improves faithfulness across various LM sizes while preserving a level of diversity comparable to predominant decoding algorithms.

## 2 Methodology

In this section, we present preliminary information on Contrastive Search (Su et al., 2022) before detailing our proposed FECS.

## 2.1 Preliminary

To address shortcomings in existing decoding methods, Su et al. (2022) propose Contrastive Search, a new decoding approach capable of generating diverse content without compromising coherency. At time step t, given an input $x _ { 0 : c + t : }$ , where $x _ { 0 : c }$ signifies the prefix context and $x _ { c : c + t }$ represents the previously generated tokens, Contrastive Search generates the next token $x _ { c + t }$ via the following formula:

$$
\begin{array} { r l } &  x _ { c + t } = \underset { v \in V ^ { ( k ) } } { \arg \operatorname* { m a x } } \Big \{ ( 1 - \alpha ) \times \underset { \mathrm { m o d e l ~ c o n f i d e n c e } } { \underbrace { p _ { \theta } \left( v \left| x _ { 0 : c + t } \right) \right.}  } \\ & { \left. \qquad - \alpha \times \underset { \mathrm { d e g e n e r a t i o n ~ p e n a l t y } } { \underbrace { \operatorname* { m a x } } } \right\} \Big \} } \end{array}
$$

Here, $V ^ { k }$ denotes a set of k candidate tokens with the top-k probability from the model’s prediction distribution $p _ { \theta } ( \cdot | x _ { 0 : c + t } )$ . The model confidence term represents the probability of the candidate token v, while the degeneration penalty term signifies the maximum value of the cosine similarity $s i m ( \cdot , \cdot )$ between candidate token v and all previously generated tokens $\{ x _ { c } , . . . , x _ { c + t - 1 } \}$

Specifically, $s i m ( \cdot , \cdot )$ employs the token representation $h _ { x _ { \mathrm { i } } }$ and $h _ { v }$ from the model’s last hidden state, calculated by appending v to $x _ { 0 : c + t }$ as model input. α serves as a pre-determined, nonnegative hyper-parameter; when α equals 0, Contrastive Search reduces to greedy decoding. Essentially, Contrastive Search preserves coherence by choosing outputs from the top-k probable candidates while also curbing degeneration behaviors such as repetitions, thereby promoting diversity.

## 2.2 Fidelity-Enriched Contrastive Search

Motivated by Contrastive Search, we extend this framework by integrating a faithfulness term that encourages factuality and reduces hallucination. Using the notations from Section 2.1, we define FECS as follows:

Consider an input $x _ { 0 : c + t }$ at time step t, where $x _ { 0 : c }$ represents the prefix context, and $x _ { c : c + t }$ is the previously generated tokens. We further decompose $x _ { 0 : c }$ into: (1) the prompts $x _ { 0 : s } ,$ , and (2) the provided source $x _ { s : c : }$ , which the output is expected to remain faithful to. FECS generates the next token $x _ { c + t }$ via the following formula:

$$
\begin{array} { r l } & { x _ { c + t } = \underset { v \in V ^ { ( k ) } } { \mathrm { a r g } \operatorname* { m a x } } \Big \{ ( 1 - \alpha - \beta ) \times \underset { \mathrm { m o d e l ~ c o n r d e n c e } } { \underline { { p _ { \theta } } } ( v \mid x _ { 0 : c + t } ) } } \\ & { \quad \quad - \alpha \times \underset { \underset { \mathrm { e \leq i \leq j \leq c + t - 1 } } { \underline { { \mathrm { m a x } } } } } { \mathrm { m a x } } \{ s i m ( h _ { v } , h _ { x _ { i } } ) \} } \\ & { \quad \quad \quad + \beta \times \underset { \underset { \mathrm { e \leq i \leq j \leq c - 1 } } { \operatorname* { m a x } } } { \mathrm { m a x } } \{ s i m ( h _ { v } , h _ { x _ { j } } ) \} \Big \} } \end{array}
$$

The newly introduced faithfulness term rewards candidate tokens exhibiting high semantic similarity to tokens in the source content. Specifically, the faithfulness term denotes the maximum value of the cosine similarity sim( , ) between the candidate token v and all source tokens $\{ x _ { s } , . . . , x _ { c - 1 } \}$ . Here, $\beta$ is also a pre-determined, non-negative hyperparameter.

## 3 Experimental Setup

## 3.1 Datasets, Models, and Configurations

We evaluate our method, FECS, on two tasks known for their susceptibility to hallucination issues: abstractive summarization and dialogue generation. For the abstractive summarization task, we adopt CNN-DailyMail (CNN-DM) dataset (Nallapati et al., 2016), a widely-used benchmark in several recent studies (Dong et al., 2020; Cao and Wang, 2021; Cao et al., 2020). The dialogue generation task employs the popular Wizard of Wikipedia (WoW) dataset (Dinan et al., 2018). The objective here is to generate responses based on given knowledge snippets, taken from Wikipedia, that are pertinent to the conversation topic.

<table><tr><td rowspan="2">Model Size</td><td rowspan="2">Method</td><td colspan="5">CNN-DM</td><td colspan="4">WoW</td></tr><tr><td>R-1</td><td>R-2</td><td>R-L</td><td>BERTSc.</td><td>FEQA</td><td>B-4</td><td>R-L</td><td>BERTSc.</td><td>Q2</td></tr><tr><td rowspan="5">1.3B</td><td>Greedy</td><td>27.89</td><td>12.14</td><td>20.37</td><td>86.54</td><td>32.38</td><td>3.76</td><td>11.44</td><td>74.40</td><td>24.37</td></tr><tr><td>Beam</td><td>28.10</td><td>14.14</td><td>20.35</td><td>84.34</td><td>23.59</td><td>7.65</td><td>17.33</td><td>76.51</td><td>36.10</td></tr><tr><td>Nucleus</td><td>20.58</td><td>5.25</td><td>13.82</td><td>84.34</td><td>15.54</td><td>1.54</td><td>10.72</td><td>72.27</td><td>12.97</td></tr><tr><td>Contrastive</td><td>30.06</td><td>11.74</td><td>20.80</td><td>86.70</td><td>32.73</td><td>4.50</td><td>15.89</td><td>74.57</td><td>25.42</td></tr><tr><td>FECS (ours)</td><td>30.06</td><td>13.07</td><td>21.80</td><td>87.02</td><td>39.87</td><td>5.37</td><td>14.73</td><td>77.59</td><td>32.08</td></tr><tr><td rowspan="5">2.7B</td><td>Greedy</td><td>28.61</td><td>12.15</td><td>20.99</td><td>86.81</td><td>37.78</td><td>4.14</td><td>13.33</td><td>70.71</td><td>26.39</td></tr><tr><td>Beam</td><td>28.83</td><td>14.28</td><td>20.71</td><td>86.63</td><td>20.89</td><td>7.64</td><td>18.79</td><td>76.58</td><td>41.26</td></tr><tr><td>Nucleus</td><td>24.48</td><td>7.14</td><td>16.73</td><td>85.62</td><td>22.62</td><td>1.46</td><td>11.19</td><td>72.19</td><td>12.60</td></tr><tr><td>Contrastive</td><td>30.33</td><td>12.17</td><td>21.38</td><td>87.08</td><td>38.38</td><td>3.80</td><td>16.32</td><td>73.63</td><td>27.52</td></tr><tr><td>FECS (ours)</td><td>28.74</td><td>12.56</td><td>21.45</td><td>87.49</td><td>45.75</td><td>9.32</td><td>22.42</td><td>75.27</td><td>45.10</td></tr><tr><td rowspan="5">6.7B / 6B</td><td>Greedy</td><td>33.77</td><td>14.59</td><td>23.95</td><td>87.47</td><td>42.46</td><td>0.27</td><td>4.48</td><td>67.79</td><td>7.14</td></tr><tr><td>Beam</td><td>29.99</td><td>14.77</td><td>21.18</td><td>86.70</td><td>24.59</td><td>0.15</td><td>4.46</td><td>74.86</td><td>9.15</td></tr><tr><td>Nucleus</td><td>27.14</td><td>8.11</td><td>17.93</td><td>85.96</td><td>22.75 40.75</td><td>1.31 0.87</td><td>9.06</td><td>71.21</td><td>13.22</td></tr><tr><td>Contrastive</td><td>33.45</td><td>13.08</td><td>23.07</td><td>87.33</td><td></td><td></td><td>9.89</td><td>72.60</td><td>14.13</td></tr><tr><td>FECS (ours)</td><td>34.80</td><td>15.08</td><td>24.86</td><td>87.75</td><td>52.01</td><td>2.48</td><td>10.32</td><td>75.03</td><td>23.12</td></tr></table>

Table 1: Experimental results comparing FECS with other decoding methods across model scales.

In our experiments involving abstractive summarization, we adopt OPT (Zhang et al., 2022) with three scales: 1.3B, 2.7B, and 6.7B. For dialogue generation, we follow the Few-Shot Bot approach (Madotto et al., 2021), using GPT-Neo 1.3B and 2.7B (Black et al., 2021), along with GPT-J 6B (Wang and Komatsuzaki, 2021). All experiments are conducted with few-shot prompting, using two shots.<sup>3</sup> We compare FECS with Contrastive Search, Greedy Decoding, Beam Search, and Nucleus Sampling. For Beam Search, we set the beam size to 4; for Nucleus Sampling, p = 0.95; and for Contrastive Search, $( k , \alpha ) = ( 4 , 0 . 6 )$ . For FECS, we retain the same α value as Contrastive Search, setting (k, α, β) = (4, 0.3, 0.3) without hyper-parameter tuning.

## 3.2 Evaluation Metrics

Our evaluation process employs the following metrics:

Standard Metrics. For assessing the quality of summarization, we employ ROUGE (Lin, 2004). For dialogue generation, we use ROUGE-L and

BLEU-4 (Papineni et al., 2002). In addition, we also report BERTScore (Zhang et al., 2019) on both tasks for a more advanced soft metric.

Faithfulness Metrics. To measure factuality in summarization, we use FEQA (Durmus et al., 2020) following prior studies (Aralikatte et al., 2021; Chen et al., 2021). Higher FEQA scores indicate greater faithfulness of the summary to the source article. For evaluating dialogue, we employ $Q ^ { 2 }$ (Honovich et al., 2021), a question-answering (QA) based metric designed for assessing factual consistency in knowledge-grounded dialogue generation. Both FEQA and $Q ^ { 2 }$ exhibit strong correlations with human judgments.

Diversity Metric. For both summarization and dialogue tasks, we evaluate the diversity of the generated text x by calculating

$$
\operatorname { d i v e r s i t y } ( x ) = \prod _ { n = 2 } ^ { 4 } ( 1 . 0 - { \frac { \mathrm { R e p } { - } n ( x ) } { 1 0 0 } } )
$$

where Rep-n(x) measures the proportion of n-gram repetitions in x, and is calculated as

$$
{ \mathrm { R e p } } { \cdot } n ( x ) = ( 1 - \frac { | \mathrm { u n i q u e } { \cdot } n { \cdot } \mathrm { g r a m } ( x ) | } { | \mathrm { t o t a l } { \cdot } n { \cdot } \mathrm { g r a m } ( x ) | } ) \times 1 0 0 
$$

A higher diversity score suggests the model outputs exhibit less degeneration (Welleck et al., 2019; Su et al., 2022).

<table><tr><td rowspan=1 colspan=1>Article</td></tr><tr><td rowspan=1 colspan=1>West Hamare discussing a deal forJamaican starlet</td></tr><tr><td rowspan=2 colspan=1>DeShane Beckfordafter he impressed on trial. The skilful</td></tr><tr><td rowspan=2 colspan=1>to train with West Ham&#x27;s academyearlier this monthandhas impressed coaches after spending two weeks with theclub. Beckford also has offers from clubs in Belgium. [...]The Hammers will have the cheapest pricing strategy inthe Barclays Premier League in a bid to fill the 54,000 ca-pacity stadium when they make the switch for the 2016-17season.</td></tr><tr><td rowspan=1 colspan=1>emyearlier this monthand</td></tr><tr><td rowspan=1 colspan=1>Summary by Contrastive Search</td></tr><tr><td rowspan=1 colspan=1>West Ham are discussing DeShane Beckford.Jamaican starletimpressed on trial atUpton Park.</td></tr><tr><td rowspan=1 colspan=1>Summary by FECS</td></tr><tr><td rowspan=1 colspan=1>West Ham are discussing a deal for Jamaican starletDeShane BeckfordBeckfordimpressed on trial atWest Hamearlier thismonth.</td></tr></table>

Table 2: An actual example of news summaries generated by Contrastive Search and FECS on an article from CNN-DailyMail. Text highlighted in green indicates factual information; red indicates hallucination not supported by the article.

## 4 Experimental Results

## 4.1 Faithfulness

Table 1 presents the results for abstractive summarization and dialogue generation. For abstractive summarization, FECS achieves substantial improvements on the factuality score across all scales, with 7.14%, 7.37%, and 9.55% increases for the 1.3B, 2.7B, and 6.7B models, respectively. Moreover, FECS records strong results in the ROUGE score and outperforms all other methods at the 6.7B scale. For dialogue generation, on the 1.3B scale, all stochastic algorithms, including FECS, fall short of Beam Search in most metrics. However, FECS surpasses other stochastic algorithms in terms of BLEU-4 and $Q ^ { 2 }$ . Upon scaling up to 2.7B and 6B, FECS outperforms all methods substantially in terms of BLEU-4, ROUGE-L, and Q<sup>2</sup>. Notably, the 6B model performs worse than its smaller counterparts, consistent with previous findings (Madotto et al., 2021).

Compared to Contrastive Search, FECS exhibits a superior ability to focus on entities within the source material, emphasizing factual information more comprehensively. As evident in Figure 2, FECS provides more complete information—comparing “Jamaican starlet DeShane Beckford” versus “DeShane Beckford”—and generates output more comprehensively, evidenced by Contrastive Search’s failure to produce the time phrase “earlier this month". Furthermore, when factual entities are already present in the previous output, the degeneration penalty can inadvertently increase hallucinations. For instance, the term “Upton Park” produced by Contrastive Search lacks support from the source, whereas the correct output should be the previously generated “West Ham”. In this case, FECS accurately reproduces “West Ham”. Building on the framework of Contrastive Search, FECS not only inherits its properties of coherency and diversity (avoidance of degeneration) but also fosters the utilization of tokens that faithfully represent the provided source content.

<table><tr><td>Dataset</td><td>Model Size</td><td>Faithfulness</td><td>Diversity</td></tr><tr><td rowspan="3">CNN-DM</td><td>1.3B</td><td>+21.83%</td><td>-5.00%</td></tr><tr><td>2.7B</td><td>+19.20%</td><td>-0.20%</td></tr><tr><td>6.7B</td><td>+27.63%</td><td>-1.10%</td></tr><tr><td rowspan="3">WoW</td><td>1.3B</td><td>+26.20%</td><td>-35.00%</td></tr><tr><td>2.7B</td><td>+63.88%</td><td>-11.20%</td></tr><tr><td>6B</td><td>+63.62%</td><td>-3.30%</td></tr></table>

Table 3: Relative improvements in faithfulness and reduction of diversity of FECS over Contrastive Search.

## 4.2 Diversity

As we discussed in Section 1, model outputs must balance faithfulness and diversity. To better understand the impact of our proposed faithfulness reward on these two facets in the context of the original Contrastive Search, we calculated the improvements in faithfulness and the reductions in diversity based on the results from both the proposed FECS and the Contrastive Search.<sup>4</sup> Table 3 presents these evaluations. With the CNN-DailyMail dataset, FECS notably enhances faithfulness while marginally affecting diversity. Especially when the model size exceeds 2.7B, the decrease in diversity ranges only from 0.2% to 1.1%. These findings suggest that FECS successfully negotiates the faithfulness-diversity trade-off in abstractive summarization. Contrastingly, in the Wizard of Wikipedia dataset, FECS shows greater improvements in faithfulness and lesser reductions in diversity as the model size increases. Specifically, when the model size reaches 6.7B, FECS demonstrates a 63.62% improvement in faithfulness and experiences a mere 3.3% decrease in diversity. This implies that FECS performs more effectively when larger LMs are employed in dialogue generation tasks.

<table><tr><td rowspan="2">Method</td><td colspan="3">CNN-DM</td><td colspan="3">WoW</td></tr><tr><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>1.3B</td><td>2.7B</td><td>6B</td></tr><tr><td>Greedy</td><td>1.32</td><td>2.66</td><td>2.42</td><td>1.79</td><td>2.58</td><td>3.84</td></tr><tr><td>Beam</td><td>3.32</td><td>5.73</td><td>5.15</td><td>2.41</td><td>3.41</td><td>4.76</td></tr><tr><td>Nucleus</td><td>1.31</td><td>2.52</td><td>2.34</td><td>1.78</td><td>2.69</td><td>3.79</td></tr><tr><td>Contrastive</td><td>3.55</td><td>6.47</td><td>6.53</td><td>2.84</td><td>4.34</td><td>5.27</td></tr><tr><td>FECS (ours)</td><td>4.20</td><td>7.47</td><td>8.16</td><td>2.91</td><td>4.29</td><td>5.28</td></tr></table>

Table 4: The averaged decoding speed (sec) per instance using different decoding methods across model scales. As observed, FECS is comparable to Contrastive Search.

## 4.3 Analysis

Latency. To assess the decoding latency of our proposed FECS objective, we report the average decoding time (sec) per instance in Table 4. The results are averaged across 100 randomly selected instances. As observed in both the dialogue generation and abstractive summarization tasks, FECS and Contrastive Search perform comparably and slightly slower than beam search. Greedy and nucleus are the fastest.

The role of α. To establish a more comprehensive baseline, we evaluate FECS against Contrastive Search with different values of α on the 6.7B model. Intuitively, a smaller α value (i.e., a lower degree of diversity) might contribute to a more factual performance. However, as shown in Table 5 lowering α only improves faithfulness marginally and with essentially the same rouge scores. On the contrary, FECS retains a high level of diversity and achieves superior performance on both FEQA and standard metrics, indicating the effectiveness of our newly introduced β term.

## 5 Human Evaluation

In addition to the automatic evaluation, we also perform human evaluation to assess the faithfulness of our proposed FECS on the abstractive summarization task. We compare FECS against Contrastive Search, and ask annotators to vote which response is considered more faithful to the provided source (i.e., the text to be summarized). Specifically, we randomly sample 20 instance for each of the three model sizes, with a total of 60 instances for the evaluation. More details including the full evaluation protocol are provided in Appendix A.2. We present the results in Figure 2. As observed, FECS shows superior results, recording more than 60% of the votes, and outperforms Contrastive Search with more than twice the votes. The results support the outcome of automatic evaluation, suggesting our proposed FECS is able to generated contents which are more faithful to the provided source.

<table><tr><td rowspan="2">Metric</td><td colspan="4">Contrastive Search α</td><td>FECS (α, β)</td></tr><tr><td>0.6</td><td>0.4</td><td>0.2</td><td>0.0</td><td>(0.3, 0.3)</td></tr><tr><td>R-1</td><td>33.45</td><td>34.14</td><td>33.92</td><td>33.77</td><td>34.80</td></tr><tr><td>R-2</td><td>13.08</td><td>14.17</td><td>14.43</td><td>14.59</td><td>15.08</td></tr><tr><td>R-L</td><td>23.07</td><td>23.91</td><td>23.97</td><td>23.95</td><td>24.86</td></tr><tr><td>Diversity</td><td>94.21</td><td>90.13</td><td>88.07</td><td>83.57</td><td>93.18</td></tr><tr><td>FEQA</td><td>40.75</td><td>41.12</td><td>42.37</td><td>42.46</td><td>52.01</td></tr></table>

Table 5: Comparison of FECS and Contrastive Search with different values of α.

![](images/bbe9586e1322f399a63d5d07444ce1aa53ea4a05fc0ec3f669f77eb6f29ea50b.jpg)  
Figure 2: Human evaluation results comparing the faithfulness of FECS against Contrastive Search(CS) on the abstractive summarization task. FECS outperforms Contrastive Search, receiving more than twice the votes.

## 6 Conclusion

This paper introduces a novel decoding approach, Fidelity-Enriched Contrastive Search (FECS), designed to enhance faithfulness in text generation. Our experimental results on abstractive summarization and dialogue generation demonstrated the efficacy of FECS. It consistently improved faithfulness across various LM scales while preserving a level of diversity that is comparable to other leading decoding algorithms. Particularly when using larger LMs, it notably enhances faithfulness with only a minor impact on diversity. This indicates that FECS performs effectively when larger LMs are employed in dialogue generation tasks. In the future, we plan to explore how FECS performs with different kinds of source content, including erroneous or ambiguous inputs.

## Limitations

Firstly, while FECS presents an improvement in faithfulness and diversity trade-off, its performance could be influenced by the quality of the source content. The assumption that source content is always correct and complete may not hold true in all scenarios, particularly in cases where the input data is ambiguous, incomplete, or erroneous. Secondly, the faithfulness assessment is primarily quantitative, based on FEQA and $Q ^ { 2 }$ established metrics. Although these metrics provide an essential standard for comparing models, they may not capture all nuanced aspects of faithfulness, such as the preservation of subtle implications or subjective information.

## Acknowledgments

We thank the reviewers for their insightful comments. This research was supported by JSPS KAKENHI Grant Number 23K16956 and a project JPNP20006, commissioned by the New Energy and Industrial Technology Development Organization (NEDO). This work was also partially supported by National Science and Technology Council, Taiwan, under grants MOST 110-2221-E-002-128-MY3, 110-2634-F-002-050-, and NSTC 111-2634-F-002- 023-, and Ministry of Education (MOE) in Taiwan, under grants NTU-112L900901.

## References

Rahul Aralikatte, Shashi Narayan, Joshua Maynez, Sascha Rothe, and Ryan McDonald. 2021. Focus attention: Promoting faithfulness and diversity in summarization. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6078–6095, Online. Association for Computational Linguistics.

Sourya Basu, Govardana Sachitanandam Ramachandran, Nitish Shirish Keskar, and Lav R Varshney. 2020. Mirostat: A neural text decoding algorithm that directly controls perplexity. arXiv preprint arXiv:2007.14966.

Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. 2021. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow. If you use this software, please cite it using these metadata.

Meng Cao, Yue Dong, Jiapeng Wu, and Jackie Chi Kit Cheung. 2020. Factual error correction for abstractive summarization models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6251–6258, Online. Association for Computational Linguistics.

Shuyang Cao and Lu Wang. 2021. CLIFF: Contrastive learning for improving faithfulness and factuality in abstractive summarization. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6633–6649, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Ziqiang Cao, Furu Wei, Wenjie Li, and Sujian Li. 2018. Faithful to the original: Fact aware neural abstractive summarization. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 32.

Sihao Chen, Fan Zhang, Kazoo Sone, and Dan Roth. 2021. Improving faithfulness in abstractive summarization with contrast candidate generation and selection. In Proceedings of the 2021 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 5935–5941, Online. Association for Computational Linguistics.

Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. 2018. Wizard of wikipedia: Knowledge-powered conversational agents. arXiv preprint arXiv:1811.01241.

Yue Dong, Shuohang Wang, Zhe Gan, Yu Cheng, Jackie Chi Kit Cheung, and Jingjing Liu. 2020. Multifact correction in abstractive text summarization. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9320–9331, Online. Association for Computational Linguistics.

Esin Durmus, He He, and Mona Diab. 2020. FEQA: A question answering evaluation framework for faithfulness assessment in abstractive summarization. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 5055– 5070, Online. Association for Computational Linguistics.

Nouha Dziri, Andrea Madotto, Osmar Zaïane, and Avishek Joey Bose. 2021. Neural path hunter: Reducing hallucination in dialogue systems via path grounding. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 2197–2214, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889–898, Melbourne, Australia. Association for Computational Linguistics.

Katja Filippova. 2020. Controlled hallucinations: Learning to generate faithfully from noisy data. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 864–870, Online. Association for Computational Linguistics.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2019. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751.

Or Honovich, Leshem Choshen, Roee Aharoni, Ella Neeman, Idan Szpektor, and Omri Abend. 2021. $q ^ { 2 } \colon$ Evaluating factual consistency in knowledgegrounded dialogues via question generation and question answering. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 7856–7870, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Luyang Huang, Lingfei Wu, and Lu Wang. 2020. Knowledge graph-augmented abstractive summarization with semantic-driven cloze reward. In Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics, pages 5094–5107, Online. Association for Computational Linguistics.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38.

Philipp Koehn and Rebecca Knowles. 2017. Six challenges for neural machine translation. In Proceedings of the First Workshop on Neural Machine Translation, pages 28–39, Vancouver. Association for Computational Linguistics.

Chenliang Li, Bin Bi, Ming Yan, Wei Wang, and Songfang Huang. 2021. Addressing semantic drift in generative question answering with auxiliary extraction. In Proceedings of the 59th Annual Meeting of the Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 942–947, Online. Association for Computational Linguistics.

Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. 2016. A diversity-promoting objective function for neural conversation models. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 110–119, San Diego, California. Association for Computational Linguistics.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Andrea Madotto, Zhaojiang Lin, Genta Indra Winata, and Pascale Fung. 2021. Few-shot bot: Promptbased learning for dialogue systems. arXiv preprint arXiv:2110.08118.

Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020. On faithfulness and factuality in abstractive summarization. In Proceedings of the 58th Annual Meeting of the Association for

Computational Linguistics, pages 1906–1919, Online. Association for Computational Linguistics.

Ramesh Nallapati, Bowen Zhou, Cicero dos Santos, Çaglar Gulçehre, and Bing Xiang. 2016.˘ Abstractive text summarization using sequence-to-sequence RNNs and beyond. In Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning, pages 280–290, Berlin, Germany. Association for Computational Linguistics.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Vikas Raunak, Arul Menezes, and Marcin Junczys-Dowmunt. 2021. The curious case of hallucinations in neural machine translation. In Proceedings of the 2021 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 1172–1183, Online. Association for Computational Linguistics.

Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. 2018. Object hallucination in image captioning. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4035–4045, Brussels, Belgium. Association for Computational Linguistics.

Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021. Retrieval augmentation reduces hallucination in conversation. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 3784–3803, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Yixuan Su and Nigel Collier. 2023. Contrastive search is what you need for neural text generation. Transactions on Machine Learning Research.

Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier. 2022. A contrastive framework for neural text generation. In Advances in Neural Information Processing Systems.

Ben Wang and Aran Komatsuzaki. 2021. Gpt-j-6b: A 6 billion parameter autoregressive language model.

Hongmin Wang. 2019. Revisiting challenges in data-totext generation with fact grounding. In Proceedings ofthe 12th International Conference on Natural Language Generation, pages 311–322, Tokyo, Japan. Association for Computational Linguistics.

Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. 2019. Neural text generation with unlikelihood training. arXiv preprint arXiv:1908.04319.

![](images/1741d8146271874ea8122108594e7bc60fd1b5f04f031eef1546feccd687365a.jpg)  
Figure 3: An example prompt of the CNN-DailyMail dataset for the abstractive summarzation task.

## A.2 Details of Human Evaluation

The full human evaluation protocol is presented in Figure 5. We invite three graduate-level students proficient in English for the evaluation for the annotations. As our task does not require specific domain expertise, the payment is determined by the

<table><tr><td rowspan="2">Model Size</td><td rowspan="2">Method</td><td colspan="4">CNN-DM</td><td colspan="4">WoW</td></tr><tr><td>Rep-2</td><td>Rep-3</td><td>Rep-4</td><td>Diversity</td><td>Rep-2</td><td>Rep-3</td><td>Rep-4</td><td>Diversity</td></tr><tr><td rowspan="5">1.3B</td><td>Greedy</td><td>16.22</td><td>12.80</td><td>11.75</td><td>64.47</td><td>55.33</td><td>54.89</td><td>55.21</td><td>9.03</td></tr><tr><td>Beam</td><td>9.82</td><td>5.65</td><td>4.43</td><td>81.32</td><td>41.22</td><td>41.28</td><td>41.98</td><td>20.03</td></tr><tr><td>Nucleus</td><td>5.33</td><td>2.06</td><td>1.41</td><td>91.41</td><td>3.31</td><td>1.17</td><td>0.63</td><td>94.96</td></tr><tr><td>Contrastive</td><td>5.68</td><td>2.82</td><td>2.07</td><td>89.76</td><td>6.13</td><td>4.13</td><td>3.52</td><td>86.83</td></tr><tr><td>FECS (ours)</td><td>7.60</td><td>4.37</td><td>3.45</td><td>85.31</td><td>17.91</td><td>17.04</td><td>17.08</td><td>56.47</td></tr><tr><td rowspan="5">2.7B</td><td>Greedy</td><td>19.65</td><td>16.98</td><td>16.22</td><td>55.89</td><td>36.57</td><td>34.46</td><td>33.52</td><td>27.64</td></tr><tr><td>Beam</td><td>8.72</td><td>4.76</td><td>3.69</td><td>83.73</td><td>30.35</td><td>29.22</td><td>29.05</td><td>34.98</td></tr><tr><td>Nucleus</td><td>5.79</td><td>3.07</td><td>2.42</td><td>89.11</td><td>2.74</td><td>0.92</td><td>0.49</td><td>95.89</td></tr><tr><td>Contrastive</td><td>4.13</td><td>1.82</td><td>1.25</td><td>92.95</td><td>3.22</td><td>2.03</td><td>1.67</td><td>93.23</td></tr><tr><td>FECS (ours)</td><td>4.40</td><td>1.84</td><td>1.15</td><td>92.76</td><td>7.10</td><td>5.78</td><td>5.40</td><td>82.80</td></tr><tr><td rowspan="5">6.7B /6B</td><td>Greedy</td><td>8.11</td><td>5.09</td><td>4.18</td><td>83.57</td><td>45.07</td><td>44.22</td><td>44.41</td><td>17.03</td></tr><tr><td>Beam</td><td>8.15</td><td>4.29</td><td>3.26</td><td>85.04</td><td>15.53</td><td>14.75</td><td>14.80</td><td>61.35</td></tr><tr><td>Nucleus</td><td>4.47</td><td>2.03</td><td>1.40</td><td>92.28</td><td>2.52</td><td>0.83</td><td>0.44</td><td>96.25</td></tr><tr><td>Contrastive</td><td>3.45</td><td>1.46</td><td>0.98</td><td>94.21</td><td>0.71</td><td>0.18</td><td>0.06</td><td>99.05</td></tr><tr><td>FECS (ours)</td><td>4.05</td><td>1.76</td><td>1.15</td><td>93.18</td><td>2.63</td><td>1.07</td><td>0.55</td><td>95.80</td></tr></table>

Table 6: The evaluation results of repetition and diversity on FECS and other decoding methods across model scales.

## Task: Abstractive Summarization

Given two summaries (Summary\_A and Summary\_B), you should determine which one is more faithful to the provided Source, and fill in “A” or “B” in the Faithful column.

Degree of faithfulness (from most faithful to least faithful)

1. All information presented in the summary can be supported by the source.

○ If there is a tie, choose the one with more correct information or is more comprehensive/complete.

2. The summary contains information which can not be supported by the source.

○ If there is a tie, choose the one with less information that can not be supported by the source.

3. The summary contains information which contradicts the source.

○ If there is a tie, choose the one with less information that contradicts the source.

4. If the two summaries are not rankable (e.g., they are exactly the same), please fill in “T” in the Faithful column.

Figure 5: The human evaluation protocol for the abstractive summarization task.