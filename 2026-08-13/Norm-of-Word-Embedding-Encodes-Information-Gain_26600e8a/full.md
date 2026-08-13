# Norm of Word Embedding Encodes Information Gain

Momose Oyama<sup>1,3</sup>

Sho Yokoi<sup>2,3</sup>

Hidetoshi Shimodaira<sup>1,3</sup>

<sup>1</sup>Kyoto University

<sup>2</sup>Tohoku University

<sup>3</sup>RIKEN AIP

oyama.momose@sys.i.kyoto-u.ac.jp,

yokoi@tohoku.ac.jp, shimo@i.kyoto-u.ac.jp

## Abstract

Distributed representations of words encode lexical semantic information, but what type of information is encoded and how? Focusing on the skip-gram with negative-sampling method, we found that the squared norm of static word embedding encodes the information gain conveyed by the word; the information gain is defined by the Kullback-Leibler divergence of the co-occurrence distribution of the word to the unigram distribution. Our findings are explained by the theoretical framework of the exponential family of probability distributions and confirmed through precise experiments that remove spurious correlations arising from word frequency. This theory also extends to contextualized word embeddings in language models or any neural networks with the softmax output layer. We also demonstrate that both the KL divergence and the squared norm of embedding provide a useful metric of the informativeness of a word in tasks such as keyword extraction, proper-noun discrimination, and hypernym discrimination.

## 1 Introduction

The strong connection between natural language processing and deep learning began with word embeddings (Mikolov et al., 2013; Pennington et al., 2014; Bojanowski et al., 2017; Schnabel et al., 2015). Even in today’s complex models, each word is initially converted into a vector in the first layer. One of the particularly interesting empirical findings about word embeddings is that the norm represents the relative importance of the word while the direction represents the meaning of the word (Schakel and Wilson, 2015; Khodak et al., 2018; Arefyev et al., 2018; Pagliardini et al., 2018; Yokoi et al., 2020).

This study focuses on the word embeddings obtained by the skip-gram with negative sampling (SGNS) model (Mikolov et al., 2013). We show theoretically and experimentally that the Euclidean norm of embedding for word w, denoted as $\| u _ { w } \| .$ is closely related to the Kullback-Leibler (KL) divergence of the co-occurrence distribution $p ( \cdot | w )$ of a word w for a fixed-width window to the unigram distribution $p ( \cdot )$ of the corpus, denoted as

<table><tr><td colspan="2">Top 10</td><td colspan="2">Bottom 10</td></tr><tr><td>word</td><td>KL(w)</td><td>word</td><td>KL(w)</td></tr><tr><td>rajonas</td><td>11.31</td><td>the</td><td>0.04</td></tr><tr><td>rajons</td><td>10.82</td><td>in</td><td>0.04</td></tr><tr><td>dicrostonyx</td><td>10.31</td><td>and</td><td>0.04</td></tr><tr><td>dasyprocta</td><td>10.27</td><td>of</td><td>0.05</td></tr><tr><td>stenella</td><td>10.24</td><td>a</td><td>0.07</td></tr><tr><td>pesce</td><td>10.22</td><td>to</td><td>0.09</td></tr><tr><td>audita</td><td>10.09</td><td>by</td><td>0.09</td></tr><tr><td>landesverband</td><td>10.05</td><td>with</td><td>0.10</td></tr><tr><td>auditum</td><td>9.96</td><td>for</td><td>0.10</td></tr><tr><td>factum</td><td>9.84</td><td>S</td><td>0.10</td></tr></table>

Table 1: Top 10 words and bottom 10 words sorted by the value of KL(w) in the text8 corpus with word frequency $n _ { w } \ge 1 0$

$$
\mathrm { K L } ( w ) : = \mathrm { K L } ( p ( \cdot | w ) \parallel p ( \cdot ) ) .
$$

In Bayesian inference, the expected KL divergence is called information gain. In this context, the prior distribution is p( ), and the posterior distribution is $p ( \cdot | w )$ . The information gain represents how much information we obtain about the context word distribution when observing w. Table 1 shows that the 10 highest values of KL(w) are given by contextspecific informative words, while the 10 lowest values are given by context-independent words.

Fig. 1 shows that $\| u _ { w } \| ^ { 2 }$ is almost linearly related to KL(w); this relationship holds also for a larger corpus of Wikipedia dump as shown in Appendix G. We prove in Section 4 that the square of the norm of the word embedding with a whitening-like transformation approximates the KL divergence<sup>1</sup>. The main results are explained by the theory of the exponential family of distributions (Barndorff-Nielsen, 2014; Efron, 1978, 2022; Amari, 1982).

![](images/509779f88c304f2cba97d7f2126337df1435800262de4984105252da085760d3.jpg)  
Figure 1: Linear relationship between the KL divergence and the squared norm of word embedding for the text8 corpus computed with 100 epochs. The color represents word frequency $n _ { w }$ . Plotted for all vocabulary words, but those with $n _ { w } < 1 0$ were discarded. A regression line was fitted to words with $n _ { w } > 1 0 ^ { 3 }$ . Other settings are explained in Section 4.2 and Appendix A.

Empirically, the KL divergence, and thus the norm of word embedding, are helpful for some NLP tasks. In other words, the notion of information gain, which is defined in terms of statistics and information theory, can be used directly as a metric of informativeness in language. We show this through experiments on the tasks of keyword extraction, proper-noun discrimination, and hypernym discrimination in Section 7.

In addition, we perform controlled experiments that correct for word frequency bias to strengthen the claim. The KL divergence is heavily influenced by the word frequency $n _ { w }$ , the number of times that word w appears in the corpus. Since the corpus size is finite, although often very large, the KL divergence calculated from the co-occurrence matrix of the corpus is influenced by the quantization error and the sampling error, especially for low-frequency words. The same is also true for the norm of word embedding. This results in bias due to word frequency, and a spurious relationship is observed between word frequency and other quantities. Therefore, in the experiments, we correct the word frequency bias of the KL divergence and the norm of word embedding.

The contributions of this paper are as follows:

• We showed theoretically and empirically that the squared norm of word embedding obtained by the SGNS model approximates the information gain of a word defined by the KL divergence. Furthermore, we have extended this theory to encompass contextualized embeddings in language models.

• We empirically showed that the bias-corrected KL divergence and the norm of word embedding are similarly good as a metric of word informativeness.

After providing related work (Section 2) and theoretical background (Section 3), we prove the theoretical main results in Section 4. In Section 5, we extend this theory to contextualized embeddings. We then explain the word frequency bias (Section 6) and evaluate KL(w) and $\| u _ { w } \| ^ { 2 }$ as a metric of word informativeness in the experiments of Section 7.

## 2 Related work

## 2.1 Norm of word embedding

Several studies empirically suggest that the norm of word embedding encodes the word informativeness. According to the additive compositionality of word vectors (Mitchell and Lapata, 2010), the norm of word embedding is considered to represent the importance of the word in a sentence because longer vectors have a larger influence on the vector sum. Moreover, it has been shown in Yokoi et al. (2020) that good performance of word mover’s distance is achieved in semantic textual similarity (STS) task when the word weights are set to the norm of word embedding, while the transport costs are set to the cosine similarity. Schakel and Wilson (2015) claimed that the norm of word embedding and the word frequency represent word significance and showed experimentally that proper nouns have embeddings with larger norms than function words. Also, it has been experimentally shown that the norm of word embedding is smaller for less informative tokens (Arefyev et al., 2018; Kobayashi et al., 2020).

## 2.2 Metrics of word informativeness

Keyword extraction. Keywords are expected to have relatively large amounts of information. Keyword extraction algorithms often use a metric of the “importance of words in a document” calculated by some methods, such as TF-IDF or word co-occurrence (Wartena et al., 2010). Matsuo and Ishizuka (2004) showed that the $\chi ^ { 2 }$ statistics computed from the word co-occurrence are useful for keyword extraction. The $\chi ^ { 2 }$ statistic is closely related to the KL divergence (Agresti, 2013) since $\chi ^ { 2 }$ statistic approximates the likelihood-ratio chisquared statistic $G ^ { 2 } = 2 n _ { w } \mathrm { K L } ( w )$ when each document is treated as a corpus.

Hypernym discrimination. The identification of hypernyms (superordinate words) and hyponyms (subordinate words) in word pairs, e.g., cat and munchkin, has been actively studied. Recent unsupervised hypernym discrimination methods are based on the idea that hyponyms are more informative than hypernyms and make discriminations by comparing a metric of the informativeness of words. Several metrics have been proposed, including the KL divergence of the co-occurrence distribution to the unigram distribution (Herbelot and Ganesalingam, 2013), the Shannon entropy (Shwartz et al., 2017), and the median entropy of context words (Santus et al., 2014).

Word frequency bias. Word frequency is a strong baseline metric for unsupervised hypernym discrimination. Discriminations based on several unsupervised methods with good task performance are highly correlated with those based simply on word frequency (Bott et al., 2021). KL divergence achieved 80% precision but did not outperform the word frequency (Herbelot and Ganesalingam, 2013). WeedsPrec (Weeds et al., 2004) and SLQS Row (Shwartz et al., 2017) correlate strongly with frequency-based predictions, calling for the need to examine the frequency bias in these methods.

## 3 Theoretical background

In this section, we describe the KL divergence (Section 3.2), the probability model of SGNS (Section 3.3), and the exponential family of distributions (Section 3.4) that are the background of our theoretical argument in the next section.

## 3.1 Preliminary

Probability distributions. We denote the probability of a word w in the corpus as $p ( w )$ and the unigram distribution of the corpus as $p ( \cdot )$ . Also, we denote the conditional probability of a word $w ^ { \prime }$ co-occurring with w within a fixed-width window as $p ( w ^ { \prime } | w )$ , and the co-occurrence distribution as $p ( \cdot | w )$ . Since these are probability distributions, $\begin{array} { r } { \sum _ { w \in V } p ( w ) = \sum _ { w ^ { \prime } \in V } p ( w ^ { \prime } | w ) = 1 } \end{array}$ , where $V$ is the vocabulary set of the corpus. The frequencyweighted average of $p ( \cdot | w )$ is again the unigram distribution $p ( \cdot )$ , that is,

$$
p ( \cdot ) = \sum _ { w \in V } p ( w ) p ( \cdot | w ) .\tag{1}
$$

Embeddings. SGNS learns two different embeddings with dimensions d for each word in $V { : }$ word embedding $u _ { w } \in \mathbb { R } ^ { d }$ for $w \in V$ and context embedding $v _ { w ^ { \prime } } \in \mathbb { R } ^ { d }$ for $w ^ { \prime } \in V$ . We denote the frequency-weighted averages of $u _ { w }$ and $v _ { w ^ { \prime } }$ as

$$
\bar { u } = \sum _ { w \in V } p ( w ) u _ { w } , \quad \bar { v } = \sum _ { w ^ { \prime } \in V } p ( w ^ { \prime } ) v _ { w ^ { \prime } } .\tag{2}
$$

We also use the centered vectors

$$
\hat { u } _ { w } : = u _ { w } - \bar { u } , \quad \hat { v } _ { w ^ { \prime } } : = v _ { w ^ { \prime } } - \bar { v } .
$$

3.2 KL divergence measures information gain The distributional semantics (Harris, 1954; Firth, 1957) suggests that “similar words will appear in similar contexts” (Brunila and LaViolette, 2022). This implies that the conditional probability distribution $p ( \cdot | w )$ represents the meaning of a word $w .$ The difference between $p ( \cdot | w )$ and the marginal distribution $p ( \cdot )$ can therefore capture the additional information obtained by observing w in a corpus.

A metric for such discrepancies of information is the KL divergence of $p ( \cdot | w )$ to $p ( \cdot )$ , defined as

$$
\mathrm { K L } ( p ( \cdot | w ) \parallel p ( \cdot ) ) = \sum _ { w ^ { \prime } \in V } p ( w ^ { \prime } | w ) \log \frac { p ( w ^ { \prime } | w ) } { p ( w ^ { \prime } ) } .
$$

In this paper, we denote it by $\operatorname { K L } ( w )$ and call it the KL divergence of word w. Since $p ( \cdot )$ is the prior distribution and $p ( \cdot | w )$ is the posterior distribution given the word $w ,$ KL(w) can be interpreted as the information gain of word w (Oladyshkin and Nowak, 2019). Since the joint distribution of $w ^ { \prime }$ and w is $p ( w ^ { \prime } , w ) = p ( w ^ { \prime } | w ) p ( w )$ , the expected value of $\operatorname { K L } ( w )$ is expressed as

$$
\begin{array} { r l } { { } } & { { \displaystyle \sum _ { w \in V } p ( w ) \mathrm { K L } ( w ) } } \\ { { } } & { { = \displaystyle \sum _ { w \in V } \displaystyle \sum _ { w ^ { \prime } \in V } p ( w ^ { \prime } , w ) \log \frac { p ( w ^ { \prime } , w ) } { p ( w ^ { \prime } ) p ( w ) } . } } \end{array}
$$

This is the mutual information $I ( W ^ { \prime } , W )$ of the two random variables $W ^ { \prime }$ and W that correspond to $w ^ { \prime }$ and w, respectively<sup>2</sup>. $I ( W ^ { \prime } , W )$ is often called information gain in the literature.

## 3.3 The probability model of SGNS

The SGNS training utilizes the Noise Contrastive Estimation (NCE) (Gutmann and Hyvärinen, 2012) to distinguish between $p ( \cdot | w )$ and the negative sampling distribution $q ( \cdot ) \ \propto \ p ( \cdot ) ^ { 3 / 4 }$ For each co-occurring word pair $( w , w ^ { \prime } )$ in the corpus, ν negative samples $\{ w _ { i } ^ { \prime \prime } \} _ { i = 1 } ^ { \nu }$ are generated, and we aim to classify the $\nu + 1$ samples $\{ w ^ { \prime } , w _ { 1 } ^ { \prime \prime } , \ldots , w _ { \nu } ^ { \prime \prime } \}$ as either a positive sample generated from $w ^ { \prime }$ $p ( w ^ { \prime } | w )$ or a negative sample generated from $w ^ { \prime \prime }$ 2 $q ( w ^ { \prime \prime } )$ . The objective of SGNS (Mikolov et al., 2013) involves computing the probability of $w ^ { \prime }$ being a positive sample using a kind of logistic regression model, which is expressed as follows (Gutmann and Hyvärinen, 2012):

$$
\frac { p ( w ^ { \prime } | w ) } { p ( w ^ { \prime } | w ) + \nu q ( w ^ { \prime } ) } = \frac { 1 } { 1 + e ^ { - \langle u _ { w } , v _ { w ^ { \prime } } \rangle } } .\tag{3}
$$

To gain a better understanding of this formula, we can cross-multiply both sides of (3) by the denominators:

$$
p ( w ^ { \prime } | w ) ( 1 + e ^ { - \langle u _ { w } , v _ { w ^ { \prime } } \rangle } ) = p ( w ^ { \prime } | w ) + \nu q ( w ^ { \prime } ) ,
$$

and rearrange it to obtain:

$$
p ( w ^ { \prime } | w ) = \nu q ( w ^ { \prime } ) e ^ { \langle u _ { w } , v _ { w ^ { \prime } } \rangle } .\tag{4}
$$

We assume that the co-occurrence distribution satisfies the probability model (4). This is achieved when the word embeddings $\{ u _ { w } \}$ and $\{ v _ { w ^ { \prime } } \}$ perfectly optimize the SGNS’s objective, whereas it holds only approximately in reality.

## 3.4 Exponential family of distributions

We can generalize (4) by considering an instance of the exponential family of distributions (Lehmann and Casella, 1998; Barndorff-Nielsen, 2014; Efron, 2022), given by

$$
p ( w ^ { \prime } | u ) : = q ( w ^ { \prime } ) \exp ( \langle u , v _ { w ^ { \prime } } \rangle - \psi ( u ) ) ,\tag{5}
$$

where $u \in \mathbb { R } ^ { d }$ is referred to as the natural parameter vector, $v _ { w ^ { \prime } } \in \mathbb { R } ^ { d }$ represents the sufficient statistics (treated as constant vectors here, while tunable parameters in SGNS model), and the normalizing function is defined as

$$
\psi ( u ) : = \log \sum _ { w ^ { \prime } \in V } q ( w ^ { \prime } ) \exp ( \langle u , v _ { w ^ { \prime } } \rangle ) ,
$$

ensuring that $\begin{array} { r } { \sum _ { w ^ { \prime } \in V } p ( w ^ { \prime } | u ) = 1 } \end{array}$ for any $u \in \mathbb { R } ^ { d }$ The SGNS model (4) is interpreted as a special case of the exponential family

$$
p ( w ^ { \prime } | w ) = p ( w ^ { \prime } | u _ { w } )
$$

for $\textit { u } = \textit { u } _ { w }$ with constraints $\psi ( u _ { w } ) = - $ log ν for $w \in V ;$ ; the model (5) is a curved exponential family when the parameter value u is constrained as $\psi ( u ) = - \log \nu _ { : }$ , but we do not assume it in the following argument.

This section outlines some well-known basic properties of the exponential family of distributions, which have been established in the literature (Barndorff-Nielsen, 2014; Efron, 1978, 2022; Amari, 1982). For ease of reference, we provide the derivations of these basic properties in Appendix J.

The expectation and the covariance matrix of $v _ { w ^ { \prime } }$ with respect to $w ^ { \prime } \sim p ( w ^ { \prime } | u )$ are calculated as the first and second derivatives of $\psi ( u )$ , respectively. Specifically, we have

$$
\eta ( u ) : = \frac { \partial \psi ( u ) } { \partial u } = \sum _ { w ^ { \prime } \in V } p ( w ^ { \prime } | u ) v _ { w ^ { \prime } } ,\tag{6}
$$

$$
\begin{array} { r l } & { G ( u ) : = \cfrac { \partial ^ { 2 } \psi ( u ) } { \partial u \partial u ^ { \top } } = } \\ & { \displaystyle \sum _ { w ^ { \prime } \in V } p ( w ^ { \prime } | u ) ( v _ { w ^ { \prime } } - \eta ( u ) ) ( v _ { w ^ { \prime } } - \eta ( u ) ) ^ { \top } . } \end{array}\tag{7}
$$

The KL divergence of $p ( \cdot | u _ { 1 } )$ to $p ( \cdot | u _ { 2 } )$ for two parameter values $u _ { 1 } , u _ { 2 } \in \mathbb { R } ^ { d }$ is expressed as

$$
\begin{array} { r l } & { \mathrm { K L } ( p ( \cdot | u _ { 1 } ) \parallel p ( \cdot | u _ { 2 } ) ) = } \\ & { \quad \quad \langle u _ { 1 } - u _ { 2 } , \eta ( u _ { 1 } ) \rangle - \psi ( u _ { 1 } ) + \psi ( u _ { 2 } ) . } \end{array}\tag{8}
$$

The KL divergence is interpreted as the squared distance between two parameter values when they are not very far from each other. In fact, the KL divergence (8) is expressed approximately as

$$
\begin{array} { r l } & { 2 \mathrm { K L } ( p ( \cdot | u _ { 1 } ) \parallel p ( \cdot | u _ { 2 } ) ) } \\ & { \qquad \simeq \left( u _ { 1 } - u _ { 2 } \right) ^ { \top } G ( u _ { i } ) \left( u _ { 1 } - u _ { 2 } \right) } \end{array}\tag{9}
$$

for $i = 1 , 2$ . Here, the equation holds approximately by ignoring higher order terms of $O ( \Vert u _ { 1 } -$ $u _ { 2 } \| ^ { 3 } )$ . For more details, refer to Amari (1982, p. 369), Efron (2022, p. 35). More generally, $G ( u )$ is the Fisher information metric, and (9) holds for a wide class of probability models (Amari, 1998).

## 4 Squared norm of word embedding approximates KL divergence

In this section, we theoretically explain the linear relationship between KL(w) and $\| u _ { w } \| ^ { 2 }$ observed in Fig. 1 by elaborating on additional details of the exponential family of distributions (Section 4.1) and experimentally confirm our theoretical results (Section 4.2).

## 4.1 Derivation of theoretical results

We assume that the unigram distribution is represented by a parameter vector $u _ { 0 } \in \mathbb { R } ^ { d }$ and

$$
p ( w ^ { \prime } ) = p ( w ^ { \prime } | u _ { 0 } ) .\tag{10}
$$

By substituting $u _ { 1 }$ and $u _ { 2 }$ with $u _ { w }$ and $u _ { 0 }$ respectively in (9), we obtain

$$
2 \mathrm { K L } ( w ) \simeq ( u _ { w } - u _ { 0 } ) ^ { \top } G ( u _ { w } - u _ { 0 } ) .\tag{11}
$$

Here $G : = G ( u _ { 0 } )$ is the covariance matrix of $v _ { w ^ { \prime } }$ with respect to $w ^ { \prime } \sim p ( w ^ { \prime } )$ , and we can easily compute it from (7) as

$$
G = \sum _ { w ^ { \prime } \in V } p ( w ^ { \prime } ) ( v _ { w ^ { \prime } } - \bar { v } ) \big ( v _ { w ^ { \prime } } - \bar { v } \big ) ^ { \top } ,
$$

because $\eta ( u _ { 0 } ) = \bar { v }$ from (2) and (6). However, it is important to note that the value of $u _ { 0 }$ is not trained in practice, and thus we need an estimate of $u _ { 0 }$ to compute $u _ { w } - u _ { 0 }$ on the right-hand side of (11).

We argue that $u _ { w } - u _ { 0 }$ in (11) can be replaced by $u _ { w } - \bar { u } = \hat { u } _ { w }$ so that

$$
\mathrm { 2 K L } ( w ) \simeq \hat { u } _ { w } ^ { \top } G \hat { u } _ { w } .\tag{12}
$$

For a formal derivation of (12), see Appendix K. Intuitively speaking, u¯ approximates $u _ { 0 }$ , because u¯ corresponds to $p ( \cdot )$ in the sense that u¯ is the weighted average of $u _ { w }$ as seen in (2), while $p ( \cdot )$ is the weighted average of $p ( \cdot | u _ { w } )$ as seen in (1).

To approximate $u _ { 0 } ,$ , we could also use $u _ { w }$ of some representative words instead of using u¯. We expect $u _ { 0 }$ to be very close to some $u _ { w }$ of stopwords such as $\cdot _ { a } ,$ and $\cdot _ { t h e } ,$ since their $p ( \cdot | u _ { w } )$ are expected to be very close to $p ( \cdot )$

Let us define a linear transform of the centered embedding as

$$
\tilde { u } _ { w } : = G ^ { \frac { 1 } { 2 } } \hat { u } _ { w } ,\tag{13}
$$

i.e., the whitening $o f u _ { w }$ with the context embed-$d i n g ^ { 3 }$ , then (12) is now expressed<sup>4</sup> as

$$
2 \mathrm { K L } ( w ) \simeq \| \tilde { u } _ { w } \| ^ { 2 } .\tag{14}
$$

Therefore, the square of the norm of the word embedding with the whitening-like transformation in (13) approximates the KL divergence.

![](images/cf5c7ede584fb191a85647b3123b9bf1fb1f200171db6dd59da5b95b98da4254.jpg)  
Figure 2: Confirmation of (11). The slope coefficient of 0.909, which is close to 1, indicates the validity of the theory.

![](images/00a67b9f8060fcf2eac709e9e2b34fff35543c136a54d7013a2071dabaae6496.jpg)  
Figure 3: Confirmation of (12) and (14). The slope coefficient of 1.384, which is close to 1, suggests the validity of the theory.

## 4.2 Experimental confirmation of theory

The theory explained so far was confirmed by an experiment on real data.

Settings. We used the text8 corpus (Mahoney, 2011) with the size of $N = 1 7 . 0 \times 1 0 ^ { 6 }$ tokens and $| V | = 2 5 4 \times 1 0 ^ { 3 }$ vocabulary words. We trained 300-dimensional word embeddings $( u _ { w } ) _ { w \in V }$ and $( v _ { w ^ { \prime } } ) _ { w ^ { \prime } \in V }$ by optimizing the objective of SGNS model (Mikolov et al., 2013). We also computed the KL divergence $( \mathrm { K L } ( w ) ) _ { w \in V }$ from the co-occurrence matrix. These embeddings and KL divergence are used throughout the paper. See $\mathsf { A p - }$ pendix A for the details of the settings.

Details of Fig. 1. First, look at the plot of KL(w) and $\| u _ { w } \| ^ { 2 }$ in Fig. 1 again. Although $u _ { w }$ are raw word embeddings without the transformation (13), we confirm good linearity $\| u _ { w } \| ^ { 2 } \propto \mathrm { K L } ( w )$ . A regression line was fitted to words with $n _ { w } > 1 0 ^ { 3 }$ where low-frequency words were not very stable and ignored. The coefficient of determination $R ^ { 2 } = 0 . 8 3 1$ indicates a very good fitting.

Adequacy of theoretical assumptions. In Fig. 1, the minimum value of $\operatorname { K L } ( w )$ is observed to be very close to zero. This indicates that $p ( \cdot | w )$ for the most frequent $w$ is very close to $p ( \cdot )$ in the corpus, and that the assumption (10) in Section 4.1 is adequate.

Confirmation of the theoretical results. To confirm the theory stated in (11), we thus estimated $u _ { 0 }$ as the frequency-weighted average of word vectors corresponding to the words the, of, and . These three words were selected as they are the top three words in the word frequency $n _ { w }$ . Then the correctness of (11) was verified in Fig. 2, where the slope coefficient is much closer to 1 than 0.048 of Fig. 1. Similarly, the fitting in Fig. 3 confirmed the theory stated in (12) and (14), where we replaced $u _ { 0 }$ by u¯.

Experiments on other embeddings. In $\mathsf { A p - }$ pendix G, the theory was verified by performing experiments using a larger corpus of Wikipedia dump (Wikimedia Foundation, 2021). In $\mathsf { A p - }$ pendix H, we also confirmed similar results using pre-trained fastText (Bojanowski et al., 2017) and SGNS (Li et al., 2017) embeddings.

## 5 Contextualized embeddings

The theory developed for static embeddings of the SGNS model is extended to contextualized embeddings in language models, or any neural networks with the softmax output layer.

## 5.1 Theory for language models

The final layer of language models with weights $v _ { w ^ { \prime } } \in \mathbb { R } ^ { d }$ and bias $b _ { w ^ { \prime } } \in \mathbb { R }$ is expressed for contextualized embedding $u \in \mathbb { R } ^ { d }$ as

$$
y _ { w ^ { \prime } } = \langle u , v _ { w ^ { \prime } } \rangle + b _ { w ^ { \prime } } ,
$$

and the probability of choosing the word $w ^ { \prime } \in V$ is calculated by the softmax function

$$
p _ { \mathrm { s o f t m a x } } ( w ^ { \prime } | u ) = \frac { e ^ { y _ { w ^ { \prime } } } } { \sum _ { w \in V } e ^ { y _ { w } } } .\tag{15}
$$

Comparing (15) with (5), the final layer is actually interpreted as the exponential family of distributions with $q ( w ^ { \prime } ) = \bar { e } ^ { b _ { w ^ { \prime } } } / \sum _ { w \in V } e ^ { b _ { w } }$ so that $p _ { \mathrm { s o f t m a x } } ( w ^ { \prime } | u ) = p ( w ^ { \prime } | u )$ Thus, the theory for SGNS based on the exponential family of distributions should hold for language models.

However, we need the following modifications to interpret the theory. Rather than representing the co-occurrence distribution, $p ( \cdot | u )$ now signifies the word distribution at a specific token position provided with the contextualized embedding $u .$ Instead of the frequency-weighted average $\bar { u } =$ $\textstyle \sum _ { w \in V } p ( w ) u _ { w } .$ , we redefine $\textstyle { \bar { u } } : = \sum _ { i = 1 } ^ { N } u _ { i } / N$ as the average over the contextualized embeddings $\{ u _ { i } \} _ { i = 1 } ^ { N }$ calculated from the training corpus of the language model. Here, $u _ { i }$ denotes the contextualized embedding computed for the i-th token of the training set of size N. The information gain of contextualized embedding u is

$$
\mathrm { K L } ( u ) : = \mathrm { K L } ( p ( \cdot | u ) \parallel p ( \cdot ) ) .
$$

With these modifications, all the arguments presented in Sections 3.4 and 4.1, along with their respective proofs, remain applicable in the same manner (Appendix L), and we have the main result (14) extended to contextualized embeddings as

$$
\begin{array} { r } { 2 \mathrm { K L } ( u ) \simeq \lVert \tilde { u } \rVert ^ { 2 } , } \end{array}\tag{16}
$$

where the contextualized version of the centering and whitening are expressed as ${ \hat { u } } : = u - { \bar { u } }$ and $\tilde { u } : = G ^ { \frac { 1 } { 2 } } \hat { u }$ , respectively.

## 5.2 Experimental confirmation of theory

![](images/bdf868e59c3ba42e2edb257f9d7ebc2fee92ef3adc6a6e65a9fc6bc6239a7f7c.jpg)

![](images/3a81e11dcab325fe8ae49c917af62d1a3e7b64200c3e1fa32ca85d70f867740f.jpg)  
Figure 4: Linear relationship between the KL divergence and the squared norm of contextualized embedding for RoBERTa and Llama 2. The color represents token frequency.

We have tested four pre-trained language models: BERT (Devlin et al., 2019), RoBERTa (Liu et al., 2019), GPT-2 (Radford et al., 2019), and Llama 2 (Touvron et al., 2023) from Hugging Face transformers library (Wolf et al., 2020). Since the assumption (10) may not be appropriate for these models, we first computed $u _ { 0 } ~ =$ $\mathrm { a r g m i n } _ { u \in \{ u _ { 1 } , \ldots , u _ { N } \} } \mathrm { K L } ( u )$ , and used $p ( \cdot | u _ { 0 } )$ as a substitute for $p ( \cdot )$ when verifying the linear relationship between KL(u) and $\Vert u - u _ { 0 } \Vert ^ { 2 }$ . Fig. 4 demonstrates that the linear relationship holds approximately for RoBERTa and Llama 2. All results, including those for BERT and GPT-2, as well as additional details, are described in Appendix I. While not as distinct as the result from SGNS in Fig. 1, it was observed that the theory suggested by (16) approximately holds true in the case of contextualized embeddings from language models.

![](images/7a11fef19576913ca35bfd41f579b697c95c22c22fceadf1d2cd32a025db67a1.jpg)  
Figure 5: KL divergence computed with four different procedures plotted against word frequency $n _ { w }$ for the same words in Fig. 1. ‘raw’, ‘shuffle’, and ‘round’ are KL(w), $\overline { { \mathrm { K L } } } ( w )$ , and $\mathrm { K L } _ { 0 } ( w )$ , respectively. ‘lower 3 percentile’ is the lower 3-percentile point of KL(w) at each word frequency bin.

## 6 Word frequency bias in KL divergence

The KL divergence is highly correlated with word frequency. In Fig. 5, ‘raw’ shows the plot of KL(w) against $n _ { w }$ . The KL divergence tends to be larger for less frequent words. A part of this tendency represents the true relationship that rarer words are more informative and thus tend to shift the co-occurrence distribution from the corpus distribution. However, a large part of the tendency, particularly for low-frequency words, comes from the error caused by the finite size N of the corpus. This introduces a spurious relationship between KL(w) and $n _ { w } ,$ , causing a direct influence of word frequency. The word informativeness can be better measured by using the KL divergence when this error is adequately corrected.

## 6.1 Estimation of word frequency bias

Preliminary. The word distributions $p ( \cdot )$ and $p ( \cdot | w )$ are calculated from a finite-length corpus. The observed probability of a word w is $p ( w ) =$ $n _ { w } / N$ , where $\begin{array} { r } { N = \sum _ { w \in V } n _ { w } } \end{array}$ . The observed probability of a context word $w ^ { \prime }$ co-occurring with w is $\begin{array} { r } { p ( w ^ { \prime } | w ) ~ = ~ \left.  { n _ { w , w ^ { \prime } } } \right/ \sum _ { w ^ { \prime \prime } \in V }  { n _ { w , w ^ { \prime \prime } } } } \end{array}$ , where $( n _ { w , w ^ { \prime } } ) _ { w , w ^ { \prime } \in V }$ is the co-occurrence matrix. We computed $n _ { w , w ^ { \prime } }$ as the number of times that $w ^ { \prime }$ appears within a window of $\pm h$ around w in the corpus. Note that the denominator of $p ( w ^ { \prime } | w )$ is $\textstyle \sum _ { w ^ { \prime \prime } \in V } n _ { w , w ^ { \prime \prime } } = 2 h n _ { w }$ if the endpoints of the corpus are ignored.

Sampling error (‘shuffle’). Now we explain how word frequency directly influences the KL divergence. Consider a randomly shuffled corpus, i.e., words are randomly reordered from the original corpus (Montemurro and Zanette, 2010; Tanaka-Ishii, 2021). The unigram information, i.e., $n _ { w }$ and $p ( \cdot )$ , remains unchanged after shuffling the corpus. On the other hand, the bigram information, i.e., $n _ { w , w ^ { \prime } }$ and $p ( \cdot | w )$ , computed for the shuffled corpus is independent of the co-occurrence of words in the original corpus. In the limit of $N \to \infty , p ( \cdot | w ) = p ( \cdot )$ holds and $\mathrm { K L } ( w ) = 0$ for all $w \in V$ in the shuffled corpus. For finite corpus size N, however, $p ( \cdot | w )$ deviates from $p ( \cdot )$ because $( n _ { w , w ^ { \prime } } ) _ { w ^ { \prime } \in V }$ is approximately interpreted as a sample from the multinomial distribution with parameter $p ( \cdot )$ and $2 h n _ { w }$

In order to estimate the error caused by the direct influence of word frequency, we generated 10 sets of randomly shuffled corpus and computed the average of $\operatorname { K L } ( w )$ , denoted as $\overline { { \mathrm { K L } } } ( w )$ , which is shown as ‘shuffle’ in Fig. 5. $\overline { { \mathrm { K L } } } ( w )$ does not convey the bigram information of the original corpus but does represent the sampling error of the multinomial distribution. For sufficiently large $N _ { \ast }$ we expect $\overline { { \mathrm { K L } } } ( w ) \approx 0$ for all $w \in V$ . However, $\overline { { \mathrm { K L } } } ( w )$ is very large for small $n _ { w }$ in Fig. 5.

Sampling error (‘lower 3 percentile’). Another computation of $\overline { { \mathrm { K L } } } ( w )$ faster than ‘shuffle’ was also attempted as indicated as ‘lower 3 percentile in Fig. 5. This represents the lower 3-percentile point of KL(w) in a narrow bin of word frequency $n _ { w } .$ First, 200 bins were equally spaced on a logarithmic scale in the interval from 1 to max $\left( n _ { w } \right)$ Next, each bin was checked in order of decreasing $n _ { w }$ and merged so that each bin had at least 50 data points. This method allows for faster and more robust computation of $\overline { { \mathrm { K L } } } ( w )$ directly from $\operatorname { K L } ( w )$ of the original corpus without the need for shuffling.

Quantization error (‘round’). There is another word frequency bias due to the fact that the cooccurrence matrix only takes integer values; it is indicated as ‘round’ in Fig. 5. This quantization error is included in the sampling error estimated by $\overline { { \mathrm { K L } } } ( w )$ , so there is no need for further correction. See Appendix C for details.

## 6.2 Correcting word frequency bias

We simply subtracted $\overline { { \mathrm { K L } } } ( w )$ from $\operatorname { K L } ( w )$ . The sampling error $\overline { { \mathrm { K L } } } ( w )$ was estimated by either ‘shuffle’ or ‘lower 3 percentile’. We call

$$
\Delta \mathrm { K L } ( w ) : = \mathrm { K L } ( w ) - \overline { { \mathrm { K L } } } ( w )\tag{17}
$$

as the bias-corrected KL divergence. The same idea using the random word shuffling has been applied to an entropy-like word statistic in an existing study (Montemurro and Zanette, 2010).

## 7 Experiments

In the experiments, we first confirmed that the KL divergence is indeed a good metric of the word informativeness (Section 7.1). Then we confirmed that the norm of word embedding encodes the word informativeness as well as the KL divergence (Section 7.2). Details of the experiments are given in Appenices D, E, and F.

As one of the baseline methods, we used the Shannon entropy of $p ( \cdot | w )$ , defined as

$$
H ( w ) = - \sum _ { w ^ { \prime } \in V } p ( w ^ { \prime } | w ) \log p ( w ^ { \prime } | w ) .
$$

It also represents the information conveyed by w as explained in Appendix B.

<table><tr><td>Dataset</td><td>random</td><td> $n _ { w }$ </td><td> $n _ { w } H ( w )$ </td><td> $n _ { w } \mathrm { K L } ( w )$ </td></tr><tr><td>Krapivin2009</td><td>0.86</td><td>6.17</td><td>6.13</td><td>9.59</td></tr><tr><td>theses100</td><td>0.97</td><td>9.69</td><td>9.79</td><td>12.31</td></tr><tr><td>fao780</td><td>1.61</td><td>11.77</td><td>11.84</td><td>15.39</td></tr><tr><td>SemEval2010</td><td>1.67</td><td>9.52</td><td>9.50</td><td>11.10</td></tr><tr><td>Nguyen2007</td><td>1.90</td><td>10.56</td><td>10.57</td><td>12.84</td></tr><tr><td>PubMed</td><td>2.89</td><td>8.28</td><td>8.25</td><td>11.93</td></tr><tr><td>citeulike180</td><td>4.01</td><td>18.20</td><td>18.18</td><td>17.98</td></tr><tr><td>wiki20</td><td>4.15</td><td>9.32</td><td>9.23</td><td>19.90</td></tr><tr><td>fao30</td><td>4.92</td><td>15.92</td><td>17.05</td><td>36.88</td></tr><tr><td>Schutz2008</td><td>8.36</td><td>22.32</td><td>22.83</td><td>20.93</td></tr><tr><td>kdd</td><td>10.14</td><td>18.27</td><td>18.24</td><td>10.08</td></tr><tr><td>Inspec</td><td>10.54</td><td>16.31</td><td>16.22</td><td>14.61</td></tr><tr><td>WWW</td><td>12.08</td><td>21.20</td><td>21.11</td><td>12.76</td></tr><tr><td>SemEval2017</td><td>14.16</td><td>19.86</td><td>19.62</td><td>20.85</td></tr><tr><td>KPCrowd</td><td>39.64</td><td>25.73</td><td>25.82</td><td>40.47</td></tr></table>

Table 2: MRR of keyword extraction experiment. For complete results on MRR and P@5, see Tables 6 and 7, respectively, in Appendix D.

## 7.1 KL divergence represents the word informativeness

Through keyword extraction tasks, we confirmed that the KL divergence is indeed a good metric of the word informativeness.

Settings. We used 15 public datasets for keyword extraction for English documents. Treating each document as a “corpus”, vocabulary words were ordered by a measure of informativeness, and Mean Reciprocal Rank (MRR) was computed as an evaluation metric. When a keyword consists of two or more words, the worst value of rank was used. We used specific metrics, namely ‘random’, $n _ { w } .$ $n _ { w } H ( w )$ and $n _ { w } \mathrm { K L } ( w )$ , as our baselines. These metrics are computed only from each document without relying on external knowledge, such as a dictionary of stopwords or a set of other documents. For this reason, we did not use other metrics, such as TF-IDF, as our baselines. Note that $\| u _ { w } \| ^ { 2 }$ was not included in this experiment because embeddings cannot be trained from a very short “corpus”.

Results and discussions. Table 2 shows that $n _ { w } \mathrm { K L } ( w )$ performed best in many datasets. Therefore, keywords tend to have a large value of $n _ { w } \mathrm { K L } ( w )$ , and thus $p ( \cdot | w )$ is significantly different from $p ( \cdot )$ . This result verifies the idea that keywords have significantly large information gain.

## 7.2 Norm of word embedding encodes the word informativeness

We confirmed through proper-noun discrimination tasks (Section 7.2.1) and hypernym discrimination tasks (Section 7.2.2) that the norm of word embedding, as well as the KL divergence, encodes the word informativeness, and also confirmed that correcting the word frequency bias improves it.

In these experiments, we examined the properties of the raw word embedding $u _ { w }$ instead of the whitening-like transformed word embedding $\tilde { u } _ { w }$ From a practical standpoint, we used $u _ { w }$ , but experiments using $\tilde { u } _ { w }$ exhibited a similar trend.

Correcting word frequency bias. In the same way as (17), we correct the bias of embedding norm and denote the bias-corrected squared norm as $\Delta \| u _ { w } \| ^ { 2 } : = \| u _ { w } \| ^ { 2 } - \overline { { \| u _ { w } \| ^ { 2 } } }$ . We used the ‘lower 3 percentile’ method of Section 6.1 for $\Delta \| u _ { w } \| ^ { 2 }$ because the recomputation of embeddings for the shuffled corpus is prohibitive. Other bias-corrected quantities, such as $\Delta \mathrm { K L } ( w )$ and $\Delta H ( w )$ , were computed from 10 sets of randomly shuffled corpus.

## 7.2.1 Proper-noun discrimination

Settings. We used 10561 proper nouns, 123 function words, 4771 verbs, and 2695 adjectives that appeared in the text8 corpus not less than 10 times.

<table><tr><td></td><td> $n _ { w }$ </td><td> $H ( w )$ </td><td> $\operatorname { K L } ( w )$ </td><td> $\| u _ { w } \| ^ { 2 }$ </td><td> $\Delta H ( w )$ </td><td> $\Delta \mathrm { K L } ( w )$ </td><td> $\Delta \| u _ { w } \| ^ { 2 }$ </td></tr><tr><td>proper nouns vs. verbs</td><td>0.519</td><td>0.582</td><td>0.651</td><td>0.656</td><td>0.715</td><td>0.826</td><td>0.842</td></tr><tr><td>proper nouns vs. adjectives</td><td>0.543</td><td>0.581</td><td>0.613</td><td>0.626</td><td>0.645</td><td>0.699</td><td>0.728</td></tr></table>

Table 3: Binary classification of part-of-speech. Values are the ROC-AUC (higher is better). See Fig. 9 in Appendix E for histograms of measures.

![](images/12b8f6c70026d53edcb8fdd080a4394e8ddea498ad3baac5c89b0b564f083564.jpg)

![](images/242a20554c20972ece9ef48047e68812064b5592a540a3325ae57864f0a4cf79.jpg)  
Figure 6: The bias-corrected KL divergence $\Delta \mathrm { K L } ( w )$ and the bias-corrected squared norm of word embedding $\Delta \| u _ { w } \| ^ { 2 }$ are plotted against word frequency $n _ { w }$ . Each dot represents a word; 10561 proper nouns (red dots), 123 function words (blue dots), and 4771 verbs (green dots). The same plot for adjectives, which is omitted in the figure, produced a scatterplot that almost overlapped with the verbs.

<table><tr><td colspan="3"> $n _ { \mathrm { h y p e r } } / n _ { \mathrm { h y p o } }$ </td></tr><tr><td></td><td> $> 1$ </td><td> $< 1$  ave.</td></tr><tr><td>random</td><td>50.00 100.00</td><td>50.00 50.00</td></tr><tr><td> $n _ { w }$ </td><td>0.00</td><td>50.00</td></tr><tr><td>WeedsPrec</td><td>95.05 7.61</td><td>51.33</td></tr><tr><td>SLQS Row</td><td>95.20 13.70</td><td>54.45</td></tr><tr><td>SLQS</td><td>82.69 42.82</td><td>62.76</td></tr><tr><td> $\operatorname { K L } ( w )$ </td><td>96.46 17.84</td><td>57.15</td></tr><tr><td> $\| u _ { w } \| ^ { 2 }$ </td><td>94.07 24.89</td><td>59.48</td></tr><tr><td> $\Delta \mathrm { W e e d s P r e c }$ </td><td>46.53 51.88</td><td>49.20</td></tr><tr><td> $\Delta \mathrm { S L Q S }$  Row</td><td>59.75 43.14</td><td>51.44</td></tr><tr><td> $\Delta \mathrm { S L Q S }$ </td><td>50.41 69.06</td><td>59.74</td></tr><tr><td> $\Delta \mathrm { K L } ( w )$ </td><td>65.90 62.94</td><td>64.42</td></tr><tr><td> $\Delta \parallel u _ { w } \parallel ^ { 2 }$ </td><td>75.86 61.81</td><td>68.84</td></tr></table>

Table 4: Accuracy of hypernym-hyponym classification; the unweighted average over the four datasets. See Table 9 in Appendix F for the complete result.

We used $n _ { w } , H ( w ) , \mathrm { K L } ( w )$ , and $\| u _ { w } \| ^ { 2 }$ as a measure for discrimination. The performance of binary classification was evaluated by ROC-AUC.

Results and discussions. Table 3 shows that ∆KL(w) and $\Delta \| u _ { w } \| ^ { 2 }$ can discriminate proper nouns from other parts of speech more effectively than alternative measures. A larger value of $\Delta \mathrm { K L } ( w )$ and $\Delta \| u _ { w } \| ^ { 2 }$ indicates that words appear in a more limited context. Fig. 6 illustrates that proper nouns tend to have larger $\Delta \mathrm { K L } ( w )$ and $\Delta \| u _ { w } \| ^ { 2 }$ values when compared to verbs and function words.

## 7.2.2 Hypernym discrimination

Settings. We used English hypernym-hyponym pairs extracted from four benchmark datasets for hypernym discrimination: BLESS (Baroni and Lenci, 2011), EVALution (Santus et al., 2015), Lenci/Benotto (Lenci and Benotto, 2012), and Weeds (Weeds et al., 2014). Each dataset was divided into two parts by comparing $n _ { w }$ of hypernym and hyponym to remove the effect of word frequency. In addition to ‘random’ and $n _ { w } ,$ we used WeedsPrec (Weeds and Weir, 2003; Weeds et al., 2004), SLQS Row (Shwartz et al., 2017) and SLQS (Santus et al., 2014) as baselines.

Results and discussions. Table 4 shows that $\Delta \| u _ { w } \| ^ { 2 }$ and $\Delta \mathrm { K L } ( w )$ were the best and the second best, respectively, for predicting hypernym in hypernym-hyponym pairs. Correcting frequency bias remedies the difficulty of discrimination for the $n _ { \mathrm { h y p e r } } < n _ { \mathrm { h y p o } }$ part, resulting in an improvement in the average accuracy.

## 8 Conclusion

We showed theoretically and empirically that the KL divergence, i.e., the information gain of the word, is encoded in the norm of word embedding. The KL divergence and, thus, the norm of word embedding has the word frequency bias, which was corrected in the experiments. We then confirmed that the KL divergence and the norm of word embedding work as a metric of informativeness in NLP tasks.

## Limitations

• The important limitation of the paper is that the theory assumes the skip-gram with negative sampling (SGNS) model for static word embeddings or the softmax function in the final layer of language models for contextualized word embeddings.

• The theory also assumes that the model is trained perfectly, as mentioned in Section 3.3. When the assumption is violated, the theory may not hold. For example, the training is not perfect when the number of epochs is insufficient, as illustrated in Appendix G.

## Ethics Statement

This study complies with the ACL Ethics Policy<sup>5</sup>.

## Acknowledgements

We would like to thank Junya Honda and Yoichi Ishibashi for the discussion and the anonymous reviewers for their helpful advice. This study was partially supported by JSPS KAKENHI 22H05106, 23H03355, JST ACT-X JPMJAX200S, and JST CREST JPMJCR21N3.

## References

Alan Agresti. 2013. Categorical Data Analysis, 3rd edition. John Wiley & Sons.

Shun-Ichi Amari. 1982. Differential Geometry of Curved Exponential Families-Curvatures and Information Loss. The Annals of Statistics, 10(2):357 – 385.

Shun-Ichi Amari. 1998. Natural gradient works efficiently in learning. Neural computation, 10:251–276.

Makoto Aoshima, Dan Shen, Haipeng Shen, Kazuyoshi Yata, Yi-Hui Zhou, and James S Marron. 2018. A survey of high dimension low sample size asymptotics. Australian & New Zealandjournal ofstatistics, 60:4– 19.

Nikolay Arefyev, Pavel Ermolaev, and Alexander Panchenko. 2018. How much does a word weigh? weighting word embeddings for word sense induction. ArXiv 1805.09209.

A. R. Aronson, O. Bodenreider, H. F. Chang, S. M. Humphrey, J. G. Mork, S. J. Nelson, T. C. Rindflesch, and W. J. Wilbur. 2000. The NLM indexing initiative. In Proceedings of the AMIA Symposium, pages 17– 21.

Sanjeev Arora, Nadav Cohen, Wei Hu, and Yuping Luo. 2019. Implicit regularization in deep matrix factorization. Advances in Neural Information Processing Systems, 32.

Giusepppe Attardi. 2015. Wikiextractor. https:// github.com/attardi/wikiextractor.

Isabelle Augenstein, Mrinal Das, Sebastian Riedel, Lakshmi Vikraman, and Andrew McCallum. 2017. SemEval 2017 task 10: ScienceIE - extracting keyphrases and relations from scientific publications. In Proceedings ofthe 11th International Workshop on Semantic Evaluation (SemEval-2017), pages 546– 555, Vancouver, Canada. Association for Computational Linguistics.

Ole Barndorff-Nielsen. 2014. Information and exponentialfamilies: in statistical theory. John Wiley & Sons.

Marco Baroni and Alessandro Lenci. 2011. How we BLESSed distributional semantic evaluation. In Proceedings ofthe GEMS 2011 Workshop on GEometrical Models ofNatural Language Semantics, pages 1–10, Edinburgh, UK. Association for Computational Linguistics.

Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. 2017. Enriching word vectors with subword information. Transactions of the Associationfor Computational Linguistics, 5:135–146.

Thomas Bott, Dominik Schlechtweg, and Sabine Schulte im Walde. 2021. More than just frequency? demasking unsupervised hypernymy prediction methods. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 186– 192, Online. Association for Computational Linguistics.

Mikael Brunila and Jack LaViolette. 2022. What company do words keep? revisiting the distributional semantics of J.R. firth & zellig Harris. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4403–4417, Seattle, United States. Association for Computational Linguistics.

Ciprian Chelba, Tomás Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. 2014. One billion word benchmark for measuring progress in statistical language modeling. In INTER-SPEECH 2014, 15th Annual Conference of the International Speech Communication Association, Singapore, September 14-18, 2014, pages 2635–2639. ISCA.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages

4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Bradley Efron. 1978. The geometry of exponential families. The Annals ofStatistics, 6:362–376.

Bradley Efron. 2022. Exponential Families in Theory and Practice. Cambridge University Press.

Stefan Evert. 2005. The statistics of word cooccurrences: word pairs and collocations. Ph.D. thesis, University of Stuttgart.

J. R. Firth. 1957. A synopsis of linguistic theory 1930- 55. Studies in Linguistic Analysis (special volume of the Philological Society), 1952-59:1–32.

Sujatha Das Gollapalli and Cornelia Caragea. 2014. Extracting keyphrases from research papers using citation networks. Proceedings ofAAAI Conference on Artificial Intelligence, 28(1).

Michael Gutmann and Aapo Hyvärinen. 2012. Noisecontrastive estimation of unnormalized statistical models, with applications to natural image statistics. J. Mach. Learn. Res., 13:307–361.

Zellig Harris. 1954. Distributional structure. Word, 10(2-3):146–162.

Aurélie Herbelot and Mohan Ganesalingam. 2013. Measuring semantic content in distributional vectors. In Proceedings ofthe 51st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 440–445, Sofia, Bulgaria. Association for Computational Linguistics.

Anette Hulth. 2003. Improved automatic keyword extraction given more linguistic knowledge. In Proceedings ofthe 2003 Conference on Empirical Methods in Natural Language Processing, pages 216–223.

Sungkyu Jung and J Stephen Marron. 2009. PCA consistency in high dimension, low sample size context. The Annals ofStatistics, 37:4104 – 4130.

Mikhail Khodak, Nikunj Saunshi, Yingyu Liang, Tengyu Ma, Brandon Stewart, and Sanjeev Arora. 2018. A la carte embedding: Cheap but effective induction of semantic feature vectors. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12–22, Melbourne, Australia. Association for Computational Linguistics.

Su Nam Kim, Olena Medelyan, Min-Yen Kan, and Timothy Baldwin. 2010. SemEval-2010 task 5 : Automatic keyphrase extraction from scientific articles. In Proceedings of the 5th International Workshop on Semantic Evaluation, pages 21–26, Uppsala, Sweden. Association for Computational Linguistics.

Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, and Kentaro Inui. 2020. Attention is not only a weight: Analyzing transformers with vector norms. In Proceedings of the 2020 Conference on Empirical

Methods in Natural Language Processing (EMNLP), pages 7057–7075, Online. Association for Computational Linguistics.

Mikalai Krapivin, Aliaksandr Autaeu, and Maurizio Marchese. 2009. Large dataset for keyphrases extraction. Technical Report DISI-09-055, University of Trento.

Erich L Lehmann and George Casella. 1998. Theory of point estimation. Springer New York, NY.

Alessandro Lenci and Giulia Benotto. 2012. Identifying hypernyms in distributional semantic spaces. In \*SEM 2012: The First Joint Conference on Lexical and Computational Semantics – Volume 1: Proceedings ofthe main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012), pages 75–79, Montréal, Canada. Association for Computational Linguistics.

Bofang Li, Tao Liu, Zhe Zhao, Buzhou Tang, Aleksandr Drozd, Anna Rogers, and Xiaoyong Du. 2017. Investigating different syntactic context types and context representations for learning word embeddings. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 2421–2431, Copenhagen, Denmark. Association for Computational Linguistics.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692.

Matt Mahoney. 2011. About the test data. http:// mattmahoney.net/dc/textdata.html.

Luís Marujo, Márcio Viveiros, and João Paulo da Silva Neto. 2011. Keyphrase cloud generation of broadcast news. Proceedings ofAnnual Conference ofthe International Speech Communication Association, pages 2393–2396.

Y. Matsuo and M. Ishizuka. 2004. Keyword extraction from a single document using word co-occurrence statistical information. International Journal on Artificial Intelligence Tools, 13(01):157–169.

Alyona Medelyan. 2015. Keyword extraction datasets. https://github.com/zelandiya/ keyword-extraction-datasets.

Olena Medelyan, Eibe Frank, and Ian H. Witten. 2009. Human-competitive tagging using automatic keyphrase extraction. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing, pages 1318–1327, Singapore. Association for Computational Linguistics.

Olena Medelyan and Ian H. Witten. 2008. Domainindependent automatic keyphrase indexing with small training sets. Journal ofthe American Society for Information Science and Technology, 59(7):1026– 1040.

Olena Medelyan, Ian H Witten, and David Milne. 2008. Topic indexing with Wikipedia. In Proceedings of the AAAI WikiAI workshop, volume 1, pages 19–24.

Tomás Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems, pages 3111–3119.

Jeff Mitchell and Mirella Lapata. 2010. Composition in distributional models of semantics. Cognitive Science, 34(8):1388–1429.

Marcelo A. Montemurro and Damiá n H. Zanette. 2010. Towards the quantification of the semantic information encoded in written language. Advances in Complex Systems, 13(2):135–153.

Thuy Dung Nguyen and Min-Yen Kan. 2007. Keyphrase extraction in scientific publications. In Asian Digital Libraries. Looking Back 10 Years and Forging New Frontiers, pages 317–326, Berlin, Heidelberg. Springer Berlin Heidelberg.

Sergey Oladyshkin and Wolfgang Nowak. 2019. The connection between bayesian inference and information theory for model selection, information gain and experimental design. Entropy, 21(11):1081.

Matteo Pagliardini, Prakhar Gupta, and Martin Jaggi. 2018. Unsupervised learning of sentence embeddings using compositional n-gram features. In Proceedings ofthe 2018 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 528–540, New Orleans, Louisiana. Association for Computational Linguistics.

Jeffrey Pennington, Richard Socher, and Christopher Manning. 2014. GloVe: Global vectors for word representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1532–1543, Doha, Qatar. Association for Computational Linguistics.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Enrico Santus, Alessandro Lenci, Qin Lu, and Sabine Schulte im Walde. 2014. Chasing hypernyms in vector spaces with entropy. In Proceedings of the 14th Conference of the European Chapter of the Associationfor Computational Linguistics, volume 2: Short Papers, pages 38–42, Gothenburg, Sweden. Association for Computational Linguistics.

Enrico Santus, Frances Yung, Alessandro Lenci, and Chu-Ren Huang. 2015. EVALution 1.0: an evolving semantic dataset for training and evaluation of distributional semantic models. In Proceedings of the 4th Workshop on Linked Data in Linguistics: Resources and Applications, pages 64–69, Beijing, China. Association for Computational Linguistics.

Adriaan M. J. Schakel and Benjamin J. Wilson. 2015. Measuring word significance using distributed representations of words. ArXiv 1508.02297.

Tobias Schnabel, Igor Labutov, David Mimno, and Thorsten Joachims. 2015. Evaluation methods for unsupervised word embeddings. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 298–307, Lisbon, Portugal. Association for Computational Linguistics.

A. T. Schutz. 2008. Keyphrase extraction from single documents in the open domain exploiting linguistic and statistical methods. Master’s thesis, National University of Ireland.

Vered Shwartz, Enrico Santus, and Dominik Schlechtweg. 2017. Hypernyms under siege: Linguistically-motivated artillery for hypernymy detection. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, pages 65–75, Valencia, Spain. Association for Computational Linguistics.

Kumiko Tanaka-Ishii. 2021. Statistical Universals of Language. Springer.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Christian Wartena, Rogier Brussee, and Wout Slakhorst. 2010. Keyword extraction using word co-occurrence. In 2010 Workshops on Database and Expert Systems Applications, pages 54–58.

Julie Weeds, Daoud Clarke, Jeremy Reffin, David Weir, and Bill Keller. 2014. Learning to distinguish hypernyms and co-hyponyms. In Proceedings ofCOLING 2014, the 25th International Conference on Computational Linguistics: Technical Papers, pages 2249– 2259, Dublin, Ireland. Dublin City University and Association for Computational Linguistics.

Julie Weeds and David Weir. 2003. A general framework for distributional similarity. In Proceedings of the 2003 Conference on Empirical Methods in Natural Language Processing, pages 81–88.

Julie Weeds, David Weir, and Diana McCarthy. 2004. Characterising measures of lexical distributional similarity. In COLING 2004: Proceedings of the 20th International Conference on Computational Linguistics, pages 1015–1021, Geneva, Switzerland. COL-ING.

Wikimedia Foundation. 2021. English wikipedia dump data. Accessed on: 15-June-2021.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen,

<table><tr><td>Dimensionality</td><td>300</td></tr><tr><td>Epochs</td><td>100</td></tr><tr><td>Window size h</td><td>10</td></tr><tr><td>Negative samples ν</td><td>5</td></tr><tr><td>Learning rate</td><td>0.025</td></tr><tr><td>Min count</td><td>1</td></tr></table>

Table 5: SGNS parameters.

Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Sho Yokoi, Ryo Takahashi, Reina Akama, Jun Suzuki, and Kentaro Inui. 2020. Word rotator’s distance. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2944–2960, Online. Association for Computational Linguistics.

## A Settings for computation of word embeddings and KL divergence

Corpus. We used the text8 (Mahoney, 2011), which is an English corpus data with the size of $N = 1 7 . 0 { \times } 1 0 ^ { 6 }$ tokens and $| V | = 2 5 4 \times 1 0 ^ { 3 }$ vocabulary words. We used all the tokens<sup>6</sup> separated by spaces for word embeddings and KL divergence.

Training of the SGNS model. Word embeddings were trained<sup>7</sup> by optimizing the same objective function used in Mikolov et al. (2013). Parameters used to train SGNS are summarized in Table 5. The learning rate shown is the initial value, which we decreased linearly to the minimum value of $1 . 0 \times$ $1 0 ^ { - 4 }$ during the learning process. The negative sampling distribution was specified as

$$
q ( w ) \propto ( n _ { w } ) ^ { \frac { 3 } { 4 } } .
$$

The elements of $u _ { w }$ were initialized by the uniform distribution over [ 0.5, 0.5] divided by the dimensionality of the embedding, and the elements of $v _ { w }$ were initialized by zero.

![](images/187be624bcecc95be03f32eda76268218fd92084fee750787ba419000f117275.jpg)  
Figure 7: The Shannon entropy and the squared norm of word embedding. Settings are the same as in Fig. 1.

![](images/db2725ad041b174753d15047b84c38e739ea009e48ceb6838036db54bff7e7a8.jpg)  
Figure 8: The self-information and the squared norm of word embedding. Settings are the same as in Fig. 1.

Computation of KL divergence. The value of $\operatorname { K L } ( w )$ was computed from $p ( \cdot | w )$ and $p ( \cdot )$ using the definition in Section 3.2 with the convention that 0 log 0 = 0. The word probability $p ( w ^ { \prime } )$ and the co-occurrence probability $p ( w ^ { \prime } | w )$ were computed from the word frequency $n _ { w }$ and the cooccurrence matrix $( n _ { w , w ^ { \prime } } ) _ { w , w ^ { \prime } \in V }$ , respectively, as described in Section 6. The co-occurrence matrix was computed with the window size $h = 1 0$

Word set for visualization. We have used 47 $1 0 ^ { 3 }$ words with $n _ { w } \ge 1 0 ^ { 1 }$ for the plots of Figs. 1 to 5. Except for Fig. 5, extreme points, up to 0.5% for each axis, were truncated to set the plot range. Word embeddings and KL divergence are not very stable for low-frequency words. For this reason, we used 1820 words with $n _ { w } > 1 0 ^ { 3 }$ to fit the simple linear regression model using the least squares method.

## B Other quantities of information theory

In addition to KL divergence, two other information theoretic quantities are discussed here.

## B.1 Shannon entropy

The Shannon entropy of $p ( \cdot | w )$ , defined as

$$
H ( w ) = - \sum _ { w ^ { \prime } \in V } p ( w ^ { \prime } | w ) \log p ( w ^ { \prime } | w ) ,
$$

also represents information conveyed by $w .$ In this paper, we call it the Shannon entropy of word w. $H ( w )$ is closely related to $\operatorname { K L } ( w )$ . The Shannon entropy of $p ( \cdot | w )$ can be written as

$$
H ( w ) = \log | V | - \mathrm { K L } ( p ( \cdot | w ) \parallel \operatorname { u n i f } ( \cdot ) ) ,
$$

meaning that $- H ( w )$ measures how much the cooccurrence distribution shifts from the uniform distribution (i.e., uni $\operatorname { f } ( w ^ { \prime } ) = 1 / | V | )$ . Thus, $H ( w )$ and $\operatorname { K L } ( w )$ have different reference distributions.

## B.2 Self-information

A much naive way of measuring the information of a word is the self-information of the event that the word w is sampled from $p ( \cdot )$ , defined as

$$
I ( w ) = - \log p ( w ) .
$$

The expected value $\scriptstyle \sum _ { w \in V } p ( w ) I ( w )$ is the Shannon entropy of $p ( \cdot )$ . Since $p ( w )$ was computed as $p ( w ) = n _ { w } / N$

$$
I ( w ) = \log N - \log n _ { w }
$$

actually looks at the word frequency $n _ { w }$ in the log scale.

## B.3 Relation to word embedding

$H ( w )$ and $I ( w )$ were computed with the same settings as in Section 4.2 and Appendix A. They were plotted along with $\| u _ { w } \| ^ { 2 }$ as shown in Fig. 7 and Fig. 8, respectively. Compared with KL(w), the relationships are less clear with $R ^ { 2 } \approx 0 . 4$ . From this experiment, we see that $\operatorname { K L } ( w )$ better represents $\| u _ { w } \| ^ { 2 }$ than $H ( w )$ and $I ( w )$

## C Quantization error

The co-occurrence matrix $( n _ { w , w ^ { \prime } } ) _ { w , w ^ { \prime } \in V }$ is sparse with many zero values at rows of w with small $n _ { w } .$ The effect of quantization error caused by $n _ { w , w ^ { \prime } }$ taking only integer values cannot be ignored for low-frequency words. This effect is part of the sampling error, but we try to isolate the quantization error here. Let us redefine $n _ { w , w ^ { \prime } } : = \mathrm { r o u n d } ( 2 h n _ { w } p ( w ^ { \prime } ) )$ and compute the KL divergence, denoted as $\mathrm { K L } _ { 0 } ( w )$ , which is shown as ‘round’ in Fig. 5. If there is no rounding errors, $p ( w ^ { \prime } | w ) = p ( w ^ { \prime } )$ so that $\mathrm { K L } _ { 0 } ( w ) = 0$ . In reality, however, $\mathrm { K L } _ { 0 } ( w )$ is non-negligible for words with small $n _ { w } ,$ and this effect can be corrected by $\mathrm { K L } ( w ) - \mathrm { K L } _ { 0 } ( w )$

## D Details of experiment in Section 7.1

In this experiment, we confirmed that humanannotated keywords of documents were observed at the top of the ranking calculated by the discrepancy between $p ( \cdot | w )$ and $p ( \cdot )$ .

Datasets. For the experiment of keyword extraction, we used 15 datasets in English<sup>8</sup>. Each entry consists of a pair of document and gold keywords. Table 6 includes information on the size (the number of documents) and the type of documents.

Preparation. Each document in the datasets was tokenized by NLTK’s word\_tokenize function. Then, each word was stemmed using NLTK’s PorterStemmer, and all characters were converted to lowercase. The same preprocessing of stemming and lowercase was also applied to the gold keywords. However, we did not remove stopwords in preprocessing to see if the informativeness measures could remove unnecessary stopwords by themselves. The co-occurrence matrix for each document was computed with the window size $h = 1 0$ . Note that only a subset $V ^ { \prime } \subset V$ of the vocabulary set described below was used for stable computation of $p ( w ^ { \prime } | w )$ $w ^ { \prime } \in V ^ { \prime } , w \in V$ . For constructing $V ^ { \prime } .$ , all the words $w \in V$ were sorted in decreasing order of $n _ { w }$ , and the cumulative frequency $\begin{array} { r } { c _ { i } \ = \ \sum _ { j = 1 } ^ { i } n _ { w _ { j } } } \end{array}$ up to the i-th frequent word were computed for $i = 1 , 2 , \dots , | V |$ . Then $V ^ { \prime } = \{ w _ { 1 } , \ldots , w _ { i } \}$ was defined with the smallest i such that $c _ { i } \geq N / 3$

Methods. In each document, word ranking lists were created by sorting its vocabulary words using the informativeness measures. For ‘random’, the ranking list is simply a random shuffle of the vocabulary words. For $n _ { w } H ( w )$ , words were ranked in increasing order. For other measures, words were ranked in decreasing order. We multiply $n _ { w }$ to KL(w) because $G ^ { 2 } = 2 n _ { w } \mathrm { K L } ( w )$ is appropriate for testing the null hypothesis that $p ( \cdot | w ) = p ( \cdot )$ $n _ { w } H ( w )$ is also interpreted as a test statistic for testing the null hypothesis that $p ( \cdot | w ) = \operatorname { u n i f } ( \cdot )$ We also included the $\chi ^ { 2 }$ statistic (Matsuo and Ishizuka, 2004), which is related to KL(w) as $\chi ^ { 2 } \approx G ^ { 2 }$ for sufficiently large $n _ { w }$

<table><tr><td>Dataset</td><td>Size</td><td>Type</td><td>random</td><td> $n _ { w }$ </td><td> $n _ { w } H ( w )$ </td><td> $\chi ^ { 2 } ( w )$ </td><td> $n _ { w } \mathrm { K L } ( w )$ </td></tr><tr><td>Krapivin2009</td><td>2304</td><td>article</td><td>0.86</td><td>6.17</td><td>6.13</td><td>8.00</td><td>9.59</td></tr><tr><td>theses100</td><td>100</td><td>article</td><td>0.97</td><td>9.69</td><td>9.79</td><td>9.31</td><td>12.31</td></tr><tr><td>fao780</td><td>779</td><td>article</td><td>1.61</td><td>11.77</td><td>11.84</td><td>11.04</td><td>15.39</td></tr><tr><td>SemEval2010</td><td>243</td><td>article</td><td>1.67</td><td>9.52</td><td>9.50</td><td>8.40</td><td>11.10</td></tr><tr><td>Nguyen2007</td><td>209</td><td>article</td><td>1.90</td><td>10.56</td><td>10.57</td><td>9.78</td><td>12.84</td></tr><tr><td>PubMed</td><td>500</td><td>article</td><td>2.89</td><td>8.28</td><td>8.25</td><td>9.91</td><td>11.93</td></tr><tr><td>citeulike180</td><td>183</td><td>article</td><td>4.01</td><td>18.20</td><td>18.18</td><td>10.03</td><td>17.98</td></tr><tr><td>wiki20</td><td>20</td><td>report</td><td>4.15</td><td>9.32</td><td>9.23</td><td>12.82</td><td>19.90</td></tr><tr><td>fao30</td><td>30</td><td>article</td><td>4.92</td><td>15.92</td><td>17.05</td><td>29.47</td><td>36.88</td></tr><tr><td>Schutz2008</td><td>1231</td><td>article</td><td>8.36</td><td>22.32</td><td>22.83</td><td>13.14</td><td>20.93</td></tr><tr><td>kdd</td><td>755</td><td>abstract</td><td>10.14</td><td>18.27</td><td>18.24</td><td>9.71</td><td>10.08</td></tr><tr><td>Inspec</td><td>2000</td><td>abstract</td><td>10.54</td><td>16.31</td><td>16.22</td><td>13.75</td><td>14.61</td></tr><tr><td>WWW</td><td>1330</td><td>abstract</td><td>12.08</td><td>21.20</td><td>21.11</td><td>11.67</td><td>12.76</td></tr><tr><td>SemEval2017</td><td>493</td><td>paragraph</td><td>14.16</td><td>19.86</td><td>19.62</td><td>19.18</td><td>20.85</td></tr><tr><td>KPCrowd</td><td>500</td><td>news</td><td>39.64</td><td>25.73</td><td>25.82</td><td>39.02</td><td>40.47</td></tr></table>

Table 6: MRR of keyword extraction experiment.

Evaluation metrics. We used MRR and P@5 as evaluation metrics for the keyword prediction task.

MRR is the average of the reciprocals of gold keywords’ ranks. The numbers in the tables were multiplied by 100. For each document, we used the best-ranked keyword, i.e., the minimum value of the ranks of correct answers. If a keyword is given as a phrase consisting of two or more words, the rank of the keyword is defined by the worstranked word. For example, the rank of "New York" is 10 if the ranks of "new" and "york" are 3 and 10, respectively.

P@5 is the average percentage of correct answers that appear in the top five words of the ranked list. For each document, the number of gold keywords in the top five words was computed and divided by 5. For a keyword consisting of two or more words, it is regarded as a correct answer only when all the words are included in the top five words. Thus the percentage can be larger than 100 if several gold keywords share the same words.

Results. Table 6 shows MRR, and Table 7 shows P@5 of the experiment. Datasets were sorted in the increasing order of MRR of the random baseline in both tables. Table 2 in Section 7.1 is a summary of Table 6. Small values of MRR or P@5 of the random baseline indicate the extent of difficulty of the keyword extraction. Datasets with the article type are difficult, and the dataset with the news type is the easiest. In the difficult datasets, $n _ { w } \mathrm { K L } ( w )$ performed best in almost all datasets.

## E Details of experiment in Section 7.2.1

In this experiment, we confirmed that proper nouns tend to have larger values of ∆KL(w) and $\Delta \| u _ { w } \|$ compared to other parts of speech.

Datasets. We used 10561 proper nouns, 123 function words, 4771 verbs, and 2695 adjectives that appeared in the text8 corpus not less than 10 times $( n _ { w } ~ \ge ~ 1 0 )$ . The parts of speech of these words were identified by NLTK’s POS tagger. Proper nouns are tagged as {NN, NNS}, verbs are tagged as {VB, VBD, VBG, VBN, VBP, VBZ}, adjectives are tagged as {JJ, JJS, JJR}, and function words are tagged as {IN, PRP, PRP\$, WP, WP\$, DT, PDT, WDT, CC, MD, RP}. Proper nouns were restricted to those found in the 61711 words of the English Proper nouns database<sup>9</sup>.

<table><tr><td>Dataset</td><td>Size</td><td>Type</td><td>random</td><td> $n _ { w }$ </td><td> $n _ { w } H ( w )$ </td><td> $\chi ^ { 2 } ( w )$ </td><td> $n _ { w } \mathrm { K L } ( w )$ </td></tr><tr><td>Krapivin2009</td><td>2304</td><td>article</td><td>0.11</td><td>0.80</td><td>0.83</td><td>2.37</td><td>3.12</td></tr><tr><td>theses100</td><td>100</td><td>article</td><td>0.16</td><td>3.40</td><td>3.60</td><td>3.80</td><td>5.40</td></tr><tr><td>fao780</td><td>779</td><td>article</td><td>0.28</td><td>3.70</td><td>3.72</td><td>3.75</td><td>5.52</td></tr><tr><td>SemEval2010</td><td>243</td><td>article</td><td>0.23</td><td>1.89</td><td>1.81</td><td>2.63</td><td>4.28</td></tr><tr><td>Nguyen2007</td><td>209</td><td>article</td><td>0.42</td><td>3.44</td><td>3.54</td><td>4.40</td><td>5.74</td></tr><tr><td>PubMed</td><td>500</td><td>article</td><td>0.54</td><td>2.08</td><td>2.00</td><td>2.96</td><td>3.76</td></tr><tr><td>citeulike180</td><td>183</td><td>article</td><td>0.90</td><td>12.02</td><td>11.69</td><td>4.37</td><td>8.52</td></tr><tr><td>wiki20</td><td>20</td><td>report</td><td>0.70</td><td>1.00</td><td>1.00</td><td>7.00</td><td>10.00</td></tr><tr><td>fao30</td><td>30</td><td>article</td><td>1.53</td><td>9.33</td><td>8.67</td><td>14.67</td><td>18.00</td></tr><tr><td>Schutz2008</td><td>1231</td><td>article</td><td>2.37</td><td>14.77</td><td>15.22</td><td>5.20</td><td>10.93</td></tr><tr><td>kdd</td><td>755</td><td>abstract</td><td>3.07</td><td>8.98</td><td>9.14</td><td>2.12</td><td>2.28</td></tr><tr><td>Inspec</td><td>2000</td><td>abstract</td><td>2.84</td><td>7.32</td><td>6.85</td><td>5.09</td><td>5.68</td></tr><tr><td>WWW</td><td>1330</td><td>abstract</td><td>3.78</td><td>10.98</td><td>10.89</td><td>2.33</td><td>3.07</td></tr><tr><td>SemEval2017</td><td>493</td><td>paragraph</td><td>4.10</td><td>13.35</td><td>12.78</td><td>8.88</td><td>9.33</td></tr><tr><td>KPCrowd</td><td>500</td><td>news</td><td>21.75</td><td>18.37</td><td>18.33</td><td>21.25</td><td>24.33</td></tr></table>

Table 7: P@5 of keyword extraction experiment.

<table><tr><td>∆KL(w)</td><td>Word Examples</td></tr><tr><td>Top  $( 0 \% \sim 1 0 \% )$ </td><td>HONDA, INTERPOL, Gabon, Yin, VAR, IMF, Benin, BO, Bene, GB</td></tr><tr><td>Middle (45%～55%)</td><td>Pete, Dee, Wine, Tony, Bogart, Alice, Cliff, Madonna, Dover, Leopold</td></tr><tr><td>Bottom  $( 9 0 \% \sim 1 0 0 \% )$ </td><td>storm, haven, sale, miracle, discover, Phillip, duty, prohibition, capitol, comfort</td></tr></table>

Table 8: Randomly sampled proper nouns for each range of informativeness measured by the KL divergence.

Preparation. We computed $n _ { w } , \ \mathrm { K L } ( w )$ and $\| u _ { w } \| ^ { 2 }$ from the text8 corpus as described in $\mathsf { A p - }$ pendix A. $H ( w )$ was also computed in the same way as KL(w). For their bias-corrected versions, we used the ‘shuffle’ method in Section 6.1 for $\Delta \mathrm { K L } ( w )$ and $\Delta H ( w )$ , and the ‘lower 3 percentile method for $\Delta \| u _ { w } \| ^ { 2 }$ . We used these measures for the binary classification of part-of-speech.

Methods. Proper nouns tend to have large values of $n _ { w } , \mathrm { K L } ( w )$ and $\| u _ { w } \| ^ { 2 }$ , or small values of $H ( w )$ as seen in Fig. 9. Therefore, each word is classified as a proper noun if a measure is larger (or smaller) than a threshold value. We performed two sets of binary classification experiments: proper nouns vs. verbs, and proper nouns vs. adjectives.

Evaluation metrics. Since the classification depends on the threshold value, we used ROC-AUC to evaluate the classification performance. ROC-AUC was computed by Scipy’s roc\_curve function.

Results. Table 3 in Section 7.2.1 shows the ROC-AUC of the classification task, confirming the good performance of ∆KL(w) and $\Delta \| u _ { w } \| ^ { 2 }$

Table 8 shows randomly sampled proper nouns with $1 0 ^ { 1 } ~ \le ~ n _ { w } ~ \le ~ 1 0 ^ { 3 }$ and specific ranges of ∆KL(w); since our experiment is case-insensitive, some selected words were actually considered as common nouns, such as storm and haven. We observed that common nouns tend to have small KL values. On the other hand, words with large KL values include context-specific nouns, such as company names, suggesting that they are more informative.

## F Details of experiment in Section 7.2.2

In this experiment, we confirmed that ∆KL(w) and $\Delta \| u _ { w } \| ^ { 2 }$ tend to have a smaller value for hypernym in hypernym-hyponym pairs.

Datasets. Among the hypernym-hyponym pairs in each dataset, we used those consisting of words that appear in the text8 corpus. Specifically, we used 1336 pairs from the 1337 pairs of the BLESS dataset (Baroni and Lenci, 2011), 3635 pairs from the 3637 pairs of the EVALution dataset (Santus et al., 2015), 1760 pairs from the 1933 pairs of the Lenci/Benotto dataset (Lenci and Benotto,

![](images/5b34a55cfa7940ec4830e3af34983a82799ba72cf663a630e3eaba019de107ce.jpg)

![](images/993985651a4a33c67641995d333fbb47574c0f00782ce6e8ecea19e293862c0d.jpg)

![](images/0b61787919034c93d4d04196dae5c1348f457eba2e04756b909b05d506bcdccd.jpg)

![](images/593538110c68e9170d2566f1b5773b3a30639f37ee2deda3797b142e3d0300af.jpg)

![](images/75b32e358396666e0a6bdff7a21de9cf63013d0829296812cfa119f2b71ff8f9.jpg)

![](images/f999eebabf8a9e6c223f3c731aa8e1651a20c9696dcf9b8c311b9e2916a9e878.jpg)

![](images/75fc10d2ef90d83a4780c9e98e2d6d71f5ef1559ebdee01fa710ceac8ac6f7c1.jpg)  
Figure 9: Histogram of each measure used for binary classification of part-of-speech. Plotted for 10561 proper nouns (red) and 4771 verbs (green) in the text8 corpus.

2012), 1427 pairs from the 1427 pairs of the Weeds dataset (Weeds et al., 2014). Each dataset was divided into two parts: the $n _ { \mathrm { h y p e r } } > n _ { \mathrm { h y p o } }$ part and the $n _ { \mathrm { h y p e r } } < n _ { \mathrm { h y p o } } \ : \mathrm { p a r t } .$

Preparation. We computed $n _ { w } , n _ { w , w ^ { \prime } } , H ( w )$ KL $( w ) , \| u _ { w } \| ^ { 2 } , \Delta H ( w ) , \Delta \mathrm { K L } ( w )$ , and $\Delta \| u _ { w } \| ^ { 2 }$ from the text8 corpus as described in Appendices A and E.

Methods. We considered the binary classification of hypernym given a hypernym-hyponym pair. Using $\mathrm { K L } ( w ) , \| u _ { w } \| ^ { 2 } , \Delta \mathrm { K L } ( w )$ , or $\Delta \| u _ { w } \| ^ { 2 }$ as a measure of informativeness, the word with a smaller value of the measure was predicted as hypernym.

Baseline methods to predict hypernym given a word pair $( w _ { 1 } , w _ { 2 } )$ are described below.

• Random is the random classification. The accuracy is 50%.

• Word Frequency chooses the word with larger $n _ { w }$ as hypernym.

• WeedsPrec (Weeds and Weir, 2003; Weeds et al., 2004) is based on the distributional inclusion hypothesis that the context of hyponym is included in the context of its hypernym. The weighted inclusion of word w<sub>2</sub> in the context of word $w _ { 1 }$ is formulated as

$$
\mathrm { W e e d s P r e c } ( w _ { 1 } , w _ { 2 } ) = \frac { \sum _ { w ^ { \prime } \in V _ { w _ { 1 } \cap w _ { 2 } } } n _ { w _ { 1 } , w ^ { \prime } } } { \sum _ { w ^ { \prime } \in V } n _ { w _ { 1 } , w ^ { \prime } } } ,
$$

where $V _ { w _ { 1 } \cap w _ { 2 } } = \{ w ^ { \prime } \in V \mid n _ { w _ { 1 } , w ^ { \prime } } > 0 \land$ $n _ { w _ { 2 } , w ^ { \prime } } > 0 \}$ . w<sub>1</sub> is predicted as hypernym if

WeedsPrec(w<sub>1</sub>, w<sub>2</sub>) < WeedsPrec(w<sub>2</sub>, w<sub>1</sub>).

• SLQS Row (Shwartz et al., 2017) compares the Shannon entropy. w<sub>1</sub> is predicted as hypernym if

$$
S L Q S _ { R o w } ( w _ { 1 } , w _ { 2 } ) : = 1 - \frac { H ( w _ { 1 } ) } { H ( w _ { 2 } ) } < 0 ,
$$

or equivalently $H ( w _ { 1 } ) > H ( w _ { 2 } )$

• SLQS (Santus et al., 2014) compares the median entropy of context words defined as

$$
\begin{array} { r } { E ( w ) = \mathrm { M e d i a n } _ { c \in C _ { w } } H ( c ) . } \end{array}
$$

w<sub>1</sub> is predicted as hypernym if

$$
S L Q S ( w _ { 1 } , w _ { 2 } ) : = 1 - \frac { E ( w _ { 1 } ) } { E ( w _ { 2 } ) } < 0 ,
$$

or equivalently $E ( w _ { 1 } ) > E ( w _ { 2 } )$ . Note that $C _ { w }$ is the set of most strongly associated context words of w, as determined by positive local mutual information (Evert, 2005). We used $| C _ { w } | = 5 0$

• ∆WeedsPrec is the bias-corrected version of WeedsPrec computed by the method in Section 6.2. $\overline { { \mathrm { W e e d s P r e c } } } ( w _ { 1 } , w _ { 2 } )$ is the average of $\mathrm { W e e d s P r e c } ( w _ { 1 } , w _ { 2 } )$ for 10 randomly shuffled corpora, and $\Delta \mathrm { W e e d s P r e c } ( w _ { 1 } , w _ { 2 } )$ =

<table><tr><td rowspan="2"></td><td colspan="4"> $n _ { \mathrm { h y p e r } } > n _ { \mathrm { h y p o } }$ </td><td colspan="4"> $n _ { \mathrm { h y p e r } } < n _ { \mathrm { h y p o } }$ </td><td rowspan="2">average</td></tr><tr><td>BLESS 763</td><td>EVAL 2394</td><td>LB 1324</td><td>Weeds 1022</td><td>BLESS 573</td><td>EVAL 1241</td><td>LB 436</td><td>Weeds 405</td></tr><tr><td>size random</td><td>50.00</td><td>50.00</td><td>50.00</td><td>50.00</td><td>50.00</td><td>50.00</td><td>50.00</td><td>50.00</td><td>50.00</td></tr><tr><td>frequency</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>50.00</td></tr><tr><td>WeedsPrec</td><td>93.97</td><td>94.78</td><td>96.45</td><td>95.01</td><td>4.54</td><td>8.30</td><td>8.49</td><td>9.14</td><td>51.33</td></tr><tr><td>SLQS Row</td><td>96.46</td><td>91.73</td><td>96.60</td><td>95.99</td><td>7.68</td><td>21.19</td><td>12.84</td><td>13.09</td><td>54.45</td></tr><tr><td>SLQS</td><td>87.94</td><td>84.04</td><td>83.16</td><td>75.64</td><td>52.53</td><td>46.25</td><td>40.14</td><td>32.35</td><td>62.76</td></tr><tr><td> $\operatorname { K L } ( w )$ </td><td>98.43</td><td>94.74</td><td>96.98</td><td>95.69</td><td>16.93</td><td>21.11</td><td>16.51</td><td>16.79</td><td>57.15</td></tr><tr><td> $\| u _ { w } \| ^ { 2 }$ </td><td>98.17</td><td>93.69</td><td>94.49</td><td>89.92</td><td>28.27</td><td>27.56</td><td>22.25</td><td>21.48</td><td>59.48</td></tr><tr><td> $\Delta \mathrm { W e e d s P r e c }$ </td><td>35.78</td><td>46.07</td><td>50.83</td><td>53.42</td><td>57.77</td><td>49.88</td><td>50.00</td><td>49.88</td><td>49.20</td></tr><tr><td> $\Delta \mathrm { S L Q S }$  Row</td><td>57.54</td><td>59.19</td><td>58.08</td><td>64.19</td><td>47.64</td><td>40.21</td><td>41.74</td><td>42.96</td><td>51.44</td></tr><tr><td> $\Delta \mathrm { S L Q S }$ </td><td>55.83</td><td>55.93</td><td>50.45</td><td>39.43</td><td>73.30</td><td>66.00</td><td>72.25</td><td>64.69</td><td>59.74</td></tr><tr><td> $\Delta \mathrm { K L } ( w )$ </td><td>84.80</td><td>71.39</td><td>58.61</td><td>48.83</td><td>71.38</td><td>56.16</td><td>61.24</td><td>62.96</td><td>64.42</td></tr><tr><td> $\Delta \| u _ { w } \| ^ { 2 }$ </td><td>91.87</td><td>75.23</td><td>72.73</td><td>63.60</td><td>74.69</td><td>58.26</td><td>55.05</td><td>59.26</td><td>68.84</td></tr></table>

Table 9: Accuracy of hypernym classification. For each method, ∆Method is the bias-corrected version. We divided each dataset into two parts based on the word frequencies of hypernym $( n _ { \mathrm { h y p e r } } )$ and hyponym $( n _ { \mathrm { h y p o } } )$ Dataset EVAL denotes EVALution.

WeedsPrec(w<sub>1</sub>, w<sub>2</sub>)  WeedsPrec(w<sub>1</sub>, w<sub>2</sub>). $w _ { 1 }$ is predicted as hypernym if

$$
\begin{array} { r } { \Delta \mathrm { W e e d s P r e c } ( w _ { 1 } , w _ { 2 } ) } \\ { < \Delta \mathrm { W e e d s P r e c } ( w _ { 2 } , w _ { 1 } ) . } \end{array}
$$

• ∆SLQS Row is the bias-corrected version of SLQS Row. $w _ { 1 }$ is predicted as hypernym if ${ \Delta H ( w _ { 1 } ) > \Delta H ( w _ { 2 } ) }$

• ∆SLQS is the bias-corrected version of SLQS. w<sub>1</sub> is predicted as hypernym if $\Delta E ( w _ { 1 } ) > \Delta E ( w _ { 2 } )$ , where

$$
\Delta E ( w ) = \mathrm { M e d i a n } _ { c \in C _ { w } } \Delta H ( c ) .
$$

Evaluation metrics. The classification accuracy of each method was computed separately for the $n _ { \mathrm { h y p e r } } > n _ { \mathrm { h y p o } }$ part and for the $n _ { \mathrm { h y p e r } } < n _ { \mathrm { h y p o } }$ part of each dataset. Then, we calculated the unweighted average of the accuracy over the four datasets for each part and for both parts.

Results. Table 9 shows the classification accuracy. Table 4 in Section 7.2.2 is a summary of Table 9. Looking at the overall accuracy, $\Delta \| u _ { w } \| ^ { 2 }$ and $\Delta \mathrm { K L } ( w )$ were the best and the second best, respectively, for predicting hypernym in hypernymhyponym pairs.

## G Results on Wikipedia dump

We used the Wikipedia dump (Wikimedia Foundation, 2021)<sup>10</sup> with the size of $N = 2 4 . 0 \times 1 0 ^ { 8 }$ tokens and $| V | = 6 4 5 \times 1 0 ^ { 4 }$ vocabulary words, which was preprocessed by Wikiextractor (Attardi, 2015). The training of the SGNS model and the computation of KL divergence were performed as in Appendix A using the same setting<sup>11</sup>. For plotting the results, we used 50,000 words randomly sampled from the 1,114,207 vocabulary words with $n _ { w } \ge 1 0$ . For fitting the regression line, we used 2,662 words with $n _ { w } > 1 0 ^ { 3 }$

Fig. 10 shows the word embeddings of the Wikipedia dump computed with the same setting as that of the text8 corpus. The left panel of Fig. 10 is very similar to Fig. 1, confirming that the result for the text8 corpus is reproduced for the Wikipedia dump. The right panel of Fig. 10 corresponds to Fig. 8 with the axes exchanged and the $\log _ { 1 0 } n _ { w }$ axis rescaled. Again, the two plots are very similar.

However, the result changes when the epoch of training is reduced, thus the optimization is insufficient. Fig. 11 shows the word embeddings of the Wikipedia dump, but the epoch was reduced to 10.

![](images/daed3d3a43e3a12cd9d290d6f9ef4b518d50080379de6529f94c61cad00fce84.jpg)

![](images/2f0d98bb8fa9029681f4e427e4b19d25a51d14bfd48522ba434008888efd7c52.jpg)  
Figure 10: Word embeddings of Wikipedia dump computed with 100 epochs.

![](images/fe6ce5171b951a2291bd531f23912795825338445483aa590f9b8382d8e693c7.jpg)

![](images/b206b6afe11ceb5a764e380e631e99b6a5fd90975e16875c83444ad3fe966ffb.jpg)

Figure 11: Word embeddings of Wikipedia dump computed with 10 epochs.  
![](images/85be2447fe269f9b322a8d228d921eff8d3f2f6b5e3d783226d15a3288cd0d41.jpg)

![](images/6e2c3083378590e0e995d6a32cd9bcfbe652f0d0c87dad9955a47043f8f5105c.jpg)  
Figure 12: Two pre-trained word embeddings. Each regression line was fitted to all the points in the scatterplot.

![](images/47b768f6b4718c6e9136285b357d1c5f91303d1de07c4c57477f899ccd1185da.jpg)

![](images/2621f83cc0fb7d03121eef3f6c828e356410a9f255f9cefaf3edc6db6f598ffc.jpg)

![](images/aa8545291893b44687065a25cdd8dde6b0100b64571b9e8621ec97fa62599282.jpg)

![](images/35c59d5f4eda81f1d8ce78ef28aedbb9babbabf7824640ddddd9d2f20a4dbd42.jpg)  
Figure 13: Linear relationship between the KL divergence and the squared norm of contextualized embedding for BERT, RoBERTa, GPT-2, and Llama 2. The color represents token frequency.

In the left panel, the linear relationship was not reproduced. Looking at the right panel, the norm of embedding reduces for low-frequency words with $n _ { w } < 1 0 0 ;$ plots of the same shape are also found in the literature (Schakel and Wilson, 2015; Arefyev et al., 2018; Pagliardini et al., 2018; Khodak et al., 2018). This is considered a consequence of insufficient optimization epochs; the norm of parameters tends to be smaller due to the implicit regularization (Arora et al., 2019), thus the trained parameters do not satisfy the ideal SGNS model (4) very well, particularly for low-frequency words.

## H Results on pre-trained word embeddings

In this section, we show that the linear relationship between the KL divergence and the squared norm of word embedding holds also for pre-trained word embeddings.

## H.1 Pre-trained fastText embeddings

We used Wiki word vectors provided by Bojanowski et al. (2017). These 300-dimensional embeddings are trained for 5 epochs on Wikipedia with the fastText model. We used the same KL divergence as in Appendix G, which was calculated on the Wikipedia dump corpus. Results are shown in the left panel of Figure 12, where we randomly selected 10,000 words that appeared not less than $1 0 ^ { 4 }$ times in the Wikipedia dump.

## H.2 Pre-trained SGNS embeddings

We used pre-trained SGNS vectors provided by Li et al. (2017). These 500-dimensional embeddings are trained for 2 epochs on Wikipedia with the SGNS model. We used the same KL divergence as in Appendix G, which was calculated on the Wikipedia dump corpus. Results are shown in the right panel of Figure 12, where we randomly selected 10,000 words that appeared not less than $1 0 ^ { 4 }$ times in the Wikipedia dump.

## I Results on contextualized embeddings

Settings. For the experiment of contextualized word embeddings, we used embeddings obtained from the final layer of BERT, RoBERTa, GPT-2, and Llama 2. We obtained 2000 sentences from One Billion Word Benchmark (Chelba et al., 2014) and input them into each language model to get contextualized embeddings of all tokens. Special tokens at the beginning and end of tokenized inputs, if any, were excluded.

<table><tr><td rowspan="2"></td><td colspan="2">raw</td><td colspan="2">whitened</td></tr><tr><td> $R ^ { 2 }$ </td><td>COS</td><td> $R ^ { 2 }$ </td><td>COS</td></tr><tr><td>BERT</td><td>0.183</td><td>0.952</td><td>0.003</td><td>0.898</td></tr><tr><td>RoBERTa</td><td>0.557</td><td>0.977</td><td>0.196</td><td>0.943</td></tr><tr><td>GPT-2</td><td>0.054</td><td>0.812</td><td>0.431</td><td>0.905</td></tr><tr><td>Llama 2</td><td>0.112</td><td>0.902</td><td>0.127</td><td>0.894</td></tr></table>

Table 10: Linear relationship strength between KL divergence and squared norm of language model contextual word embeddings. Presented are coefficients of determination $( R ^ { 2 } )$ and uncentered correlation coefficients (cosine similarity) for both raw and whitened embeddings. Larger values indicate better performance.

Results. Looking at the scatterplots in Fig. 13, approximate linear relationships can be observed in BERT, RoBERTa, and Llama 2, but in GPT-2, the linear relationship is somewhat weaker. According to the values in Table 10, whitening improves the linear relationship for GPT-2 and Llama 2, but it worsens for BERT and RoBERTa, and the effect of whitening is not clear-cut. While there is still room for discussion, overall, an approximate linear relationship between KL divergence and the squared norm of contextual embeddings appears to hold.

## J Basic properties of the exponential family of distributions

The expectation and covariance matrix. The first and second derivatives of $\psi ( u )$ are computed as

$$
\begin{array} { r l r } {  { \frac { \partial \psi ( u ) } { \partial u } = e ^ { - \psi ( u ) } \frac { \partial } { \partial u } \sum _ { w ^ { \prime } \in V } q ( w ^ { \prime } ) e ^ { \langle u , v _ { w ^ { \prime } } \rangle } } } \\ & { } & { \qquad = e ^ { - \psi ( u ) } \sum _ { w ^ { \prime } \in V } q ( w ^ { \prime } ) v _ { w ^ { \prime } } e ^ { \langle u , v _ { w ^ { \prime } } \rangle } } \\ & { } & { \qquad = \sum _ { w ^ { \prime } \in V } p ( w ^ { \prime } | u ) v _ { w ^ { \prime } } , } \end{array}
$$

$$
\begin{array} { l } { { \displaystyle \frac { d ^ { 2 } \mathrm { S y } ( x ) } { d t } = \frac { \partial } { \partial u _ { \alpha } ^ { \prime } } ( e ^ { - \mathrm { i } \omega ( x ) } ) \sum _ { \alpha = K } q ( u ( \omega ) ) _ { \alpha \alpha } e ^ { - \mathrm { i } \omega ( x ) \omega _ { \alpha } t } ) ^ { \top } } } \\ { { \displaystyle = \mathrm { e } ^ { - \mathrm { i } \omega ( x ) } \cdot \frac { \partial } { \partial u _ { \alpha } } \sum _ { \alpha = K } q ( x ) \mathrm { s y } _ { \alpha \alpha } ^ { \top } e ^ { \mathrm { i } \omega ( x ) \omega _ { \alpha } t } } } \\ { ~ - ~ \frac { \partial \mathrm { e } ^ { - \mathrm { i } \omega ( x ) } } { \partial t } \sum _ { \alpha = K } q ( x ) \mathrm { s y } _ { \alpha \alpha } ^ { \top } e ^ { \mathrm { i } \omega ( x ) \omega _ { \alpha } t } } \\ { ~ - ~ e ^ { - \mathrm { i } \omega ( x ) } \sum _ { \alpha = K } q ( x ) e ^ { - \mathrm { i } \omega ( x ) } e ^ { - \mathrm { i } \omega ( x ) \omega _ { \alpha } t } } \\ { ~ - ~ \frac { \mathrm { Y } ( x ) } { \partial u _ { \alpha } } \sum _ { \alpha = K } q ( x ) \mathrm { s y } _ { \alpha \alpha } ^ { \top } e ^ { \mathrm { i } \omega ( x ) \omega _ { \alpha } t } e ^ { - \mathrm { i } \omega ( x ) \omega _ { \alpha } t } } \\ { ~ - ~ \frac { \mathrm { Y } ( x ) } { \partial u _ { \alpha } } ~ e ^ { \mathrm { i } \omega ( x ) } \cdot \sum _ { \alpha = K } q ( x ) e ^ { - \mathrm { i } \omega ( x ) \omega _ { \alpha } t } e ^ { - \mathrm { i } \omega ( x ) \omega _ { \alpha } t } } \\ { ~ - ~ \sum _ { \alpha = K } q ( x ) \mathrm { i n t e r } _ { \alpha } ^ { \top } e ^ { - \mathrm { i } \omega ( x ) \omega _ { \alpha } t } e ^ { - \mathrm { i } \omega ( x ) \omega _ { \alpha } t } } \\  ~ - ~  \end{array}
$$

showing (6) and (7), respectively.

KL divergence. For computing the KL divergence, first note that

$$
\begin{array} { r l r } {  { \log { \frac { p ( w ^ { \prime } | u _ { 1 } ) } { p ( w ^ { \prime } | u _ { 2 } ) } } } } \\ & { } & { = \langle u _ { 1 } - u _ { 2 } , v _ { w ^ { \prime } } \rangle - \psi ( u _ { 1 } ) + \psi ( u _ { 2 } ) } \end{array}
$$

from (4). Thus, the KL divergence is

$$
\begin{array} { r l r } {  { \mathrm { K L } ( p ( \cdot | u _ { 1 } ) \parallel p ( \cdot | u _ { 2 } ) ) = } } \\ & { \sum _ { w ^ { \prime } \in V } p ( w ^ { \prime } | u _ { 1 } ) \Big ( \langle u _ { 1 } - u _ { 2 } , v _ { w ^ { \prime } } \rangle - \psi ( u _ { 1 } ) + \psi ( u _ { 2 } ) \Big ) } \\ & { = \langle u _ { 1 } - u _ { 2 } , \eta ( u _ { 1 } ) \rangle - \psi ( u _ { 1 } ) + \psi ( u _ { 2 } ) , } & { ( 1 8 ) } \end{array}
$$

showing (8).

Approximation of KL divergence. Next, we consider the Taylor expansion of $\psi ( u )$ at $u = u _ { 1 }$ By ignoring higher order terms of $O ( \Vert u - u _ { 1 } \Vert ^ { 3 } )$ we have

$$
\begin{array} { l } { \displaystyle \psi ( \boldsymbol { u } ) \simeq \psi ( u _ { 1 } ) + \frac { \partial \psi ( \boldsymbol { u } ) } { \partial \boldsymbol { u } ^ { \top } } \bigg \vert _ { u _ { 1 } } ( \boldsymbol { u } - \boldsymbol { u } _ { 1 } ) } \\ { \displaystyle \qquad + \frac { 1 } { 2 } ( \boldsymbol { u } - \boldsymbol { u } _ { 1 } ) ^ { \top } \frac { \partial ^ { 2 } \psi ( \boldsymbol { u } ) } { \partial \boldsymbol { u } \partial \boldsymbol { u } ^ { \top } } \bigg \vert _ { u _ { 1 } } ( \boldsymbol { u } - \boldsymbol { u } _ { 1 } ) . } \end{array}
$$

Using (6) and $( 7 )$ , we can rewrite this expression for $u = u _ { 2 }$ as

$$
\begin{array} { l } { \displaystyle \psi ( \boldsymbol { u } _ { 2 } ) \simeq \psi ( \boldsymbol { u } _ { 1 } ) + \langle \boldsymbol { u } _ { 2 } - \boldsymbol { u } _ { 1 } , \eta ( \boldsymbol { u } _ { 1 } ) \rangle } \\ { \displaystyle \qquad + \frac { 1 } { 2 } ( \boldsymbol { u } _ { 2 } - \boldsymbol { u } _ { 1 } ) ^ { \top } G ( \boldsymbol { u } _ { 1 } ) ( \boldsymbol { u } _ { 2 } - \boldsymbol { u } _ { 1 } ) , } \end{array}\tag{19}
$$

and substituting it into (18), we obtain

$$
\begin{array} { r l } & { 2 \mathrm { K L } ( p ( \cdot | u _ { 1 } ) \parallel p ( \cdot | u _ { 2 } ) ) } \\ & { \qquad \simeq ( u _ { 1 } - u _ { 2 } ) ^ { \top } G ( u _ { 1 } ) ( u _ { 1 } - u _ { 2 } ) , } \end{array}\tag{20}
$$

showing (9) for $i \ = \ 1$ . Considering the Taylor expansion of $G ( u )$ at $u \ : = \ : u _ { 2 }$ , each element of $G ( u _ { 1 } ) \mathrm { { ~ i s ~ } } G _ { i j } ( u _ { 1 } ) = G _ { i j } ( u _ { 2 } ) + O ( \vert \vert u _ { 1 } - u _ { 2 } \vert \vert )$ Thus we can rewrite the right hand side of (20) as $\left( u _ { 1 } - u _ { 2 } \right) ^ { \top } ( G ( u _ { 2 } ) + O ( \left. u _ { 1 } - u _ { 2 } \right. ) ) \left( u _ { 1 } - u _ { 2 } \right) \simeq$ $\left( u _ { 1 } - u _ { 2 } \right) ^ { \top } G ( u _ { 2 } ) \left( u _ { 1 } - u _ { 2 } \right) + O ( \left. u _ { 1 } - u _ { 2 } \right. ^ { 3 } )$ . Therefore, we have shown that (9) holds for both $i = 1$ and $i = 2$

## K High-dimensional random vectors

Random vector setting. In this section, we adopt a probabilistic viewpoint and treat the elements of vectors u and v as random variables denoted by $u ^ { i }$ and $v ^ { i }$ for $i = 1 , \ldots , d$ to estimate the orders of magnitude of various quantities, such as vector norms. Although the embedding vectors $\{ u _ { w } \} _ { w \in V } .$ $\{ v _ { w ^ { \prime } } \} _ { w ^ { \prime } \in V }$ are not random variables, the random variable setting is justified when we randomly sample words w and $w ^ { \prime }$ from a large corpus and set $u = u _ { w }$ and $v = v _ { w ^ { \prime } }$ . To simplify the analysis, we assume that the vector elements are distributed independently. While we could relax this assumption by imposing the spherical condition (Jung and Marron, 2009; Aoshima et al., 2018), we leave this extension for future work.

We aim to discuss the relative magnitudes of vectors, so rescaling the vectors does not affect the argument. Therefore, we assume that each element is proportional to d− $^ { - 1 / 2 } .$ , i.e., $u ^ { i } = O _ { p } ( d ^ { - 1 / 2 } )$ $v ^ { i } = O _ { p } ( d ^ { - 1 / 2 } )$ . The squared norm of u is $\| u \| ^ { 2 } =$ $\begin{array} { r } { \sum _ { i = 1 } ^ { d } ( \dot { u } ^ { i } ) ^ { 2 } = O _ { p } ( d \cdot ( d ^ { - 1 / 2 } ) ^ { 2 } ) = O _ { p } ( 1 ) } \end{array}$ , and the norm itself is also $\| u \| = ( \| u \| ^ { 2 } ) ^ { 1 / 2 } = O _ { p } ( 1 )$ Here $O _ { p } ( 1 )$ means that the magnitude of the vector remains bounded even if the dimension d increases. The same applies to v, i.e., $\| v \| = O _ { p } ( 1 )$ . The inner product of u and v is also $\begin{array} { r } { \langle u , v \rangle = \sum _ { i = 1 } ^ { d } u ^ { i } v ^ { i } = } \end{array}$ $O _ { p } ( d \cdot ( d ^ { - 1 / 2 } ) ^ { 2 } ) = O _ { p } ( 1 )$ . Throughout this section, we consider magnitudes up to $O ( d ^ { - 1 } )$ and ignore higher order terms of $O ( d ^ { - 3 / 2 } )$ for sufficiently large d.

Inner product with centered vector. Each element of centered vector $u \mathrm { ~ - ~ } \bar { u }$ is $u ^ { i } - \bar { u } ^ { i } \ =$ $O _ { p } ( d ^ { - 1 / 2 } )$ , thus $\begin{array} { r } { \| u - \bar { u } \| ^ { 2 } = \sum _ { i = 1 } ^ { d } ( u ^ { i } - \bar { u } ^ { i } ) ^ { 2 } = } \end{array}$ $\bar { O _ { p } } ( d \cdot ( d ^ { - 1 / 2 } ) ^ { 2 } ) = O _ { p } ( 1 )$ . However, the inner product

$$
\langle u - \bar { u } , v \rangle = O _ { p } ( d ^ { - 1 / 2 } ) ,\tag{21}
$$

i.e., it tends to zero as $d \mathrm { ~  ~ { ~  ~ } ~ } \infty .$ . Similarly, $\langle u , v - \bar { v } \rangle ~ = ~ O _ { p } ( d ^ { - 1 / 2 } )$ . To show (21), note that $\begin{array} { r } { \mathbb { E } ( u ^ { i } - \bar { u } ^ { i } ) = \sum _ { w \in V } p ( w ) ( u _ { w } ^ { i } - \bar { u } _ { w } ^ { i } ) = 0 } \end{array}$ Thus, $\mathbb { E } ( ( u ^ { i } - \bar { u } ^ { i } ) v ^ { i } ) \stackrel { \sim } { = } \mathbb { E } ( u ^ { i } - \bar { u } ^ { i } ) \mathbb { E } ( v ^ { i } ) = 0 \mathrm { . }$ The variance is ${ \mathbb E } ( ( ( u ^ { i } - \bar { u } ^ { i } ) v ^ { i } ) ^ { 2 } ) ~ = ~ { \mathbb E } ( ( u ^ { i } -$ $\bar { u } ^ { i } ) ^ { 2 } ) \mathbb { E } ( ( v ^ { i } ) ^ { 2 } ) = O ( d ^ { - 1 } \cdot d ^ { - 1 } ) = O ( d ^ { - 2 } )$ . Therefore, $\mathbb { E } ( \langle u - \bar { u } , v \rangle ) = 0 .$ , and $\mathbb { E } ( \langle u - \bar { u } , v \rangle ^ { 2 } ) =$ $\begin{array} { r l r } { \mathbb { E } ( \sum _ { i = 1 } ^ { d } ( u ^ { i } ~ - ~ \bar { u } ^ { i } ) v ^ { i } ) ^ { 2 } ) } & { { } = } & { \sum _ { i = 1 } ^ { d } \mathbb { E } ( ( ( u ^ { i } ~ - } \end{array}$ $\begin{array} { r } { \bar { u } ^ { i } ) v ^ { i } \dot { \textgreater } ^ { 2 } ) + \sum _ { i \neq j } \mathbb { E } ( ( u ^ { i } - \bar { u } ^ { i } ) v ^ { i } ) \mathbb { E } ( ( \bar { u } ^ { j } - \bar { u } ^ { j } ) v ^ { j } ) = } \end{array}$ $O ( d \cdot d ^ { - 2 } ) + \bar { 0 } = O ( d ^ { - 1 } )$ . This proves (21).

u¯ approximates $\mathbf { \Delta } \mathbf { u _ { 0 } }$ . Regarding $v ,$ we used only the property $v ^ { i } = O _ { p } ( d ^ { - 1 / 2 } )$ when deriving (21). $\mathrm { S o , }$ the result does not change if we replace v by $v - \bar { v } \colon \langle u - \bar { u } , v - \bar { v } \rangle = O _ { p } ( d ^ { - 1 / 2 } )$ . However, the result changes if we further replace u by u<sub>0</sub>:

$$
\langle u _ { 0 } - \bar { u } , v - \bar { v } \rangle = O _ { p } ( d ^ { - 1 } ) ,\tag{22}
$$

meaning that u¯ approximates $u _ { 0 }$ . To show this, we first prepare another presentation of (5) as follows. Since $p ( w ^ { \prime } ) = q ( w ^ { \prime } ) \exp ( \langle u _ { 0 } , v _ { w ^ { \prime } } \rangle - \psi ( u _ { 0 } ) ) , ( 5 )$ is expressed as $p ( w ^ { \prime } | u ) = p ( w ^ { \prime } ) \exp ( \langle u - u _ { 0 } , v _ { w ^ { \prime } } \rangle -$ $\psi ( u ) + \psi ( u _ { 0 } ) )$ by canceling out $q ( w ^ { \prime } )$ . We substitute $\psi ( u )$ by (19) with $u _ { 1 } = u _ { 0 } , u _ { 2 } = u$ to obtain

$$
\begin{array} { r } { p ( w ^ { \prime } | u ) \simeq p ( w ^ { \prime } ) \exp ( \langle u - u _ { 0 } , v _ { w ^ { \prime } } - \bar { v } \rangle } \\ { - \frac { 1 } { 2 } ( u - u _ { 0 } ) ^ { \top } G ( u - u _ { 0 } ) ) . \quad \quad } \end{array}\tag{23}
$$

In the above, $\langle u - u _ { 0 } , v _ { w ^ { \prime } } - \bar { v } \rangle = O _ { p } ( d ^ { - 1 / 2 } )$ , and $\begin{array} { r } { ( u - u _ { 0 } ) ^ { \top } G ( u - u _ { 0 } ) = \sum _ { w ^ { \prime } \in V } ( u - u _ { 0 } ) ^ { \top } ( v _ { w ^ { \prime } } - } \end{array}$ $\begin{array} { r c l } { \bar { v } ) ( v _ { w ^ { \prime } } - \bar { v } ) ^ { \top } ( u - u _ { 0 } ) p ( w ^ { \prime } ) } & { = } & { \sum _ { w ^ { \prime } \in V } \langle u - \frac { } { } \partial _ { v } u _ { 0 } } \end{array}$ $u _ { 0 } , v _ { w ^ { \prime } } - \bar { v } \rangle ^ { 2 } p ( w ^ { \prime } ) = { \cal O } ( d ^ { - 1 } )$ , because $\langle u \mathrm { ~ - ~ }$ $u _ { 0 } , v _ { w ^ { \prime } } - \bar { v } \rangle = O _ { p } ( d ^ { - 1 / 2 } )$

Next, we consider (1) and let $\begin{array} { r l } { p ( w ^ { \prime } | w ) } & { { } = } \end{array}$ $p ( w ^ { \prime } | u _ { w } )$ with (23).

$$
\begin{array} { r l } & { p ( w ^ { \prime } ) = \displaystyle \sum _ { w \in V } p ( w ^ { \prime } | u _ { w } ) p ( w ) } \\ & { \qquad \quad \simeq p ( w ^ { \prime } ) \displaystyle \sum _ { w \in V } \exp \Big [ \langle u _ { w } - u _ { 0 } , v _ { w ^ { \prime } } - \bar { v } \rangle } \\ & { \qquad \quad - \frac { 1 } { 2 } \big ( u _ { w } - u _ { 0 } \big ) ^ { \top } G \big ( u _ { w } - u _ { 0 } \big ) \Big ] p ( w ) . } \end{array}
$$

This holds for any $w ^ { \prime } ,$ thus $\begin{array} { r } { \sum _ { w \in V } \exp [ \cdot \cdot \cdot ] p ( w ) \simeq } \end{array}$ 1. By considering the Taylor expansion of the summand above, we have exp $[ \cdots ] = 1 + \langle u _ { w } -$ $\begin{array} { r } { u _ { 0 } , v _ { w ^ { \prime } } - \bar { v } \rangle - \frac { 1 } { 2 } ( u _ { w } - u _ { 0 } ) ^ { \top } G ( u _ { w } - u _ { 0 } ) + \frac { 1 } { 2 } \langle u _ { w } - } \end{array}$ $u _ { 0 } , v _ { w ^ { \prime } } - \bar { v } \rangle ^ { 2 } + O _ { p } ( d ^ { - 3 / 2 } )$ . Therefore, by taking

the summation, we have

$$
\begin{array} { r l } & { \langle \bar { u } - u _ { 0 } , v - \bar { v } \rangle } \\ & { \mathrm { ~ \ - \ } \frac { 1 } { 2 } \displaystyle \sum _ { w \in { \cal V } } ( u _ { w } - u _ { 0 } ) ^ { \top } G ( u _ { w } - u _ { 0 } ) p ( w ) } \\ & { \mathrm { \ ~ \ } + \frac { 1 } { 2 } \displaystyle \sum _ { w \in { \cal V } } \langle u _ { w } - u _ { 0 } , v - \bar { v } \rangle ^ { 2 } p ( w ) \simeq 0 , } \end{array}\tag{24}
$$

where we have replaced $v _ { w ^ { \prime } }$ by v to clarify that $w ^ { \prime }$ is arbitrary. Here, $\begin{array} { r } { \sum _ { w \in V } ( u _ { w } - u _ { 0 } ) ^ { \top } G ( u _ { w } - } \end{array}$ $u _ { 0 } ) p ( w ) = O _ { p } ( d ^ { - 1 } )$ and $\begin{array} { r } { \sum _ { w \in V } \langle u _ { w } - u _ { 0 } , v - } \end{array}$ $\bar { v } \rangle ^ { 2 } p ( w ) = O _ { p } \bar { ( d ^ { - 1 } ) }$ , thus we have proved (22).

In addition to showing (22), we can also obtain an explicit formula for ${ \langle u _ { 0 } ~ - ~ \bar { u } , v }$ $\bar { v } \rangle$ The second term in (24) is $\sum { _ { w \in V } ( u _ { w } \ - }$ $\begin{array} { r c l } { u _ { 0 } ) ^ { \top } G ( u _ { w } \mathrm { ~ - ~ } u _ { 0 } ) p ( w ) } & { = } & { \sum _ { w \in V } \mathrm { t r } G ( u _ { w } \mathrm { ~ - ~ } } \end{array}$ $u _ { 0 } ) ( u _ { w } - u _ { 0 } ) ^ { \top } p ( w ) = \mathrm { t r } G H .$ , where

$$
H : = \sum _ { w \in V } p ( w ) ( u _ { w } - u _ { 0 } ) ( u _ { w } - u _ { 0 } ) ^ { \top } .\tag{25}
$$

The third term in (24) is $\begin{array} { r } { \sum _ { w \in V } \langle u _ { w } - u _ { 0 } , v \ - } \end{array}$ $\begin{array} { r } { \bar { v } \rangle ^ { 2 } p ( w ) ~ = ~ \sum _ { w \in V } ( v ~ - ~ \bar { v } ) ^ { \top } \bar { ( u _ { w } ~ - ~ u _ { 0 } ) } ( u _ { w } ~ - } \end{array}$ $u _ { 0 } ) ^ { \top } ( v - \bar { v } ) = ( \bar { v } - \bar { v } ) ^ { \top } H ( v - \bar { v } )$ . Therefore, we obtain

$$
\begin{array} { r } { \langle u _ { 0 } - \bar { u } , v - \bar { v } \rangle \simeq \frac { 1 } { 2 } ( v - \bar { v } ) ^ { \top } H ( v - \bar { v } ) } \\ { - \frac { 1 } { 2 } \mathrm { t r } G H . } \end{array}\tag{26}
$$

Interstingly, (26) shows that all the context embeddings $\{ v _ { w ^ { \prime } } \} _ { w ^ { \prime } \in V }$ are constrained to a qudractic surface in $\mathbb { R } ^ { d }$

Proof of (12). First note that

$$
\begin{array} { r l } & { ( u - u _ { 0 } ) ^ { \top } G ( u - u _ { 0 } ) } \\ & { = ( u - \bar { u } + \bar { u } - u _ { 0 } ) ^ { \top } G ( u - \bar { u } + \bar { u } - u _ { 0 } ) } \\ & { = ( u - \bar { u } ) ^ { \top } G ( u - \bar { u } ) + ( \bar { u } - u _ { 0 } ) ^ { \top } G ( \bar { u } - u _ { 0 } ) } \\ & { \phantom { = } + 2 ( u - \bar { u } ) ^ { \top } G ( \bar { u } - u _ { 0 } ) . } \end{array}
$$

Using (22), the magnitude of the remaining terms is obtained as follows. $( \bar { u } - u _ { 0 } ) ^ { \top } G ( \bar { u } -$ $\begin{array} { r } { u _ { 0 } ) = \sum _ { w ^ { \prime } \in V } ( \bar { u } - u _ { 0 } ) ^ { \top } ( v _ { w ^ { \prime } } - \bar { v } ) ( v _ { w ^ { \prime } } - \bar { v } ) ^ { \top } ( \bar { u } - } \end{array}$ $\begin{array} { r } { u _ { 0 } ) p ( w ^ { \prime } ) \stackrel {  } { = } \sum _ { w ^ { \prime } \in V } \langle \bar { u } - u _ { 0 } , v _ { w ^ { \prime } } - \bar { v } \rangle ^ { 2 } p ( w ^ { \prime } ) = } \end{array}$ $O ( ( d ^ { - 1 } ) ^ { 2 } ) = O ( d ^ { - 2 } )$ . Similarly, $( u - \bar { u } ) ^ { \top } G ( \bar { u } -$ $\begin{array} { r } { u _ { 0 } ) = \sum _ { w ^ { \prime } \in V } ( u - \bar { u } ) ^ { \top } ( v _ { w ^ { \prime } } - \bar { v } ) ( v _ { w ^ { \prime } } - \bar { v } ) ^ { \top } ( \bar { u } - } \end{array}$ $\begin{array} { r } { u _ { 0 } ) p ( w ^ { \prime } ) = \sum _ { w ^ { \prime } \in V } \langle u - \bar { u } , v _ { w ^ { \prime } } - \bar { v } \rangle \langle v _ { w ^ { \prime } } - \bar { v } , \bar { u } - } \end{array}$ $u _ { 0 } \rangle p ( w ^ { \prime } ) = O ( d ^ { - 1 / 2 } \cdot d ^ { - 1 } ) = O ( d ^ { - 3 / 2 } )$ . Therefore, we have shown that

$$
\begin{array} { l } { { ( u - u _ { 0 } ) ^ { \top } G ( u - u _ { 0 } ) } } \\ { { = ( u - \bar { u } ) ^ { \top } G ( u - \bar { u } ) + O _ { p } ( d ^ { - 3 / 2 } ) , } } \end{array}
$$

where the magnitude of $( u - u _ { 0 } ) ^ { \top } G ( u - u _ { 0 } )$ and $( u - \bar { u } ) ^ { \top } G ( u - \bar { u } )$ is $O _ { p } ( d ^ { - 1 } )$ , and u is arbitrary $u _ { w }$ . Combining this with (11) proves (12).

## L Technical details of the contextualized embeddings

We need only the following additional modifications. The equation (1) for the unigram distribution $p ( w )$ is replaced by

$$
p ( \cdot ) = \sum _ { i = 1 } ^ { N } p ( \cdot | u _ { i } ) / N .
$$

The definition (25) for the matrix H in Appendix K is replaced by

$$
H : = \sum _ { i = 1 } ^ { N } ( u _ { i } - u _ { 0 } ) ( u _ { i } - u _ { 0 } ) ^ { \top } / N .
$$

These modifications simply replace the average weighted by word frequency $p ( w )$ on the vocabulary set V with the simple average over $\{ u _ { i } \} _ { i = 1 } ^ { N }$ For a sufficiently large corpus size N of the training set, the distribution of $\{ u _ { i } \} _ { i = 1 } ^ { N }$ is approximated by a density function $\pi ( u )$ of contextualized embedding u. Therefore, the simple average is interpreted as the expectation with respect to $\pi ( u )$ . Consequently, we can also employ an alternate approach to the definition: $\textstyle { \bar { u } } = \int u \pi ( u ) d u$ $\begin{array} { r } { p ( \cdot ) = \int p ( \cdot | u ) \pi ( u ) } \end{array}$ du and $\begin{array} { r } { H = \int ( u - u _ { 0 } ) ( u - } \end{array}$ $u _ { 0 } ) ^ { \top } \pi ( u ) d u$