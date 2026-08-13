# Revisiting the Optimality of Word Lengths

Tiago Pimentel<sup>1,2</sup> Clara Meister<sup>2</sup> Ethan Gotlieb Wilcox<sup>2</sup>

Kyle Mahowald<sup>3</sup> Ryan Cotterell<sup>2</sup>

<sup>1</sup>University of Cambridge <sup>2</sup>ETH Zürich <sup>3</sup>University of Texas, Austin {tiago.pimentel, clara.meister, ethan.wilcox, ryan.cotterell}@inf.ethz.ch mahowald@utexas.edu

## Abstract

Zipf (1935) posited that wordforms are opti mized to minimize utterances’ communicative costs. Under the assumption that cost is given by an utterance’s length, he supported this claim by showing that words’ lengths are in versely correlated with their frequencies. Communicative cost, however, can be operational ized in different ways. Piantadosi et al. (2011) claim that cost should be measured as the dis tance between an utterance’s information rate and channel capacity, which we dub the channel capacity hypothesis (CCH) here. Following this logic, they then proposed that a word’s length should be proportional to the expected value of its surprisal (negative log-probability in context). In this work, we show that Piantadosi et al.’s derivation does not minimize CCH’s cost, but rather a lower bound, which we term CCH . We propose a novel derivation, suggest ing an improved way to minimize CCH’s cost. Under this method, we find that a language’s word lengths should instead be proportional to the surprisal’s expectation plus its variance-tomean ratio. Experimentally, we compare these three communicative cost functions: Zipf’s, CCH , and CCH. Across 13 languages and several experimental settings, we find that length is better predicted by frequency than either of the other hypotheses. In fact, when surprisal’s expectation, or expectation plus variance-to-mean ratio, is estimated using better language models, it leads to worse word length predictions. We take these results as evidence that Zipf’s longstanding hypothesis holds.

![](images/dd73f9892fccfd374803b028c44053eeab86f19905141636757d4f2c72dae070.jpg)

0 https://github.com/tpimentelms/ optimality-of-word-lengths

## 1 Introduction

Zipf proposed the idea that languages are optimized to minimize their expected utterance length (Zipf, 1935).<sup>1</sup> Under this hypothesis, a word’s length should be inversely proportional to its frequency. Indeed, this relationship has been attested across a wide variety of the world’s languages (Grzybek, 2015; Bentz and Ferrer-i-Cancho, 2016, inter alia).

![](images/f7d8e7af9dd525803502dce3af8c7ca5a05228333ffbe81594a8e7f3f119428b.jpg)  
Figure 1: Mean squared error achieved by a linear model predicting real word lengths under the three hypotheses (lower is better).

In subsequent work, Piantadosi et al. (2011) offered a complementary account of communicative cost. Starting from the hypothesis that information rate should be roughly constant during communication (UID; Fenk and Fenk, 1980; Levy and Jaeger, 2007), they argue that word lengths should make information rates as close as possible to a hypothetical channel capacity, where the channel refers to the means by which information is transferred from one person to another. We term this the channel capacity hypothesis (CCH).<sup>2</sup> They conclude that lengths should be proportional to a word’s expected surprisal instead.<sup>3</sup>

As the communicative efficiency of language provides important insights into human cognition (Gibson et al., 2019), Piantadosi et al.’s finding that word lengths are better explained by average surprisal than frequency has been influential. However, there are shortcomings: First, the manner in which Piantadosi et al. finds a solution which minimizes the cost associated with CCH is not formally specified. And second, Piantadosi et al.’s empirical results have been shown to be sensitive to a number of methodological decisions, such as the choice of text-encoding (e.g., ascii vs. unicode), the inclusion of non-conventional wordforms and other orthographic conventions of a language (Meylan and Griffiths, 2021; Levshina, 2022). Thus, there remain fundamental open questions about the relationship between communicative efficiency and word length. Here, we aim to clarify both theoretical and empirical aspects of this relationship.

Theoretically, we offer a novel, formal derivation of Piantadosi et al.’s claim. We find that Piantadosi et al. (2011) optimize not for the objective under the CCH, but for a lower bound on it instead; we call this the $\mathrm { C C H _ { \downarrow } }$ objective. We then provide a closed-form expression for the function that determines word lengths under CCH: Word lengths should be proportional to the expected surprisal plus its variance-to-mean ratio. Importantly, we derive the solution above by framing the problem of assigning wordforms as the optimization of a cost function.<sup>4</sup> By instantiating this optimization problem with the objectives posited by each hypothesis (ZIPF, CCH, and $\mathrm { C C H _ { \downarrow } ) }$ we can compute their word length predictions within a single, unified framework.

Empirically, we offer a large-scale comparison of $Z \mathrm { I P F } ^ { \prime } \mathbf { S } ,$ $\mathrm { C C H _ { \downarrow } \mathrm { ^ { \circ } s } , }$ , and CCH’s word length predictions across 13 typologically diverse languages. Notably, we use neural language models to estimate words’ surprisals, which provides more accurate estimates than the n-gram models relied on by prior work on this topic (Piantadosi et al., 2011; Meylan and Griffiths, 2021; Levshina, 2022). We find strong evidence (see Fig. 1) that languages are optimized to minimize their utterance lengths: A word’s frequency (ZIPF’s prediction) offers stronger predictive power over word lengths than either the surprisal’s expected value $\mathrm { ( C C H _ { \downarrow } ) _ { s } }$ prediction) or expected surprisal plus variance-tomean ratio (CCH’s prediction). We conclude that Zipf’s longstanding theory stands strong.

## 2 The Lexicalization Problem

Zipf (1935, 1949) posited that the lexicon is optimized for communication, taking the needs of both speakers and listeners into account. In this section, we formalize a slice of this optimization problem. First, we assume a fixed (but potentially infinite) vocabulary $\mathcal { W }$ of words, each of which we denote as $w \in \mathcal { W }$ , and a fixed alphabet Σ. Given a vocabulary and alphabet, we define a lexicon as a function that outputs a wordform for each word; we denote a lexicon as $\phi : \mathcal { W } \to \Sigma ^ { * }$ and a wordform as $\phi ( w ) \in \Sigma ^ { * }$ . Note that we distinguish between a word, which is an abstract notion or concept, and its wordform, which is its orthophonological realization. Further, let $p ( w , c )$ be a language’s joint probability distribution over these words and their prior linguistic context $c \in \mathcal { W } ^ { * } . ^ { 5 }$ Finally, let cost $[ \phi ] ( w , c )$ be a cost function that, given a lexicon, outputs the communicative cost of a word in context. It is often suggested that the only attribute of a wordform $\phi ( w )$ that the function cost[ϕ] is concerned with is its length $| \phi ( w ) |$ , where $| \cdot | : \Sigma ^ { * } \to \mathbb { Z } _ { + }$ . We now define the optimization problem proposed by Zipf as follows. Definition 1. The lexicalization problem is the task of finding an optimal lexicon $\phi ^ { * }$ , which minimizes cost[ϕ]. This lexicon can be described formally as the solution to

$$
\begin{array} { r } { \phi ^ { * } = \underset { \phi } { \mathrm { a r g m i n } } \underset { p ( w , c ) } { \mathrm { ~ E ~ } } [ \phi ] ( w , c ) } \\ { s u b j e c t t o \phi \in \Phi _ { \ell } } \end{array}\tag{1}
$$

where $\Phi _ { \ell }$ is the set of valid ϕ for language ℓ.

There are many assumptions that one could make about $\Phi _ { \ell } \mathbf { \ ' } _ { \mathbf { s } }$ characteristics. We make a few explicit in the following remark.

Remark 1. We take the set $\Phi _ { \ell }$ to include all lexicons which: 1 only produce phonotactically valid wordforms,<sup>6</sup> 2 respect morphological composi-$t i o n , ^ { 7 } a n d \textcircled { 3 }$ are uniquely decodable.<sup>8</sup>

Another implicit constraint $\textcircled{4}$ regarding valid $\phi -$ which comes from our specification of the output space of ϕ—is that these mappings only produce integer-length wordforms.

In the subsequent sections, we consider relaxations of eq. (1) to arrive at simple solutions regarding the lengths provided by optimal lexica. Specifically, we partially relax constraint $\textcircled{1}$ and fully relax constraint $\textcircled{2}$ when deriving a lexicon with minimal utterance length. Further, when deriving optimal results for both CCH and $\mathrm { C C H _ { \downarrow } }$ , we also fully relax constraints 1 , 3 , and $\textcircled{4} . 9$ Note that, as in all optimization problems, removing constraints always yields a lower bound on the expected cost we obtain under an optimal lexicon.<sup>10</sup>

## 3 Revisiting Zipf’s Law of Abbreviation

Zipf (1935, 1949) posited a specific form that the cost function in eq. (1) should take. Concretely, he posited that lexica were optimized with the goal of minimizing speakers’ utterance lengths, which can be written as cost $[ \phi ] ( w , c ) = | \phi ( w ) |$ in our notation. In an attempt to formalize his position, he proposed his eponymous law of abbreviation:

$$
| \phi _ { \mathrm { z i p f } } ( w ) | \propto - \log p ( w )\tag{2}
$$

Over the years, Zipf’s law of abbreviation has been empirically investigated numerous times (Wimmer et al., 1994; Sigurd et al., 2004; Kanwal et al., 2017; Koplenig et al., 2022; Levshina, 2022; Petrini et al., 2022, 2023). We now present a formal derivation of Zipf’s law of abbreviation by viewing it as an instantiation of the lexicalization problem.

Hypothesis 1. Zipf’s hypothesis predicts that communication is made optimal by the mapping $\phi _ { \mathrm { z i p f } }$ that satisfies:

$$
\phi _ { \mathrm { z i p f } } = \underset { \phi } { \mathrm { a r g m i n } } \ \underset { p ( w , c ) } { \mathrm { E } } \ | \phi ( w ) |\tag{3}
$$

If we relax constraints 1 and $\textcircled{2}$ in Remark 1, then the optimal solution to eq. (3) can be achieved by Huffman coding (Huffman, 1952).<sup>11</sup> We know that this optimal solution’s word lengths respect:

$$
| \phi _ { \mathrm { z i p f } } ( w ) | \leq - \log _ { | \Sigma | } p ( w ) + 1\tag{4}
$$

which can be roughly approximated as $| \phi _ { \mathrm { z i p f } } ( w ) | \approx - \log _ { | \Sigma | } p ( w )$ . Unfortunately, empirical evidence suggests that this solution, which suggests the proportionality constant in eq. (2) equals 1, is not representative of how natural languages behave (Pimentel et al., 2021c). It thus gives us little insight into how actual wordforms should behave.

Fortunately, we can derive a more interesting result where the proportionality in eq. (2) still holds by only partially relaxing 1 from Remark 1. We first assume a very simplistic model of phonotactics. Given an alphabet Σ of phones, let $L _ { \ell } \subset \Sigma ^ { * }$ be the set of phonotactically valid wordforms in language ℓ. Note that this assumes deterministic phonotactics (Gorman, 2013; Dai and Futrell, 2021).<sup>12</sup> Further, define PREFIXES $\begin{array} { r } { \left( L _ { \ell } \right) \stackrel { \mathrm { d e f } } { = } \left\{ \alpha _ { < t } \ : | \ : \alpha \in L _ { \ell } , t \leq \right. } \end{array}$ $| \alpha | \}$ to be the set of all prefixes in this language.

Assumption 1. The constant phonotactic assumption assumes there exists a $K \in \mathbb { Z } _ { > 0 }$ such that $K \leq | \Sigma |$ and, for every string $\alpha { \in } \mathrm { P R E F I X E S } ( L _ { \ell } )$ there exist exactly K symbols $\{ \sigma _ { k } \} _ { k = 1 } ^ { K }$ for which $\alpha \sigma _ { k } \in \mathrm { P R E F I X E S } ( L _ { \ell } )$

In words, Assumption 1 says that there are exactly K valid symbols with which every phonotactically valid prefix can be extended. Given this assumption, we can now find a solution to eq. (3), which only partially relaxes the phonotactic constraint in Remark 1.

Theorem 1. The minimization problem given in Hypothesis 1 with constraint $\textcircled{2}$ relaxed can be solved by Huffman coding<sup>13</sup> with K symbols. The optimal solution is given by

$$
| \phi _ { \mathrm { z i p f } } ( w ) | = | \phi _ { \mathrm { h u f f } _ { K } } ( w ) |\tag{5a}
$$

$$
\leq - { \frac { 1 } { \log _ { | \Sigma | } K } } \log _ { | \Sigma | } p ( w ) + 1\tag{5b}
$$

Proof. The proof is available in App. C.

Theorem 1 makes precise the sense in which we claim to have derived Zipf’s law of abbreviation. Under the rough approximation $| \phi _ { \mathrm { z i p f } } ( w ) | ~ \approx$ $- \frac { \log _ { | \Sigma | } p ( w ) } { \log _ { | \Sigma | } K }$ , the proportionality in eq. (2) is realized through the unknown constant $1 / \log _ { | \Sigma | } K$

## 4 Revisiting Piantadosi et al. (2011)

What’s wrong with Zipf’s law of abbreviation? The solution in eq. (5) is only optimal if one believes that $\mathrm { c o s t } [ \phi ] ( w , c ) = | \phi ( w ) |$ is the true objective underlying the lexicalization problem. However, more recent work on communicative efficiency (e.g., Piantadosi et al., 2009, 2011) has proposed that speakers may intend to optimize another objective instead. Specifically, one can take the perspective that language is an exchange of information via a noisy communication channel, where information is operationalized as a word’s surprisal $\mathrm { H } ( w \mid c ) = - \log p ( w \mid c )$ . This channel has an inherent capacity ${ \mathfrak { C } } \in \mathbb { R } _ { > 0 }$ at which information can be transmitted while the receiver is still able to effectively decode the underlying message. Under this perspective, optimal communication happens when a word’s information rate $( \frac { \mathrm { H } ( w | c ) } { | \phi ( w ) | }$ , in bits per character) is kept as close as possible to C. A word’s channel deviation is then the difference between its information rate and channel capacity. This hypothesis can thus be stated within the framework of the lexicalization problem by defining the $\mathrm { c o s t } [ \phi ] ( w , c )$ of a lexicon as a function of the channel deviation.

Hypothesis 2. The channel capacity hypothesis predicts that communication is made optimal by the mapping $\phi _ { \mathrm { c c h } }$ that satisfies:

$$
\phi _ { \mathrm { c c h } } = \underset { \mathcal { \phi } } { \mathrm { a r g m i n } } \ \underset { p ( w , c ) } { \mathrm { E } } { \mathrm { d i s t } } \bigg ( \frac { \mathrm { H } ( w \mid c ) } { \lvert \phi ( w ) \rvert } , \mathfrak { C } \bigg )\tag{6}
$$

where dist $( x , y )$ is afunction that quantifies how far x is from $y . ^ { 1 4 }$

Intuitively, eq. (6) penalizes lexica where the length of a word causes its information rate to deviate from the channel capacity. Thus, $\phi _ { \mathrm { c c h } }$ will generate word lengths which produce information rates that are as uniform as possible. It follows that it can be categorized under the larger umbrella of the uniform information density hypothesis (UID; Fenk and Fenk, 1980; Levy and Jaeger, 2007). As discussed by Meister et al. (2021), however, UID has several potential interpretations, only one of which involves maximizing the use of a communication channel. Here, we will only discuss it under this perspective, and assume that its operationalization is given by eq. (6).

## 4.1 Optimal Word Lengths

The exact solution to eq. (6) depends on the choice of dist. In this section, we assume a quadratic distance function, i.e., dist $( x , { \mathfrak { C } } ) \ = \ ( x - { \mathfrak { C } } ) ^ { 2 }$ Efficient lexica should thus minimize the expected value of the square of the channel deviation under $p ( w , c )$ (i.e., its mean squared error). We now derive a closed-form expression for CCH-optimal word lengths under this cost function. As in Theorem 1, we relax the morphological 2 constraint. Beyond this, we also relax the phonotactic 1 , unique-decodability 3 , and the integer-length 4 constraints. Note that, unlike in Theorem 1, we need to relax 4 here because we have no efficient combinatorial algorithm to solve eq. (6).

Theorem 2. Under Hypothesis 2, if we relax $\textcircled{1} , \textcircled{2 } ,$ 3 and 4 , the optimal word lengths are given by

$$
\left| \phi _ { \mathrm { c c h } } ( w ) \right| = \frac { 1 } { \mathfrak { C } } \frac { \mathrm { E } } { \mathrm { \Upsilon } _ { p ( c | w ) } } \left[ \mathrm { H } ^ { 2 } ( w \mid c ) \right]\tag{7}
$$

Proof. The proof is available in App. D.

We note that Eq. (7) is equivalent to the expected surprisal plus a variance-to-mean ratio.<sup>15</sup>

## 4.2 Choices of Distance

In the above section, we assumed a quadratic penalty for a word’s channel deviation. There is, however, no inherent reason why dist should be quadratic. We thus examine alternative ways to quantify the deviation between a word’s information rate and the channel capacity. Different choices of dist will then each define a cost function through cost $\begin{array} { r } { [ \phi ] ( w , c ) = \mathrm { d i s t } \Big ( \frac { \mathrm { H } ( w | c ) } { | \phi ( w ) | } , \mathfrak { C } \Big ) } \end{array}$

Any specific utterance should fall in one of three cases: First, a word’s information rate may be at capacity, i.e., when $\begin{array} { r } { \frac { \mathrm { H } ( w | c ) } { | \phi ( w ) | } = \mathfrak { C } } \end{array}$ . In this case, there are no CCH-based costs. As the capacity is a real number, however, this is virtually impossible in practice. Second, information rate may be below capacity. This will imply an opportunity cost on communication: speakers will need more time to produce their utterances than desired, which is not ideal from the perspective of communicative efficiency (Levy and Jaeger, 2007; Kanwal, 2018).

Third, information rate may be above capacity. This again implies a cost on communication; since communicating above a channel’s capacity is provably noisy (Shannon, 1948), there may be communication faults which will either lead to the wrong meaning being conveyed, or will require a potential retransmission of the message.

The quadratic distance function that we have proposed above assumes a symmetric cost, where communication above or below capacity are equally harmful. It is, however, reasonable to assume that the cost associated with communicating above capacity may be higher than the opportunity cost of communicating below it. This leads us to propose costs based on the following generalized distance function:

$$
\mathrm { d i s t } ( x , { \mathfrak { C } } ) = \left\{ \begin{array} { r l r } { \lambda ( x - { \mathfrak { C } } ) ^ { 2 } } & { { \mathbf { i f } } x > { \mathfrak { C } } } \\ { ( x - { \mathfrak { C } } ) ^ { 2 } } & { { \mathbf { e l s e } } } \end{array} \right.\tag{8}
$$

where $\lambda \in \mathbb { R } _ { > 0 }$ . Under this generalized distance function, any value $\lambda > 1$ will imply a larger penalty for communicating above than below capacity. Further, when $\lambda = 1$ we recover the symmetric quadratic distance function proposed earlier.

Notably, when assuming this generalized distance function, there is no capacity-agnostic closedform value to which word lengths should be proportional. Here, we find CCH -optimal lengths with a two step process: (i) given a large set of surprisal values paired with their word lengths, we find what the optimal capacity is for a language; (ii) we then use a gradient descent-based optimizer to find the optimal lengths under that capacity.

## 4.3 Piantadosi et al.’s (2011) Lower Bound

In their paper, Piantadosi et al. offer a similar argument to the one proposed in this section. They state, however, that the optimal word lengths follow:

$$
| \phi _ { \mathrm { c c h } _ { \downarrow } } ( w , c ) | \propto \mathrm { H } ( w \mid C )\tag{9}
$$

where $\mathrm { H } ( w \mid \ C )$ is the surprisal of word $w ,$ marginalized over all contexts. While Piantadosi et al. intended to find a solution which minimizes the cost associated with CCH, they actually do something else. We find that Piantadosi et al.’s proposal optimizes a different instantiation of the lexicalization problem, one that does not use the objective that formally corresponds to the CCH hypothesis.<sup>16</sup> We give the objective Piantadosi et al.’s proposal is the solution to below as its own hypothesis.

Hypothesis 3. Piantadosi et al. predict that communication is made optimal by the mapping $\phi _ { \mathrm { c c h _ { \downarrow } } }$ that satisfies:

$$
\begin{array} { l } { \phi _ { \mathrm { c c h } _ { \downarrow } } = \underset { \phi } { \mathrm { a r g m i n } } \underset { p ( w , c ) } { \mathrm { E } } \mathrm { d i s t } \bigg ( \frac { \mathrm { H } ( w \mid C ) } { \vert \phi ( w ) \vert } , \mathfrak { c } \bigg ) } \\ { s u b j e c t t o \phi \in \Phi _ { \ell } } \end{array}\tag{10}
$$

We now give the connection between Hypothesis 3 and eq. (9) in the following theorem.

Theorem 3. Under Hypothesis 3, if we further relax $\textcircled{1} , \textcircled{2 } , \textcircled{3 }$ and $\textcircled{4}$ , the optimal word lengths are given by

$$
| \phi _ { \mathrm { c c h } _ { \downarrow } } ( w ) | = \frac { 1 } { \mathfrak { C } } \mathrm { H } ( w \mid C )\tag{11}
$$

Proof. Using $\phi = \phi _ { \mathrm { c c h _ { \perp } } }$ as given by eq. (11), we get $\mathrm { d i s t } ( \cdot , { \mathfrak { C } } ) = 0$ for all words when evaluating the objective in eq. (10). By definition, this is the minimum for any dist. ■

Note that dist $\left( \frac { \mathrm { H } ( w | C ) } { | \phi ( w ) | } , \mathfrak { C } \right)$ is constant with respect to individual contexts $c .$ Thus, the expectation in eq. (10) can simply be taken over the unigram distribution, $p ( w )$ . Moreover, if dist is a convex function, then, we can use Jensen’s inequality to show that eq. (10) lower-bounds eq. (6).<sup>17</sup> We therefore denote Piantadosi et al.’s hypothesis and solution $\mathrm { C C H } _ { \downarrow }$

Proposition 1. Given a convex distfunction and any $\phi \in \Phi _ { \ell } ,$ , the cost optimized by CCH in Hypothesis 3 lower-bounds CCH’s cost in Hypothesis 2

$$
\begin{array} { r l } & { \mathrel { \phantom { = } } \displaystyle \operatorname { E } _ { p ( w , c ) } \mathrm { d i s t } \bigg ( \frac { \mathrm { H } ( w \mid c ) } { \lvert \phi ( w ) \rvert } , \mathfrak { C } \bigg ) } \\ & { \qquad \geq \underset { p ( w , c ) } { \mathrm { E } } \mathrm { d i s t } \bigg ( \frac { \mathrm { H } ( w \mid C ) } { \lvert \phi ( w ) \rvert } , \mathfrak { C } \bigg ) } \end{array}\tag{12}
$$

Proof. The proof is available in App. E.

We now provide an example to show how $\mathrm { C C H _ { \downarrow } \mathrm { ^ { \circ } s } }$ solution does not minimize dist $\begin{array} { r } { \left( \frac { \mathrm { H } ( w | c ) } { | \phi ( w ) | } , \mathfrak { C } \right) } \end{array}$ under the distribution $p ( w , c )$

Example 1. Let there be a word with a surprisal of 2 bits in ten distinct contexts, and a surprisal of24 bits in a single context; assume all eleven contexts are equiprobable. The word’s average surprisal is thus 4 bits $( i . e . , \ \frac { 1 0 { \cdot } 2 { + } 2 4 } { 1 1 } )$ . Further, assume we have a channel with capacity ${ \mathfrak { C } } = 2 .$ . According to Theorem $^ { 3 , }$ we have $\begin{array} { r } { | \phi _ { \mathrm { c c h } _ { \perp } } ( w ) | = \frac { \mathrm { H } ( w | C ) } { \mathfrak { C } } = 2 , } \end{array}$ which under the CCH objective (eq. (6)) gives us an expected cost of10 (i.e., $\textstyle { \frac { 1 0 } { 1 1 } } \big ( \frac { 2 } { 2 } - 2 \big ) ^ { 2 } + \frac { 1 } { 1 1 }  \big ( \frac { 2 4 } { 2 } - 2 \big ) ^ { 2 } \big )$ $H$ we choose word lengths according to Theorem 2 instead, we get that the length should be $| \phi _ { \mathrm { c c h } } ( w ) | = 7$ . This results in a cost under the CCH objective of roughly 2.86.

## 5 Experimental Setup

## 5.1 Estimating Word Length Predictions

To evaluate the different hypotheses, we test how well their predictions about word lengths align with the lengths of real languages’ wordforms. These predictions require computing surprisals (either unigram or contextual), which are defined according to the true probability distribution $p$ (either as a function of $p ( w )$ , or $p ( w \mid c ) ;$ the distribution $p$ is defined more precisely in App. A). While we do not have access to the true probability distribution $p ,$ we do have samples from it. We use the following estimators of eqs. (5), (7) and (11):

$$
| \widehat { \phi _ { \mathrm { z i p f } } ( w ) } | = - \log q ( w )\tag{13a}
$$

$$
| \widehat { \phi _ { \mathrm { c c h } _ { \downarrow } } ( w ) } | = - \frac { 1 } { | \mathcal { D } _ { w } | } \sum _ { c ^ { \prime } \in \mathcal { D } _ { w } } \log q ( w \mid c ^ { \prime } )\tag{13b}
$$

$$
\vert \widehat { \phi _ { \mathrm { c c h } } ( w ) } \vert = - \frac { c ^ { \prime } \overbrace { \sum _ { \mathrm { \Gamma } } \mathrm { l o g } q ( w \mid c ^ { \prime } ) } ^ { \sum } ) ^ { 2 } } { \sum _ { c ^ { \prime } \in \mathcal { D } _ { w } } \log q ( w \mid c ^ { \prime } ) }\tag{13c}
$$

where $\mathcal { D } _ { w } = \{ c ^ { \prime } \mid ( c ^ { \prime } , w ^ { \prime } ) \in \mathcal { D } , w ^ { \prime } = w \}$ , and $\mathcal { D }$ is our corpus, which we assume to be sampled from the distribution $p .$ In practice, our corpus $\mathcal { D }$ is composed of data from one out of 13 languages from 5 language families in Wiki40B (Guo et al., 2020).

Distribution $q$ is our estimate of $p ,$ which we implement using language models. We use: normalized count statistics to estimate the unigram distribution $p ( w )$ , and transformer models for $p ( w \mid c )$ . Our data and models are described in detail in App. B.<sup>18</sup> Note that we omit unknown constants from eqs. (13a) to (13c) because we only consider scale-invariant evaluation metrics.

## 5.2 Evaluation Metrics

Even with access to the true $p ,$ comparing the word length predictions of the different theories above would be non-trivial. Language evolution is a dynamic and noisy process: Even if one of the above optimization pressures has acted during the creation of languages’ lexica, it is unlikely that they are perfectly optimal with respect to that pressure. We thus cannot simply evaluate whether languages match our predictions exactly. Rather, we can instead measure if the general trends predicted by the different hypotheses match the trends observed in natural language. We will rely on a number of metrics to evaluate our results. Taken together these metrics should allow us to draw conclusions on which theory (if any) best correlates with observed word lengths.

Spearman Correlation. First, we follow prior work (Piantadosi et al., 2011; Meylan and Griffiths, 2021; Levshina, 2022) and use the Spearman correlation to assess the quality of each word-length hypothesis. A positive attribute of this correlation is that it can account for nonlinear relationships, potentially accounting for non-linear optimization obstacles. This metric, however, has a significant drawback: Namely, all wordforms contribute equally to its computation. If we evaluate large enough corpora using Spearman correlations, we will therefore consider vocabularies mostly dominated by low-frequency and uncommon wordforms, such as typos, specialized terms, and names. Yet arguably, when evaluating the different hypotheses, a given word should be weighted according to its usage (i.e, frequency) in a given language, as this is the case in our various optimization problems; a word’s impact on the lexicalization problem’s objective is a function of its frequency. This is perhaps one of the reasons why prior work has limited their analyses to only consider a subset of the most common words per language (Piantadosi et al., 2011), a design choice that we likewise employ in our main experiments.

Pearson Correlation. As a second metric, we evaluate the Pearson correlation between our predictions and actual word lengths. Pearson’s correlation has similar drawbacks to Spearman’s, differing from it only in that its value reflects the strength of linear relationships.

Weighted Mean Squared Error (MSE). As a third metric, we use weighted MSE, which avoids the main drawbacks of the previous metrics. We fit linear regression models (without a bias term) to predict a language’s word lengths using our ZIPF, CCH, or CCH estimators as the sole predictor. Importantly, we weight each squared error term by that words’ frequency (both during this model’s training and evaluation). This design choice makes our method more robust to the set of words being evaluated, since the inclusion of exponentially many low-frequency words should not substantially affect weighted MSE. Note that this procedure is equivalent to measuring the predictive power of each hypothesis, while assuming eqs. (5), (7) and (11) predict an expected length, and that word lengths are normally distributed around these expected values.

## 6 Results

Our main results are presented in Fig. 1 and 2. In short, Fig. 1 shows that words’ frequencies offer stronger predictive power of word lengths (as evinced by smaller MSE) than either of the surprisal-dependent metrics. This result provides evidence for ZIPF’s hypothesis over either CCH or CCH . This result is particularly surprising since we improve on CCH’s optimal word length predictions, but ZIPF’s hypothesis still provides the best predictions.<sup>19</sup> A similar result can be seen in Fig. 2, where frequency offers the strongest correlation with lengths (in terms of both Pearson and Spearman), in all languages but English. Notably, in our results, some languages even have a negative correlation between the two surprisal-based measures and actual word lengths. We now turn to analyzing different methodological choices that could impact our results.

## 6.1 Sensitivity to Tokenization

The first design choice that we analyze here is the choice of tokenizer that we use to preprocess our data. As cross-entropies are necessarily larger or equal to entropies,<sup>20</sup> it is reasonable to expect that our language model surprisal estimates may be, on average, larger than true surprisals. While we do not know the exact per-token nature of this difference, it is conceivable that using UnigramLM tokenization could compound it: On average, longer words will naturally decompose into more subword units, and so when adding subword surprisals, the total error of a word’s surprisal estimate may correlate with its number of subword units.

![](images/3c53a15c05a21bf2f9c7c608c8b544621683341ee8988a6d7dfd59797225dba2.jpg)  
Figure 2: Pearson and Spearman correlation of the three hypotheses in the analyzed languages (higher is better).

To assess the impact of this potential systematic error in our estimates, we thus re-train our models using a vocabulary of 32kfull words, replacing any word not in this set with an unk symbol, which is necessary when working with finite vocabularies. Under this model, all analyzed words are encoded using a single “subword” unit. We then re-analyze the three hypotheses as before. In Fig. 3 (top), we see that a word’s frequency is still a better predictor of its length than the quantities put forth by other hypotheses. Further, in the only case in which CCH offers better predictions than ZIPF (English, as evinced by higher Spearman correlations), their performance difference is now lower than before.<sup>21</sup>

We also estimate ZIPF’s unigram surprisals using tokenized counts, i.e., where we count subword tokens to estimate frequencies instead of directly counting the full words in our training set. We then estimate the suprisal of a word as the sum of the surprisals of its subwords, thus assuming independence between them. We display these new results in Fig. 3 (bottom) under the name Zipf (subwords). We see here that this tokenization scheme increases our measured correlations, and Zipf (subwords) presents the strongest correlations in all languages. Perhaps surprisingly, tokenization seems to not influence MSE as much.

## 6.2 Sensitivity to Word Filtering Protocol

Next, we analyze our results’ sensitivity with respect to how we select the set of words we analyze. Specifically, for our analyses so far we have only considered words whose wordform is composed exclusively of characters in its language’s alphabet. We now run similar analyses, but including either: All white-space-separated words in a language’s test set, or all white-space-separated words with no punctuation symbols.<sup>22</sup> We denote these conditions as: $\Sigma _ { \alpha } ^ { * }$ when selecting alphabet-only words, $\Sigma _ { 0 } ^ { * }$ when selecting no-punctuation words, and $\Sigma ^ { * }$ when selecting all words. We display results under each condition in Fig. 4. We see that across these various protocols, ZIPF’s hypothesis remains the most predictive.<sup>23</sup>

![](images/fee11b6af2830922c335a874b3648c6da01022ef0e806429fe845ed9a36c92e0.jpg)

![](images/68802a7f71ce7eb30a8d8a6b1bcda787bc1093bab44c06f6e15d6fc10dcb22b1.jpg)

![](images/4ca964e8d2341c259eafb76d164663afee3547d838837aad69cdac5bfddcb187.jpg)

![](images/54aaeb880497706d6fc62886d0a3269739cca62d82a2d223671ac5d3f58d4441.jpg)  
Figure 3: MSE and Spearman correlation when surprisals are estimated using either: full words directly (top), or adding subword surprisals (bottom). Note that when using full words, Zipf and Zipf (subword) are the same.

![](images/3edd144f3065ff6296a6135eb7718d4f1c0cab3c732f3565f225c0f118d08c99.jpg)  
Figure 4: Average MSE across languages when hypotheses are evaluated using different word filtering protocols.

Additionally, we consider the impact of including only the top 25k most frequent words in our analysis. In Fig. 5, we present MSE values computed when using sets composed from the top 10k most frequent words, to entire test sets. Notably, we again see that frequency remains the best predictor of word length. In App. G’s Fig. 10, we display results per language for MSE and Spearman correlation. There, we see that MSE rates frequency best on all languages and across all evaluated setups. Spearman correlation evaluated on few word types similarly rates frequency over CCH or $\mathrm { C C H _ { \downarrow } }$ predictions (again, except on English). When evaluated on full test-sets, Spearman correlation shows a less straightforward conclusion: While ZIPF still achieves the highest correlation in most languages, $\mathrm { C C H _ { \downarrow } }$ achieves stronger correlations in Italian, Spanish and Russian. At this stage, however, the evaluated sets are dominated by lowfrequency words, which may not be representative of the evaluated languages.

![](images/6d4df646759d24d3da91eaa28f5b155e5bc2db0dfaf43852a96c355b736dd9f7.jpg)  
Figure 5: Average MSE across languages when hypotheses are evaluated on different number of word types.

## 6.3 Sensitivity to Model Quality

Finally, we investigate how our model quality influences our results. We train new models on subsets of our training sets to get language models of different qualities. We then use these models to assess whether there is a relationship between model quality and a hypothesis’ predictive power. In addition to the models estimated using the full training sets, we thus train 7 new transformer and unigram models per language, each using from 1 million to 1 billion training tokens in log-uniform intervals. We plot the predictive power of each hypothesis (ZIPF’s, CCH ’s and CCH) vs. the language model’s cross-entropy in Fig. 6.<sup>24</sup> Unintuitively, surprisal estimates of better models (i.e., with lower cross-entropies) provide worse predictors of word length. An additional analysis suggests that the surprisal estimates of worse language models are more strongly correlated with frequency (see Fig. 7 in App. G), which may justify this unituitive result since frequencies are most predictive of word lengths in our experiments. ZIPF’s hypothesis, on the other hand, is robust to the quality of the used unigram model.

![](images/4230375fb36db204af5b3eb8589af04a82d2d42932be6fe9d73efebe1241b12d.jpg)  
Figure 6: MSE correlation as a function of the crossentropy of models used to get surprisal estimates.

## 6.4 Sensitivity to Cost Function

In our last set of experiments, we analyze the impact of our choice of quadratic cost function in our results. Using the generalized cost function in eq. (8), we derive optimal word length predictions using values of λ from 1 to 5 in 0.25 intervals. We present their MSE and Spearman correlations in App. G’s Fig. 13. While there seems to be a slight tendency for CCH to be more predictive for larger values of λ, ZIPF still has the most predictive power of the different hypotheses.

## 7 Discussion

The answer to what drives the distribution of word lengths in lexica has long been considered important for understanding the evolution and function of language (see Gibson et al., 2019 for a review). Across multiple languages and various methodological choices, our results support Zipf’s law of abbreviation over other potential explanations as a driving factor in the development of lexica.

These findings deviate from Piantadosi et al., who found average surprisal to be a stronger predictor of word lengths. We hypothesize that this is because of methodological choices. Specifically, Piantadosi et al. derive surprisal estimates from language models that are now outdated (in terms of their quality), and we found that, when CCH’s predictions were computed using worse surprisal estimates, they had stronger correlations with length than when using better estimates. Like prior work on this topic (Meylan and Griffiths, 2021; Levshina, 2022), our analyses suggest the sensitivity of Piantadosi et al.’s results to methodological choices.

What do these results tell us about the communicative optimization of natural language lexica? In short, our results suggest lexica are optimized to minimize expected utterance lengths. Notably, other linguistic properties may be optimized towards other notions of communicative efficiency. While a word’s duration is mainly determined by its wordform, speakers can still modulate this duration to a certain extent; such a modulation could target CCH. In fact, prior work has shown a correlation between surprisal and duration (Bell et al., 2003; Aylett and Turk, 2004; Pimentel et al., 2021a).

## 8 Conclusion

In this paper, we formalize the problem of assigning wordforms based on different notions of communicative efficiency, which we term the lexicalization problem. Under this framework, we describe the optimization problem related to the channel capacity hypothesis, and, in doing so, we show that Piantadosi et al.’s predictions optimized for only a lower bound on CCH, rather than on the true objective. Further, while considering relaxed versions of the lexicalization problem, we derive optimal word length values for Zipf’s hypothesis and CCH. We then empirically evaluate CCH’s, CCH ’s and ZIPF’s predictions in 13 languages. Our results strongly support ZIPF’s hypothesis: Word lengths are optimized to minimize utterance lengths.

## Limitations

A limitation of our work is that, when deriving optimal word lengths under CCH and CCH , we relax: the phonotactic 1 , morphological composition 2 , unique decodability 3 and the integer-length 4 requirements. In the case of 3 , if a language’s channel capacity is large, this might lead to poorer predictions under both these theories. Deriving optimal word lengths while considering this constraint is left as an open problem for future work. In the case of 4 , it is arguably unrealistic to consider continuous-length wordforms. This issue could be addressed by using a linear program to solve problems of the form eq. (1). This, as well as considering the role of phonotactics 1 and morphological composition 2 in CCH, is likewise left for future work. Further, we note that while we relax all four constraints to derive CCH- and CCH -optimal word lengths, we only relax 2 (and partially 1 ) to derive ZIPF-optimal lengths. This could realistically impact the fact that Zipf’s hypothesis seems to have more predictive power over word lengths.

Another limitation is that our analyses focus solely on written data from Wikipedia. We recommend future work investigates how these findings generalize to spoken or signed languages, and to other text genres. Finally, while we use a typologically diverse sample of languages, it is still skewed towards Eurasian languages. This is because the large amount of text needed for training state-of-the-art language models— necessary to estimate entropy—are not available in many languages. Expanding the set of languages analyzed here would be necessary to confirm the generality of our results.

## Acknowledgements

We thank the anonymous reviewers and metareviewer for their feedback on this paper. Tiago Pimentel also thanks Hope McGovern and Simone Teufel for helpful comments on different stages of writing this manuscript. Tiago Pimentel is funded by a Facebook PhD Fellowship. Clara Meister is funded by a Google PhD Fellowship. Ethan Gotlieb Wilcox would like to acknowledge support from an ETH Postdoctoral Fellowship.

## References

Matthew Aylett and Alice Turk. 2004. The smooth signal redundancy hypothesis: A functional explana-

tion for relationships between redundancy, prosodic prominence, and duration in spontaneous speech. Language and Speech, 47(1):31–56.

Alan Bell, Daniel Jurafsky, Eric Fosler-Lussier, Cynthia Girand, Michelle Gregory, and Daniel Gildea. 2003. Effects of disfluencies, predictability, and utterance position on word form variation in English conversation. The Journal of the Acoustical Society ofAmerica, 113(2):1001–1024.

Christian Bentz and Ramon Ferrer-i-Cancho. 2016. Zipf’s law of abbreviation as a language universal. In Proceedings of the Leiden Workshop on Capturing Phylogenetic Algorithms for Linguistics. Universität Tübingen.

Noam Chomsky. 2002. An interview on minimalism. Cambridge University Press.

Uriel Cohen Priva. 2015. Informativity affects consonant duration and deletion rates. Laboratory Phonology, 6(2):243–278.

Thomas M. Cover and Joy A. Thomas. 2005. Elements ofInformation Theory. John Wiley & Sons, Ltd.

Huteng Dai and Richard Futrell. 2021. Simple induction of (deterministic) probabilistic finite-state automata for phonotactics by stochastic gradient descent. In Proceedings ofthe 18th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology, pages 167–176, Online. Association for Computational Linguistics.

August Fenk and Gertraud Fenk. 1980. Konstanz im Kurzzeitgedächtnis - Konstanz im sprachlichen Informationsfluß? Zeitschriftfür Experimentelle und Angewandte Psychologie, 27(3):400–414.

Edward Gibson, Richard Futrell, Steven T. Piantadosi, Isabelle Dautriche, Kyle Mahowald, Leon Bergen, and Roger Levy. 2019. How efficiency shapes human language. Trends in Cognitive Sciences, 23(5):389– 407.

Kyle Gorman. 2013. Generative Phonotactics. Ph.D. thesis, University of Pennsylvania.

Peter Grzybek. 2015. Word length. In The Oxford Handbook ofthe Word, pages 89–119. Oxford University Press.

Mandy Guo, Zihang Dai, Denny Vrandeciˇ c, and Rami´ Al-Rfou. 2020. Wiki-40B: Multilingual language model dataset. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 2440– 2452, Marseille, France. European Language Resources Association.

Bruce Hayes and Colin Wilson. 2008. A maximum entropy model of phonotactics and phonotactic learning. Linguistic Inquiry, 39(3):379–440.

David A. Huffman. 1952. A method for the construction of minimum-redundancy codes. Proceedings ofthe IRE, 40(9):1098–1101.

Jasmeen Kanwal. 2018. Word length and the principle of least effort: language as an evolving, efficient code for information transfer. Ph.D. thesis, The University of Edinburgh, Edinburgh, UK.

Jasmeen Kanwal, Kenny Smith, Jennifer Culbertson, and Simon Kirby. 2017. Zipf’s law of abbreviation and the principle of least effort: Language users optimise a miniature lexicon for efficient communication. Cognition, 165:45–52.

Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In International Conference on Learning Representations.

Alexander Koplenig, Marc Kupietz, and Sascha Wolfer. 2022. Testing the relationship between word length, frequency, and predictability based on the German reference corpus. Cognitive Science, 46(6):e13090.

Taku Kudo. 2018. Subword regularization: Improving neural network translation models with multiple subword candidates. In Proceedings of the 56th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 66–75, Melbourne, Australia. Association for Computational Linguistics.

Natalia Levshina. 2022. Frequency, informativity and word length: Insights from typologically diverse corpora. Entropy, 24(2).

Roger P. Levy and Tim Florian Jaeger. 2007. Speakers optimize information density through syntactic reduction. In Advances in Neural Information Processing Systems, pages 849–856.

T. Linder, V. Tarokh, and K. Zeger. 1997. Existence of optimal prefix codes for infinite source alphabets. IEEE Transactions on Information Theory, 43(6):2026–2028.

Clara Meister, Tiago Pimentel, Patrick Haller, Lena Jäger, Ryan Cotterell, and Roger Levy. 2021. Revisiting the uniform information density hypothesis. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 963– 980, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Stephan C. Meylan and Thomas L. Griffiths. 2021. The challenges of large-scale, web-based language datasets: Word length and predictability revisited. Cognitive Science, 45(6):e12983.

Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. fairseq: A fast, extensible toolkit for sequence modeling. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations), pages 48–53, Minneapolis, Minnesota. Association for Computational Linguistics.

Sonia Petrini, Antoni Casas-i-Muñoz, Jordi Cluet-i-Martinell, Mengxue Wang, Christian Bentz, and Ramon Ferrer-i-Cancho. 2022. The optimality of word lengths. Theoretical foundations and an empirical study. arXiv preprint arXiv:2208.10384.

Sonia Petrini, Antoni Casas-i-Muñoz, Jordi Cluet-i-Martinell, Mengxue Wang, Christian Bentz, and Ramon Ferrer-i-Cancho. 2023. Direct and indirect evidence of compression of word lengths. Zipf’s law of abbreviation revisited. arXiv preprint arXiv:2303.10128.

Steven T. Piantadosi, Harry Tily, and Edward Gibson. 2011. Word lengths are optimized for efficient communication. Proceedings ofthe National Academy of Sciences, 108(9):3526–3529.

Steven T. Piantadosi, Harry Tily, and Edward Gibson. 2012. The communicative function of ambiguity in language. Cognition, 122(3):280–291.

Steven T. Piantadosi, Harry J. Tily, and Edward Gibson. 2009. The communicative lexicon hypothesis. In Proceedings of the Annual Meeting of the Cognitive Science Society, volume 31, pages 2582–2587.

Tiago Pimentel, Rowan Hall Maudslay, Damian Blasi, and Ryan Cotterell. 2020. Speakers fill lexical semantic gaps with context. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4004–4015, Online. Association for Computational Linguistics.

Tiago Pimentel, Clara Meister, Elizabeth Salesky, Simone Teufel, Damián Blasi, and Ryan Cotterell. 2021a. A surprisal–duration trade-off across and within the world’s languages. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 949–962, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Tiago Pimentel, Clara Meister, Simone Teufel, and Ryan Cotterell. 2021b. On homophony and rényi entropy. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 8284–8293, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Tiago Pimentel, Irene Nikkarinen, Kyle Mahowald, Ryan Cotterell, and Damián Blasi. 2021c. How (non-)optimal is the lexicon? In Proceedings ofthe 2021 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 4426–4438, Online. Association for Computational Linguistics.

Claude E. Shannon. 1948. A mathematical theory of communication. The Bell System Technical Journal, 27(3):379–423.

Bengt Sigurd, Mats Eeg-Olofsson, and Joost Van Weijer. 2004. Word length, sentence length and frequency – Zipf revisited. Studia Linguistica, 58(1):37–52.

Sean Trott and Benjamin Bergen. 2020. Why do human languages have homophones? Cognition, 205:104449.

Sean Trott and Benjamin Bergen. 2022. Languages are efficient, but for whom? Cognition, 225:105094.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Gejza Wimmer, Reinhard Köhler, Rüdiger Grotjahn, and Gabriel Altmann. 1994. Towards a theory of word length distribution. Journal of Quantitative Linguistics, 1(1):98–106.

George K. Zipf. 1935. The Psychobiology of Language. London: Routledge.

George K. Zipf. 1949. Human Behavior and the Principle of Least Effort. Addison-Wesley Press.

## A Defining p(w, c)

In this section, we explicitly define $p ( w , c )$ . We do this in terms of a more standard notation in language modeling. We define a sequence of words as $s \in { \mathcal { S } }$ where $\mathcal { S } \stackrel { \mathrm { d e f } } { = } \mathcal { W } ^ { * } \circ \{ \mathrm { e o s } \}$ . We then assume a distribution over such sequences $p ( s )$ . We can now define $p ( w , c )$ as:

$$
p ( w , c ) \propto \sum _ { s \in \mathcal { S } } p ( s ) \sum _ { t = 1 } ^ { T } \mathbb { 1 } \{ w = s _ { t } , c = { s } _ { < t } \}\tag{14}
$$

In words, a word–context pair is as frequent as it would be in natural language, where once a sequence is uttered, all its word–context pairs are observed jointly. This is not the only possible definition of $p ( w , c )$ , but it is the one we opt for here.

Note that the distribution $p ( w , c )$ might thus not be well defined for all distributions $p ( \pmb { s } )$ , as the normalizing constant in this definition might diverge. For instance, this distribution will not be well-defined for a language model over alphabet $\mathcal { W } = \{ a \}$ , where $\textstyle p ( a ^ { n } ) = { \frac { 6 } { \pi ^ { 2 } n ^ { 2 } } }$ for $n \geq 1$ and $p ( \varepsilon ) = 0$ , as its sequences’ average length diverges.

## B Data and Models

Data. The corpora used throughout our analyses come from Wiki40B (Guo et al., 2020). This dataset is composed of cleaned text from Wikipedia articles in more than 40 languages, out of which we select a subset of 13 for our analysis. Our selection includes: German, Greek, English, Spanish, Estonian, Finnish, Hebrew, Italian, Korean, Dutch,

Norwegian, Russian, and Turkish. These span five language families: Afro-Asiatic, Indo-European, Koreanic, Turkic, and Uralic. The data for each language comes pre-split into a training, validation and test set. We fit our models using the first two sets, while performing our analyses exclusively on the test-sets. As discussed above, the set of analyzed words may make a large difference in the measured correlations. In our main set of experiments, we filter the set of words we analyze to only include wordforms composed of characters in the language’s alphabet.<sup>25</sup> Table 1 (in App. F) includes the number of word types and tokens used per language in our analyses.

Models. To estimate the unigram distribution $p ( w )$ , we use a simple MLE estimator: the (normalized) count statistics from our training set. To estimate contextual probabilities $p ( w \mid c )$ we use an autoregressive language model $p _ { \theta }$ Specifically, we train monolingual transformers in each language using fairseq (Ott et al., 2019) with its default language modeling hyper-parameters. Our transformers (Vaswani et al., 2017) have 6 layers, a hidden size of 512, and 8 attention heads per layer. Further, they can attend to a context size of at most 512 tokens, and we train them with a dropout of 0.1, and a batch size of 64. We optimize our models using Adam (Kingma and Ba, 2015) with a learning rate of $5 \times 1 0 ^ { - 4 }$ , weight decay of 0.01, and 4k warmup steps. In our main set of experiments, we further pre-tokenize each language’s text using language-specific tokenizers fit (using the UnigramLM algorithm; Kudo, 2018) on their respective training sets, with a vocabulary of 32k subword units. We then compute per-word surprisals by adding the surprisals of all the subwords that the word is composed of. (We also consider other tokenization schemes, as described in §6.1.)

## C Proof of Theorem 1

Before proving Theorem 1, we provide a lemma which will be useful for it. In words, we prove a length-preserving bijection between $L _ { \ell }$ and $\Delta ^ { * }$ for an alphabet K such that $| \Delta | = K$

Lemma 1. Under the constant phonotactic assumption, there exists an alphabet $\Delta$ with cardinality K such that, for every $N \ge 0 , \Delta ^ { N }$ is isomorphic to $L _ { \ell } ^ { ( N ) }$ , where $L _ { \ell } ^ { ( \mathrm { \bar { N } } ) }$ is the set of phonotactically valid wordforms with length N.

Proof. First, it is clear that $| \Delta ^ { N } | = | \Delta | ^ { N } = K ^ { N }$ We now prove the same for $\dot { L } _ { \ell } ^ { ( N ) }$ by induction.

Base case $( N = 0 )$ . The set of 0-length phonotactically valid strings includes only the empty string $\{ \varepsilon \}$ . It follows that: $| L _ { \ell } ^ { ( 0 ) } | = 1 = K ^ { 0 }$

Inductive step $( N > 0 )$ . By the inductive hypothesis, we have that $| L _ { \ell } ^ { ( N - 1 ) } | ~ = ~ K ^ { N - 1 }$ . By Assumption 1, each element in $L _ { \ell } ^ { ( N - 1 ) }$ has $K$ possible continuations in $L _ { \ell } ^ { ( N ) }$ . It follows that $| L _ { \ell } ^ { ( N ) } | = | L _ { \ell } ^ { ( N - 1 ) } | K = K ^ { N }$

Since $\Delta ^ { N }$ and $L _ { \ell } ^ { ( N ) }$ have the same number of elements for every $N \geq 0$ , there exists an isomorphism between them. ■

Given the lemma above, we are now in a position to prove Theorem 1.

Theorem 1. The minimization problem given in Hypothesis 1 with constraint $\textcircled{2}$ relaxed can be solved by Huffman coding with K symbols. The optimal solution is given by

$$
| \phi _ { \mathrm { z i p f } } ( w ) | = | \phi _ { \mathrm { h u f f } _ { K } } ( w ) |\tag{5a}
$$

$$
\leq - { \frac { 1 } { \log _ { | \Sigma | } K } } \log _ { | \Sigma | } p ( w ) + 1\tag{5b}
$$

Proof. Since $L _ { \ell } ^ { ( N ) }$ is isomorphic to $\Delta ^ { N }$ for every $N \geq 0$ , there exists a length-preserving bijection ψ between $L _ { \ell }$ and $\Delta$ (by Lemma 1). By Huffman’s (1952) algorithm, we can construct an encoding that satisfies

$$
| \psi ( \phi _ { \mathrm { z i p f } } ( w ) ) | \leq - \log _ { | \Delta | } p ( w ) + 1\tag{15}
$$

However, because $\psi$ is length-preserving, $| \psi ( \phi _ { \mathrm { z i p f } } ( w ) ) | = | \phi _ { \mathrm { z i p f } } ( w ) |$ . As an upper bound, we thus have

$$
| \phi _ { \mathrm { z i p f } } ( w ) | \leq - \log _ { | \Delta | } p ( w ) + 1
$$

$$
- \frac { 1 } { \log _ { | \Sigma | } K } \log _ { | \Sigma | } p ( w ) + 1\tag{16a}
$$

(16b)

## D Proof of Theorem 2

Theorem 2. Under Hypothesis 2, if we relax 1 , 2 , 3 and $\textcircled{4} ,$ the optimal word lengths are given by

$$
| \phi _ { \mathrm { c c h } } ( w ) | = \frac { 1 } { \mathfrak { C } } \frac { \mathrm { E } } { \mathrm { \oint _ { \Gamma } } [ \mathrm { H } ( w \mid c ) ] }\tag{7}
$$

Proof. We can easily derive these optimal word lengths from eq. (6) by taking its derivative with respect to a specific word’s length, and setting it to zero. First, we rewrite it for mathematical convenience as:

$$
\operatorname { E } _ { p ( w ) } \operatorname { E } _ { p ( c | w ) } \left( { \frac { \mathrm { H } ( w \mid c ) } { | \phi ( w ) | } } - \mathfrak { C } \right) ^ { 2 }\tag{17}
$$

where we make the quadratic cost function explicit. We note this function is convex, and so if we find a point where its derivative is zero, we also find its global minimum. We now take its derivative with respect to a specific word’s length $| \phi ( w ) |$ and set this derivative to zero:

$$
p ( w ) \operatorname * { E } _ { p ( c | w ) } \left[ 2 \left( \frac { \mathrm { H } ( w \mid c ) } { | \phi ( w ) | } - \mathfrak { C } \right) \frac { \mathrm { H } ( w \mid c ) } { | \phi ( w ) | ^ { 2 } } \right] = 0\tag{18}
$$

where we note that all terms involving other words will have derivative zero (with respect to this specific word w’s length). As the expectation is a linear operation, we can rewrite this equation as:

$$
\operatorname { E } _ { p ( c | w ) } \left[ { \frac { \mathrm { H } ^ { 2 } ( w \mid c ) } { | \phi ( w ) | ^ { 3 } } } \right] = \operatorname { E } _ { p ( c | w ) } \left[ \mathfrak { C } { \frac { \mathrm { H } ( w \mid c ) } { | \phi ( w ) | ^ { 2 } } } \right]\tag{19}
$$

Note that both the length and capacity are constant with respect to the expectation over contexts. Isolating the length term, thus, we get:

$$
| \phi ( w ) | = \frac { 1 } { \mathfrak { C } } \frac { \mathrm {  ~ \cal ~ E ~ } } { \mathrm {  ~ \cal ~ E ~ } } \bigl [ \mathrm { H } ^ { 2 } ( w \mid c ) \bigr ]\tag{20}
$$

This completes the proof.

## E Proof of Proposition 1

Proposition 1. Given a convex dist function and any $\phi \in \Phi _ { \ell } ,$ , the cost optimized by CCH in Hypothesis 3 lower-bounds CCH’s cost in Hypothesis 2

$$
\begin{array} { r l } & { \mathrel { \phantom { = } } \displaystyle \operatorname { E } _ { \boldsymbol { v } ^ { c } \to \big } \mathrm { d i s t } \Bigg ( \frac { \mathrm { H } ( \boldsymbol { w } \mid \boldsymbol { c } ) } { \lvert \phi ( \boldsymbol { w } ) \rvert } , \mathfrak { e } \Bigg ) } \\ & { \qquad \geq \underset { \boldsymbol { p } ( \boldsymbol { w } , \boldsymbol { c } ) } { \mathrm { E } } \mathrm { d i s t } \Bigg ( \frac { \mathrm { H } ( \boldsymbol { w } \mid \boldsymbol { C } ) } { \lvert \phi ( \boldsymbol { w } ) \rvert } , \mathfrak { e } \Bigg ) } \end{array}\tag{12}
$$

Proof. It can be easily shown by Jensen’s inequal-

ity that for any choice of ϕ:

$$
\operatorname { E } _ { p ( w , c ) } \mathrm { d i s t } \bigg ( \frac { \mathbf { H } ( w \mid c ) } { \vert \phi ( w ) \vert } , \mathfrak { C } \bigg )\tag{21a}
$$

$$
= \operatorname { E } _ { p ( w ) } \operatorname { E } _ { p ( c | w ) } \mathrm { d i s t } \bigg ( \frac { \mathrm { H } ( w \mid c ) } { | \phi ( w ) | } , \mathfrak { C } \bigg )\tag{21b}
$$

$$
\geq \underset { p ( w ) } { \mathrm { ~ E ~ } } \operatorname { d i s t } \left( \frac { \underset { p ( c | w ) } { \mathrm { ~ E ~ } } [ \boldsymbol { \mathrm { H } } ( w \mid c ) ] } { | \phi ( w ) | } , \mathfrak { C } \right)\tag{21c}
$$

$$
= \operatorname { E } _ { p ( w ) } \mathrm { d i s t } \biggl ( \frac { \mathbf { H } ( w \mid C ) } { | \phi ( w ) | } , \mathfrak { C } \biggr )\tag{21d}
$$

which completes the proof.

## F Data Statistics

We provide dataset statistics in Table 1.

## G Further Results

For a more detailed reading, we provide MSE and Spearman correlation plots similar to Fig. 1 and $2 \mathrm { { : } }$ but as bar plots in Fig. 8. We also provide perlanguage results:

• as a function of the word filtering protocol used in our analysis in Fig. 9;

• as a function of the number of word types included in our analysis in Fig. 10;

• as a function of our language model’s crossentropy in Fig. 11; and

• as a function of the number of tokens used to train our language models and to get word count statistics in Fig. 12.

We also provide results when CCH is defined using generalized distfunctions, i.e., for several values of λ, in Fig. 13. Finally, we show the Spearman correlation between CCH and $\mathrm { C C H _ { \downarrow } }$ and unigram surprisal as a function of the used language model’s quality in Fig. 7.

![](images/a84305b9b369b358a4a80967193d4b5c802ed4a499e7fd5e6410c9cda520b180.jpg)  
Figure 7: Spearman correlation with unigram surprisal as a function of the cross-entropy of models used to get surprisal estimates

<table><tr><td colspan="4"></td><td colspan="2">None</td><td colspan="2">No Punctuation</td><td colspan="2">Only in Alphabet</td></tr><tr><td>Language</td><td>Family</td><td>ISO code</td><td>BPC</td><td># Types</td><td># Tokens</td><td># Types</td><td># Tokens</td><td># Types</td><td># Tokens</td></tr><tr><td>German</td><td>Indo-European</td><td>de</td><td>0.99</td><td>2,093,524</td><td>32,142,917</td><td>1,027,594</td><td>27,565,045</td><td>896,752</td><td>26,301,145</td></tr><tr><td>Greek</td><td>Indo-European</td><td>el</td><td>1.06</td><td>267,625</td><td>2,244,964</td><td>145,073</td><td>1,954,950</td><td>120,361</td><td>1,862,380</td></tr><tr><td>English</td><td>Indo-European</td><td>en</td><td>1.09</td><td>2,419,694</td><td>78,392,487</td><td>748,109</td><td>66,881,077</td><td>609,839</td><td>65,261,138</td></tr><tr><td>Spanish</td><td>Indo-European</td><td>es</td><td>1.04</td><td>993,894</td><td>21,472,091</td><td>380,407</td><td>18,852,688</td><td>332,668</td><td>18,458,823</td></tr><tr><td>Estonian</td><td>Uralic</td><td>et</td><td>1.23</td><td>250,860</td><td>999,296</td><td>145,003</td><td>797,817</td><td>123,863</td><td>736,860</td></tr><tr><td>Finnish</td><td>Uralic</td><td>fi</td><td>1.03</td><td>566,755</td><td>2,741,783</td><td>333,229</td><td>2,249,003</td><td>318,514</td><td>2,156,132</td></tr><tr><td>Hebrew</td><td>Afro-Asiatic</td><td>he</td><td>1.41</td><td>497,550</td><td>4,153,846</td><td>230,867</td><td>3,394,959</td><td>208,077</td><td>3,299,727</td></tr><tr><td>Italian</td><td>Indo-European</td><td>it</td><td>1.04</td><td>823,397</td><td>14,500,421</td><td>297,153</td><td>12,371,701</td><td>269,005</td><td>12,056,925</td></tr><tr><td>Korean</td><td>Koreanic</td><td>ko</td><td>2.40</td><td>538,093</td><td>1,953,812</td><td>385,948</td><td>1,622,552</td><td>331,060</td><td>1,460,230</td></tr><tr><td>Dutch</td><td>Indo-European</td><td>nl</td><td>1.00</td><td>534,101</td><td>6,811,124</td><td>246,458</td><td>5,989,335</td><td>227,466</td><td>5,768,607</td></tr><tr><td>Norwegian</td><td>Indo-European</td><td>no</td><td>1.14</td><td>325,983</td><td>2,672,869</td><td>176,089</td><td>2,318,082</td><td>164,401</td><td>2,253,985</td></tr><tr><td>Russian</td><td>Indo-European</td><td>ru</td><td>1.06</td><td>1,474,777</td><td>15,824,324</td><td>707,806</td><td>13,073,143</td><td>546,179</td><td>11,922,147</td></tr><tr><td>Turkish</td><td>Turkic</td><td>tr</td><td>1.14</td><td>285,988</td><td>1,705,030</td><td>163,352</td><td>1,371,183</td><td>148,522</td><td>1,306,751</td></tr></table>

Table 1: Wiki40B data statistics.

![](images/d14dd488806b2ac9194c148fb34252ba55db812c95d2b4aeed2d5988ad0e24f2.jpg)

![](images/3935f9cab43e8966ad63f622f2b7d6b452dda080846550d1d7cbaef7d85b7b5c.jpg)

Figure 8: MSE and Spearman correlation of the three hypotheses in the analyzed languages.  
![](images/3ce1744c3286336ee2e20f19481a136891b2eabf98d9606fd02b7b83ac2d58a8.jpg)

Figure 9: MSE and Spearman correlation when hypotheses are evaluated while filtering test set words based on different protocols.  
![](images/93a9dc99637b8e25465378df5d11346d5614c693b0016a38e0bc2e079fab46cb.jpg)  
Figure 10: MSE and Spearman correlation when hypotheses are evaluated on different number of word types.

![](images/864ba08fa5ed2add8c7012d341bdb3b02338e4e7c0b0e4f2cb20b0a07ee3ca1a.jpg)  
Language Model's Cross-entropy

![](images/66b332e9b682a0f0c3d843de7d9bc1454614b1594fe401f119f0b8d4d2ac6529.jpg)  
Language Model's Cross-entropy  
Figure 11: MSE and Spearman correlation as a function of the cross-entropy of models used to get surprisal estimates.

![](images/a81a21dfcabba98721e278f069bc559d21a595714a45b93687d51d4d87ae94b1.jpg)  
# Train Tokens

![](images/1855e700f9b10db2840a95e539dd60ec90b7b75b76c7c89434f8d3c7fccf0f3a.jpg)  
# Train Tokens  
Figure 12: Results as a function of the number of tokens used to train language models and get count statistics.

![](images/7eebd66050918f5b3509a6addd3ad1db0d7d875cc3265c1436399f18ba6df3f8.jpg)

![](images/a5543c5e494ab15daaf55b27e83c14c07be8573715f85b01e3f4ad067d4e0fe6.jpg)  
Figure 13: MSE and Spearman correlation as a function of the generalized dist’s parameter λ (in the x-axis).