# Sparse Universal Transformer

Shawn Tan<sup>1</sup> <sup>\*</sup> tanjings@mila.quebec

Zhenfang Chen<sup>2</sup> zfchen@ibm.com

Yikang Shen<sup>2</sup> <sup>\*</sup> yikang.shen@ibm.com

Aaron Courville<sup>1</sup> courvila@iro.umontreal.ca

Chuang Gan<sup>2</sup> chuangg@ibm.com

<sup>1</sup>Mila, University of Montreal

## Abstract

The Universal Transformer (UT) is a variant of the Transformer that shares parameters across its layers. Empirical evidence shows that UTs have better compositional generalization than Vanilla Transformers (VTs) in formal language tasks. The parameter-sharing also affords it better parameter efficiency than VTs. Despite its many advantages, scaling UT parameters is much more compute and memory intensive than scaling up a VT. This paper proposes the Sparse Universal Transformer (SUT), which leverages Sparse Mixture of Experts (SMoE) and a new stick-breaking-based dynamic halting mechanism to reduce UT’s computation complexity while retaining its parameter efficiency and generalization ability. Experiments show that SUT achieves the same performance as strong baseline models while only using half computation and parameters on WMT’14 and strong generalization results on formal language tasks (Logical inference and CFQ). The new halting mechanism also enables around 50% reduction in computation during inference with very little performance decrease on formal language tasks.

## 1 Introduction

Recent theoretical work has pointed out that finitedepth Transformers have an issue of expressibility that will result in failure to generalize (Hahn, 2020; Hao et al., 2022; Merrill et al., 2022; Liu et al., 2022). Delétang et al. (2022) ran several neural architectures on a suite of different synthetic languages generated from different levels of the Chomsky hierarchy and empirically confirmed these results, showing that VTs have difficulty generalizing to Regular languages. Universal Transformers (UTs; Dehghani et al. 2018) are Transformers that share parameters at every layer of the architecture. Csordás et al. (2021) performed several compositional generalization experiments on VTs and UTs along with absolute and relative position embeddings, and showed that UTs with relative positional embeddings performed better on these tasks.

![](images/5f464fa7566d685d59e21d93ecaade918aa5d3ec7b45e685f4f50488ebdec0e4.jpg)  
Figure 1: A VT has separate Transformer blocks for each layer, with different parameters. For a UT with the same number of parameters, the UT block will be 3 times the dimensions of each VT block. Running this block for 3 layers would then incur approximately 9 times the runtime memory. Using SMoEs can recover approximately the same computational cost as the VT.

However, the task of scaling UTs is challenging due to its computation complexity (Kaplan et al., 2020; Tay et al., 2022; Takase and Kiyono, 2021). Consider a VT with P parameters for each layer and L layers. Evaluating such a VT has computation complexity associated with the model size LP. A size-equivalent UT would have a UT block with LP parameters and computation complexity of approximately LP to run the block once. To run such a UT for equivalent L layers would incur a complexity of L<sup>2</sup>P. This increased computation complexity directly translates to increased training and inference time. According to Takase and Kiyono (2021), UT requires two times the training time and far more GPU memory than VT in WMT English-German translation task.

Sparsely activated neural networks were introduced to reduce the computation complexity of large models. These networks activate parts of the network conditioned on the input, computing only parts of the model, thereby disentangling the number of parameters from the computation complexity. This method allows for drastically increasing the number of parameters without proportionally increasing the computation complexity. Shazeer et al. (2017) introduced Sparse Mixture of Experts (SMoE), using the top-k operator to allow for sparse computation of experts. This allows for replacing the FeedForword (FFD) layer in the Transformer with an ensemble of $E _ { \mathrm { f f d } }$ FFDs, but only k FFDs (where $k \ < \ E )$ would have to be evaluated, conditioned on the input. Zhang et al. (2022) then introduced the Mixture of Attention heads (MoA), which allows Transformers to replace its Multihead Attention (MHA) layer with an ensemble of $E _ { \mathrm { a t t } }$ attention heads and only activates k heads condition on the input, further sparsifying the model.

This paper introduces the Sparse Universal Transformer (SUT), which applies the above sparse computation techniques to UT. Additionally, we replace the per-position halting mechanism in UT with a new stick-breaking formulation that has a probabilistic interpretation, allowing us to introduce an Adaptive Computation Time (ACT; Graves 2016) penalty to minimize layer use. It also provides an easy way to adjust the trade-off between the amount of computation and model performance during inference, further improving the efficiency of the SUT at inference time.

To demonstrate effective scaling, we perform experiments on WMT’14 English to German translation, showing that an SUT can achieve better performance for the same parameter count, while incurring less computation cost than an equivalent dense UT. Since the UT setting is a specific case of SUT, we show on the Compositional Freebase Questions (CFQ; Keysers et al. 2019) tasks that UTs have better compositional generalization properties, improving upon CFQ results from Csordás et al. (2021). Using the Logical Inference task (Bowman et al., 2015), we analyse the behaviour of our UT on length and compositional generalization. Finally, we show that the halting mechanism can be used to achieve further efficiency during inference time, and study the trade-off between efficiency and performance.

![](images/26c830baaf015c3dafc4f10af642a887f499c334b281d99838f26ed64f6b1f05.jpg)  
Figure 2: Example of the compositional generalization splits from (Shen et al., 2019). The combination of not and and are never seen in successive combination during training, and a VT may learn a shortcut that prevents generalisation during test.

## 2 Background & Related Work

Overcoming VT limitations with UT Dziri et al. (2023) and Liu et al. (2022) find that Vanilla Transformers learn shortcuts for tasks that require multistep compositional operations, and fail to generalize on larger instances of the problem that require more steps. Theoretical results have also shown that Vanilla Transformers have limitations in what they can compute that support these findings (Hahn, 2020; Hao et al., 2022; Merrill et al., 2022). Universal Transformers (Dehghani et al., 2018) are Transformers with tied weights across all layers, and an additional halting mechanism to decide when to stop. In an ideal scenario of infinite layers (now that all layers have the same parameters) the UT, like the Neural GPU (Kaiser and Sutskever, 2015), is Turing-complete, which overcomes many of the abovementioned issues.

In practice, even with limited depth, UTs have exhibited properties that afford them better performance in compositional generalization tasks (Csordás et al., 2021). UTs allow operations learned in the Transformer during training to be depth-order invariant. If some operations during training are learned to be performed in a certain order, during test time, the UT could generalize to an unseen order of operations.

Challenges with Scaling the UT Despite these compositional abilities, performance tends to decrease on real-world tasks when using UTs. AL-BERT (Lan et al., 2019) improved parameter efficiency by sharing parameters across layers. This was motivated by an observation that Transformers tend to learn to perform similar operations in the layers, and that sharing these parameters would reduce this $\mathrm { r e d u n d a n c y } ^ { 1 }$ . However, the authors observe a dip in performance when sharing parameters, contrary to Dehghani et al. (2018).

Could the issue be one of model capacity? Experiments with ALBERT show that scaling up AL-BERT can outperform the BERT baseline, even on real-world tasks (Lan et al., 2019). Kaplan et al. (2020) also show that a shared-parameter Transformer has better scaling properties in terms of parameter-to-performance, but poorer properties in terms of computation-to-performance, since parameter count causes the computation to increase. Tay et al. (2022) scale up different sequence models, and remark on difficulties with scaling up UTs, limiting the experiments they can perform on UT. Takase and Kiyono (2021) outline several strategies of scaling up shared-parameter Transformers to deal with these issues by using different parametersharing schemes.

Our experiments show that SMoE techniques can be applied successfully to the UT to scale it up, achieving the UT’s parameter efficiency while not incurring the same computation costs. We also perform experiments that support the compositional generalization claims of prior work, and provide better baselines for those tasks.

## 3 Method

Like UT, we reuse the same SUT block for every layer of the Transformer. Within each SUT block, we use SMoEs to achieve sparsity for the feedforward network (FFD) and attention heads separately. We use the Mutual Information Maximization loss proposed in Chen et al. (2022) and modified for unsupervised tasks in Shen et al. (2023). Finally, we propose a stick-breaking process formulation of dynamic halting, which affects how the attention mechanism works in the SUT, and the Adaptive Computation Time (ACT) auxiliary loss we use to minimize the number of layers used.

## 3.1 Sparse Mixture of Experts

A Mixture of Experts module consists of E submodules $f _ { 1 } , \ldots , f _ { E }$ . There is also a gating network, which we will denote by $g ( e \mid \mathbf { h } )$ – for any input h to the MoE module, the gating network would predict a distribution over the $E$ experts. When $k \ < \ E$ , we refer to this as a Sparse Mixture of Experts (SMoE), and $g ( e \mid \mathbf { h } ) > 0$ for only k experts, while maintaining that $\begin{array} { r } { \sum _ { e } ^ { E } g ( e \mid \mathbf { h } ) = 1 } \end{array}$ The final output of the SMoE is then given by $\begin{array} { r } { y = \sum _ { e = 1 } ^ { E } g ( e \mid \mathbf { h } ) \cdot f _ { e } ( \mathbf { h } ) } \end{array}$ , where $g ( e \mid \mathbf { h } ) = 0 .$ $f _ { e } ( { \bf h } )$ will not need to be evaluated, reducing computation cost during training and inference. We replace the Feed-forward layer (FFD) in the Transformer block with a mixture of FFDs. Each Mixture of FFD can be described with a 3-tuple, $( E , k , D )$ : E experts, k for the number of experts to be used in the top-k operation, and D for the dimension of the hidden layer for each FFD expert. For the attention layer, we use the Mixture of Multihead Attention (MoMHA) proposed by Zhang et al. (2022) and Shen et al. (2023). Each MoMHA module can be described by a 5-tuple, (E, k, H, D, W), with E representing the number of experts, K representing the parameter k in a top-k operation, H representing the number of attention heads per expert, and D for the dimensions per head. Like in MoA, MoMHA maintains only a single set of H key-value projections shared among the experts, while there are E query projections of H heads each. W represents the relative position embedding window size, parameterizing 2W + 1 embeddings for W positions before and after the present position. Figure $3 \ : ( L e f t )$ shows the schematic of a SUT block.

This technique has been used to reduce computation costs both during training and inference time for large models.

Mutual Information Maximisation Like other models that rely on conditional activation, auxiliary losses are needed in order to aid learning a module that decides which experts are activated, and to ensure that all experts are used, balancing the load for processing. For this, we use the Mutual Information Maximization introduced in Chen et al. (2022) for the auxiliary loss (to be maximised):

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { M I M } } = \underbrace { \sum _ { e = 1 } ^ { E } g ( e ) \log g ( e ) } _ { \displaystyle - H ( e ) } } \\ { \displaystyle \underbrace { - \frac { 1 } { | \mathcal { X } | } \sum _ { \mathbf { h } \in \mathcal { X } } \sum _ { e = 1 } ^ { E } g ( e \mid \mathbf { h } ) \log g ( e \mid \mathbf { h } ) } _ { H ( e \mid \mathbf { h } ) } , } \end{array}\tag{1}
$$

where,

$$
g ( e ) = \frac { 1 } { | \mathcal { X } | } \sum _ { \mathbf { h } \in \mathcal { X } } g ( e | \mathbf { h } )
$$

![](images/03e2016155a5c22470a1dae215343218b6ae72a30c94d7a61b5b5d8b668524a8.jpg)  
Figure 3: $L e f t { \mathrm { : } }$ Schematic of a SUT block. Right: While the input of each SUT block is the output of the previous layer, the attention mechanism attends to the halted state of the timestep. When the halting probability exceeds $\alpha _ { \mathrm { t h r e s h } }$ , the hidden state is simply copied. Finally, the halted state is used as the output of the SUT.

. Specifically, we use the unsupervised version proposed by Shen et al. (2023) that assumes a uniform distribution over all tokens and layers, resulting in the following auxiliary objective. In the SUT setting, the gating network is used $| { \mathcal { X } } | = L$ T times, where L is the number of layers, and $T$ is the number of timesteps.

Intuitively, the entropy term increases the entropy of the marginal probability of the gating network predictions, which at its maximum means that the weight for each gating network across the entire minibatch is uniform. The conditional entropy term decreases the conditional entropy, which causes the prediction of the gating network to be sharp, and also penalizes the uniform distribution solution for the gating network.

## 3.2 Stick-breaking Dynamic Halting

There have been several methods for imbuing models with the ability to make a prediction without having to use all layers of the model (Graves, 2016; Tan and Sim, 2016; Dehghani et al., 2018; Elbayad et al., 2019; Schuster et al., 2022). Motivations for this include: (1) different inputs require different amounts of iteration to make a prediction, (2) reducing computation cost.

UT implements a similar mechanism, but the UT version of halting is difficult to interpret. Here we choose a principled version of the dynamic halting mechanism based on the stick-breaking process, viewing it as a probability distribution. First, $\hat { \alpha } _ { l } ^ { ( t ) }$ are the halting probabilities predicted by halt $( \mathbf { h } _ { l } ^ { ( \dot { t } ) } )$

Algorithm 1 Halting mechanism at a given   
timestep t   
for l = 1 to L do   
if $\begin{array} { r } { \sum _ { l ^ { \prime } = 1 } ^ { l - 1 } \alpha _ { l ^ { \prime } } ^ { ( t ) } < \alpha _ { \mathrm { t h r e s h } } } \end{array}$ then   
$\hat { \alpha } _ { l - 1 } ^ { ( t ) } = \mathrm { h a l t } ( \mathbf { h } _ { l - 1 } ^ { ( t ) } )$   
$\alpha _ { l - 1 } ^ { ( t ) } = \hat { \alpha } _ { l - 1 } ^ { ( t ) } \prod _ { \prime \prime = 1 } ^ { l - 2 } ( 1 - \hat { \alpha } _ { l ^ { \prime } } ^ { ( t ) } )$   
$\mathbf { a } _ { l } ^ { ( t ) } = \mathrm { A t t e n t i o n } ( \underbrace { \mathbf { h } _ { l - 1 } ^ { ( t ) } } _ { Q } , \underbrace { \mathbf { S } _ { l - 1 } } _ { K } , \underbrace { \mathbf { S } _ { l - 1 } } _ { V } )$   
$\mathbf { h } _ { l } ^ { ( t ) } = \mathrm { F e e d F o r w a r d } ( \mathbf { h } _ { l - 1 } ^ { ( t ) } , \mathbf { a } _ { l } ^ { ( t ) } )$   
$\begin{array} { r } { { \bf s } _ { l } ^ { ( t ) } = \left( 1 - \sum _ { l ^ { \prime } = 1 } ^ { l - 1 } \alpha _ { l ^ { \prime } } ^ { ( t ) } \right) \cdot { \bf h } _ { l } ^ { ( t ) } } \end{array}$   
$+ \left( \sum _ { l ^ { \prime } = 1 } ^ { \prime } \alpha _ { l ^ { \prime } } ^ { ( t ) } \cdot \mathbf { h } _ { l ^ { \prime } } ^ { ( t ) } \right)$   
else   
$\mathbf { \tilde { h } } _ { l } ^ { ( t ) } = \mathbf { h } _ { l - 1 } ^ { ( t ) }$   
$\mathbf { s } _ { l _ { . . . } } ^ { ( t ) } = \mathbf { s } _ { l - 1 } ^ { ( t ) }$   
end if   
end for

, a function which is implemented by an MLP that takes in the previous layer’s embedding. Then, the probability of any layer halting is computed by

$$
\alpha _ { l } ^ { ( t ) } = \hat { \alpha } _ { l } ^ { ( t ) } \prod _ { l ^ { \prime } = 1 } ^ { l - 1 } ( 1 - \hat { \alpha } _ { l ^ { \prime } } ^ { ( t ) } ) .\tag{2}
$$

A similar formulation is described in Graves (2016) and Tan and Sim (2016). Algorithm 1 shows how the mechanism is implemented at any given timestep. $\mathbf { h } _ { l } .$ <sub>1</sub> is the output of the previous layer for the current timestep.

Conditioned on the fact that we are computing $\mathbf { h } _ { l } ^ { ( t ) }$ , time-step t must not have halted before or at $l - 1$ . So we can use $\mathbf { h } _ { l } ^ { ( t ) }$ , the unhalted state, as input to the computation of the attention query of the block. However, since time-step t can attend to all other timesteps, and it these other steps may have halted, we use the halted states $\mathbf { S } _ { l - 1 }$ for the previous layers.

However, because the halting is a ‘soft’ decision, we can relax the requirement for evaluating all possible halted states and use the expected halted state as a substitute. Previous halting mechanisms use a ‘gating’ mechanism of convex sums between previously gated outputs and the current step’s output $\mathbf { h } _ { l } = \alpha _ { l } \cdot \hat { \mathbf { h } } _ { l } + ( 1 - \alpha _ { l } ) \cdot \mathbf { h } _ { l - }$ <sub>1</sub> (Dehghani et al., 2018). This can lead to vanishingly small gradients going up the layers as $( 1 - \alpha _ { l } )$ multiplies. We can instead compute the expected halted embedding at any l,

$$
\mathbf { s } _ { l } ^ { ( t ) } = \underbrace { \left( 1 - \sum _ { l ^ { \prime } = 1 } ^ { l - 1 } \alpha _ { l ^ { \prime } } ^ { ( t ) } \right) \cdot \mathbf { h } _ { l } ^ { ( t ) } } _ { \mathrm { p r e v i o u s ~ l a y e r ~ i f ~ n o t ~ h a l t e d } } + \underbrace { \sum _ { l ^ { \prime } = 1 } ^ { l - 1 } \alpha _ { l ^ { \prime } } ^ { ( t ) } \mathbf { \Delta } \mathbf { h } _ { l ^ { \prime } } ^ { ( t ) } } _ { \mathrm { h a l t e d ~ a t } < l }\tag{3}
$$

If $\alpha _ { l } ^ { ( t ) } = 1$ for some $l , \ \mathbf { s } _ { l } ^ { ( t ) } = \mathbf { h } _ { l } ^ { ( t ) }$ , recovering the behavior of the discrete halting decision. We use $\mathbf { s } _ { l } ^ { ( t ) }$ as input to the attention key and value transformations.

This probabilistic interpretation also allows us to impose a loss on the expected number of layers used at each step, biasing the model towards fewer iterations, thereby saving computational cost.

$$
\mathcal { L } _ { \mathrm { A C T } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sum _ { l = 1 } ^ { L } \alpha _ { l } ^ { ( t ) } \cdot l .\tag{4}
$$

We use a threshold $\alpha _ { \mathrm { t h r e s h } } = 0 . 9 9 9$ , such that the cumulative sum of the halting probabilities has exceeded this, no computation will be performed for that time step, and the previous layer’s embeddings will be copied. Due to the routing operation required in the implementation fo SMoEs, we can simply route halted states to a “No $\mathrm { O p } ^ { \mathrm { , } \mathrm { , } }$ expert, leading to real savings in computation cost when halting hits the threshold early. We find that adjusting this threshold after training can maintain performance while saving computation steps.

## 4 Experiments

First, we show that we can scale the UT with SUT on the WMT’14 English-German (Bojar et al., 2014) translation task. We then ran experiments on Compositional Freebase Questions (CFQ; Keysers et al. 2019) to test for compositional generalization properties. To further analyze the behaviour of the model under compositional generalization settings, we test our model on the Logical inference task from (Bowman et al., 2015). All experiments were implemented within the Fairseq framework (Ott et al., 2019) <sup>2</sup>.

Table 1: BLEU score on WMT14 En-De translation datasets. MACs (Multiply–Accumulate Operations)<sup>1</sup> measures the computational complexity of each model. <sup>a</sup>Vaswani et al. (2017), <sup>b</sup>Liu et al. (2020), <sup>c</sup>Peng et al. (2020), <sup>d</sup>Zhang et al. (2022), <sup>e</sup>Dehghani et al. (2018), <sup>f</sup>Myle et al. (2018), <sup>g</sup>Wu et al. (2018)
<table><tr><td>Model</td><td>#Params</td><td>BLEU</td><td> $\mathbf { M A C s } ^ { 1 }$ </td></tr><tr><td>Transformer baseª</td><td>65M</td><td>27.3</td><td>604M</td></tr><tr><td>Admin 6L-6Lb</td><td>61M</td><td>27.7</td><td>604M</td></tr><tr><td> ${ \bf M A E } { \cdot } 7 ^ { c }$ </td><td>63M</td><td>28.4</td><td></td></tr><tr><td>MoA based</td><td>65M</td><td>28.4</td><td>628M</td></tr><tr><td> $\mathrm { U T } ^ { e }$ </td><td>65M</td><td>28.9</td><td></td></tr><tr><td>UT base + SB halting SUT base</td><td>64M 66M</td><td>29.3 29.2</td><td>1998M 787M</td></tr><tr><td>Transformer bigf</td><td>210M</td><td>29.3</td><td>2090M</td></tr><tr><td> $\mathbf { L i g h t C o n } \mathbf { \bar { v } } ^ { g }$ </td><td>202M</td><td>28.9</td><td>1750M²</td></tr><tr><td>DynamicConv9</td><td></td><td></td><td></td></tr><tr><td>Admin  $1 8 \mathrm { L } \mathrm { - } 1 8 \mathrm { L } ^ { h }$ </td><td>213M</td><td>29.7</td><td>1790M²</td></tr><tr><td></td><td>151M</td><td>29.0</td><td>1490M</td></tr><tr><td>Admin  $6 0 \mathrm { L } \mathrm { - } 1 2 \mathrm { L } ^ { i }$ </td><td>256M</td><td>30.1</td><td>2550M</td></tr><tr><td>MoA bigd</td><td>200M</td><td>29.4</td><td>1220M</td></tr><tr><td>UT big + SB halting</td><td>105M</td><td>29.6</td><td>3707M</td></tr><tr><td>SUT big</td><td>110M</td><td>29.4</td><td>787M</td></tr></table>

Table 2: Ablation Study. “– MIM loss” means replacing the MIM loss with the load balancing loss used in (Fedus et al., 2021). “– MoMHA” means replacing MoMHA with the MoA introduced in (Zhang et al., 2022).
<table><tr><td>Model</td><td>Valid loss</td><td>BLEU</td><td>#Params</td></tr><tr><td>SUT base</td><td>2.192</td><td>29.2</td><td>66M</td></tr><tr><td>- MIM loss</td><td>2.221</td><td>28.9</td><td>66M</td></tr><tr><td>- MoMHA</td><td>2.232</td><td>28.7</td><td>66M</td></tr><tr><td>– ACT loss</td><td>2.217</td><td>29.0</td><td>66M</td></tr><tr><td>– halting</td><td>2.219</td><td>29.1</td><td>65M</td></tr></table>

## 4.1 English to German Translation

We perform experiments on the WMT’14 English-German translation dataset (Bojar et al., 2014). We use the pre-processing from Liu et al. (2020). We use a joined dictionary and share all word embeddings of the encoder and decoder. For evaluation, we average the last 5 best models according to their negative log-likelihood scores. We report the BLEU scores (Papineni et al., 2002), and also report the MACs (Multiply-Accumulate Operations) to evaluate the runtime computational costs of the different models. MACs of previous models were computed in Zhang et al. (2022).

Table 3: FFD Expert-Word co-occurrences.
<table><tr><td>Exp.</td><td>6</td><td>17</td><td>41</td><td>46</td></tr><tr><td rowspan="6">5 Top</td><td>a</td><td>he</td><td>ed</td><td>team</td></tr><tr><td>their</td><td>they</td><td>ing</td><td>children</td></tr><tr><td>his</td><td>his</td><td>ted</td><td>police</td></tr><tr><td>this</td><td>He</td><td>y</td><td>devices</td></tr><tr><td>an</td><td>you</td><td>red</td><td>system</td></tr><tr><td>Det.</td><td>Pronouns</td><td>Suffixes</td><td>Nouns</td></tr></table>

The results are reported in Table 1. We compare against strong baselines while accounting for the number of parameters in these models. In addition, we train two UTs by setting $E = 1 , k = 1$ , and parameterizing the FFD and Attention layers with parameters to match our 65M, and 110M setting for SUT. The SUTs and UTs both demonstrate good parameter efficiency when compared to previous models. In the 110M parameter class, SUT and UT perform at around 29.4 and 29.6 BLEU respectively, while previous models require 200M parameters. While the SUT does not perform as well as the UT, but the computations required during runtime could be as low as one-fifth of UT. Also, because we keep k constant for SUT, the MACs stays constant as SUT scales up.

We ran experiments removing different aspects of the model and its training process, including: MIM auxiliary loss, Mixture of MHA, the ACT loss, and the halting mechanism. The results are in Table 2. The introduction of multiple heads to the MoA was crucial in seeing performance gains on this task, as well as having the MIM loss as a load-balancing auxiliary objective. Interestingly, halting does contribute as much of a performance gain as it does in CFQ.

Additionally, we compute the top 5 tokens that occur in conjunction with each expert, regardless of layers, and find that certain associations exist. We pick several experts in Table 3 that show a clear sign of co-occurring with tokens that seem to show a pattern. This suggests that while there may be redundancy between the experts, groups of experts can specialize on certain tasks, resulting in some modularity. Future work can investigate if such modularity can result in more robust generalization.

## 4.2 Compositional Freebase Questions

We run experiments on the Compositional Freebase Questions (CFQ; Keysers et al. 2019) dataset to determine the compositional generalization abilities of the SUT. This is a translation task from natural language to a SPARQL query. As an example, the sequence Who wrote M1 and wrote a film would be translated to the target sequence SELECT DISTINCT ?x0 WHERE { ?x0 a people.person ?x0 film.writer.film ?x1 M1 . ?x1 a film.film }. CFQ tests for compositional generalization using the notion of compound divergence, which measures how different the training set and test set are in terms of combinations of tokens, which they refer to as compounds. To our knowledge, the current best-performing models either finetune a pretrained language model or, use knowledge about the task to design a suitable prompt for a large language model (Drozdov et al., 2022). While the prompting approach is extremely effective at the CFQ task, we view the task as a benchmark for compositional generalization in general and should be viewed in concert with other experiments, especially real-world data (like translation). When using domain knowledge of the task in the prompt, the results may indicate better performance with a specific approach for CFQ (and perhaps other SQL translation tasks) but might be difficult to extrapolate to other settings.

In our experiments, we use preprocessing scripts from Zheng and Lapata (2021). The scripts perform preprocessing to the target sequence that simplifies the target sequence the same way performed in Furrer et al. (2020). Accordingly, we train a baseline Transformer on the transformed target. We performed a search on the SUT hyperparameters, using the MCD1 validation set, and the best-performing set of parameters are Attention $( E = 1 , k = 1 , H = 8 , D = 6 4 , W = 1 )$ and FFD $\left( E = 1 , k = 1 , D = 1 0 2 4 \right)$ , which corresponds to the UT setting. Refer to Appendix A for further details. Since CFQ is a relatively small task, larger scale is not a factor and might suggest that expert specialization may not be as helpful. The results are shown in Table 4. In cases with and without halting, the model already outperforms previous benchmarks, including the UT baseline from Bergen et al. (2021). For a fairer comparison, we use the same hyperparameters as our UT implementation, we modify our UT implementation to be more similar to the T5-based UT in Bergen et al. (2021). These changes include: the bucketed relative position bias used by T5, and going from post layer-norm to pre layer-norm. While this results in much improved results compared to the original paper, our implementation of UT still outperforms it.

Table 4: CFQ Results. Results on UT are an average of 5 runs on different seeds.
<table><tr><td>Model</td><td>Pretraining</td><td>MCD1</td><td>MCD2</td><td>MCD3</td><td> $\operatorname { A v g } .$ </td><td>MACs1</td></tr><tr><td>T5-based UT (Bergen et al., 2021)</td><td>X</td><td>42.7</td><td>9.5</td><td>11.6</td><td>21.3</td><td>1154M</td></tr><tr><td>Edge Transformer (Bergen et al., 2021)</td><td>x</td><td>47.7</td><td>13.1</td><td>13.2</td><td>24.7</td><td>6504M</td></tr><tr><td>Transformer (Keysers et al., 2019)</td><td>X</td><td>42.5</td><td>11.2</td><td>10.6</td><td>21.4</td><td>1154M</td></tr><tr><td>T5 (Furrer et al., 2020)</td><td>√</td><td>61.6</td><td>31.3</td><td>33.3</td><td>42.1</td><td>1154M</td></tr><tr><td>Roberta (Zheng and Lapata, 2021)</td><td>√</td><td>60.6</td><td>33.6</td><td>36.0</td><td>43.4</td><td>1660M</td></tr><tr><td>Dangle (Zheng and Lapata, 2021)</td><td>√</td><td>78.3</td><td>59.5</td><td>60.4</td><td>66.1</td><td>51033M</td></tr><tr><td>T5-based UT (ours)</td><td>X</td><td> $6 8 . 3 \pm 2 . 9$ </td><td> $4 3 . 1 \pm 1 . 5$ </td><td> $4 5 . 7 \pm { 1 . 8 }$ </td><td> $5 2 . 3 \pm 1 . 6$ </td><td>441M</td></tr><tr><td>UT w/o halting</td><td>X</td><td> $7 1 . 0 \pm 3 . 5$ </td><td> $4 8 . 6 \pm 2 . 3$ </td><td> $5 1 . 3 \pm 0 . 2$ </td><td> $5 6 . 9 \pm 1 . 5$ </td><td>654M</td></tr><tr><td>UT with halting</td><td>X</td><td> $7 2 . 4 \pm 3 . 5$ </td><td> $5 1 . 1 \pm 1 . 8$ </td><td> $5 1 . 7 \pm 2 . 3$ </td><td> $5 8 . 4 \pm 1 . 2 $ </td><td>654M</td></tr></table>

Table 5: Test accuracy of the models, trained on operation lengths of 6, with their out-of-distribution results shown here (lengths 7-12). LSTM baseline from Bowman et al. (2015), and Transformer baseline from Shen et al. (2019)
<table><tr><td>Model</td><td colspan="6">Number of Operations</td><td colspan="3">Comp. Gen.</td></tr><tr><td></td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>A</td><td>B</td><td>C</td></tr><tr><td>LSTM</td><td>88</td><td>84</td><td>80</td><td>78</td><td>71</td><td>69</td><td>80</td><td>60</td><td>59</td></tr><tr><td>Transformer</td><td>51</td><td>52</td><td>51</td><td>51</td><td>51</td><td>48</td><td>53</td><td>51</td><td>51</td></tr><tr><td>SUT</td><td>98</td><td>97</td><td>94</td><td>90</td><td>88</td><td>81</td><td>97</td><td>94</td><td>52</td></tr></table>

The Dangle (Zheng and Lapata, 2021) model, which beats our model, also requires re-running the encoder for every token decoded. This is an expensive process, but given that both our method and Dangle perform well at this task, is additional evidence that iterative processes are beneficial for compositional generalization.

## 4.3 Logical Inference

We use the logical inference task from (Bowman et al., 2015) as a test bench for UT. Despite the apparent simplicity of the language, the task inherently requires the model to learn the hierarchical structure of the problem. Each instance of the task comprises of two logical statements, and the goal is to predict if the statements are equivalent, contradictory, disjoint, or entail in either direction. For example, given $s _ { 1 } = a$ and s<sub>2</sub> = a ( or b ), then $s _ { 1 } ~ \subset ~ s _ { 2 }$ . The crux of the task is in training the model on sequences that have 0-6 logical operators and evaluating it on sequences that have 7-12 operators. Given our sequence-to-sequence setting, we convert the task into a translation task. The model takes sentence1 #SEP# sentence2 as its source sentence, with the target sentence being the single-token label for that pair.

We train a 12 layer model with Attention $( E =$ $1 2 , k = 4 , H = 2 , D = 3 2 , W = 1 )$ and FFD (E = 12, K = 4, D = 128) and halting. Refer to Appendix A for further details. Training a 12- layer Vanilla Transformer achieves approximately the same results as in Shen et al. (2019), so we report their results. Our results in Table 5 confirm the findings of Tran et al. (2018), showing that with recurrence in SUTs, we are able to generalize to longer sequences of the task. While there are other models that induce a tree structure that performs exceedingly well on the task, we wanted to evaluate our model against other popular architectures. The LSTM is a strong baseline, and we find that UT outperforms it in generalization. We also evaluate UTs on the compositional generalization splits as proposed in (Shen et al., 2019), where the splits A, B, and C are in increasing difficulty. The results show that UTs are able to generalize better for the A and B splits, outperforming the LSTM and VT. Split C is still presents a challenge for the Transformer variants.

Additionally, we compute the average halting depth for the test data segmented by operator counts. Because more operators require more nesting of expressions in the sequences, more recursion is required to properly parse the sequence. As expected, in Figure 4, the average halting depth increases as more operators are used. The operator count for these clauses are correlated with length, which suggests that SUTs may be suited to generalize for length. We include further experiments on length generalization in the Appendix Table 8.

![](images/c217f2a34d9505b2f54180997df254250b210f5fa31010eeec1c9122053c8cea.jpg)  
Figure 4: The average dynamic halting depth of the UT model as the number of operators increases in the test set. The model learns to think more when the problem is harder.

## 4.4 Post-training Computation Reduction

Does lowering $\alpha _ { \mathrm { t h r e s h } }$ after training cause the model to halt earlier, saving computation? How much would that cost us in terms of accuracy?

We estimate the skipped SUT block computations given different values of $\alpha _ { \mathrm { t h r e s h } } \quad \in$ $\{ 0 . 1 , 0 . 2 , \ldots , 0 . 9 \}$ by looking at the halting patterns of the decoder given the ground truth sourcetarget pairs. We pass the source-target pair into the model and analyze the halting patterns of the model, giving us a rough estimate of how much computation would be saved as a percentage of computing all layers of the SUT.

Logical Inference We observe the resulting performance on the hardest split of the test set with 12 operations. Due to the already saturated halting pattern, the halting probability $\alpha _ { l }$ spikes rapidly from close to 0 to higher values, resulting in a near constant  50% reduction of the computation time regardless of the threshold.

CFQ Using the MCD1 test split of the dataset, and our best-performing model on MCD1, we perform the $\alpha _ { \mathrm { t h r e s h } }$ adjustment. The halting patterns reflect the repeated structure of SQL, using fewer steps for ‘.‘ and ‘WHERE‘, while the main bulk of the region within {...} requires more SUT steps before halting. Surprisingly, when $0 . 8 \leq \alpha _ { \mathrm { t h r e s h } } \leq$ 0.999, the accuracy remains fairly constant. An estimated 33% of the computation steps were skipped at $\alpha _ { \mathrm { t h r e s h } } = 0 . 8 . \ \mathrm { A t } \ \alpha _ { \mathrm { t h r e s h } } = 0 . 1$ , there is a slight increase in the number of computed steps, which is possible since halting earlier will result in different embeddings, and result in different halting decisions in other timesteps. Overall, the results suggest that we can save about 20% of the SUT computation steps without any drop in accuracy, and about 50% for a 0.2% decrease.

![](images/5d6965bcbd1e707ced7c5c8204589b51611be6c7eb05de253207d1c2a9a8d979.jpg)

![](images/5856afdd8e211981c8aa6db9e8f1b4066e9c7115ef14635311104c34f8cc2948.jpg)  
Figure 5: Above: Plot of $\begin{array} { r } { 1 - \sum _ { l ^ { \prime } = 1 } ^ { l - 1 } \alpha _ { l ^ { \prime } } ^ { ( t ) } } \end{array}$ , for an example Logical Inference input — x-axis: timesteps, y-axis: layers. This visualizes the halting pattern of the model: dark blue represents halted, while yellow represents active. Below: Efficiency vs. Performance tradeoff curves when $\alpha _ { \mathrm { t h r e s h } }$ is adjusted.

English-German Translation For this larger dataset, we find that these translation models halt much later, suggesting that the translation task requires more computational steps than the 6-layer SUT we used. However, further increasing the number of layers to 12 layers does not bring about much benefit, as evidenced by the halting in Figure 4, which is an example of the halting mechanism using nearly all layers. For comparison, Admin 60L-12L model, requires a 60-layer encoder to achieve its performance. Even when $\alpha _ { \mathrm { t h r e s h } } = 1$ the skipped computation steps remain at about 33%, compared to 80% in the CFQ task. We find that we can reduce the computation by 9% while still retaining a BLEU score of 29.1.

![](images/02667bab294bc11ac11883d7cac05f2e5cfbff88233abe1b9edc6dc783290897.jpg)

![](images/0e98c35d43c848fe037529494a5f1da867e41576ccafac38a45800dee3bbefb2.jpg)  
Figure 6: Halting plot and trade-off curves for CFQ. (See Figure 5 for description)

## 5 Conclusion

We show that it is possible to scale up the UT via SUTs, and SUTs outperforms models of the same capacity in the WMT’14 English-to-German translation task. The recursive nature of both UTs and SUTs allows for better inductive biases, which we have demonstrated in synthetic tasks like CFQ and logical inference. VTs have been shown to be poor at these compositional generalization tasks without additional domain knowledge. The stick-breaking dynamic halting mechanism also allows post-training adjustment of computation cost, which is a boon for deployment at scale.

Limitations While the experiments in this paper show the desirable generalization properties of UTs, there are some aspects of compositional generalization that SUTs do not solve. Importantly, while we demonstrate scaling UTs up via SMoEs, further experiments on larger settings are needed to ascertain viability in large scale systems. Other issues may also crop up in the further scaling of SUTs, but we believe there is ample literature to draw on to finding solutions for these problems.

![](images/0feab526d291f1623d29c0e32e3d287616969d232ebbb6ce7b55dcaf8b6d0e2f.jpg)

![](images/70830fbf7d7010965c3c91e3642394dbb3656149a83975cc4740ae1cc4397560.jpg)  
Figure 7: Halting plot and trade-off curves for English-German Translation. (See Figure 5 for description)

## References

Leon Bergen, Timothy O’Donnell, and Dzmitry Bahdanau. 2021. Systematic generalization with edge transformers. Advances in Neural Information Processing Systems, 34:1390–1402.

Ondˇrej Bojar, Christian Buck, Christian Federmann, Barry Haddow, Philipp Koehn, Johannes Leveling, Christof Monz, Pavel Pecina, Matt Post, Herve Saint-Amand, et al. 2014. Findings of the 2014 workshop on statistical machine translation. In Proceedings of the ninth workshop on statistical machine translation, pages 12–58.

Samuel R Bowman, Christopher D Manning, and Christopher Potts. 2015. Tree-structured composition in neural networks without tree-structured architectures. arXiv preprint arXiv:1506.04834.

Zitian Chen, Yikang Shen, Mingyu Ding, Zhenfang

Chen, Hengshuang Zhao, Erik Learned-Miller, and Chuang Gan. 2022. Mod-squad: Designing mixture of experts as modular multi-task learners. arXiv preprint arXiv:2212.08066.

Kenneth Church and Patrick Hanks. 1990. Word association norms, mutual information, and lexicography. Computational linguistics, 16(1):22–29.

Róbert Csordás, Kazuki Irie, and Jürgen Schmidhuber. 2021. The devil is in the detail: Simple tricks improve systematic generalization of transformers. arXiv preprint arXiv:2108.12284.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. 2018. Universal transformers. arXiv preprint arXiv:1807.03819.

Grégoire Delétang, Anian Ruoss, Jordi Grau-Moya, Tim Genewein, Li Kevin Wenliang, Elliot Catt, Marcus Hutter, Shane Legg, and Pedro A Ortega. 2022. Neural networks and the chomsky hierarchy. arXiv preprint arXiv:2207.02098.

Andrew Drozdov, Nathanael Schärli, Ekin Akyürek, Nathan Scales, Xinying Song, Xinyun Chen, Olivier Bousquet, and Denny Zhou. 2022. Compositional semantic parsing with large language models. arXiv preprint arXiv:2209.15003.

Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jian, Bill Yuchen Lin, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena D Hwang, et al. 2023. Faith and fate: Limits of transformers on compositionality. arXiv preprint arXiv:2305.18654.

Maha Elbayad, Jiatao Gu, Edouard Grave, and Michael Auli. 2019. Depth-adaptive transformer. arXiv preprint arXiv:1910.10073.

William Fedus, Barret Zoph, and Noam Shazeer. 2021. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity.

Daniel Furrer, Marc van Zee, Nathan Scales, and Nathanael Schärli. 2020. Compositional generalization in semantic parsing: Pre-training vs. specialized architectures. arXiv preprint arXiv:2007.08970.

Alex Graves. 2016. Adaptive computation time for recurrent neural networks. arXiv preprint arXiv:1603.08983.

Michael Hahn. 2020. Theoretical limitations of selfattention in neural sequence models. Transactions of the Associationfor Computational Linguistics, 8:156– 171.

Yiding Hao, Dana Angluin, and Robert Frank. 2022. Formal language recognition by hard attention transformers: Perspectives from circuit complexity. Transactions of the Association for Computational Linguistics, 10:800–810.

Dieuwke Hupkes, Verna Dankers, Mathijs Mul, and Elia Bruni. 2020. Compositionality decomposed: How do neural networks generalise? Journal ofArtificial Intelligence Research, 67:757–795.

Łukasz Kaiser and Ilya Sutskever. 2015. Neural gpus learn algorithms. arXiv preprint arXiv:1511.08228.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

Daniel Keysers, Nathanael Schärli, Nathan Scales, Hylke Buisman, Daniel Furrer, Sergii Kashubin, Nikola Momchev, Danila Sinopalnikov, Lukasz Stafiniak, Tibor Tihon, et al. 2019. Measuring compositional generalization: A comprehensive method on realistic data. arXiv preprint arXiv:1912.09713.

Yoon Kim. 2021. Sequence-to-sequence learning with latent neural grammars. Advances in Neural Information Processing Systems, 34:26302–26317.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. 2019. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942.

Yuanpeng Li, Liang Zhao, Jianyu Wang, and Joel Hestness. 2019. Compositional generalization for primitive substitutions. arXiv preprint arXiv:1910.02612.

Bingbin Liu, Jordan T Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. 2022. Transformers learn shortcuts to automata. arXiv preprint arXiv:2210.10749.

Xiaodong Liu, Kevin Duh, Liyuan Liu, and Jianfeng Gao. 2020. Very deep transformers for neural machine translation. arXiv preprint arXiv:2008.07772.

William Merrill, Ashish Sabharwal, and Noah A Smith. 2022. Saturated transformers are constant-depth threshold circuits. Transactions of the Association for Computational Linguistics, 10:843–856.

Ott Myle, Edunov Sergey, Grangier David, Auli Michael, et al. 2018. Scaling neural machine translation. WMT, pages 1–9.

Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. fairseq: A fast, extensible toolkit for sequence modeling. arXiv preprint arXiv:1904.01038.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th annual meeting of the Association for Computational Linguistics, pages 311–318.

Hao Peng, Roy Schwartz, Dianqi Li, and Noah A Smith. 2020. A mixture of h-1 heads is better than h heads. In Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics, pages 6566– 6577.

Jake Russin, Jason Jo, Randall C O’Reilly, and Yoshua Bengio. 2019. Compositional generalization in a deep seq2seq model by separating syntax and semantics. arXiv preprint arXiv:1904.09708.

Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Q Tran, Yi Tay, and Donald Metzler. 2022. Confident adaptive language modeling. arXiv preprint arXiv:2207.07061.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538.

Yikang Shen, Shawn Tan, Arian Hosseini, Zhouhan Lin, Alessandro Sordoni, and Aaron C Courville. 2019. Ordered memory. Advances in Neural Information Processing Systems, 32.

Yikang Shen, Zheyu Zhang, Tianyou Cao, Shawn Tan, Zhenfang Chen, and Chuang Gan. 2023. Moduleformer: Learning modular large language models from uncurated data. arXiv preprint arXiv:2306.04640.

Vsevolod Sourkov. 2018. Igloo: Slicing the features space to represent sequences. arXiv preprint arXiv:1807.03402.

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. 2016. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818–2826.

Sho Takase and Shun Kiyono. 2021. Lessons on parameter sharing across layers in transformers. arXiv preprint arXiv:2104.06022.

Shawn Tan, Yikang Shen, Timothy J O’Donnell, Alessandro Sordoni, and Aaron Courville. 2020. Re cursive top-down production for sentence generation with latent trees. arXiv preprint arXiv:2010.04704.

Shawn Tan and Khe Chai Sim. 2016. Towards implicit complexity control using variable-depth deep neural networks for automatic speech recognition. In 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5965–5969. IEEE.

Yi Tay, Mostafa Dehghani, Samira Abnar, Hyung Won Chung, William Fedus, Jinfeng Rao, Sharan Narang, Vinh Q Tran, Dani Yogatama, and Donald Metzler. 2022. Scaling laws vs model architectures: How does inductive bias influence scaling? arXiv preprint arXiv:2207.10551.

Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. 2020. Long range arena: A benchmark for efficient transformers. arXiv preprint arXiv:2011.04006.

Ke Tran, Arianna Bisazza, and Christof Monz. 2018. The importance of being recurrent for modeling hierarchical structure. arXiv preprint arXiv:1803.03585.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Felix Wu, Angela Fan, Alexei Baevski, Yann Dauphin, and Michael Auli. 2018. Pay less attention with lightweight and dynamic convolutions. In International Conference on Learning Representations.

Xiaofeng Zhang, Yikang Shen, Zeyu Huang, Jie Zhou, Wenge Rong, and Zhang Xiong. 2022. Mixture of attention heads: Selecting attention heads per token. arXiv e-prints, pages arXiv–2210.

Hao Zheng and Mirella Lapata. 2021. Disentangled sequence to sequence learning for compositional generalization. arXiv preprint arXiv:2110.04655.