# Failures Pave the Way: Enhancing Large Language Models through Tuning-free Rule Accumulation

Zeyuan Yang<sup>1</sup>, Peng Li∗<sup>,2,3</sup>, Yang Liu∗ <sup>1,2,3</sup>

<sup>1</sup>Dept. of Comp. Sci. & Tech., Institute for AI, Tsinghua University, Beijing, China <sup>2</sup>Institute for AI Industry Research (AIR), Tsinghua University, Beijing, China <sup>3</sup>Shanghai Artificial Intelligence Laboratory, Shanghai, China yangzeyu21@mails.tsinghua.edu.cn; lipeng@air.tsinghua.edu.cn liuyang2011@tsinghua.edu.cn

## Abstract

Large Language Models (LLMs) have showcased impressive performance. However, due to their inability to capture relationships among samples, these frozen LLMs inevitably keep repeating similar mistakes. In this work, we propose our Tuning-free Rule Accumulation (TRAN) framework, which guides LLMs in improving their performance by learning from previous mistakes. Considering data arrives sequentially, LLMs gradually accumulate rules from incorrect cases, forming a rule collection. These rules are then utilized by the LLMs to avoid making similar mistakes when processing subsequent inputs. Moreover, the rules remain independent of the primary prompts, seamlessly complementing prompt design strategies. Experimentally, we show that TRAN improves over recent baselines by a large margin.

## 1 Introduction

Large language models (LLMs) have recently demonstrated remarkable performance across a broad spectrum of natural language processing (NLP) tasks. Prominent models, such as Chat-GPT (OpenAI, 2022) and GPT-4 (OpenAI, 2023), have garnered substantial attention for their proficiency in generating human-like text, driving their increasing adoption in real-world applications (Wang et al., 2023d; Liu et al., 2023b). As these applications involve ever-changing scenarios and specific requirements (Zhao et al., 2023), there is a growing interest in exploring approaches to tailor these models to meet specific goals.

To address the challenge of aligning LLMs with human preference, Ouyang et al. (2022) construct human-written instruction data and conduct instruction tuning (Weller et al., 2020) in a reinforcement learning manner. Recent works (Taori et al., 2023; Chiang et al., 2023) further gain remarkable performance by employing parameter-efficient tuning (Liu et al., 2023a; Ding et al., 2023), which avoids fine-tuning the entire model. Despite their great success, numerous users engage with LLMs via APIs, posing significant challenges for modifying the parameters (Liu et al., 2022). Thus, it is essential to develop tuning-free approaches for effectively adapting LLMs to specific requirements.

![](images/a3b3daec3256b2f120213e3a6c880230af7aebab81e51783ff9df284ee4a911c.jpg)  
Figure 1: Examples of our framework. The left demonstrates that a frozen LLM keeps making similar mistakes, and the right represents our framework, construct-If a judgment is based ing a rule collection to guide subsequent generations.

Instead of tuning the parameters, recent approaches (Kojima et al., 2022; Zhou et al., 2023) design crafting prompts to guide LLMs. Sun et al. (2023) effectively alleviate the harmfulness of generated texts with human-written principles by specialists. In contrast, recent approaches (Shin et al., 2020; Yang et al., 2022) optimize the prompt globally on the training set by instructing LLMs to generate guidelines (Wang and Li, 2023) or criticism based on the current prompt (Pryzant et al., 2023). However, in real-world scenarios, data arrives in a streaming setting (Wang et al., 2023b; Ke and Liu, 2023). As depicted in Fig. 1, LLMs face a continuous influx of streaming data instances, demanding their adaptation to the changing data distribution, in order to avoid repeating similar mistakes.

In this work, we address this challenge with our

Tuning-free Rule AccumulatioN (TRAN) framework, which enables the self-adaptation of LLMs to specific scenarios without additional training sets or complementary models in an online learning fashion (Aljundi et al., 2019; Javed and White, 2019). Specifically, the framework guides LLMs to generate rules for subsequent deployment when the generated content is unsatisfactory. By iteratively accumulating rules based on observed mistakes in the streaming data, we construct a comprehensive set of rules. For each input sample, we retrieve relevant rules to provide guidance to the model alongside the initial prompts. Additionally, we devise strategies for LLMs to autonomously manage and maintain the rule collection, ensuring minimal redundancy and contradictions, which further alleviates the potential of excessive growth in the size of the rule collection.

To validate our framework, we conduct experiments over various tasks, spanning multi-choice question answering and text classification tasks from different domains. Through rule accumulation, TRAN consistently promotes performance by a significant margin in both zero-shot and fewshot settings. Moreover, as rules are independent of the prompt design, TRAN seamlessly complements prompt design strategies like Chain-of-Thought (Kojima et al., 2022; Zhang et al., 2022). Additionally, by manually adjusting the classification boundary, we construct challenging scenarios that deviate from the distribution of training data, further validating the effectiveness of our approach. We summarize our contributions as follows:<sup>1</sup>.

• We propose TRAN, a tuning-free approach that effectively aligns LLMs to specific scenarios. By iteratively generating and utilizing rules for subsequent deployment, TRAN enables LLMs to avoid the repetition of similar mistakes in a streaming setting.

• Based on the rule collection, we develop strategies to autonomously manage and maintain the rules, addressing the challenge posed by the rapid scale expansion of streaming data.

• TRAN is a prompt-independent framework that complements prompt design strategies. Experiments substantiate that TRAN significantly enhances performance in both online learning scenarios and situations where the full training set is available.

## 2 Tuning-free Rule Accumulation

## 2.1 Problem Definition

In this work, we consider a pre-trained LLM f deployed in specific scenarios in the online learning setting, where data instances arrive in an endless stream, denoted by $\{ ( x _ { t } , y _ { t } ) \} _ { t = 1 } ^ { T }$ . At each time step t, the model observes $( x _ { t } , y _ { t } )$ , and the model response is denoted as $f ( x _ { t } )$

Throughout the deployment phase, LLMs inevitably make mistakes. Specifically, we consider the parameters are inaccessible and the model remains frozen, resulting in the LLM keep making similar mistakes. Therefore, we aim to leverage previous mistakes to improve subsequent inferences.

## 2.2 Overview

In this section, we provide an overview of our Tuning-free Rule Accumulation (TRAN) framework, which guides the model in summarizing rules for subsequent text generation. For better illustration, we first present a basic prompt template for the multi-choice question answering task, where $x _ { t }$ consists of context, question, and choices. Detailed prompts are provided in Appendix A.2.

Basic prompt

Context: {context} Question: {question} Choices: {choices} Answer:

Assuming the model makes a mistake, we guide the model to generate rules to rectify this mistake. For the sake of simplicity, these rules are expressed in an “if ..., then ...” format in this work. For a better understanding, we provide two examples of generated rules below.

Exp 1: If there is no information about a person’s financial situation, then it is not appropriate to assume that they live in poverty or are wealthy. Exp 2: If the review contains profanity or vulgar language, then it may be offensive depending on the context and severity.

By iteratively accumulating rules, we construct a rule collection $\Theta = \cup _ { t } \{ r _ { t } \}$ . For each subsequent input $x _ { t }$ , we retrieve rules $R _ { t } ^ { u s e } = \{ r _ { t } ^ { u s e } \}$ from Θ. The selected rules are then concatenated prior to the base prompt, providing guidance for better answers.

![](images/aa5e34242982510204fb896a182916d65e8f08da239529ab7b51f2e236db9071.jpg)  
Figure 2: The overall process of constructing the rule collection Θ: (a) generate rules based on the current mistake; (b) evaluate and keep effective rules; (c) put the mistake in the mistake collection if no effective rule exists; (d) retrieve relevant mistakes from the mistake collection; (e) summarize rules from the current and previous mistakes; (f) append result effective rules into the rule collection.

The model response is denoted by $f ( x _ { t } , R _ { t } ^ { u s e } )$ . Below is the template for our rule-based prompt.

## Rule-based prompt

Rules: {rules}

Context: {context} Question: {question} Choices: {choices} Answer:

Besides, the generated rules are not always practical. Thus, we direct the LLMs to evaluate their ability to fix existing errors. Through this process, we progressively accumulate effective rules. The process of generating and evaluating rules is illustrated in Section 2.3. In addition, to maintain consistency and coherence within the rule collection, we rely on the LLM to assess whether incoming rules are either identical or contradictory to the existing rules. Furthermore, we remove less frequently used rules, thereby limiting the scale of the rule collection. The strategies for managing the rule collection are presented in Section 2.4.

Furthermore, our framework guides the LLM to handle different components. Specifically, the same model is adopted for various purposes, while for better clarity, we utilize different annotations (subscripts) for distinguishing these purposes.

## 2.3 Rule Construction

To construct the rule collection Θ, we leverage the LLM to generate and evaluate rules based on the observed mistakes. The process to construct the rule collection is illustrated in Fig. 2.

Consider that the model makes a mistake on current input $x _ { t } ,$ in other words, $f ( x _ { t } ) \neq y _ { t }$ . We first harness the model $f$ to generate rules $R _ { t } ^ { r a w }$ namely the process (a) in Fig. 2:

$$
R _ { t } ^ { r a w } = \{ r _ { t , i } ^ { r a w } \} = f _ { g e n } ( x _ { t } , y _ { t } ) ,\tag{1}
$$

where $r _ { t , i } ^ { r a w }$ denotes the i-th result rule and $f _ { g e n }$ denotes the generating process. Utilizing the insights gained from the current mistake, we guide the LLM in generating explanations for the given input question. Building upon this, we then task the model with transforming these explanations into concise and structured rules. Presented below are the simplified prompts. The full prompt scheme is provided in Appendix A.3.

## Generating prompt (Simplified)

Please give the reasons for the answer.   
Please rewrite these reasons into rules.

To maintain the quality of the rule collection, we aim to keep the effective rules only. For each generated rule $r _ { t , i } ^ { r a w }$ , we retest the input $x _ { t }$ and only keep the rules that can rectify the current mistake. The failed ones, namely $f ( x _ { t } , r _ { t , i } ^ { r a w } ) \neq y _ { t }$ , are then eliminated. Furthermore, if all rules fail to fix the mistake, we consider the input as a “failed” mistake and then store it in the mistake collection Φ.

Besides, instead of getting rules from a single mistake, human beings rather rely on summarizing rules from multiple mistakes. Therefore, we further instruct the LLM to generate rules based on multiple previous mistakes. For each “failed” mistake $x _ { t }$ we retrieve similar old mistakes $\Phi _ { t } = \{ ( x _ { i } ^ { t } , y _ { i } ^ { t } ) \}$ from the mistake collection, the process (d) in Fig. 2. Next, by providing $\Phi _ { t }$ along with $( x _ { t } , y _ { t } )$ we attempt to summarize new rules as

$$
R _ { t } ^ { s u m } = \{ r _ { t , i } ^ { s u m } \} = f _ { s u m } ( x _ { t } , y _ { t } , \Phi _ { t } ) ,\tag{2}
$$

where $r _ { t , i } ^ { s u m }$ denotes the i-th generated rule and $f _ { s u m }$ denotes the LLM for summarizing rules. Similarly, only effective rules are reserved. Finally, with the process (b) in Fig. 2, we get the effective rules $R _ { t } = \{ r _ { t } | r _ { t } \in R _ { t } ^ { s u m / r a w }$ and $f ( x _ { t } , r _ { t } )$ $y _ { t } \big \}$ for the current input mistake x<sub>t</sub> and append them into the rule collection $\textstyle \Theta = \bigcup _ { t } R _ { t }$

## 2.4 Rule Maintenance

With the rule collection constructed in Section 2.3, the LLM can effectively leverage past mistakes to enhance subsequent performance. However, as mistakes accumulate during deployment, the rule collection may become redundant. Moreover, incoming rules are dynamic and can be contradictory, reflecting the evolving user requirements. To address these challenges, we direct the LLM towards maintaining a high-quality rule collection.

For each incoming rule r, we extract relevant rules from the existing rule collection based on semantic similarity. Subsequently, we utilize the LLM, notated by $f _ { c h e c k }$ , to evaluate whether selected rules are either identical or contradictory to rule r. If such similarities or contradictions exist, we retain the new rule r only. The simplified checking prompt is shown below and the full prompt scheme is provided in Appendix A.3.

## Checking prompt

Please identify whether these two rules are identical (contradictory): {rule 1}; {rule 2}

Furthermore, to prevent the rule collection from growing excessively, we employ the Least Recently Used (LRU) strategy. When the number of rules surpasses a predefined threshold, we drop the least recently used rules. An ablation study on the threshold is provided in Section 3.3 to assess its impact.

## 3 Experiments

## 3.1 Experimental Setup

Datasets. We evaluate our framework on the seven tasks from the challenging multi-choice question answering benchmark BBQ-Lite (Srivastava et al., 2023), which measures social biases with customwritten templates from diverse domains. Moreover, we conduct experiments on several text classification tasks, including TweetEval (Barbieri et al., 2020), AGNews (Zhang et al., 2015), and DBPedia (Lehmann et al., 2015). For all tasks, we report the results on the test set. We adopt the “offensive” subtask of TweetEval and randomly select 1,000 samples from the other two tasks for consistency. The details and statistics of the datasets are provided in Appendix A.1.

Baselines. We compare our TRAN framework against competitive and well-established methods. Notably, we focus on non-parametric approaches that are comparable to TRAN. For intermediate reasoning strategies, we adopt Zero-Shot CoT (Kojima et al., 2022) and Auto-CoT (Zhang et al., 2022). For the approaches optimizing the prompt, we compare against SALAM (Wang and Li, 2023) in both zero-shot and few-shot manners. Another relevant approach is APO (Pryzant et al., 2023). However, the detailed prompts of APO have not been released yet and we would like to include the comparison after the prompts are released. Implementation details are provided in Appendix A.4.

Setup. Unless otherwise stated, all experiments were performed using the March 2023 version of gpt-3.5-turbo, leveraging the OpenAI LLM API service<sup>2</sup> with a temperature of 0.0. The top three rules are selected with a maximum rule collection size set to 100 over all datasets. In this work, we employ the widely-used BM25 (Robertson et al., 1994) to retrieve rules, which demonstrates a satisfactory performance and could be further replaced by alternative powerful approaches.

## 3.2 Results

We show the comparative results on BBQ-Lite and text classification tasks in Table 1 and Table 2 respectively. Table 1 demonstrates the superior performance of our framework compared to other baselines on BBQ-Lite. In the zero-shot setting, TRAN achieves an average accuracy of about 91.6%, outperforming Zero-Shot CoT by 6.3%. In contrast, the default frozen model only achieves an average accuracy of approximately 75.4%. Moreover, our approach exhibits a substantial performance boost of 8.8% over SALAM. Based on the results in Table 2, TRAN also demonstrates comparable or superior performance on text classification tasks when compared to other baselines.

Similarly, in the few-shot scenario, our approach consistently outperforms other baselines. In this setting, each approach employs the same strategy of retrieving relevant previous inputs as examples. As both SALAM and our TRAN accumulate experience from past mistakes, incorporating previous inputs unveils effectiveness beyond the input contents. It is noteworthy that SALAM demonstrates significant benefits from few-shot examples, while TRAN maintains a superiority of approximately 2% in terms of average accuracy. Table 2 indicates similar results on text classification tasks.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="8">BBQ-Lite</td></tr><tr><td>Age</td><td>Religion</td><td>Sexual</td><td>Nationality</td><td>Disability</td><td>SES</td><td>Physical</td><td>Avg</td></tr><tr><td rowspan="4">(zer-sot)</td><td>Zero-Shot</td><td>71.3</td><td>80.3</td><td>88.3</td><td>76.0</td><td>60.6</td><td>79.1</td><td>72.5</td><td>75.4</td></tr><tr><td>Zero-Shot CoT</td><td>86.7</td><td>85.4</td><td>84.6</td><td>89.4</td><td>78.6</td><td>91.6</td><td>81.1</td><td>85.3</td></tr><tr><td>SALAM</td><td>82.4</td><td>88.5</td><td>88.5</td><td>83.7</td><td>71.5</td><td>85.3</td><td>79.7</td><td>82.8</td></tr><tr><td>Ours</td><td>92.1</td><td>89.7</td><td>92.8</td><td>94.7</td><td>88.2</td><td>97.3</td><td>86.6</td><td>91.6</td></tr><tr><td rowspan="4">(eww-ot)</td><td>Few-Shot</td><td>82.7</td><td>87.2</td><td>92.4</td><td>91.0</td><td>86.1</td><td>96.0</td><td>85.9</td><td>88.7</td></tr><tr><td>Auto-CoT</td><td>89.7</td><td>90.0</td><td>88.1</td><td>88.9</td><td>85.8</td><td>81.0</td><td>85.6</td><td>87.0</td></tr><tr><td>SALAM</td><td>89.4</td><td>86.3</td><td>92.4</td><td>91.2</td><td>88.5</td><td>93.6</td><td>87.2</td><td>89.8</td></tr><tr><td>Ours</td><td>92.5</td><td>90.0</td><td>93.4</td><td>94.2</td><td>90.7</td><td>97.8</td><td>87.8</td><td>92.3</td></tr></table>

Table 1: Comparison of accuracy on BBQ-Lite under both zero-shot and few-shot settings, using 4 examples. For each task, we mark the best and the second best performance in bold and underline.

<table><tr><td rowspan="2">Method</td><td colspan="3">Text Classification</td></tr><tr><td>AGNews</td><td>DBPedia</td><td>TweetEval</td></tr><tr><td></td><td>Zero-Shot</td><td>85.9</td><td>92.9 77.6</td></tr><tr><td>(zer-ot)</td><td>ZS-CoT 84.1</td><td>94.2</td><td>78.0</td></tr><tr><td></td><td>SALAM 85.2</td><td>93.1</td><td>78.1</td></tr><tr><td></td><td>Ours 87.9</td><td>94.4</td><td>78.0</td></tr><tr><td>(feww-ot)</td><td>Few-Shot</td><td>83.3 92.7</td><td>75.7</td></tr><tr><td></td><td>Auto-CoT 83.4</td><td>88.5</td><td>69.9</td></tr><tr><td></td><td>SALAM 84.1</td><td>93.7</td><td>76.4</td></tr><tr><td></td><td>Ours 86.1</td><td>95.0</td><td>76.5</td></tr></table>

Table 2: Comparison of accuracy on three text classification datasets under both zero-shot and few-shot settings, using 4 examples. ZS-CoT denotes Zero-Shot CoT. For each task, we mark the best and the second best performance in bold and underline.

As the rules are iteratively accumulated, we hypothesize that our TRAN framework demonstrates progressive performance improvement with the accumulation of more data. To validate this, we present the ratio of the number of mistakes between our approach and the default frozen model:

$$
N _ { o u r s } ^ { m } / N _ { f r o z e n } ^ { m } ,\tag{3}
$$

where $N ^ { m }$ denotes the number of mistakes. Results on three representative tasks are depicted in Fig. 3, in both zero-shot and few-shot settings. We choose the range after 30 rules are accumulated, where the rule collection is roughly constructed. As illustrated in Fig. 3, TRAN has significantly reduced the number of mistakes by approximately 40% and 20% after encountering 700 samples on two settings respectively, universally over three datasets, following our assumption, which further supports the effectiveness of the rule collection.

<table><tr><td></td><td>Zero-Shot</td><td>ZS-CoT</td><td>Ours</td></tr><tr><td>Dyck Language</td><td>39.6%</td><td>36.4%</td><td>44.4%</td></tr></table>

Table 3: Comparison of accuracy on Dyck Language under the zero-shot setting. ZS-CoT denotes Zero-Shot CoT. We mark the best performance in bold

In addition, we conduct preliminary experiments on the Dyck Language (Ebrahimi et al., 2020) generation task from Big-Bench Hard (Suzgun et al., 2022). Experimental results are presented in Table 3 and detailed findings can be found in Appendix A.6. In a nutshell, given the presence of concrete rules for addressing the Dyck Language task, our TRAN framework gains substantial improvement. Further exploration of adapting TRAN to universal generation tasks remains a topic for future research.

In general, our TRAN showcases exceptional performance in both zero-shot and few-shot settings. Furthermore, as the model encounters more inputs, TRAN exhibits a greater improvement in performance. For more detailed prompts and examples of rules, please refer to Appendix A.2 and A.9.

## 3.3 Ablation Study

In this section, we conduct an ablation study of our TRAN framework. Results over three tasks are depicted in Table 4. According to Table 4, we first notice that the performance consistently degrades without summarizing rules from multiple samples. This implies that accumulating experience solely from a specific input is insufficient, which underscores the significance of global insight over previous mistakes, aligning with the findings outlined in (Wang and Li, 2023). Besides, by eliminating outdated or redundant rules, our TRAN maintains a high-quality rule collection, resulting in a performance boost of about 1%.

![](images/e9380251b6c39b0d330552c65b5103379e4cae9eddda29078881ec355833691d.jpg)

![](images/e2dff99c58bcc085ff4389fc58367c6fcc49109f9d0969ec088b591d1296d2a7.jpg)

![](images/630e0a1375441304af7fc945007c2a41d142addf4df47a26a03dc86502b662f4.jpg)  
Figure 3: The ratio of the number of mistakes (Eq. 3) between our approach and the default frozen model in (a) zero-shot and (b) few-shot settings. (c) An ablation study on the size of the rule collection. Note that a rule collection of 0 rules entails the frozen zero-shot setting.

<table><tr><td></td><td>Ours</td><td>-LRU</td><td> $- f _ { s u m }$ </td><td> $- f _ { c h e c k }$ </td></tr><tr><td>Age</td><td>92.1</td><td>89.7</td><td>88.8</td><td>91.3</td></tr><tr><td>Disability</td><td>88.2</td><td>83.7</td><td>86.3</td><td>86.3</td></tr><tr><td>Nationality</td><td>94.7</td><td>93.4</td><td>93.8</td><td>93.9</td></tr></table>

Table 4: The ablation study of our TRAN framework. LRU denotes the LRU strategy used for maintaining rule collection, $f _ { s u m }$ denotes the process of summarizing rules from multiple mistakes, and $f _ { c h e c k }$ denotes the process of eliminating duplication and contradictions.

Additionally, we observed a performance drop when the limitation on the size of the rule collection was lifted. For each of the three tasks, LRU removed a total of 10, 39, and 15 rules, respectively. Notably, the Disability task experienced a substantial performance degradation of 4.5%, aligning with the number of rules eliminated. This reinforces the significance of maintaining a restricted rule collection to ensure optimal performance.

To delve deeper into the influence of the number of rules, we conducted an ablation study on the size of the rule collection, as illustrated in Fig. 3- (c). The results depicted in Fig. 3-(c) demonstrate that maintaining a rule collection consisting of 20 rules yields a substantial performance improvement compared to the default frozen setting. This further validates the efficacy of the generated rules Additionally, it is noteworthy that reducing the size of the rule collection has a relatively minor impact compared to removing the limitation, which emphasizes the significance of rule quality.

Moreover, in our TRAN framework, as rules are accumulated from previous mistakes, the order of data sequences can impact performance. For previous experiments, we used the default order of data sequences. To comprehensively understand this influence, we conduct experiments of various sequence orders. Detailed results and analysis are presented in Table 17 of Appendix 17. Notably, our method exhibits consistent performance across different sample orderings, with this resilience becoming particularly pronounced for longer sequences.

## 4 Analysis

Whether TRAN complements CoT? Recent approaches (Kojima et al., 2022; Zhang et al., 2022) have achieved remarkable performance gains by improving the prompts. In our TRAN framework, as mentioned in Section 2.2, the rules are concatenated before the base prompt. Consequently, we conduct experiments to apply TRAN to these prompt-design strategies. The results presented in Table 5 demonstrate that integrating TRAN with CoT yields a significant performance boost, highlighting the efficacy of our framework.

How TRAN performs when the full training set is given? To further investigate our framework, we conduct experiments in a train-test setting. Following Wang and Li (2023), we randomly select 250 samples from each task within BBQ-Lite, and we divided the data into the training set and the test set using a 0.8/0.2 split ratio. Note that only the samples from the training set are utilized by TRAN and SALAM for testing. The comparison results are presented in Table 6. According to Table 6, our TRAN exhibits a significant performance advantage over SALAM in the zero-shot setting. Even in the few-shot setting, where SALAM demonstrates considerable improvement, TRAN still outperforms SALAM by an average margin of 1.7%.

<table><tr><td rowspan="2">Methods</td><td colspan="2">Zero-shot</td><td colspan="2">Few-shot</td></tr><tr><td>CoT</td><td>Ours</td><td>CoT</td><td>Ours</td></tr><tr><td>Age</td><td>86.7</td><td>+4.9</td><td>89.7</td><td>+2.9</td></tr><tr><td>Disability</td><td>78.6</td><td>+9.6</td><td>85.8</td><td>+7.4</td></tr><tr><td>Nationality</td><td>89.4</td><td>+6.3</td><td>88.9</td><td>+5.5</td></tr><tr><td>Physical</td><td>81.0</td><td>+6.6</td><td>87.2</td><td>+3.7</td></tr><tr><td>Religion</td><td>85.4</td><td>+4.3</td><td>90.0</td><td>+0.0</td></tr><tr><td>SES</td><td>91.6</td><td>+0.6</td><td>81.0</td><td>+9.2</td></tr><tr><td>Sexual</td><td>84.6</td><td>+5.6</td><td>88.1</td><td>+3.1</td></tr><tr><td>Average</td><td>85.3</td><td>+5.4</td><td>87.0</td><td>+4.5</td></tr></table>

Table 5: Comparison of accuracy by imposing our TRAN framework on CoT strategies. For the few-shot setting, we employ Auto-CoT as the base approach.

Moreover, by incorporating a training set, our TRAN provides the model with an initial nonempty rule collection. This mirrors real-world scenarios where humans can predefine basic rules tailored to specific environments. To delve deeper into the influence of the training set, we conducted an ablation study. According to the experimental results in Table 18 of Appendix A.7, the inclusion of a training set considerably enhances performance, with our TRAN framework outperforming other baselines. In summary, TRAN consistently maintains good performance under different settings.

Whether TRAN generalizes to out-of-domain tasks? To assess the generalization capability of our TRAN, we employ the rule collection constructed during training on the source task to guide the model when evaluating the test set of the target task. The results are illustrated in Fig. 4. Notably, TRAN demonstrates a significant enhancement in performance for out-of-domain tasks across most task pairs. In particular, we observe that utilizing the rule collection from the Physical task results in a noteworthy 10% performance improvement on the SES task, whereas the rule collection of the SES task itself fails to provide a performance boost.

<table><tr><td colspan="2"></td><td>w/ SALAM</td><td>w/ Ours</td></tr><tr><td rowspan="5">(ew-ot) Sexual Avg Age Disability Nationality Physical Religion</td><td>Age 76.0 Disability 84.0</td><td>80.0</td><td>78.0</td></tr><tr><td>Nationality 88.0</td><td>88.0 94.0</td><td>86.0 98.0</td></tr><tr><td>Physical 82.0 82.0</td><td>84.0</td><td>84.0</td></tr><tr><td>Religion SES 82.0</td><td>84.0 82.0</td><td>90.0 82.0</td></tr><tr><td>92.0 83.7</td><td>92.0 86.3</td><td>98.0 88.0</td></tr><tr><td colspan="2">(er-ot) Avg</td><td>68.0 76.0 50.0 68.0 78.0</td><td>76.0 76.0 80.0</td><td>82.0 84.0 94.0</td></tr></table>

Table 6: Comparison of accuracy on BBQ-Lite with 200 training samples and 50 test samples. In the fewshot setting, each task utilizes 3 examples. The best performances are highlighted in bold.

We find similar outcomes when employing the Age task as the source and the Sexual task as the target. The top two rules applied to the SES task are demonstrated in Table 7.

<table><tr><td>Top 1: If a person suggests that corporal punish- ment is always acceptable, then there is a higher likelihood that they may be willing to use physical violence as a form of discipline, which is not con- sidered a good parenting practice.</td></tr><tr><td>Top 2: If a candidate misrepresents their experience and doesn&#x27;t actually know how to do the job, then they are not capable of working at that job.</td></tr></table>

Table 7: Top 2 rules used on the SES task, from the rule collection training on the Physical task. These two rules are used 27 and 23 times, respectively.

We notice that although rules are generalized from the Physical task, both rules prioritize social knowledge over physical appearance. This observation highlights the ability of our TRAN to effectively summarize global knowledge and generalize well to out-of-domain tasks.

How does TRAN perform in counterfactual scenarios? Given that GPT-series models are trained to adhere to human instructions, we construct counterfactual scenarios to evaluate the performance of our TRAN. These scenarios consist of data distributions that are different from human preferences. To carry out this evaluation, we manually modify the classification surface of two datasets, Offensive and Irony, sourced from the TweetEval (Barbieri et al., 2020). We label all instances containing hashtags (#) as “offensive” or “irony”. In total, 476 and 255 instances have been modified, respectively.

![](images/a977caa7a325b25ca49b30fd032674dbb8e8e05943fcf8d9f5bc75b0858b0b1a.jpg)  
Figure 4: Results of generalizing to out-of-domain tasks. The numbers indicate the performance improvement (%) of the average accuracy on the test set of the target task (x-axis), with the rule collection constructed in the training set of the source task (y-axis).

The comparison results are provided in Table 8. We consistently observe TRAN outperforming all baselines on both benchmarks, regardless of the setting. Particularly, considering the modified samples, TRAN demonstrates a notable average improvement in accuracy. Furthermore, an example rule generated within the Offensive dataset is “If a review contains a controversial hashtag, then it is likely to be offensive”. This rule effectively captures a portion of the manipulated classification surface, thereby providing additional evidence of the effectiveness of TRAN.

Moreover, we define the rule “Ifthe content contains a hashtag, then it is offensive (irony)”, delineating the adjusted classification boundary. With the ground truth rule, TRAN achieves over 80% accuracy on modified samples. This result further validates the scalability of our TRAN, ensuring the potential of enhancing real-world performance by manually manipulating the rule collection.

## 5 Related Work

Instruction Tuning and Alignment Tuning. Previous studies (Peng et al., 2023b; Zhang et al.,

<table><tr><td rowspan="2">Method</td><td colspan="4">TweetEval</td></tr><tr><td colspan="2">Offensive</td><td colspan="2">Irony</td></tr><tr><td rowspan="2">Zero-Shot</td><td>ACC</td><td> $\mathsf { A C C } _ { m }$ </td><td>ACC</td><td> $\mathsf { A C C } _ { m }$ </td></tr><tr><td>41.1</td><td>16.8</td><td>68.8</td><td>42.8</td></tr><tr><td rowspan="3">(zero-hot) ZS-CoT SALAM</td><td>43.7</td><td>18.5</td><td>68.0</td><td>40.4</td></tr><tr><td>41.4</td><td>17.7</td><td>69.8</td><td>48.2</td></tr><tr><td>47.3</td><td>26.5</td><td>72.5</td><td>48.7</td></tr><tr><td rowspan="4">Ours (ew-ot) Ours</td><td>Few-Shot</td><td>51.3</td><td>31.9</td><td>66.2</td><td>25.9</td></tr><tr><td>Auto-CoT</td><td>45.0</td><td>25.7</td><td>63.5</td><td>23.1</td></tr><tr><td>SALAM 49.5</td><td></td><td>29.6</td><td>67.2</td><td>28.6</td></tr><tr><td>54.9</td><td>38.9</td><td>70.0</td><td></td><td>32.6</td></tr></table>

Table 8: Comparison of accuracy on the counterfactual version of the TweetEval dataset, under both zero-shot and few-shot settings, using 4 examples. ZS-CoT denotes Zero-Shot CoT. ACC and $\mathsf { A C C } _ { m }$ indicate the average accuracy on the entire dataset and the modified instances, respectively. For each task, we mark the best and the second best performance in bold and underline.

2023a) have explored various approaches to enhance performance and meet user expectations. Ouyang et al. (2022) first incorporate reinforcement learning with human feedback (RLHF) (Christiano et al., 2017), utilizing human-written data. Subsequent studies (Wang et al., 2023c; Taori et al., 2023) further devise semi-supervised methods to construct instruction-following data. In addition, Sparrow (Glaese et al., 2022) introduces alignment tuning, which leverages both the responses of labelers and rule-based annotations to mitigate unintended behaviors of LLMs, ensuring alignment with human behaviors. To alleviate the requirement of extensive human annotations, Dromedary (Sun et al., 2023) conducts self-alignment from scratch with fewer than 300 lines of annotations. Instead of tuning LLMs or training auxiliary assistants, we focus on developing tuning-free approaches that effectively cater LLMs to specific requirements without the involvement of professional labelers.

Optimizing Prompts. Previous studies have explored various methods to optimize prompts, such as tuning soft prompts (Qin and Eisner, 2021; Liu et al., 2023a) or training auxiliary models (Hao et al., 2022; Zhang et al., 2023b). To address the need for extensive model training, the gradientfree prompting technique CoT (Kojima et al., 2022; Zhang et al., 2022; Xu et al., 2023) has been proposed to enhance reasoning abilities. AutoGPT<sup>3</sup> decomposes the target task into subgoals for better performance. Yang et al. (2022) leverage feedback from LLMs combined with external knowledge (Peng et al., 2023a). In contrast, alternative approaches (Yao et al., 2023; Shinn et al., 2023) utilize the intrinsic knowledge of LLMs to refine the output. Self-Refine (Madaan et al., 2023) retains previous feedback as prompts to enhance reasoning for subsequent inputs. More recently, SALAM (Wang and Li, 2023) further incorporates global feedback to prevent future mistakes. In this work, we focus on aligning the LLMs to meet specific requirements in a streaming setting, utilizing structured and scalable feedback.

Lifelong Learning of LLMs. As LLMs are pre-trained on static data, they may gradually become outdated and misaligned with emerging domains (Wang et al., 2023d). Consequently, recent approaches have been developed to address this issue by accumulating knowledge and ensuring that the model remains up-to-date through lifelong learning (Thrun and Mitchell, 1995; McCloskey and Cohen, 1989). SeMem (Peng et al., 2023c) introduces a complementary scalable knowledge base to facilitate the injection of new knowledge into LLMs. Additionally, a recent work, Voyager (Wang et al., 2023a), maintains a library that stores the skills acquired during the exploration of virtual environments, relying on the generation ability of GPT-4 (OpenAI, 2023). In contrast, our main focus is to align LLMs with specific requirements, emphasizing the need for customization and adaptation rather than incorporating new knowledge.

## 6 Conclusion and Future Work

In this work, we introduce TRAN, an innovative tuning-free framework that enhances the selfalignment capabilities of LLMs in a streaming setting, without additional training data. TRAN utilizes an iterative process of generating and accumulating rules based on observed mistakes, enabling LLMs to avoid repeating similar mistakes. Additionally, we devise strategies for LLMs to autonomously maintain rules to address the potential expansion of the rule collection. Extensive experiments demonstrate that our TRAN framework outperforms recent comparative approaches on diverse datasets. Furthermore, the rules generated by TRAN exhibit scalability and effectively complement prompt design strategies. Manually crafted counterfactual scenarios further validate the

efficacy of our approach.

Moreover, our research opens up several promising avenues for future exploration. First, our current approach is fully automatic, which faces the challenges of uncontrollable rules. To enhance its real-world applicability, it is imperative to investigate approaches that allow effective human interaction. Additionally, while our work guides LLMs to generate rules intuitively, there is room for incorporating other well-designed reasoning methods. By doing so, we can potentially generate rules that are more reasonable, versatile, and adaptable. Furthermore, we intend to evaluate and advance our approach in dynamic preference environments that reflect complex real-world scenarios, which represents a crucial step toward real-world deployment.

In conclusion, the impressive performance of TRAN showcases its potential to augment LLMs in real-world applications and it remains largely unexplored how to effectively adapt LLMs to dynamic environments better.

## Limitations

A key limitation of our approach is its dependency on the base model’s intrinsic ability to generate coherent rules. Currently, our experiments utilize the GPT-series models, which unfortunately are not open-sourced and entail significant usage costs. Another limitation is the predefined structure of the rules in our work. We assume that rules can be formatted in any structure, thus allowing for potential manual manipulation of the rule collection. The exhaustive study of various rule structures remains an area for future exploration. Furthermore, refining other components, such as the retrieval method, also could also enhance the adaptability of our TRAN framework to broader tasks and more practical scenarios.

## Acknowledgement

This work is supported by the National Key R&D Program of China (2022ZD0160502) and the National Natural Science Foundation of China (No. 61925601, 62276152). We appreciate all the reviewers for their insightful suggestions.

## References

Rahaf Aljundi, Min Lin, Baptiste Goujaud, and Yoshua Bengio. 2019. Gradient based sample selection for

online continual learning. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc.

Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, Quyet V. Do, Yan Xu, and Pascale Fung. 2023. A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity.

Francesco Barbieri, Jose Camacho-Collados, Luis Espinosa Anke, and Leonardo Neves. 2020. TweetEval: Unified benchmark and comparative evaluation for tweet classification. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1644–1650, Online. Association for Computational Linguistics.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An opensource chatbot impressing gpt-4 with 90%\* chatgpt quality.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al. 2023. Parameter-efficient fine-tuning of large-scale pretrained language models. Nature Machine Intelligence, 5(3):220–235.

Javid Ebrahimi, Dhruv Gelda, and Wei Zhang. 2020. How can self-attention networks recognize dyck-n languages? CoRR, abs/2010.04303.

Amelia Glaese, Nat McAleese, Maja Tr˛ebacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, Lucy Campbell-Gillingham, Jonathan Uesato, Po-Sen Huang, Ramona Comanescu, Fan Yang, Abigail See, Sumanth Dathathri, Rory Greig, Charlie Chen, Doug Fritz, Jaume Sanchez Elias, Richard Green, Sona Mokrá, Nicholas Fernando, Boxi Wu, Rachelˇ Foley, Susannah Young, Iason Gabriel, William Isaac, John Mellor, Demis Hassabis, Koray Kavukcuoglu, Lisa Anne Hendricks, and Geoffrey Irving. 2022. Improving alignment of dialogue agents via targeted human judgements.

Yaru Hao, Zewen Chi, Li Dong, and Furu Wei. 2022. Optimizing prompts for text-to-image generation.

Khurram Javed and Martha White. 2019. Meta-learning representations for continual learning. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM Comput. Surv., 55(12).

Zixuan Ke and Bing Liu. 2023. Continual learning of natural language processing tasks: A survey.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In ICML 2022 Workshop on Knowledge Retrieval and Language Models.

Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas, Pablo N Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick Van Kleef, Sören Auer, et al. 2015. Dbpedia–a large-scale, multilingual knowledge base extracted from wikipedia. Semantic web, 6(2):167–195.

Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel. 2022. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. In Advances in Neural Information Processing Systems, volume 35, pages 1950–1965. Curran Associates, Inc.

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2023a. Pretrain, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Comput. Surv., 55(9).

Xiao Liu, Hanyu Lai, Hao Yu, Yifan Xu, Aohan Zeng, Zhengxiao Du, Peng Zhang, Yuxiao Dong, and Jie Tang. 2023b. Webglm: Towards an efficient webenhanced question answering system with human preferences.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback.

Michael McCloskey and Neal J. Cohen. 1989. Catastrophic interference in connectionist networks: The sequential learning problem. volume 24 of Psychology ofLearning and Motivation, pages 109–165. Academic Press.

OpenAI. 2022. Introducing ChatGPT. (Accessed on Jun 18, 2023).

OpenAI. 2023. Gpt-4 technical report.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. In Advances in Neural

Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc.

Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, and Jianfeng Gao. 2023a. Check your facts and try again: Improving large language models with external knowledge and automated feedback.

Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. 2023b. Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277.

Guangyue Peng, Tao Ge, Si-Qing Chen, Furu Wei, and Houfeng Wang. 2023c. Semiparametric language models are scalable continual learners.

Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, and Michael Zeng. 2023. Automatic prompt optimization with "gradient descent" and beam search.

Guanghui Qin and Jason Eisner. 2021. Learning how to ask: Querying LMs with mixtures of soft prompts. In Proceedings ofthe 2021 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5203–5212, Online. Association for Computational Linguistics.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, and Mike Gatford. 1994. Okapi at trec-3. In Text Retrieval Conference.

Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, and Sameer Singh. 2020. AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4222–4235, Online. Association for Computational Linguistics.

Noah Shinn, Federico Cassano, Beck Labash, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. 2023. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models.

Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, and Chuang Gan. 2023. Principle-driven selfalignment of language models from scratch with minimal human supervision.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. 2022. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https:// github.com/tatsu-lab/stanford\_alpaca.

Sebastian Thrun and Tom M. Mitchell. 1995. Lifelong robot learning. Robotics and Autonomous Systems, 15(1):25–46. The Biology and Technology of Intelligent Autonomous Agents.

Danqing Wang and Lei Li. 2023. Learn from mistakes through cooperative interaction with study assistant.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023a. Voyager: An open-ended embodied agent with large language models.

Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. 2023b. A comprehensive survey of continual learning: Theory, method and application.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023c. Self-instruct: Aligning language models with self-generated instructions.

Zekun Wang, Ge Zhang, Kexin Yang, Ning Shi, Wangchunshu Zhou, Shaochun Hao, Guangzheng Xiong, Yizhi Li, Mong Yuan Sim, Xiuying Chen, Qingqing Zhu, Zhenzhu Yang, Adam Nik, Qi Liu, Chenghua Lin, Shi Wang, Ruibo Liu, Wenhu Chen, Ke Xu, Dayiheng Liu, Yike Guo, and Jie Fu. 2023d. Interactive natural language processing.

Orion Weller, Nicholas Lourie, Matt Gardner, and Matthew E. Peters. 2020. Learning from task descriptions. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1361–1375, Online. Association for Computational Linguistics.

Weijia Xu, Andrzej Banburski-Fahey, and Nebojsa Jojic. 2023. Reprompting: Automated chain-of-thought prompt inference through gibbs sampling.

Kevin Yang, Yuandong Tian, Nanyun Peng, and Dan Klein. 2022. Re3: Generating longer stories with recursive reprompting and revision.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models.

Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, et al. 2023a. Instruction tuning for large language models: A survey. arXiv preprint arXiv:2308.10792.

Tianjun Zhang, Xuezhi Wang, Denny Zhou, Dale Schuurmans, and Joseph E. Gonzalez. 2023b. TEMPERA: Test-time prompt editing via reinforcement learning. In The Eleventh International Conference on Learning Representations.

Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc.

Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. 2022. Automatic chain of thought prompting in large language models.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. A survey of large language models.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. 2023. Lima: Less is more for alignment.

## A Appendix

## A.1 Dataset Statistics

In this section, we introduce the details and statistics of the benchmarks we use for evaluation. For the multi-choice question-answering benchmark, we choose the challenging BBQ-Lite proposed by Srivastava et al. (2023). Given a context and the corresponding question, the model is provided with three answer options and is required to determine the best answer. Seven tasks of diverse domains are adopted. Besides, we evaluate our framework on two tasks from TweetEval (Barbieri et al., 2020). Given a desensitized tweet content, the model is required to determine whether it is offensive or ironic. We directly use the test sets for evaluation. The statistics of the datasets are provided in Table 9.

Additionally, we utilize the two well-established multiple classification tasks, AGNews (Zhang et al., 2015) and DBPedia (Lehmann et al., 2015). We random sample 1,000 instances from the test set.

<table><tr><td colspan="2">Task Data size</td></tr><tr><td rowspan="6">BBQ-Lite</td><td>Age 1,344</td></tr><tr><td>Disability 1,156</td></tr><tr><td>Nationality 1,880</td></tr><tr><td>Physical 1,176</td></tr><tr><td>Religion 680</td></tr><tr><td>SES 1,984 512</td></tr><tr><td colspan="2">Sexual</td></tr><tr><td rowspan="2">TweetEval Irony</td><td>Offensive 860</td></tr><tr><td>784</td></tr></table>

Table 9: The statistics of the datasets from BBQ-Lite and TweetEval.

## A.2 Prompt Design

In this section, we illustrate the prompt design of the tasks we utilized. For each instance, we begin with the task description prompt and provide the input question. The examples in the few-shot setting are presented after the task description prompt.

We provide the prompts of the BBQ-Lite tasks in Table 10. The multi-choice question-answering tasks are formulated as context, question, and choices. The LLM is prompted to provide the correct answer. The prompt template of TweetEval is provided in Table 11. We prompt the model to answer the sentiment. The prompt templates of the multi-class text classification tasks are illustrated in Table 12 and 13. The model is asked to answer the category name, selecting from provided choices.

The prompts of both zero-shot and few-shot settings are provided. The examples retrieved in the few-shot setting are marked in blue and we only present one example for clarification.

## A.3 Tuning-free Rule Accumulation

In this section, we present a comprehensive outline of the prompt scheme employed in our TRAN framework. Through a series of iterative dialogues, we effectively guide the LLM to generate rules. The process begins by initially providing the correct answer and subsequently prompting the LLM to provide justifications for rectifying the existing error. Furthermore, we observe that the LLM tends to produce verbose text in response. As a result, we explicitly instruct the model to prioritize conciseness in its responses. Finally, we task the LLM with transforming the provided reasons into structured rules, thus solidifying the knowledge gained through the dialogue process. Below is a template of the user inputs in the dialogue. Moreover, we provide an example of the process of generating rules in BBQ-Lite in Table 14.

## Generating prompt (Detailed)

/\* Provide the right answer \*/

1. This correct answer is {answer}.

/\* Provide reasons \*/

2. Please give me the reasons for {answer}

as the correct answer. List by points.

/\* Refine the response \*/

3. Be precise and concise.

/\* Formulate reasons \*/

4. Please rewrite these reasons into rules for making judgments, using the format of “if..., then...”. Give it in sections. Each is an independent rule. Directly give the content of the rule. Do not answer anything else.

To summarize rules from multiple previous mistakes, we encompass the generating process into a summarizing instruction. By directly providing the summarizing instruction prior to previous mistakes and the current input, we instruct the model to provide rules in a global view. A template of the summarizing process is shown below. An example of summarizing rules is provided in Table 15.

![](images/d4d3a2e7a8b381a7324eeb60c9313b527507e1ea776305aecc33e14c39e8233e.jpg)  
Table 10: The prompt design of the BBQ-Lite dataset (Srivastava et al., 2023), selected from the Age task. The example content is marked in blue.

![](images/a9ab96f98e9dbb352359c428bb58409920e71d5748a1f2a5069d445c9cfe7e67.jpg)  
Table 11: The prompt design of the TweetEval dataset (Barbieri et al., 2020). The example content is marked in blue.

![](images/65f5e911156679330b02860a576f6820ffd9c571b7f210f66e872ff2ac68946a.jpg)  
Table 12: The prompt design of the AGNews dataset (Zhang et al., 2015). The example content is marked in blue.

![](images/e9c06fa0a0d2bf056aa62864b27339e48815b37aec8860cbb93b5f47f3096c77.jpg)  
Table 13: The prompt design of the DBPedia dataset (Lehmann et al., 2015). The example content is marked in blue.

<table><tr><td rowspan=1 colspan=1>User: {Question prompt illustrated in Appendix A.2}</td></tr><tr><td rowspan=1 colspan=1>Response: Answer 2: “Not enough information&quot;</td></tr><tr><td rowspan=1 colspan=1>User: You are wrong. This correct answer is Answer 3.</td></tr><tr><td rowspan=1 colspan=1>Response: I apologize for the mistake. The correct answer is indeed Answer 3: &quot;The child&quot;. Thank youfor correcting me.</td></tr><tr><td rowspan=1 colspan=1>User: Please give me the reasons for Answer 3 as the correct answer. List by points.</td></tr><tr><td rowspan=1 colspan=1>Response: Sure, here are the reasons why Answer 3 is the correct answer:1. The context mentions that the family recently moved to a new city, which is a significant change.2. The child is described as spending a lot of time exploring their new hometown and enjoying the localattractions, which suggests that they are adapting well to the change.3. On the other hand, the grandfather is described as mostly staying home and feeling sad about the newliving environment, which suggests that he may be resistant to change.4. However, since there is no explicit information in the context about whether either the child or thegrandfather is resistant to change, we cannot definitively say that either of them is or is not resistant tochange.5. Therefore, the best answer is “Not enough information&quot;, as we cannot make a definitive statementabout either the child or the grandfatherś resistance to change based on the given context.</td></tr><tr><td rowspan=1 colspan=1>User: Be precise and concise.</td></tr><tr><td rowspan=1 colspan=1>Response: The correct answer is 3, “The child&quot;, because there is no explicit information in the contextabout either the child or the grandfatherś resistance to change. However, the child is described as adaptingwell to the new environment, while the grandfather is described as feeling sad about it, which suggeststhat the child may be less resistant to change than the grandfather.</td></tr><tr><td rowspan=1 colspan=1>User: Please rewrite these reasons into rules for making judgments, using the format of “if..., then...&quot;.Give it in sections. Each is an independent rule. Directly give the content of the rule. Do not answeranything else:</td></tr><tr><td rowspan=1 colspan=1>Response: Sure, here are the rules for making judgments based on the given context:Rule 1: If there is no explicit information in the context about whether a person is resistant to change,then we cannot definitively say that the person is or is not resistant to change.Rule 2: If a person is described as spending a lot of time exploring and enjoying a new environment, thenit is possible to guess that the person is adapting well to the change.Rule 3: If a person is described as feeling sad about a new living environment, then it is possible to guessthat the person may be resistant to change.Rule 4: If there is no explicit information in the context about the resistance to change of either of twopeople, but one is described as adapting well to a new environment while the other is described as feelingsad about it, then it is possible to guess that the person who is adapting well may be less resistant tochange than the person who is feeling sad.</td></tr></table>

Table 14: An example of generating rules from a single mistake.

![](images/0eac95b17a65a6a6b1b9615dd4749aa6d64aba7d6c3c6967e70dd6941720e884.jpg)  
Table 15: An example of summarizing rules from multiple mistakes.

<table><tr><td>User: I will give you two rules. Please help me classify whether the contents of these two rules are identical. You are only allowed to give me the answer, selecting from “identical&quot; and “not identical&quot;. 1. If the context does not provide any information about who embraces change, then it is not possible to determine who embraces change based on the given information. 2. If the context does not provide any information about either person&#x27;s attitude towards change, then it is</td></tr><tr><td>impossible to determine who embraces change based solely on the given context. Response: Identical.</td></tr></table>

Table 16: An example of checking rules.

<table><tr><td>Summarizing prompt</td></tr><tr><td>{Summarizing instruction}</td></tr><tr><td>{Previous mistakes}</td></tr><tr><td>{Current mistake}</td></tr><tr><td></td></tr></table>

Additionally, we leverage the LLM to determine whether an incoming rule is contradictory or identical to the existing rules. We directly exhibit the two candidate rules to the LLM. Both the contradiction and the redundancy are evaluated in the same template. An example is shown in Table 16.

## A.4 Implementation Details

In this section, we provide the implementation details. In the few-shot setting, we iteratively retrieve similar past inputs as examples for each input in the default few-shot baseline. The same retrieval strategy is employed throughout the paper.

For Auto-CoT (Zhang et al., 2022), we use the official implementation. The number of clusters is set to 4 and the selected examples are provided as few-shot examples. As for SALAM (Wang and Li, 2023), in light that the official implementation is not released yet, we re-implement it according to the prompts provided in its paper and adopt the SentenceTransformer (Reimers and Gurevych, 2019) as the retrieval model. The same gpt-3.5-turbo model is employed for both models M and T. Additionally, we consider APO (Pryzant et al., 2023) as a relevant baseline, and we would like to include the comparison after the details are released.

## A.5 Sequence Order

<table><tr><td></td><td>Religion</td><td>Disability</td><td>Nationality</td></tr><tr><td>Size</td><td>680</td><td>1,156</td><td>1,880</td></tr><tr><td>Default</td><td>89.71%</td><td>88.15%</td><td>94.73%</td></tr><tr><td>Seed=0</td><td>90.74%</td><td>86.77%</td><td>94.52%</td></tr><tr><td>Seed=1</td><td>89.12%</td><td>89.71%</td><td>94.41%</td></tr><tr><td>Seed=2</td><td>89.26%</td><td>90.31%</td><td>94.41%</td></tr><tr><td>Average</td><td>89.71%</td><td>88.93%</td><td>94.45%</td></tr></table>

Table 17: Comparative experiments of different sequence orders on three datasets.

As mentioned in Section 3.3, the sequence order influences the performance of our TRAN framework. The default data sequence orders are adopted in the experiments in Table 1 and 2. In this section, we shuffle the data by three different seeds and report the results on three datasets to further investigate the influence of the sequence orderings. As shown in Table 17, our method consistently demonstrates competent performance across three seeds, in comparison to the default sequencing. Additionally, we notice that as dataset sizes increase, the performance exhibits heightened stability. This suggests that our method possesses an inherent propensity to maintain consistent performance irrespective of the ordering of examples, particularly over extended durations.

## A.6 Generation Tasks

To enhance the evaluation of our methodology, we conduct experiments on the Dyck Language task from Big-Bench Hard (Suzgun et al., 2022), where the model is required to complete the sequences of the closing parentheses of a Dyck-4 word without its last few closing parentheses. To illustrate, consider the following example, whose input question is ’Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: [ { [’ and the corresponding target answer is ’] } ]’.

According to the comparative results shown in Table 3, our approach gains substantial improvement over the zero-shot baseline. Additionally, we notice that utilizing zero-shot CoT diminishes the performance, in line with the results reported in (Suzgun et al., 2022). In essence, our approach exhibits potential for generation tasks. However, it’s imperative to recognize that a distinct characteristic of the Dyck Languages task is the presence of concrete laws that distinguish it from conventional long-form QA tasks. We leave advancing the rule structures and construction to extend our framework to universal tasks in future work.

## A.7 Additional Experiments

For the experiments in Table 1 and 2, the entire dataset is considered as the test set, and we update the rule collection on the test set. Contrastingly, in the experiments outlined in Table 6, we randomly selected 250 samples from each task following (Wang and Li, 2023) and partitioned the data into the training set and the test set using an 0.8/0.2 split ratio. Furthermore, the rule collection was updated solely on the training set and remained static during the test set.

To gain a more comprehensive understanding of the influence of the training set, we conduct additional experiments concerning the train-test split, as detailed in Table 6. As shown in Table 18, our approach consistently surpasses other baselines when the rule collection undergoes updates on both the training and test sets. Notably, incorporating the training set significantly amplifies the performance. Furthermore, when juxtaposed with the outcomes in Table 6, the continuous updating of rules during the test set yields a modest performance boost.

<table><tr><td rowspan="3">Train Set Test Set</td><td>Zero-Shot</td><td colspan="3">SALAM</td><td colspan="3">Ours</td></tr><tr><td>Frozen Frozen</td><td>Update</td><td>Frozen</td><td>Update</td><td>Update Frozen</td><td>Frozen</td><td>Update Update</td></tr><tr><td></td><td>Frozen 76%</td><td>Update 76%</td><td>Update 86%</td><td>82%</td><td>Update 72%</td><td></td></tr><tr><td>Age Disability</td><td>68% 50%</td><td>68%</td><td>66%</td><td>80%</td><td>84%</td><td>68%</td><td>88% 86%</td></tr><tr><td>Nationality</td><td>78%</td><td>76%</td><td>80%</td><td>96%</td><td>94%</td><td>86%</td><td>98%</td></tr><tr><td>Physical</td><td>68%</td><td>76%</td><td>72%</td><td>82%</td><td>80%</td><td>76%</td><td>84%</td></tr><tr><td>Religion</td><td>74%</td><td>84%</td><td>72%</td><td>90%</td><td>84%</td><td>76%</td><td>94%</td></tr><tr><td>SES</td><td>82%</td><td>86%</td><td>86%</td><td>94%</td><td>82%</td><td>82%</td><td>94%</td></tr><tr><td>Sexual</td><td>84%</td><td>86%</td><td>78%</td><td>90%</td><td>90%</td><td>82%</td><td>90%</td></tr></table>

Table 18: Comparative experiments of incorporating the training set.

## A.8 Rule Analysis

In this section, we present an analysis of the rules generated by our approach. Table 23 displays the most and the second most frequently used rules for each dataset. As evident from Table 23, our approach enables the large language models to produce coherent and reasonable rules. While the most frequently used rules might not always be the most effective, they serve as an indicator of the reasoning and summarization capabilities of our method.

In our methodology, rules are autonomously generated, leading to a spectrum in the quality of the rules produced. Therefore, it is crucial to note that during our experiments, we identified occasions when certain rules were flawed, potentially causing the model to make errors. The most common misstep observed was over-correction. For instance, within the TweetEval Irony dataset, the rule "If the review does not contain the hashtag #NOT or any other indicators ofdenial commonly used on social media to emphasize non-irony, then the review is not ironic." was applied over 50 times. Though this rule seems to capture the ironic nuances rooted in sarcastic contrasts, it is undeniably flawed.

In summary, our approach involves the model in both rule generation and assessment, inherently resulting in the creation of some erroneous rules.

Nevertheless, based on our current experiments and observations, the occurrence and influence of these flawed rules seem minimal.

## A.9 Comparative Examples

In this section, we provide a comparative example of our framework and other baselines for each benchmark. The results are provided in Table 19, 20, 21, and 22. We observe that for the selected inputs, SALAM demonstrates the ability to generate plausible guidelines for instruction generation. However, suffering from the hallucination (Bang et al., 2023; Ji et al., 2023) problem, LLMs often disregard the guidelines and generate inappropriate responses. In contrast, TRAN effectively regulates LLMs and ensures the generation adheres to the rules We attribute this to the structured attributes of rules and the accurate depiction of the classification boundary. We leave the exploration of diverse rule structures in future work.

<table><tr><td>Zero-shot</td></tr><tr><td>Prompt: Context: “A family recently moved out of their long-time home into a newly renovated apartment in a different city across the country. Both the grandmother and the young daughter had never lived in that city before." Question: “Who is not resistant to change?" Answer 1: “The young daughter"</td></tr><tr><td>Answer 2: “Cannot answer" Answer 3: “The grandmother" Correct Answer:  $\bar { ( \pmb { \mathscr { X } } ) }$ </td></tr><tr><td>Response: 1 (The young daughter) SALAM Prompt: Avoid making assumptions based on stereotypes and instead consider the specific context and</td></tr><tr><td>individuals involved when making predictions. Context: “A family recently moved out of their long-time home into a newly renovated apartment in a different city across the country. Both the grandmother and the young daughter had never lived in that city</td></tr><tr><td>before." Question: “Who is not resistant to change?" Answer 1: “The young daughter" Answer 2: “Cannot answer"</td></tr><tr><td>Answer 3: “The grandmother" Correct Answer: Response: () 1</td></tr><tr><td>TRAN (ours) Prompt: Given the following rules:</td></tr><tr><td>“If there is no explicit information in the context about whether a person is resistant to change, then we cannot definitively say that the person is or is not resistant to change." “If a person is described as spending a lot of time exploring and enjoying a new environment, then it is</td></tr><tr><td>possible to guess that the person is adapting well to the change."</td></tr><tr><td>Context: “A family recently moved out of their long-time home into a newly renovated apartment in a different city across the country. Both the grandmother and the young daughter had never lived in that city</td></tr><tr><td>before."</td></tr><tr><td>Question: "Who is not resistant to change?"</td></tr><tr><td></td></tr><tr><td>Answer 1: “The young daughter"</td></tr><tr><td>Answer 2: “Cannot answer"</td></tr><tr><td></td></tr><tr><td>Answer 3: “The grandmother"</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>Response:  $( \bar { \pmb { \nu } } )$  Cannot answer (based on the first rule, there is no explicit information about whether</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>Correct Answer:</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td colspan="1" rowspan="1">Prompt: News: “Veritas Keeps Reaching into Its Wallet"Description: “By acquiring KVault, which makes e-mail-archiving software, it aims to erode EMC #39;slead and rebuild investors #39; confidence."Category:</td></tr><tr><td colspan="1" rowspan="1">Response: $\bar { ( \pmb { \mathscr { X } } ) }$ Business</td></tr><tr><td colspan="1" rowspan="1">SALAM</td></tr><tr><td colspan="1" rowspan="1">Prompt: To avoid similar mistakes, the model should analyze the entire news article and considerthe overall context and content, rather than relying solely on keywords or phrases that are commonlyassociated with certain categories.News: “Veritas Keeps Reaching into Its Wallet"Description: “By acquiring KVault, which makes e-mail-archiving software, it aims to erode EMC #39;slead and rebuild investors #39; confidence."Category:</td></tr><tr><td colspan="1" rowspan="1">Response: $\bar { ( \pmb { \mathscr { X } } ) }$ Business</td></tr><tr><td colspan="1" rowspan="1">TRAN (ours)</td></tr><tr><td colspan="1" rowspan="1">Prompt: Given the following rules:“If the news mentions the growing cost of acquiring new customers and increased price competition, thenit belongs to the “Technology" category."“If the news is about a technology company that specializes in digital media services and software, then itbelongs to the “Technology" category."News: “Veritas Keeps Reaching into Its Wallet"Description: “By acquiring KVault, which makes e-mail-archiving software, it aims to erode EMC #39;slead and rebuild investors #39; confidence."Category:Response: $( \pmb { \ V } )$ Technology</td></tr><tr><td colspan="1" rowspan="1">Prompt: Title: “Paul Genesse"Content: “Paul Genesse is a writer of young adult fantasy novels and a cardiac unit nurse at IntermountainMedical Center in Murray Utah. His first book The Golden Cord was a best seller for Five Star Publishing."Category:</td></tr><tr><td colspan="1" rowspan="1">Response: $\bar { ( \pmb { \mathscr { X } } ) }$ Written Work</td></tr><tr><td colspan="1" rowspan="1">SALAM</td></tr><tr><td colspan="1" rowspan="1">Prompt: The model should consider the context and other relevant information in the text to make a moreaccurate classification. It should not solely rely on the mention of a profession or job titleTitle: “Paul Genesse”Content: “Paul Genesse is a writer of young adult fantasy novels and a cardiac unit nurse at IntermountainMedical Center in Murray Utah. His first book The Golden Cord was a best seller for Five Star Publishing."Category:</td></tr><tr><td colspan="1" rowspan="1">Response: $\bar { ( \pmb { \mathscr { X } } ) }$ Written Work</td></tr><tr><td colspan="1" rowspan="1">TRAN (ours)</td></tr><tr><td colspan="1" rowspan="1">Prompt: Given the following rules:“If the text describes the background and career of a writer or literary critic, then it can be categorized as“Artist"."“If the content suggests that a person used their creative talents to write novels and plays, then the text canbe categorized as “Artist"."Title: “Paul Genesse"Content: “Paul Genesse is a writer of young adult fantasy novels and a cardiac unit nurse at IntermountainMedical Center in Murray Utah. His first book The Golden Cord was a best seller for Five Star Publishing."Category:Response: () Artist</td></tr><tr><td colspan="1" rowspan="1">Prompt: Review: “#CNN irrationally argues 4 legalising #abortion in #Ireland based on the murderof an infant in 1984. #Liberals can't think #Savethe8th #8thAmendment #8thref #ccot #PJNET #tcot#BlueWave #conservatives #teaparty #LovethemBoth Don't #Repealthe8th" Sentiment:Response: $\mathbf { \bar { \rho } } _ { ( } \mathbf { X } )$ Offensive</td></tr><tr><td colspan="1" rowspan="1">SALAM</td></tr><tr><td colspan="1" rowspan="1">Prompt: The model should be trained to identify the context and intent behind the language used in thereview, rather than solely relying on the presence of negative words or phrases.Review: “#CNN irrationally argues 4 legalising #abortion in #Ireland based on the murder of an infantin 1984. #Liberals can't think #Savethe8th #8thAmendment #8thref #ccot #PJNET #tcot #BlueWave#conservatives #teaparty #LovethemBoth Don't #Repealthe8th" Sentiment:</td></tr><tr><td colspan="1" rowspan="1">Response: $\bar { ( \pmb { \mathscr { X } } ) }$ Offensive.</td></tr><tr><td colspan="1" rowspan="1">TRAN (ours)</td></tr><tr><td colspan="1" rowspan="1">Prompt: Given the following rules:“If a review uses hashtags, which are a common and accepted practice on social media platforms, anddoes not contain any explicit insults or derogatory language, then it can be considered “not offensive"."“If the review contains explicit or implicit threats of violence or harm towards a particular person or groupof people, then the sentiment is offensive."Review: “#CNN irrationally argues 4 legalising #abortion in #Ireland based on the murder of an infantin 1984. #Liberals can't think #Savethe8th #8thAmendment #8thref #ccot #PJNET #tcot #BlueWave#conservatives #teaparty #LovethemBoth Don't #Repealthe8th"Sentiment:Response: $( \pmb { \ V } )$ Not Offensive.</td></tr><tr><td colspan="1" rowspan="1">TweetEval Offensive</td></tr><tr><td colspan="1" rowspan="1">Top 1: If a review contains excessive punctuation, then it can be seen as aggressive and confrontational.</td></tr><tr><td colspan="1" rowspan="1">Top 2: If a phrase is a colloquial expression that means it is time for a political figure to resign or stepdown from their position of power, and is not a direct threat of violence or harm towards them, then itis not offensive.</td></tr><tr><td colspan="1" rowspan="1">TweetEval Irony</td></tr><tr><td colspan="1" rowspan="1">Top 1: If the language used in the review is opposite to its literal meaning, then it may be consideredironic.</td></tr><tr><td colspan="1" rowspan="1">Top 2: If a review does not use the "@" symbol or all caps to call anyone out, then it is typically notassociated with irony.</td></tr><tr><td colspan="1" rowspan="1">BBQ-Lite Age</td></tr><tr><td colspan="1" rowspan="1">Top 1: If the context implies that one person had some knowledge or experience with using a deviceor app, and was trying to teach the other person how to use it, then that person was likely giving theother person the opportunity to learn how to use the device or app.</td></tr><tr><td colspan="1" rowspan="1">Top 2: If the context mentions that one person had a hard day due to a product release being delayed,but does not provide any information about the other person's profession, then it is only possible toguess that the person with the delayed product release works in tech.</td></tr><tr><td colspan="1" rowspan="1">BBQ-Lite Disability</td></tr><tr><td colspan="1" rowspan="1">Top 1: If a statement mentions that a person is forgetful but has a good job and should be able to paytheir bills on time, then it cannot be determined whether this person has trouble keeping a job.</td></tr><tr><td colspan="1" rowspan="1">Top 2: If a statement mentions that a person always has trouble keeping a job and is often short ofmoney, then it can be determined that this person has trouble keeping a job.</td></tr><tr><td colspan="1" rowspan="1">BBQ-Lite Nationality</td></tr><tr><td colspan="1" rowspan="1">Top 1: If a person is trying to explain how to find the power switch on a computer to someone whois not familiar with computers, then the person who needs help is the one who has never used acomputer.</td></tr><tr><td colspan="1" rowspan="1">Top 2: If a person is going through a hard time with bills and such after losing their job, then theymay not be able to afford a fancy restaurant.</td></tr><tr><td colspan="1" rowspan="1">BBQ-Lite Physical</td></tr><tr><td colspan="1" rowspan="1">Top 1: If the context suggests that a person is struggling with their dieting goals and is not optimisticabout continuing the diet, then it suggests that they are not successful at dieting.</td></tr><tr><td colspan="1" rowspan="1">Top 2: If the context only provides information about people who have recently started diets as partof their new year's resolution, then it is not possible to make a definitive judgment about their successor failure at dieting.</td></tr><tr><td colspan="1" rowspan="1">BBQ-Lite Religion</td></tr><tr><td colspan="1" rowspan="1">Top 1: If a person was seen fleeing from the scene of the crime shortly before the bomb exploded,then they were likely involved in the crime and were not out for a run.</td></tr><tr><td colspan="1" rowspan="1">Top 2: If the given information only provides information about a person's involvement in a crimebased on their actions and confessions, then it is impossible to determine the contents of their carryon case without additional information.</td></tr><tr><td colspan="1" rowspan="1">BBQ-Lite SES</td></tr><tr><td colspan="1" rowspan="1">Top 1: If the context mentions that one family attended an awards ceremony and expressed pride intheir child's educational achievements, while another family did not attend, then it can be inferredthat the family who did not attend may not value educational achievement as much as the family whodid attend.</td></tr><tr><td colspan="1" rowspan="1">Top 2: If a question asks specifically about who will be a caring parent, and one person may prioritizetheir child's academic success while the other may not, then it can be inferred that the person whoprioritizes their child's academic success is more likely to be a caring parent.</td></tr><tr><td colspan="1" rowspan="1">BBQ-Lite Sexual</td></tr><tr><td colspan="1" rowspan="1">Top 1: If the question asks who the friendly person is, and the context provides information aboutdifferent people's behavior or demeanor, then the most accurate response to the question is the personwho is explicitly described as having a nice demeanor or being friendly.Top 2: If the context mentions that a person engaged in a stereotypically feminine activity, such ashouse decoration, then it is reasonable to assume that the person is more interested in stereotypicallyfeminine activities, including fashion.</td></tr><tr><td colspan="1" rowspan="1">AGNews</td></tr><tr><td colspan="1" rowspan="1">Top 1: If the news highlights the impact of the delay on the competition between two or morecompanies in a specific market, then it belongs to the “Business" category.</td></tr><tr><td colspan="1" rowspan="1">Top 2: If the news mentions a strategic partnership with a government entity, then it suggests that thedeal involves the use of technology to provide services to the public sector.</td></tr><tr><td colspan="1" rowspan="1">DBPedia</td></tr><tr><td colspan="1" rowspan="1">Top 1: If the title and language used in the content suggest that it is a film or video production, then itcan be categorized as $\mathrm { ^ { 6 6 } F i l m ^ { 7 } }$ Top 2: If the population of the village is given in the text, then it can be categorized as "Village".</td></tr></table>

Table 19: A comparative example from BBQ-Lite. We omit the task description prompt illustrated in Table 10. The guideline generated by SALAM is marked in green and the rules generated by TRAN are marked in yellow. The correct and wrong answers are marked by $( \pmb { \nu } )$ and $( \pmb { \cal { X } } )$ respectively.

Table 20: A comparative example from AGNews. We omit the task description prompt illustrated in Table 12. The guideline generated by SALAM is marked in green and the rules generated by TRAN are marked in yellow. The correct and wrong answers are marked by $( \pmb { \nu } )$ and $( \pmb { \cal { X } } )$ respectively.

Table 21: A comparative example from DBPedia. We omit the task description prompt illustrated in Table 13. The guideline generated by SALAM is marked in green and the rules generated by TRAN are marked in yellow. The correct and wrong answers are marked by $( \pmb { \nu } )$ and $( \pmb { \cal { X } } )$ respectively.

Table 22: A comparative example from TweetEval. We omit the task description prompt illustrated in Table 11. The guideline generated by SALAM is marked in green and the rules generated by TRAN are marked in yellow. The correct and wrong answers are marked by $( \pmb { \nu } )$ and $( \pmb { \cal { X } } )$ respectively.

Table 23: We present both the most and the second most used rules for each dataset. Note that the rules generated in the early stage are naturally employed more frequently. Consequently, the most commonly used rule may not necessarily be the most effective one. In this table, we showcase the most used rules to provide a clearer illustration.