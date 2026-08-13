# Can Language Models Laugh at YouTube Short-form Videos?

Dayoon Ko<sup>1</sup> Sangho Lee<sup>2</sup> Gunhee Kim<sup>1</sup>

<sup>1</sup>Seoul National University <sup>2</sup>Allen Institute for Artificial Intelligence dayoon.ko@vision.snu.ac.kr sanghol@allenai.org gunhee.kim@snu.ac.kr https://github.com/dayoon-ko/ExFunTube

## Abstract

As short-form funny videos on social networks are gaining popularity, it becomes demanding for AI models to understand them for better communication with humans. Unfortunately, previous video humor datasets target specific domains such as speeches or sitcoms, and mostly focus on verbal cues. We curate a usergenerated dataset of 10K multimodal funny videos from YouTube, called ExFunTube. Using a video filtering pipeline with GPT-3.5, we verify both verbal and visual elements contributing to humor. After filtering, we annotate each video with timestamps and text explanations for funny moments. Our ExFunTube is unique over existing datasets in that our videos cover a wide range of domains with various types of humor that necessitate a multimodal understanding of the content. Also, we develop a zero-shot video-to-text prompting to maximize video humor understanding of large language models (LLMs). With three diferent evaluation methods using automatic scores, rationale qual ity experiments, and human evaluations, we show that our prompting significantly improves LLMs’ ability for humor explanation.

## 1 Introduction

Today, a huge number of short-form funny videos are popularly circulated on social media platforms. Although humor often triggers instant laughter, understanding humor is not a straightforward process. Numerous studies (Hazlitt, 1845; Kant, 1786; Nerhardt, 1970; Jones, 1970; Shultz, 1972; Suls, 1972, 1983) have explored the cognitive process of humor appreciation. For instance, Hazlitt (1845) and Kant (1786) propose the incongruity theory, asserting that incongruity provokes laughter. Nerhardt (1970) further develops the idea by defining the discrepancy between expectation and content, such as punchlines or cartoons. Suls (1972) suggests the incongruity-resolution theory, positing that humor arises only when the incongruity is resolved by retrieving information from the joke, cartoon, or the perceiver’s own knowledge. Since a suficient understanding of the context is required to perceive and further resolve the incongruity, understanding humor can be challenging. Nevertheless, if AI models can understand humor, they could interact more efectively with humans by providing empathetic responses based on users’ sense of humor. Furthermore, if the models understand short-form funny videos, they can recommend videos based on users’ preferences or even generate witty titles based on video contexts.

![](images/86aafac24fc332d2a7f634dc0e0725841cbeb7b6060443d72bc642bfcb82d2fb.jpg)  
Figure 1: An example from the ExFunTube dataset. We curate funny short-form videos in various domains through a filtering pipeline that verifies both verbal and visual elements contributing to humor. Each video is annotated with timestamps and explanations for funny moments. In this example, three funny moments are identified.

Several studies (Hasan et al., 2019; Castro et al., 2019; Patro et al., 2021; Kumar et al., 2022) have collected humorous video datasets to investigate whether models can understand if a video is funny or not. However, the datasets have been gathered from a limited domain, such as speeches or sitcoms. For example, Hasan et al. (2019) collect videos from TED, where there is a single speaker, and visual cues are restricted to gestures or facial expressions. Castro et al. (2019) build the MUStARD dataset from four sitcoms, mainly from "Friends" and "Big Bang Theory," and Patro et al. (2021) collect the MHD dataset from the sitcom "Big Bang Theory." However, in sitcoms, the fixed actors follow a predetermined script on a constructed set, and the punchline plays a crucial role, so the visual elements may have less contribution to humor. Moreover, the aforementioned video datasets only have binary labels indicating whether the content is humorous or not. As binary classification may not evaluate whether a model truly understands the humor in a video, Kumar et al. (2022) collect WITS with annotated text explanations. However, this dataset is limited to sarcasm, a specific form of humor, and focuses on sarcasm explanation in dialogue. It highlights a need for a humor explanation dataset that considers visual elements more and covers general humor.

<table><tr><td>Dataset</td><td>Modality</td><td>Type</td><td>#Data Points</td><td>Data Config</td><td>Exp</td><td>Task</td></tr><tr><td>ExPUN</td><td>T</td><td>Pun</td><td>2K</td><td>{Pun, Keywords, Up to 5 scores &amp; explanations}</td><td>√</td><td>Pun Exp</td></tr><tr><td>AVH / FOR</td><td>I</td><td>Abstract Scene</td><td>3K/ 15K</td><td>{A funny image, An unfunny image, 10 funniness ratings} / {A counterpart (object replaced) image}</td><td></td><td>Image Humor Scoring &amp; Altering</td></tr><tr><td>NYCC</td><td>I,T</td><td>Cartoon</td><td>0.7K</td><td>{Cartoon, Three finalist captions, 3 annotations of locations, descriptions, uncanny descriptions, relevant entities, and explanations}</td><td>7</td><td>Cartoon Caption Exp</td></tr><tr><td>MORE</td><td>I,T</td><td>Posts</td><td>3K</td><td>{Image, Caption, 1 explanation}</td><td>√</td><td>Image Sarcasm Exp</td></tr><tr><td>MUStARD</td><td>V,A,T</td><td>Sitcom</td><td>6K</td><td>{Video, Binary (funny/unfunny) label}</td><td></td><td>Video Sarcasm BC</td></tr><tr><td>WITS</td><td>V,A,T</td><td>Sitcom</td><td>2.2K</td><td>{Video, One Explanation}</td><td></td><td>Dialogue Sarcasm Exp</td></tr><tr><td>UR-FUNNY</td><td>V,A,T</td><td>Speech</td><td>8K</td><td>{Video, Binary (funny/unfunny) label}</td><td></td><td>Video Humor BC</td></tr><tr><td>MHD</td><td>V,T</td><td>Sitcom</td><td>11K</td><td>{Video, Binary (funny/unfunny) label}</td><td></td><td>Video Humor BC</td></tr><tr><td>ExFunTube</td><td>V,A,T</td><td>Short-form Youtube videos</td><td>10K</td><td>{Video, Up to 3 timestamps &amp; explanations}</td><td>√</td><td>Video Humor Exp</td></tr></table>

Table 1: Comparison of our ExFunTube with previous humor datasets: ExPUN (Sun et al., 2022), AVH&FOR (Chandrasekaran et al., 2016), NYCC (Hessel et al., 2022), MORE (Desai et al., 2022), MUStARD (Castro et al., 2019), WITS (Kumar et al., 2022), UR-FUNNY (Hasan et al., 2019), and MHD (Patro et al., 2021) . In the Modality column, I, V, A, and T denote image, video, audio, and text, respectively. The #Data Points column shows only the number of positive (humorous) data points. The Data Config column specifies the composition of each data point. The Exp column indicates the presence of annotated explanations. In the Task column, Exp and BC are abbreviations of explanation generation and binary classification task each.

To this end, we curate ExFunTube, a dataset of funny, short-form videos with explanations. These videos are collected from user-generated YouTube videos, which are shared on the "r/youtubehaiku" subreddit. In this subreddit, users upload shortform funny videos, typically up to 30 seconds long. We develop a video filtering pipeline with GPT-3.5 (Ouyang et al., 2022), designed to exclude the videos with minimal visual impact on humor. Then, we annotate the collected videos with timestamps and text explanations of funny moments, as exemplified in Figure 1.

Recent LLMs show great performance for explaining humor present in text to some extent (Chowdhery et al., 2022). Inspired by the recent research on multimodal-informed prompting (Zeng et al., 2022), we convert video content into text, leveraging various zero-shot models on diverse modalities of the video. We provide LLMs with the text prompt as a linguistic summary of video content. Specifically, we consider two modalities of the video content: visual and audio. From the visual modality, we obtain dense video descriptions. From the audio modality, we acquire speech transcripts and sound labels. Finally, we chronologically integrate them into a text prompt that can maximize LLMs’ ability for humor explanation.

Since evaluating a model’s ability to explain humor is challenging, we report our results in three diferent ways: model-based automatic scores, rationale quality metrics with the moment localization task, and human evaluation. First, we report modelbased metrics instead of those using word overlap. Second, we conduct a rationale quality experiment, which assesses the quality of explanations from the accuracy of predicting gold labels (Wiegrefe et al., 2021). Finally, we carry out human evaluations with sampled test examples. Through these three diferent results, our prompting approach considerably improves the humor explanation performance of three important LLMs, including one zero-shot GPT-3.5 and two finetuned T5 (Rafel et al., 2020) and BART (Lewis et al., 2020).

To summarize, our key contributions are:

1. We curate ExFunTube, a dataset consisting of 10,136 user-generated, funny short-form videos. Each video is annotated with timestamps and explanations of funny moments. As compared in Table 1, our ExFunTube is unique over existing datasets in that our videos cover a wide range of domains with various types of humor that necessitate a multimodal understanding of the content.

2. We design a zero-shot video-to-text prompting that converts video content into text to maximize LLMs’ ability to explain video humor.

3. With three diferent evaluation methods of model-based lexical scores, rationale quality scores, and human evaluations, we verify that our prompting improves LLMs’ performance on humor explanation.

## 2 Related work

Humor Understanding. It has been a longstanding question whether AI models can understand humor in text, images, or videos. Early studies focused on classifying whether text (Annamoradnejad and Zoghi, 2020), images (Chandrasekaran et al., 2016), or videos (Hasan et al., 2019; Castro et al., 2019; Patro et al., 2021) are humorous or not. Some studies, such as Chandrasekaran et al. (2016), also rate the degree to which abstract scenes are perceived as humorous. However, binary classifications or ratings do not fully evaluate whether a model understands humor in detail. Recent humor studies have shifted towards having models explain humor. Sun et al. (2022) augment the SemEval 2017 Task 7 (Miller et al., 2017) with funniness ratings and explanations. Hessel et al. (2022) augment the New Yorker cartoon captions with explanations. Desai et al. (2022) propose a dataset of explanations for sarcastic captions, and Kumar et al. (2022) collect sarcastic videos from a sitcom with explanations.

Natural Language Explanation. As tasks of interest become increasingly complex, predicting labels may not be enough to evaluate the models’ true understanding. Thus, some works make models explain their decisions as an alternative. For instance, FLUTE (Chakrabarty et al., 2022) augments e-SNLI (Camburu et al., 2018) to curate figurative texts with labels for natural language inference (NLI) tasks and evaluate model-generated explanations. To evaluate model explanations, they utilize a rationale quality metric suggested by Wiegrefe et al. (2021). As word-overlap scores may be insufficient for the evaluation of explanation, Wiegrefe et al. (2021) propose a rationale quality metric that calculates the diference of prediction scores for gold labels when rationales are provided or not: Acc (IR O) Acc (I O), where I, R, and O denote input, rationale, and gold label, respectively. In addition, Sun et al. (2022) evaluate explanations by comparing the accuracy of joke classification with and without explanations: Acc (IE O) Acc (I O) where E denotes explanation. We introduce a moment localization task to compute the rationale quality score of the video explanation.

Modular Vision-Language Learning. As pretrained models become larger and are trained with extensive datasets, various multimodal comprehension tasks have been tackled by composing these pretrained models. One approach is to transform visual information into discrete text words (Zeng et al., 2022; Yang et al., 2022; Wang et al., 2022b). Zeng et al. (2022) propose a modular framework that leverages LLM to construct the input text for the subsequent model based on the output of multimodal models in the previous stage. They demonstrate performance improvements in image captioning and visual question answering (VQA) tasks. Another approach connects pretrained models through continuous feature embeddings (Patro et al., 2021; Alayrac et al., 2022; Tiong et al., 2022). Li et al. (2023a) pretrain additional lightweight modules that bridge the frozen image encoder and LLMs to eliminate the modality gap between the two frozen pretrained models. Tewel et al. (2022) connect the frozen image encoder with the frozen language decoder and evolve additional pseudo tokens during inference time to perform the video captioning task. Recently, there have been eforts to integrate these two diferent approaches. Li et al. (2023b) introduce VideoChat, a chat-centric video understanding system consisting of two modules: VideoChat-Text and VideoChat-Embed. The former generates text descriptions from the video and the latter encodes the video as embeddings. These text descriptions and embeddings are combined with a received question to form a prompt, based on which the LLM generates a response.

In our work, we combine vision-language pretrained models with LLMs through text for two uses: (i) video filtering for collecting multimodal funny videos and (ii) video-to-text generation to provide LLMs with a prompt of video content.

## 3 The ExFunTube Dataset

The ExFunTube dataset comprises 10,136 videos, each annotated with timestamps of funny moments and corresponding explanations describing why each moment is humorous. The purpose of this dataset is to evaluate the models’ ability to explain why a given video is funny as a measure of understanding video humor.

## 3.1 Video Collection and Filtering

We initially crawl all 220K videos shared on the subreddit "r/youtubehaiku,"1 where people share humorous short-form YouTube videos lasting up to 30 seconds. To ensure multimodal humor in videos, we design a four-step filtering pipeline that selects videos with both visual and verbal elements contributing to humor, as shown in Figure 2.

Video Caption and Transcript. In the first step (Figure 2 (a)), we obtain a transcript and a video caption to describe the verbal and visual elements of a video clip, respectively. We extract a video caption using a zero-shot video captioning model (Tewel et al., 2022). Since our dataset contains diverse videos such as animations and edited videos not present in previous video datasets, we choose a model that utilizes both CLIP (Radford et al., 2021) and GPT-2 (Radford et al., 2019), which are pretrained on huge Web-sourced data. We transcribe audio from the video clip using a speechto-text model Whisper (Radford et al., 2022). We remove videos with no speech or in languages other than English.

Multimodal Humor. Our goal is to collect the videos that are funny from both verbal and visual elements, instead of funny from only one modality. Thus, as shown in Figure 2 (b), we first verify that the video is verbally funny; we do this by whether GPT-3.5 can find a funny utterance given a pair of the video caption and the transcript. If GPT-3.5 detects no funny utterances, we filter out the video. Next, as shown in Figure 2 (c), we again prompt GPT-3.5 to find a funny utterance with only a transcript (i.e., no video caption). If no funny utterance is detected, then we accept this video. The rationale is that the humor of this video is multimodal; the visual caption is required to identify the fun in the video. Otherwise, if GPT-3.5 can find a funny utterance in this case, we perform a further inspection as follows.

![](images/64793c34c09dcbcb8dd47a7880bbe790a2198eb5d2b5f90a7fd9967f290c7a1f.jpg)  
Figure 2: The video filtering pipeline selects multimodal funny videos. Red boxes display the actual prompts provided to GPT-3.5. See the details in § 3.1. (a) We generate a transcript and a caption from the input video. (b) Via GPT-3.5 prompting, we filter out the video that is not funny from the transcript and caption. (c) The video is accepted if it is funny from both the transcript and caption but not from the transcript only, since its humor is multimodal. (d) GPT-3.5 generates humor explanations with or without the video caption. We remove the videos if they are too similar since their humor is not multimodal. Examples for each case are presented in the Appendix.

Diference in Explanations. In the last step (Figure 2 (d)), GPT-3.5 is prompted to generate explanations in one sentence for the two cases: when given both a video caption and a transcript and when given only a transcript. We then measure the similarity between the two explanations using the SentBERT score (Reimers and Gurevych, 2019), which embeds each sentence and calculates the cosine similarity of their embeddings. The reason for adopting the SentBERT score is that it can reflect the semantics of the entire sentence. If the score is higher than the threshold, we exclude the video since the video caption does not contribute to the humor explanation. Otherwise, the video is accepted.

Rationale of Our Pipeline. There has yet to be a method to gauge the extent and manner in which visual elements contribute to humor. In other benchmarks, the multimodality of datasets has been validated by analyzing the performance gap when visual information is either provided or not (Hasan et al., 2019; Patro et al., 2021; Kumar et al., 2022). Similarly, we collect videos that exhibit diferences in the assigned task (i.e., identifying humorous utterances by GPT-3.5) with or without visual information. In the field of NLI, previous works (Liu et al., 2022; Wiegrefe et al., 2022; Chakrabarty et al., 2022) leverage the power of LLMs such as GPT-3 (Brown et al., 2020) in creating figurative language examples or explanations for them. Likewise, we use GPT-3.5 to check the diference between generated explanations. To the best of our knowledge, this is the first approach that employs explanations for curating a dataset. Thanks to the pipeline, we can collect 21K high-quality multimodal humorous videos.

Postprocessing. To ensure that our dataset does not contain any disrespectful or harmful content towards individuals or animals, we conduct a thorough manual review of all 21K videos. We filter out the videos using the five criteria based on the safety objectives outlined by Thoppilan et al. (2022): (i) Discrimination: videos displaying discrimination based on race, gender, sexual orientation, age, or disability. (ii) Animal cruelty: videos depicting acts of animal cruelty, such as a cat falling. (iii) Dangerous goods, services, activities, or self-harm: videos featuring dangerous content like drugs, violence, or bullying. (iv) Obscenities or profanities: videos containing explicit language or sexual actions. (v) Shocking content: videos that include shocking content, such as gunshots or explosions. After the filtering, about 50% of the videos are removed, and we are left with 10,136 videos.

## 3.2 Data annotations

We crowdsource via Amazon Mechanical Turk (AMT) to annotate start and end timestamps of funny moments and provide text explanations for each moment. To participate in our dataset annotation, workers must meet the following criteria: a HIT approval rate of 99% or higher, a total of more than 10,000 approved HITs, and be located in one of the countries of AU, CA, GB, NZ, or US. We conduct a qualification test for these workers, selecting those who can efectively explain humor.

Out of 219 workers, only 60 pass the qualification test, indicating our thorough selection.

For each video, we instruct one worker first to identify up to three funny moments within a video (up to 30 seconds long) and then annotate why each moment is funny. To make workers explain both humor elements and justifications, we provide a recommended format: “[What is funny]. It is funny because [Why funny]”. We only accept responses including both descriptions (What) and justifications (Why) and reject those that lack either. Given the dificulty of the task, we ofer detailed feedback to the workers, helping them improve their performance with a high annotation standard.

As a result, we obtain 11,166 explanations, each paired with start and end timestamps of the moment. They consist of 44.3 words on average. Out of 10,136 videos, 9,222 contain one funny moment, 798 contain two, and 116 contain three. Most videos contain a single funny moment since videos are typically shorter than 30 seconds. However, given the varied content in each video, there can be any number of funny moments.

## 4 Approach

We explore an approach to explain video humor. Our idea is first to convert the video content into fine-grained text and then take advantage of recent powerful LLMs in a zero-shot manner. We design to extract as much information from videos into text as possible. Figure 3 shows a zero-shot videoto-text prompting that converts the video content into a text input to LLMs.

## 4.1 Fine-grained Text Prompts

Videos contain visual and audio modalities. The audio is further split into speech and sound. For each component, we initially generate text descriptions using state-of-the-art zero-shot models. Then, we arrange text descriptions in chronological order and use them as a prompt.

Visual. In order to populate high-quality text descriptions about the visual, we first (i) segment the video, (ii) generate multiple frame captions, and (iii) retrieve the best-matching caption with the video-to-text model.

First, we employ PySceneDetect2 to divide a video into a set of � segments based on visual changes. During the filtering pipeline (§3.1), the speech-to-text model Whisper generates timestamps for each utterance. We also use them to split the segments further, resulting in more fine-grained and semantically meaningful video segments.

![](images/0901e85aa38cc4a351e8cf57e2fd7b00c64ad81e49955d42414255c222bb097b.jpg)  
Figure 3: (a) A zero-shot video-to-text prompting for converting video content into fine-grained text (§ 4.1). For the visual modality, the video is first divided into � segments, for each of which many possible captions are generated, and the best one is chosen finally. For audio modality, a transcript with speaker separation and sound tags are obtained. (b) The fine-grained text is configured as an input prompt to LLMs (§ 4.2).

Next, we extract frames at a rate of 5fps from each of the � video segments. We generate � = 20 captions per frame using the image captioning model BLIP-2 (Li et al., 2023a) with a "Who is doing what?" prompt, which can enhance action detection. We then have a frame caption corpus (# Frames × <sup>�</sup> <sup>captions)</sup> <sup>per</sup> <sup>segment.</sup> <sup>Subsequently,</sup> <sup>we</sup> <sup>use</sup> the video-to-text model InternVideo (Wang et al., 2022a) to retrieve the caption that best matches each video segment from the respective frame corpus. Finally, we obtain one caption per segment, resulting in a total of � captions, which are fine-grained descriptions of the visual component.

Speech. We transcribe audio with Whisper (Radford et al., 2022) as done in our video filtering pipeline. We then predict the number of speakers and assign speakers to each utterance utilizing ChatGPT (OpenAI, 2023). This speaker separation helps a deep understanding of dialogue.

Sound. We extract sound tags to provide more context. We use an audio tagging model (Schmid et al., 2022) to classify the entire audio stream. We select the top 3 predicted tags that have a higher confidence value than the threshold (0.3). We concatenate the tags and insert them at the beginning of the prompt. This can provide the model with an overall atmosphere of the video.

## 4.2 Prompt Configuration and LLMs

After extracting text from visual, speech, and sound, we configure the prompt like an example of Figure 3. The prompt starts with a predefined text “Please generate ” to instruct LLMs to explain as if they are watching the video. We then include sound tags enclosed in parentheses and arrange the extracted text of speech and visuals for each video segment chronologically. To distinguish between video segments, we begin each segment with "Scene: Finally, we ask LLMs to generate an explanation of up to three sentences.

LLMs. Although any LLMs can be adopted, we use three diferent ones: finetuned T5 (Rafel et al., 2020) and BART (Lewis et al., 2020), and zero-shot GPT-3.5 text-davinci-003.

## 5 Experiments

We experiment with diferent models to see how well they explain the humor in the ExFunTube videos. We evaluate the models in three diferent ways of model-based automatic scores, rationale quality experiments, and human evaluation.

## 5.1 Experimental Setup

Baselines. We evaluate four types of explanation models. (i) Text-only LLMs generate explanations when only a transcript is provided (i.e., no use of visual). We use T5 Large and BART Large with finetuning and GPT-3.5 as a zero-shot model. (ii)

<table><tr><td rowspan="3" colspan="2"></td><td colspan="8">Automatic Score</td><td rowspan="2" colspan="2">Rationale Quality Score (↓)</td><td rowspan="2">Human Evaluation (↑)</td></tr><tr><td colspan="5">SentBERT (↑)</td><td colspan="3">ROSCOE (RA) (↑)</td></tr><tr><td>@0.7</td><td></td><td>@0.6</td><td>@0.5</td><td>@0.4</td><td>Mean</td><td>@0.8</td><td>@0.7</td><td>Mean</td><td>@0.3</td><td>@0.5</td><td>Rating</td></tr><tr><td rowspan="3">Text-Only</td><td>T5</td><td>0.154</td><td>0.355</td><td>0.585</td><td>0.795</td><td>0.534</td><td>0.406</td><td>0.871</td><td>0.780</td><td>10.3</td><td>21.9</td><td></td></tr><tr><td>BART</td><td>0.169</td><td>0.388</td><td>0.617</td><td>0.807</td><td>0.545</td><td>0.440</td><td>0.875</td><td>0.785</td><td>13.7</td><td>30.1</td><td>0.178</td></tr><tr><td>GPT-3.5</td><td>0.149</td><td>0.310</td><td>0.556</td><td>0.774</td><td>0.529</td><td>0.371</td><td>0.841</td><td>0.772</td><td>18.8</td><td>22.5</td><td>0.385</td></tr><tr><td>MAF</td><td>-</td><td>0.149</td><td>0.375</td><td>0.604</td><td>0.809</td><td>0.541</td><td>0.438</td><td>0.880</td><td>0.785</td><td>13.1</td><td>25.3</td><td>0.131</td></tr><tr><td>VideoChat-Text</td><td>GPT-3.5</td><td>0.115</td><td>0.345</td><td>0.618</td><td>0.839</td><td>0.539</td><td>0.414</td><td>0.900</td><td>0.783</td><td>13.9</td><td>26.5</td><td>-</td></tr><tr><td rowspan="3">Our Prompting</td><td>T5</td><td>0.230</td><td>0.483</td><td>0.719</td><td>0.887</td><td>0.584</td><td>0.543</td><td>0.932</td><td>0.804</td><td>2.9</td><td>12.5</td><td>=</td></tr><tr><td>BART</td><td>0.238</td><td>0.500</td><td>0.730</td><td>0.886</td><td>0.588</td><td>0.554</td><td>0.935</td><td>0.805</td><td>6.3</td><td>23.9</td><td>0.282</td></tr><tr><td>GPT-3.5</td><td>0.214</td><td>0.541</td><td>0.806</td><td>0.945</td><td>0.602</td><td>0.639</td><td>0.971</td><td>0.817</td><td>5.5</td><td>9.3</td><td>0.523</td></tr><tr><td>Gold</td><td></td><td>=</td><td>=</td><td>=</td><td>=</td><td></td><td>=</td><td>=</td><td>=</td><td>=</td><td>=</td><td>0.792</td></tr></table>

Table 2: Humor explanation results in terms of automatic scores (SentBERT and ROSCOE), rationale quality scores, and human rating. In the automatic scores, @K shows the proportion of test explanations of which scores are higher than K, and the mean column is the average score of each metric. For rationale quality scores with funny moment localization, we adopt two IoU thresholds, 0.3 and 0.5; lower scores are better. For human rating, five workers rate each of 100 randomly selected test videos from No (0), Weak No (0.25), Neutral (0.5), Weak Yes (0.75), to Yes (1). After excluding the highest and lowest scores, the remaining scores are averaged.

MAF (Kumar et al., 2022) is a multimodal end-toend model designed for video sarcasm explanation. It generates explanations by receiving features of the three components (visual, speech, and audio). We train the model on our dataset. (iii) VideoChat-Text (Li et al., 2023b) is a multimodal prompting framework that textualizes video information into text, including video/clip captions, objects contained in the video and a transcript. Given the prompt, GPT-3.5 generates explanations in a zeroshot manner. (iv) LLMs with our prompting generate explanations given a prompt created by our zero-shot video-to-text prompting, using the same LLMs as (i) of T5, BART, and GPT-3.5. Note that T5 and BART models are finetuned to generate explanations given generated prompts, while GPT-3.5 generates in a zero-shot manner.

Explanation Generation. For all finetuned models on our dataset, we employ K-fold crossvalidation as follows. We divide the entire dataset of 10,136 videos into five equal-sized subsets. In each iteration, we train the model on three subsets, use one subset for validation, and test on the remaining subset. We repeat this process five times, rotating the test subset in each iteration. Finally, we obtain predicted explanations for the entire set.

Evaluation. To compare the predicted explanation with the gold explanation for each video, we concatenate explanations for each moment into a single, unified explanation. For more details on experiments, please refer to the Appendix.

## 5.2 Results of Model-based Automatic Scores

Since the metrics based on word overlaps may fail to reflect faithfulness and plausibility as highlighted by Sun et al. (2022), we evaluate explanations using two model-based scores: SentBERT Score and ROSCOE (Golovneva et al., 2022). ROSCOE is a suite of metrics designed to evaluate the reasoning process within a chain-of-thought prompting (Wei et al., 2022). It is suitable for our explanation tasks since our goal is to uncover the reason for laughter (i.e., why is the video humorous?) Among the various scores provided by ROSCOE, we use the reasoning alignment (RA) score, which computes the contextual similarity between the hypothesis and reasoning.

Table 2 reports the model-based automatic scores of diferent methods. We show not only the mean metric values but also the proportions of the test set with scores higher than various thresholds; @� represents the proportion of data points with scores equal to or greater than �.

The results show that, except for SentBERT @0.7, GPT-3.5 with our prompting reaches the best performance. Especially, the SentBERT and ROSCOE scores with our prompting are higher than those with text-only baselines in all cases. In addition, our method outperforms the multimodal end-toend baseline MAF and the multimodal zero-shot prompting baseline VideoChat-Text. The comparison of @� metrics shows even more significant diferences, particularly for SentBERT @0.5 and

ROSCOE @0.8, where the performance margin ranges from 0.1 (BART) to 0.27 (GPT-3.5) compared to the text-only baselines. This means that using transcripts alone may not be suficient to understand the humor in our videos.

## 5.3 Results of Rationale Quality Scores

We conduct a rationale quality experiment following Wiegrefe et al. (2021) and Sun et al. (2022). Since our dataset consists of videos, unlike theirs, we adapt the experimentation scheme by evaluating the rationale quality through a moment localization task, which aims at predicting funny moments defined by their start and end timestamps in a video given the text explanation.

We use QD-DETR (Moon et al., 2023) as a localizer and divide the entire dataset into 8:1:1 splits for training (8,110), validation (1,013), and testing (1,013). During the training, the localizer is learned to predict the gold timestamp given a gold explanation. At inference, we compute the rationale quality as the prediction diference of the localizer between when given a model-generated explanation and when given a gold explanation.

Let � be a model-generated explanation, � be a gold explanation, and � be a threshold. For each test data point, we calculate the maximum IoU from the top 5 candidates given � or �, respectively denoted as $\mathrm { I o U } _ { \mathrm { M } }$ or IoU<sub>G</sub>. We use the top 5 since there can be at most three funny moments in a single video and the localization predictions can overlap with each other. We compute the diference when $\mathrm { I o U } _ { M } > \tau$ . The final score � is the sum of diferences for all test data:

$$
S = \sum _ { i = 1 } ^ { n } ( \mathrm { I o U } _ { G _ { i } } - \mathrm { I o U } _ { M _ { i } } ) \cdot \mathbb { 1 } ( \mathrm { I o U } _ { M _ { i } } > \tau ) ,
$$

where � is the number of test data points, and � is the indicator function.

Table 2 shows the results when the IoU threshold � is set to 0.3 and 0.5. A lower score is better as it is closer to the gold standard. In each LLM, the performance improves when our prompting is included compared to corresponding text-only ones. In particular, our approach improves GPT-3.5 the most, with the threshold at 0.3 resulting in a score gap of 13.3, and at 0.5, a score gap of 13.2. Again, the performance of all LLMs with our prompting is better than MAF and VideoChat-Text.

![](images/3e914e801e843356191181710194b88d354419bd3b1af8356052d0fdff364b67.jpg)  
Figure 4: Results of human preference: comparing GPT-3.5 with our prompting to text-only GPT-3.5, MAF, and Gold, respectively.

## 5.4 Results of Human Evaluations

For human evaluation, we employ 10 AMT workers using the same criteria as in the dataset annotation but excluding the ones who already participated in the annotation. We randomly select 100 videos and evaluate explanations generated by all models except baselines using T5 and VideoChat-Text, which show worse automatic scores than other text-only or multimodal baselines. We obtain human evaluations with two methods: rating and comparison.

For the rating, workers are asked to rate each explanation according to No (0), Weak No (0.25), Neutral (0.5), Weak Yes (0.75), and Yes (1) and check any shortcomings. We ask five workers for each explanation, exclude the highest and lowest scores, and take the average. For the comparison, workers compare GPT-3.5 with our prompting to (1) Text-only GPT-3.5, (2) MAF, and (3) Gold explanations and choose the better explanation. We ask five workers for each pair of comparisons.

The rating results are presented on the far right of Table 2. The scores of BART and GPT-3.5 increase by about 0.1 when our prompting is included. The comparison results are presented in Figure 4. The number of votes for text-only GPT-3.5 is significantly lower than that of GPT-3.5 with our prompting, indicating that visual information is valuable, and our prompting helps convey visual information efectively. In both rating and comparison, MAF shows lower performance than the text-only models despite being a multimodal model. This suggests that providing visual information as text to LLMs could be more efective than training the multimodal model end-to-end. Moreover, GPT-3.5 with our prompting, which shows the best results, still scores lower than Gold, indicating that understanding and explaining the humor in our dataset still remains unsolved.

![](images/bfd80c93e743330ead1359dd2b60996f36907f52af006233605240fe94fba87b.jpg)  
Figure 5: Explanation performance according to humor taxonomy. We categorize all videos into 20 humor classes and compare the performance of eight diferent baselines in terms of the SentBERT score. The humor taxonomy is arranged in descending order of proportion in our dataset.

## 5.5 Analyzing LLMs with Humor Taxonomy

We classify our dataset into a total of 20 humor categories referring to Martin and Ford (2018) and Buĳzen and Valkenburg (2004), and observe the performance of baselines by the humor taxonomy. We provide ChatGPT with 20 categories along with a brief description and one example (i.e., oneshot learning) and instruct ChatGPT to classify the video based on the given explanation. Thanks to ChatGPT’s powerful in-context learning capability, we efectively classify 10,136 videos based on their corresponding explanations.

Figure 5 shows the models’ performance by humor categories. Excluding the Jokes and Selfdeprecating classes, the performance increases with our prompting in all categories. In particular, the performance significantly increases in Clownish humor, Visual gags, and Slapsticks, which heavily reflect visual elements. This indicates that our zeroshot video-to-text prompting efectively conveys visual elements to the LLM.

## 5.6 Ablation Study

We compare the importance of each modality in humor explanation. Table 3 presents the results of SentBERT and ROSCOE scores when visual, speech, and sound components are not included in the prompt one by one. In GPT-3.5 with our prompting, the performance without the visual component drops as much as when the speech is removed, indicating that the visual component plays an important role in our dataset. Moreover, the performance decreases when either of the components is removed, which suggests that all three components are crucial for understanding and explaining humorous videos in our dataset. Additional ablation studies are presented in the Appendix.

<table><tr><td rowspan="2"></td><td colspan="4">GPT-3.5 w/ Prompting</td></tr><tr><td>w/o V</td><td>w/o T</td><td>w/o A</td><td>w/ V, T, A</td></tr><tr><td>SentBERT</td><td>0.512</td><td>0.497</td><td>0.574</td><td>0.602</td></tr><tr><td>ROSCOE (RA)</td><td>0.778</td><td>0.763</td><td>0.801</td><td>0.817</td></tr></table>

Table 3: Ablation results of GPT-3.5 with our prompting measured by SentBERT and ROSCOE scores when each modality component is removed. V, T, and A denote visual, speech, and sound, respectively.

## 6 Conclusion

We introduced ExFunTube, a dataset consisting of 10,136 user-generated videos annotated with timestamps and explanations of funny moments. Our dataset aims to assess how well AI models understand and explain video humor. We devised a zero-shot video-to-text prompting to make existing LLMs better explain the video content. With three diferent evaluation methods, we demonstrated that the humor in our dataset is multimodal, and our prompting maximized LLMs’ ability to generate explanations.

However, as the performance still falls short of human levels, our dataset remains suficiently challenging and calls for future research. Furthermore, we can consider the training of the model using user feedback for personalized humor understanding.

## Limitations

Since the copyright remains with the original owners of our dataset videos, we will only distribute URLs instead of videos.

Our method relies on the performance of existing state-of-the-art models, as we used them in a zeroshot composition. Also, our approach composes models through text, so it could also be explorable to use an adaptor-based method for prompt tuning during inference.

We measured the videos by dividing them into three modalities, but we did not consider the temporal information of sound. As timing can play a role in humor, analyzing the sound in accordance with the timeline could be helpful.

Lastly, humor is subjective, which means that our collected explanations may be subjective, too.

## Ethics Statement

We put much efort into ensuring that our dataset contains no inappropriate videos that may raise ethical issues. Based on the safety rules of Thoppilan et al. (2022), authors manually viewed each video entirely from start to end and filtered the video if there was any content that corresponded to the filtering criteria presented in the dataset postprocessing. Although we carefully reviewed all the videos, there could still be some videos that are not comfortable for someone. If such inappropriate videos are found, we will remove them in the future. Also, since we only recruit workers in AU, CA, GB, NZ, and US as mentioned in the Appendix, the cultural and geographic biases may influence humor explanations.

## Acknowledgments

We sincerely thank Jaekyeom Kim, Jaewoo Ahn, Soochan Lee, Wonkwang Lee, Yeda Song, and Jaehyeon Son for their valuable comments. We would also like to thank AMT workers for their commitment to building the ExFunTube dataset. This work was supported by the SNU-Global Excellence Research Center establishment project, Basic Science Research Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Education (RS-2023-00274280), Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00156, Fundamental research on continual meta-learning for quality enhancement of casual videos and their 3D metaverse transformation), and Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2021-0-01343, Artificial Intelligence Graduate School Program (Seoul National University)). Gunhee Kim is the corresponding author.

## References

Jean-Baptiste Alayrac, Jef Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. 2022. Flamingo: a Visual Language Model for Few-Shot Learning. In Advances in Neural Information Processing Systems, volume 35.

Issa Annamoradnejad and Gohar Zoghi. 2020. Col-BERT: Using BERT Sentence Embedding for Humor Detection. arXiv preprint arXiv:2004.12765, 1(3).

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language Models Are Few-Shot Learners. Advances in neural information processing systems, 33:1877–1901.

Moniek Buĳzen and Patti M Valkenburg. 2004. Developing a Typology of Humor in Audiovisual Media. Media psychology, 6(2):147–167.

Oana-Maria Camburu, Tim Rocktäschel, Thomas Lukasiewicz, and Phil Blunsom. 2018. e-SNLI: Natural Language Inference with Natural Language Explanations. In Advances in Neural Information Processing Systems, volume 31.

Santiago Castro, Devamanyu Hazarika, Verónica Pérez-Rosas, Roger Zimmermann, Rada Mihalcea, and Soujanya Poria. 2019. Towards Multimodal Sarcasm Detection (An \_Obviously\_ Perfect Paper). In Proceedings of the 57th Annual Meeting of the Associationfor Computational Linguistics.

Tuhin Chakrabarty, Arkadiy Saakyan, Debanjan Ghosh, and Smaranda Muresan. 2022. Flute: Figurative Language Understanding through Textual Explanations. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing.

Arjun Chandrasekaran, Ashwin K Vĳayakumar, Stanislaw Antol, Mohit Bansal, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. 2016. We Are Humor Beings: Understanding and Predicting Visual Humor. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi,

Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jef Dean, Slav Petrov, and Noah Fiedel. 2022. PaLM: Scaling Language Modeling with Pathways.

Poorav Desai, Tanmoy Chakraborty, and Md Shad Akhtar. 2022. Nice Perfume. How Long Did You Marinate in It? Multimodal Sarcasm Explanation. In Proceedings ofthe AAAI Conference on Artificial Intelligence.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of Human Language Technologies: The Annual Conference ofthe North American Chapter of the Association for Computational Linguistics.

Olga Golovneva, Moya Chen, Spencer Pof, Martin Corredor, Luke Zettlemoyer, Maryam Fazel-Zarandi, and Asli Celikyilmaz. 2022. ROSCOE: A Suite of Metrics for Scoring Step-by-Step Reasoning. arXiv preprint arXiv:2212.07919.

Md Kamrul Hasan, Wasifur Rahman, AmirAli Bagher Zadeh, Jianyuan Zhong, Md Iftekhar Tanveer, Louis-Philippe Morency, and Mohammed Ehsan Hoque. 2019. UR-FUNNY: A Multimodal Language Dataset for Understanding Humor. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-ĲCNLP).

William Hazlitt. 1845. Lectures on the English Comic Writers. 28. Wiley and Putnam.

Jack Hessel, Ana Marasović, Jena D Hwang, Lillian Lee, Jef Da, Rowan Zellers, Robert Mankof, and Yejin Choi. 2022. Do Androids Laugh at Electric Sheep? Humor "Understanding" Benchmarks from The New Yorker Caption Contest. arXiv preprint arXiv:2209.06293.

James McCoy Jones. 1970. Cognitive Factors in the Appreciation of Humor: A Theoretical and Experimental Analysis. Yale University.

Immanuel Kant. 1786. Kritik der Urteilskraft und Schriften zur Naturphilosophie, volume 5. Insel-Verlag.

Shivani Kumar, Atharva Kulkarni, Md Shad Akhtar, and Tanmoy Chakraborty. 2022. When did you become so smart, oh wise one?! Sarcasm Explanation in Multi-modal Multi-party Dialogues. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 7871–7880.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023a. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. arXiv preprint arXiv:2301.12597.

KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. 2023b. Videochat: Chat-centric video understanding.

Alisa Liu, Swabha Swayamdipta, Noah A Smith, and Yejin Choi. 2022. WANLI: Worker and Ai Collaboration for Natural Language Inference Dataset Creation. arXiv preprint arXiv:2201.05955.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled Weight Decay Regularization. In International Conference on Learning Representations.

Rod A Martin and Thomas Ford. 2018. The Psychology ofHumor: An Integrative Approach. Academic press.

Tristan Miller, Christian F Hempelmann, and Iryna Gurevych. 2017. Semeval-2017 Task 7: Detection and Interpretation of English Puns. In Proceedings ofthe 11th International Workshop on Semantic Evaluation (SemEval-2017).

WonJun Moon, Sangeek Hyun, SangUk Park, Dongchan Park, and Jae-Pil Heo. 2023. Query-Dependent Video Representation for Moment Retrieval and Highlight Detection. arXiv preprint arXiv:2303.13874.

Göran Nerhardt. 1970. Humor and Inclination to Laugh: Emotional Reactions to Stimuli of Diferent Divergence from a Range of Expectancy. Scandinavian Journal ofPsychology, 11(1):185–195.

OpenAI. 2023. ChatGPT. Generated with GPT-3 technology.

Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Badri N Patro, Mayank Lunayach, Deepankar Srivastava, Hunar Singh, Vinay P Namboodiri, et al. 2021. Multimodal Humor Dataset: Predicting Laughter Tracks for Sitcoms. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning Transferable Visual Models from Natural Language Supervision. In International conference on machine learning. PMLR.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2022. Robust Speech Recognition via Large-scale Weak Supervision. arXiv preprint arXiv:2212.04356.

Alec Radford, Jefrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language Models are Unsupervised Multitask Learners. OpenAI blog, 1(8):9.

Colin Rafel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal ofMachine Learning Research, 21:1–67.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-ĲCNLP).

Florian Schmid, Khaled Koutini, and Gerhard Widmer. 2022. Eficient Large-scale Audio Tagging via Transformer-to-CNN Knowledge Distillation. arXiv preprint arXiv:2211.04772.

Thomas R Shultz. 1972. The role of incongruity and resolution in children’s appreciation of cartoon humor. Journal ofExperimental Child Psychology, 13(3):456– 477.

Jerry Suls. 1983. Cognitive Processes in Humor Appreciation. Handbook of Humor Research: Volume 1: Basic Issues, pages 39–57.

Jerry M Suls. 1972. A Two-Stage Model for the Appreciation of Jokes and Cartoons: An Information-Processing Analysis. The psychology ofhumor: Theoretical perspectives and empirical issues, 1:81–100.

Jiao Sun, Anjali Narayan-Chen, Shereen Oraby, Alessandra Cervone, Tagyoung Chung, Jing Huang, Yang Liu, and Nanyun Peng. 2022. ExPUNations: Augmenting Puns with Keywords and Explanations. arXiv preprint arXiv:2210.13513.

Yoad Tewel, Yoav Shalev, Roy Nadler, Idan Schwartz, and Lior Wolf. 2022. Zero-Shot Video Captioning with Evolving Pseudo-Tokens. arXiv preprint arXiv:2207.11100.

Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. 2022. LaMDA: Language Models for Dialog Applications. arXiv preprint arXiv:2201.08239.

Anthony Meng Huat Tiong, Junnan Li, Boyang Li, Silvio Savarese, and Steven CH Hoi. 2022. Plugand-Play VQA: Zero-shot VQA by Conjoining Large Pretrained Models with Zero Training. arXiv preprint arXiv:2210.08773.

Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. 2022a. InternVideo: General Video Foundation Models via Generative and Discriminative Learning. arXiv preprint arXiv:2212.03191.

Zhenhailong Wang, Manling Li, Ruochen Xu, Luowei Zhou, Jie Lei, Xudong Lin, Shuohang Wang, Ziyi Yang, Chenguang Zhu, Derek Hoiem, Shih-Fu Chang, Mohit Bansal, and Heng Ji. 2022b. Language Models with Image Descriptors are Strong Few-Shot Video-Language Learners. In Advances in Neural Information Processing Systems, volume 35.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. arXiv preprint arXiv:2201.11903.

Sarah Wiegrefe, Jack Hessel, Swabha Swayamdipta, Mark Riedl, and Yejin Choi. 2022. Reframing Human-AI Collaboration for Generating Free-Text Explanations. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 632–658.

Sarah Wiegrefe, Ana Marasović, and Noah A Smith. 2021. Measuring Association Between Labels and Free-Text Rationales. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing.

Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Yumao Lu, Zicheng Liu, and Lĳuan Wang. 2022. An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3081–3089.

Andy Zeng, Maria Attarian, Brian Ichter, Krzysztof Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael Ryoo, Vikas Sindhwani, Johnny Lee, Vincent Vanhoucke, and Pete Florence. 2022. Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language. arXiv.

## A Experimental Details

Video Filtering Pipeline. In the video filtering pipeline, we utilize a zero-shot video captioning model from Tewel et al. (2022), a speech-to-text model Whisper (Radford et al., 2022), and GPT-3.5 (Ouyang et al., 2022). For the video captioning model, we optimize pseudo tokens for 25 iterations at inference time to guide the pretrained GPT-2 (Radford et al., 2019) with the CLIP ViT-L/14 image encoder (Radford et al., 2021). We use AdamW optimizer (Loshchilov and Hutter, 2019) with a learning rate of 0.008 and an L2 weight decay of 0.003. For Whisper, we use the large-v2 model. For GPT-3.5, we use text-davinci-003 and set the temperature to 0 for funny utterance detection and 0.3 for explanation generation.

Video-to-Text Prompting. During the prompting stage, we use BLIP-2 (Li et al., 2023a), Intern-Video (Wang et al., 2022a), Whisper, ChatGPT (OpenAI, 2023), and an audio-tagging model from Schmid et al. (2022). We use the coco-pretrained BLIP-2 model with nucleus sampling. For Intern-Video, we use CLIP ViT-L/14 as the image encoder. We set the temperature to 0.3 for ChatGPT, and we use the mn40\_as model for audio tagging.

Explanation Generation. To generate explanations with baseline models, we finetune T5 (Rafel et al., 2020) and BART (Lewis et al., 2020) with a batch size of 4 for 5 epochs. We use the AdamW optimizer with a learning rate of 2e-5 and an L2 weight decay of 0.01. Additionally, we train MAF (Kumar et al., 2022), a multimodal end-to-end model with an adaptor added to BART, with a batch size of 4 for 20 epochs. We use the AdamW optimizer with an L2 weight decay of 1e-4, and the learning rate is set to 5e-8 for BART parameters and 5e-7 for the remaining parameters. We use BART Large for all models.

Rationale Quality Experiments. For the rationale quality experiments with moment localization, we train QD-DETR (Moon et al., 2023) with a batch size of 128 for 200 epochs. We use the AdamW optimizer with a learning rate of 1e-5 and an L2 weight decay of 1e-4. We optimize with the moment retrieval loss consisting of the L1 loss, the cross-entropy loss and the generalized IoU loss. We use the loss balancing terms of 10, 1 and 2 for each of them, respectively. We do not use the saliency loss. We use the bert-base-uncased model (Devlin et al., 2019) as the text encoder with the max query length set to 400 and CLIP ViT-L/14 as the video encoder. We sample video frames at a rate of 1 fps.

<table><tr><td colspan="3">SentBERT ROSCOE (RA)</td></tr><tr><td></td><td>w/o V</td><td>0.540 0.783</td></tr><tr><td rowspan="3">w/o T T5 w/ Prompting</td><td>0.463</td><td>0.753</td></tr><tr><td>w/o A 0.578</td><td>0.801</td></tr><tr><td>w/ V, T, A 0.584</td><td>0.804</td></tr><tr><td rowspan="4">BART w/ Prompting</td><td>w/o V</td><td>0.551</td><td>0.788</td></tr><tr><td>w/o T</td><td>0.497</td><td>0.767</td></tr><tr><td>w/o A</td><td>0.587</td><td>0.805</td></tr><tr><td>w/ V, T, A</td><td>0.588</td><td>0.805</td></tr></table>

Table 4: Ablation results of T5 and BART with our prompting measured by SentBERT and ROSCOE scores when each modality component is removed. V, T, and A denote visual, speech, and sound, respectively.

Except for the aforementioned hyperparameters, we use the default values for all models.

## B Additional Ablation Study

We conduct ablation experiments on BART and T5 with our prompting as well, and the results are as shown in Table 4. Similar to the results of GPT-3.5 with our prompting, using all modalities achieves the best performance, and there is a certain degree of performance decrease when the visual component is removed.

## C Crowdsourcing Details

We use three diferent user interfaces of Amazon Mechanical Turk (AMT) for (i) annotating the timestamps and explanations of funny moments, and the human evaluation of (ii) rating and (iii) comparison, as shown in Figures 6-8, respectively. We guarantee AMT workers receive fair wages of approximately \$18 per hour. Additionally, we allocate about \$2 as compensation for each data point and grant additional wages to workers contributing extended time and efort.

## D Case Study

Figures 9-12 show representative videos accepted or excluded by our video filtering pipeline. Figures 13-18 provide several examples to demonstrate humor explanations that our baseline models actually generate. We color-code relevant (blue) and irrelevant (red) information contained in generated explanations. LLMs with our prompting, especially GPT-3.5, correctly explain the funny moments in Figures 13-16 while text-only LLMs and MAF fail to. All the models fail to explain humorous moments in Figures 17-18.

## Temporal localization & Explanation

Please enter the shortest possible time range between 1 second and 5 seconds, and explain why the video is funny including visual information and funny remarks in the speech. If there are more than one moments, please add an input field and enter them individually. Filling fields a lot isn't necessarily a good thing. Write only what makes you laugh particularly

![](images/3754b65ae776abcc1207d30178a968bd2c04733c60a86efc8639af232cb467ed.jpg)

For your convenience, we provide video subtitles. This could be wrong.

Video subtitles

What is my purpose? Pass the butter. Thank you. What is my purpose?. You passed butter. Oh my god.

20

24

The robot realizes his entire purpose in 'life' is to "pass butter" and reacts with a verbal "oh my God" while slumping in dismay. It's funny, because it's a reaction we'd normally attribute to a human and not a robot

Start time (s…

End time (se…

Explanation

Add field +

If your answer is not being submitted, please check the requirements below.

1. Start time and end time entered must be less than length of the video

2. Start time must be less than end time.

3. Enter the time range as short as possible. (The maximum limit was removed by reflecting the feedback.)

4. The explanation must be at least 100 characters.

(Optional) Please let us know if anything was unclear, if you experienced any issues, or if you have any feedback for us

## Submit

Figure 6: A user interface for annotating timestamps and explanations of humorous moments. Workers are asked to watch a video, identify up to three funny moments, and provide the start/end timestamps along with the explanation for each moment.

![](images/cc2081c9155b170f8b501b310c616dfb9f7c8f3755cc3b7833418b06a2f1ab47.jpg)

## [Your Work]

![](images/14797d672a21bc23fb62a13b0b120ca8f9346c61ba4f8dc7a8f56f1a8703c750.jpg)

## Rate Explanations

Please read each explanation and rate it as "No", "Weak No", "Neutral", "Weak Yes", or "Yes" based on the criterion and then check shortcomings of each explanation. More details on the shortcomings are provided below

## Descriptions of Shortcomings

\- Missed Humor : The explanation fails to identify the key aspect of the humor in the video.

\- Too Vague : The explanation isn't specific about the funny parts of the video.

\- Incorrect Details : The explanation has wrong information or misinterprets the funny elements

\- Lacking Context : The explanation misses important context or background information.

\- Too Verbose : The explanation is overly lengthy, making it harder to understand the humor.

## Explanation #1.

The delivery of the line "I got the coffees!" is delivered with an exaggerated enthusiasm, which is funny in itself. The response of "Ah!" is delivered with a surprised and relieved expression, which adds to the comedic effect. The combination of the two lines creates a humorous moment that elicits a laugh from the viewer.

Q1. Does the explanation clearly describe the reasons why it is funny including video content?

![](images/159d767f0df5c5fa70cfbb0b32134d931d6f36d83cbf20b7e76b48daffb68e94.jpg)

![](images/54fedb2414501dbb4912e5a115037140d221395b1d65af5a1303170ac6f6774d.jpg)

![](images/06b6f345495f88575fc738c788d1231ec5644df05c09f84d181ed4186032d7bc.jpg)

![](images/8f9439f59e9807345ac5d1ddb07fd31e5029f8f80b47153a86f80f6eac1cd069.jpg)

![](images/f8607f74eeaec01890942bbb687e813294417b22a4c5aec03ecd9a37966c786c.jpg)

Q2. What are the shortcomings of the explanation? (Check all.)

![](images/592078681c8c90277920fc84d7f30794c977e1df80cebb51d761b10297495b94.jpg)

![](images/3955ca3f44ddea9721160a8efc3ca1a37c7698435b91c640986aa541a15cc18a.jpg)

![](images/ae03d8d081fe090cb1ba29e4c6a0ec6275829efedebd02d220ba89cccc46cc2d.jpg)

Figure 7: A user interface for human evaluation through rating. Workers are asked to rate the explanation on a scale of No, Weak No, Neutral, Weak Yes, to Yes, and to choose any shortcomings if present.

[Your Work]  
![](images/2bd5fed1d9cecbc4adbcabfdc0f820f001c333938df7b019fd75b9d339d012f1.jpg)

Choose a Better Explanation

Please read the following three pairs of explanations and for each pair choose the one that explains better why the video is funny.

Explanation #1.

The delivery of the line "I got the coffees!" is delivered with an exaggerated enthusiasm, which is funny in itself. The response of "Ah!" is delivered with a surprised and relieved expression, which adds to the comedic effect. The combination of the two lines creates a humorous moment that elicits a laugh from the viewer.

Explanation #2.

The video starts with a man on a skateboard holding onto a pile of cups, and then the audio caption of a skateboard breaking in the background of a speech is heard. The man then falls to the ground by the trash cans, and the next scene shows a guy on a skateboard wearing a bandana. The last scene shows the same skateboarder walking next to a truck, which implies that he was the one who broke the skateboard.

Q. Which one provides more clear explanation of why it is funny, including video content?

Figure 8: A user interface for human evaluation through comparison. Workers are asked to compare GPT-3.5 with our prompting to text-only GPT-3.5, MAF, and Gold, respectively, and select the superior one.

Video shows the brain’s response to a cartoon.

![](images/b860c5c04fb0bcdada6dd90c3dc4ee179ecf65ed1708ae9469a0b01605d24f60.jpg)

![](images/86e208d7ad73b324ebfe124b972a3b6eb4acc369f9d22001e9b20f3fea086083.jpg)  
Video showing hostage taker detonating in a crowded mall during the game stealth gameplay mode of multiplayer.

![](images/dabb68bbac308d847392ddbf42941d2160f992d6a77512180bc0c8194ecf5c7a.jpg)

We’ve done it. Hey.

Figure 9: An example of a video excluded in the second step (Figure 2 (b)) of the filtering pipeline.  
![](images/7846821d1df478c8511e7c4514d2a60d26afaaa78367f103b16a0afea70d257b.jpg)

![](images/39c758a8eb2c7aa73c18e0af5ed243fd5bcb5d46e54e71f50dc3584341783f67.jpg)

![](images/746a9a6f0c396d60be5ce4921f00ec8798a7a5c4fb2650b3ee0a3a59d39ec9e9.jpg)

Hey, why did they call it Mordor? That place seemed like it had a reasonable number of doors. You know you can't get rid of me. That won't stop me from trying."

<table><tr><td>Funny Utterance w/ V</td><td>You know you can&#x27;t get rid of me.</td></tr><tr><td>Funny Utterance w/o V</td><td>You know you can&#x27;t get rid of me.</td></tr><tr><td>Explanation w/V</td><td>The video is funny because the brain&#x27;s response to the cartoon is unexpected and the subtitles are humorous.</td></tr><tr><td>Explanation w/o V</td><td>The video is funny because it is a humorous take on the classic fantasy setting of Mordor, with the protagonist&#x27;s attempts to escape being thwarted by the sheer number of doors.</td></tr><tr><td>SentBERT score 0.49</td><td></td></tr></table>

Figure 10: An example of a video accepted in the fourth step (Figure 2 (d)) of the filtering pipeline.

![](images/ce28d5cf5dc74cf598aeaa5ddf6693076e6fd4fca3a2bbafbbcb4cda6ac5f48e.jpg)

![](images/ff0fd25aa152c8c7c58561baf4fa2768e01e2b72c3c1ae4fa82a9aa7d30392e9.jpg)  
Video shows a man in the insect world, and it's.

![](images/56dad6c9c61f2a1bb1435d0fdda0f2eaaf66dc0145d9fd5c604a47a2cd5462f3.jpg)

<table><tr><td>Funny Are you feeling warm all of a sudden? Utterance w/ V</td></tr><tr><td>Funny X Utterance w/o V</td></tr></table>

Figure 11: An example of a video accepted in the third step (Figure 2 (c)) of the filtering pipeline.

![](images/545b1a7aa9e3fee4f3b6dda10f68a1c9850e83d7fe5bc956f6f822e35329eb9c.jpg)

![](images/d2e3e3856d3a56581efc1ff4d5064bfecd7ff9fe4c9c1deae4dfcb2ea4e030a7.jpg)  
Video shows newspaper ads from the time of a century ago.

![](images/886b0cabd2b8203477ec069ae74dbef3b9ec0c0fa1d226cd8e03215c435448ce.jpg)

Alright, what's up everybody? I just randomly stumbled upon this article in an old issue of the Delta Colorado Independent. This is from October 19th, 1923. Someone dreamed the other night that he was living in the year 2023 and people were going on strike because they only got \$125 a day while the price of eggs had gone up to \$10 a dozen. Ooh, spooky, right?

<table><tr><td>Funny Utterance w/ V</td><td>Ooh, spooky, right?</td></tr><tr><td>Funny Utterance w/o V</td><td>Ooh, spooky, right?</td></tr><tr><td>Explanation w/V</td><td>The video is funny because it is a humorous take on the absurdly high prices of eggs in the future compared to the wages of the time.</td></tr><tr><td>Explanation w/o V</td><td>The video is funny because it is absurd to think that people would go on strike for only getting $125 a day in the future, when the price of eggs has gone up to</td></tr><tr><td>SentBERT score 0.83</td><td>$10 a dozen.</td></tr></table>

Figure 12: An example of a video excluded in the fourth step (Figure 2 (d)) of the filtering pipeline.

![](images/71980649b73564f6e14ddee63528924b48f665a065e0ac390c87461887f029eb.jpg)

<table><tr><td><img src="images/27c2f244d8e6defbd0fa09611d15c52e01f9dc0fd7eca08e11b3a47a626ab3e3.jpg"/></td><td>Hey Luke, sit. Luke, dandelion. AHHH! Luke, look what you did to the dandelion. Gus, look dandelion. AH-HA- HA-HA! Look what he did to the dandelion. I was trying to give him a flower. AYE! MY DANDELION!</td></tr><tr><td>T5</td><td>A man is sitting in a chair and he says &quot;look what you did to the dandelion&quot; and then he screams. It is funny because it is unexpected and because he is screaming at the same time</td></tr><tr><td>BART</td><td>The video is funny because it presents a close up capture of a dandelion that has been knocked over by a man that is attempting to give it a flower. Furthermore, this scene is comical because the man featured seems to be comically flailing his arms and legs in such a way that makes it appear as if he is trying to dance to the beat of the song playing in the background. The video is also funny when the man says, &quot;I was just trying to give him a flower,&quot; because it is clear that he is not in the mood to dance, but instead is merely trying to get the attention of his</td></tr><tr><td>GPT- 3.5 MAF The video is funny because it presents a clip of a man</td><td>The video is funny because of the contrast between the speaker&#x27;s expectation of Luke&#x27;s reaction to the dandelion and his actual reaction. The speaker&#x27;s shock and dismay at Luke&#x27;s destruction of the dandelion is comically exaggerated, and the speaker&#x27;s surprise is punctuated by the loud &quot;Aye!&quot; when the dandelion is destroyed. The speaker&#x27;s over-the-top reaction to the destruction of the dandelion is what makes the video funny.</td></tr><tr><td>T5</td><td>attempting to give a dandelion to another man, but instead of giving him a flower, the man throws it to the ground, which is unexpected. Furthermore, it is comical because it is unexpected to see a man throw a flower at another man. A man is holding dandelions and a dog is shown</td></tr><tr><td>w/P BART</td><td>eating them. It is funny because the dog is so aggressive and the man is trying to get him to sit, but he is not doing so well. A man gives his dog a dandelion and the dog starts</td></tr><tr><td>w/P GPT- 3.5</td><td>licking it. It is funny because it is unexpected, and also because the dog seems to enjoy it. The video is funny because the white dog, Luke, is initially obedient when told to sit, but then proceeds</td></tr><tr><td>w/P</td><td>to eat the dandelion instead of iust sniffing it. This causes another person to scream in shock, leading to a chain reaction of laughter and more dogs being fed dandelions. The unexpected behavior of the dogs and the exaggerated reactions of the people make the video humorous.</td></tr><tr><td>Gold</td><td>It&#x27;s funny because the white dog is shown a dandelion and then the dog eats the dandelion unexpectedly. It&#x27;s funny because the black and white dog is shown a dandelion and does the same thing as the dog and eats the dandelion. Also, the man&#x27;s intention was simply to give the dog a flower, not for the dog to eat it. The dog turns and notices the dandelion, then goes over and eats the dandelion from the man&#x27;s hand. It&#x27;s funny because of the man&#x27;s exaggerated reaction.</td></tr></table>

Figure 13: (Correct) An example of explanation generation results. GPT-3.5 with our prompting correctly describes the unexpected behavior of dogs and the exaggeration of the people that provoke laughter. 2

![](images/b392f3b7af520adeca40642d2269b8d9c63fcdabc910dff41e8d32c8748eef8c.jpg)

![](images/c34b50d3e8418a8eae1305195e6f60c8a9bc1a987836fc9deae4670f68026ff7.jpg)

Get the muscles developed. And the muscles. She pushed a lot more.
<table><tr><td>T5</td><td>The video is funny because it presents a comical moment where a man is attempting to push his body to the point where he is unable to do so. Furthermore, the man&#x27;s remark is comical because he is attempting to make a joke about how hard he is pushing his body to the point where he is unable to do so.</td></tr><tr><td>BART</td><td>The video is funny because it presents a close up capture of a female individual that appears to be attempting to &quot;push a lot more.&quot; Furthermore, this scene is comical because it is clear that the individual featured is attempting to demonstrate a &quot;push-a-thon&quot; in order to &quot;get the muscles developed,&quot; which is why it is laughable to see that the woman featured is actually attempting to push herself to the extent that she is actually &quot;pushing&quot; herself.</td></tr><tr><td>GPT- 3.5</td><td>The speaker&#x27;s enthusiasm and repetition of the phrase &quot;muscles&quot; is comical, as if they are trying to emphasize the importance of the muscles. The speaker&#x27;s exaggerated emphasis on the word &quot;pushed&quot; also adds to the humor as it implies that the person they are referring to is putting in a lot of effort. The speaker&#x27;s enthusiasm and exaggeration makes the video funny.</td></tr><tr><td>MAF</td><td>The video is funny because it presents a clip of a man saying, &quot;She pushed a lot more.&quot; It is comical because it is unexpected to see a man say this, and also because the video is edited to make it seem as if the man is implying that the woman pushed him harder than she actually did.</td></tr><tr><td>T5 w/P</td><td>A man is riding a lawnmower and he is talking about getting his muscles developed, and then he says &quot;she pushed a lot more.&quot; It is funny because he is talking about pushing his muscles, and also because he is talking about pushing his muscles. A man is riding a lawnmower next to a bunch of</td></tr><tr><td>BART w/P GPT-</td><td>sheep, and he says, &quot;She pushed a lot more.&quot; It is funny because it is unexpected to see someone push a mower like that, and also because the man says it in a way that sounds like he is bragging. The visual of a person riding a lawnmower with a</td></tr><tr><td>3.5 w/P</td><td>herd of sheep is humorous in itself. The dialogue between the two speakers adds to the comedic effect as they discuss the man&#x27;s muscles and how much he has pushed. The absurdity of the situation makes it even funnier.</td></tr><tr><td>Gold</td><td>It&#x27;s funny to me because the implication is that lawnmower is sheep powered, because the sheep are pushing it, instead of horse powered. It&#x27;s also weird to see sheep doing that.</td></tr></table>

Figure 14: (Correct) An example of explanation generation results. LLMs with our prompting describe the verbal (“She pushed a lot more”) and visual elements (“a person riding a lawnmower with a herd of sheep”) contributing to humor.

![](images/c517d7fa89c689f148f62f549788ec30c8364c6f406deeeaa8b5d29a9eb0651f.jpg)

![](images/517f610e3f090b953aeceaf01cdefa15dd40a40554baf6237d205c3704c42f31.jpg)

![](images/538f505bd78a877ed18aa656637bc566c884f49cc2b7055491fe65a50e4fde65.jpg)

<table><tr><td><img src="images/99a11f85a894ea3d2d1cc2a02a69743e9fb126db9a8c030484afc3a9bbf450da.jpg"/></td><td>Hey, Mr. Bumblebee. Thanks for the honey. High five. Oh, yeah, there.</td></tr><tr><td>T5</td><td>The video is funny because it features a man who is attempting to enter an aquarium by yelling &quot;Aquarium, Aquarium, Aquarium, Aquarium,”</td></tr><tr><td>BART</td><td>It&#x27;s funny because the narrator keeps saying &quot;I want to go to the aquarium&quot; over and over again, as if he&#x27;s trying to convince the audience that they should go to an aquarium.</td></tr><tr><td>GPT- 3.5</td><td>The speaker in the video is saying the same phrase over and over again in a silly, sing-song voice. The repetition of the phrase is humorous, and the speaker&#x27;s enthusiasm for wanting to go to the aquarium is endearing. The way they keep asking &quot;somebody&quot; to take them to the aquarium</td></tr><tr><td>MAF</td><td>is especially funny. It&#x27;s funny because the man is talking about going to the aquarium, but he says he wants to go to someone else&#x27;s aquarium.</td></tr><tr><td>T5 w/P</td><td>A man is holding a goldfish in his hand and he says &quot;Aquarium, aquarium, I wanna go to the aquarium.&quot; The fish then starts to squirm and he says &quot;Okay, aquarium, somebody.&quot; It is funny because the man is trying to get the fish to go to the aquarium, but the fish is not interested in</td></tr><tr><td>BART w/P</td><td>going. It&#x27;s funny because the narrator keeps saying &quot;aquarium&quot; over and over again, but the fish doesn&#x27;t seem to care.</td></tr><tr><td>GPT- 3.5 w/P</td><td>The video is funny because the goldfish is portrayed as a character that wants to go to the aquarium. The goldfish is seen swimming around the tank and looking up at the camera with its big eyes, as if it is asking for help. The video ends with an ad for the Oregon Coast Aquarium in Newport, making it even funnier.</td></tr><tr><td>Gold</td><td>The goldfish is swimming around in his bowl saying, &quot;I want to go to the aquarium.&quot; It is funny because he is a goldfish and he&#x27;s already in a tank full of water.</td></tr></table>

Figure 15: (Correct) An example of explanation generation results. Except for LLMs with our prompting, the presence of a goldfish (or fish) is not mentioned. Particularly, GPT-3.5 with our prompting accurately describes the character and behavior of the goldfish, and even mentions the content of the concluding advertisement.

![](images/b859998eeef4c07d425d322f227bf81ee0cf37527eb6cb6c127a88380358758a.jpg)

![](images/d5e54ee1fda347e6f176b9518e5194ca831ec09403646d2c3ed55fd7edc2bd31.jpg)

![](images/d8e54106df5a34680f5cba68a11e5a6b2f0d2ccd585c631330de09116f5ded85.jpg)

![](images/a0dca9f285a73fb48593c8435a150f352debc82b00db84e84a22225d96a6b4a9.jpg)

<table><tr><td><img src="images/81d96ba9d122a18afc066902256bcf1e16c16197cefab63d8ca978215554c9b0.jpg"/></td><td>I like the color. You like the color? Yeah. I had a mixed reviews on the color. Some people said they like it, some people said they don&#x27;t like it. And I said, well what color is your baguette?</td></tr><tr><td>T5</td><td>The man is describing a baguette that he bought and he says he liked the color, and then he says &quot;some people said they like it, some people said they don&#x27;t like it.&quot; It is funny because he is implying that people don&#x27;t like the color of the baguette.</td></tr><tr><td>BART</td><td>A man asks a woman what color is her baguette. She replies, &quot;I had a mixed reviews on the color, some people said they don&#x27;t like it.&quot; The man then asks her, &quot;And I said, well what color?&quot; It is funny because it is unexpected.</td></tr><tr><td>GPT- 3.5 humor.</td><td>The speaker&#x27;s response to the question of whether people like the color or not is hilarious. The unexpected and absurd question of what color their baguette is catches the audience off guard and elicits a burst of laughter. The speaker&#x27;s deadpan delivery of the line only adds to the</td></tr><tr><td>MAF</td><td>A man says &quot;I had a mixed reviews on the color of my baguette.&quot; It is funny because it is unexpected.</td></tr><tr><td>T5 w/P</td><td>The man in the cartoon says &quot;I like the color, you like the color?&quot; and then the car is towed away. It&#x27;s funny because the car is a very bright color and the man in the cartoon is a very nerdy looking guy.</td></tr><tr><td>BART w/P GPT-</td><td>A man says &quot;I had a mixed reviews on the color&quot; while looking at a car. This is funny because it&#x27;s a play on the word mixed. The video is funny because the speaker is making</td></tr><tr><td>3.5 w/P</td><td>a joke about the color of the car, asking what color the person&#x27;s baguette is in comparison. The visuals add to the humor, showing the man in sunglasses and orange shirt next to the orange sports car, and the tow truck driver with his tool. The sound of the car at the beginning also adds to the comedic effect.</td></tr><tr><td>Gold</td><td>The video is funny because it features an animated clip featuring a man that is towing a luxury car while in the presence of its presumably wealthy owner that makes the sly comment in stating, &quot;I had mixed reviews on the color. Some people say they like it, some people said they don&#x27;t like it. I said well, what color is your Baguette?&quot; This remark is laughable because it implies that the man is rich and the person that is questioning the car color is not and is likely not in ownership of a Baguette, so therefore, their opinion is irrelevant.</td></tr></table>

Figure 16: (Correct) An example of explanation generation results. Text-only LLMs do not mention a car that has a similar color to a Baguette. Meanwhile, LLMs with our prompting provide details about the car in the scene. Note that GPT-3.5 with our prompting can explain the sarcasm related to the baguette like in Gold.

![](images/0802343c7f3c68e35e7ae8d03cd66454a903f1d076b47c92d4fe11a4648f41db.jpg)

![](images/b88585795bb0ebfe13f2ad9cdb9b70bc3e65dbace67393c51f05e5fd9ab524b1.jpg)

![](images/fae1c2531a885fd597deefd0ffc6ca98d22caa455d8a47b82851431e17dc620e.jpg)

![](images/6db648243e55b522ca4a8af443079980863cc597cdc2516807902483190b9197.jpg)

<table><tr><td><img src="images/614a7de44799611ce14a73998cb5b9ba67a8b059068a9a117ee74b7d8ba487c4.jpg"/></td><td>Hey man, is it raining today? Uh... Yeah, it&#x27;s sprinkling.</td></tr><tr><td>T5</td><td>It&#x27;s funny because the man is asking if it&#x27;s raining and the weatherman says it&#x27;s sprinkling, which is unexpected.</td></tr><tr><td>BART</td><td>It&#x27;s funny because the man asks his friend if it&#x27;s raining and the friend replies, &quot;Yeah, it&#x27;s sprinkling&quot; and then the camera zooms in on the ground where it is actually raining.</td></tr><tr><td>GPT- 3.5</td><td>The video is funny because the man asking the question is standing outside in the rain, yet he still has to ask if it&#x27;s raining. The person he&#x27;s asking looks at him incredulously and simply replies &quot;Yeah, it&#x27;s sprinkling&quot;. The contrast between the man&#x27;s obliviousness and the other person&#x27;s deadpan response is humorous.</td></tr><tr><td>MAF</td><td>It&#x27;s funny because the man asks the other man if it&#x27;s raining today, and the man replies &quot;yeah, it&#x27;s sprinkling&quot;.</td></tr><tr><td>T5 w/P</td><td>The man asks the other man if it&#x27;s raining, and he replies, &quot;uh, it&#x27;s sprinkling.&quot; It is funny because he is implying that rain is bad and he is implying that rain is bad.</td></tr><tr><td>BART w/P</td><td>A man asks his friend if it&#x27;s raining today and his friend replies, &quot;yeah, it&#x27;s sprinkling.&quot; It is funny because it is unexpected that someone would say that.</td></tr><tr><td>GPT- 3.5 w/P</td><td>The humor in this video comes from the absurdity of the situation. The man is driving while eating an ice cream cone, which is already a bit ridiculous, but then the question about the weather adds to the humor. The deadpan response of &quot;yeah, it&#x27;s sprinkling&quot; adds to the absurdity and makes the situation even funnier.</td></tr><tr><td>Gold</td><td>What is funny is that the man in the car puts his plain white icecream cone outside his window and when he brings it back inside the car you can see that it is covered with sprinkles. Another man had asked him if he thought is was raining today and he uses his sprinkled cone to show that it is &quot;sprinkling&quot;. Making a joke of the two meanings for the word sprinkles creates the humor and the unexpected site of the icecream cone suddenly covered with candy sprinkles.</td></tr></table>

Figure 17: (Incorrect) An example of explanation generation results. Unlike both text-only BART and GPT-3.5 say it’s raining, GPT-3.5 with our prompting correctly mentions a man eating an ice cream cone in his car. However, it fails to explain the use of the “sprinkled cone” for a pun. 2

![](images/98b06d4c011d55e311978e7c29b37a561e9d304edf21b81b9cc5f247d01971f3.jpg)

![](images/aeb83decf012ce854ab65c526c814380409d77555cd065c78c3c81d346cd1531.jpg)

![](images/cc9f0137a717222bba971f7adc9834ed6bc292fd12b59a04f581c4e548085f52.jpg)

![](images/ccbe0a218759a1c7a576c29f61eb782a6177ffc732ad2414a18dc805b3b84ce0.jpg)

![](images/63e82c61a9ef269d25be9e233e0bf06e86b3b78bfee2df5d55a0e08f7b19ad88.jpg)

<table><tr><td>T5</td><td>The video is funny because it presents a comical scene of a man attempting to eat a bumblebee, which is a surprisingly common occurrence in nature. Furthermore, the man&#x27;s attempt to eat a bumblebee is comical because it is a slapstick comedy that is comically portrayed in the clip.</td></tr><tr><td>BART</td><td>The video is funny because it presents a close up capture of a male individual that is attempting to high five a &quot;Bumblebee&quot; character. Furthermore, this clip is comical because the clip is edited to make it appear as though the male individual is actually attempting to &quot;high five&quot; the Bumblebee character, which is a silly notion because it is absurd.</td></tr><tr><td>GPT- 3.5</td><td>The video is funny because the speaker is talking to a bumblebee as if it is a person. The speaker then gives the bumblebee a high five, which is a humorous gesture as bumblebees are not able to reciprocate the gesture. The speaker then says &quot;Oh, yeah, there&quot; in a satisfied tone, making the situation even funnier.</td></tr><tr><td>MAF</td><td>The video is funny because it features a clip of a man giving a high five to a bee, which is unexpected. Furthermore, this clip is comical because the man&#x27;s high-fiving the bee is unexpected, and also because the bee&#x27;s reaction to the high-five is not what you would expect from a bee.</td></tr><tr><td>T5 w/P</td><td>A man says &quot;hey Mr. Bumblebee, thanks for the honey. High five.&quot; It is funny because he is pointing at a bumblebee and he is pointing at a flower.</td></tr><tr><td>BART w/P</td><td>The video is funny because it captures a clip of a male individual attempting to high five a bumblebee that appears to be attempting to get away from him. Furthermore, this clip is comical because the male individual is attempting to capture the attention of the bee but is comically unsuccessful in his attempt to do so. The scene is funny because the man is trying to</td></tr><tr><td>GPT- 3.5 w/P</td><td>high five a bumblebee, which is an impossible task. The bumblebee is oblivious to the man&#x27;s attempts and is instead focusing on the daisies. The juxtaposition of the man&#x27;s excitement and the bee&#x27;s obliviousness is humorous.</td></tr><tr><td>Gold</td><td>It&#x27;s funny because you expect the bee to sting the man but the bee gives the man a high five instead and then puts his arm back.</td></tr></table>

Figure 18: (Incorrect) An example of explanation generation results. Unlike text-only LLMs, both MAF and LLMs with our prompting correctly identify Bumblebee as a bee, not a character. However, they incorrectly generate explanations saying that Bumblebee fails to perform a “high-five,” which difers from Gold.