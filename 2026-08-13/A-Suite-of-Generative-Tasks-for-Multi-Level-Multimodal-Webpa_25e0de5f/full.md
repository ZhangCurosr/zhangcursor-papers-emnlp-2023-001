# A Suite of Generative Tasks for Multi-Level Multimodal Webpage Understanding

Andrea Burns1\* Krishna Srinivasan2 Joshua Ainslie2 Geoff Brown2

Bryan A. Plummer1 Kate Saenko1,3 Jianmo Ni2 Mandy Guo2

1Boston University, 2Google, 3FAIR

{aburns4, bplum, saenko}@bu.edu, {krishnaps, jainslie, geoffbrown, jianmon, xyguo}@google.com

## Abstract

Webpages have been a rich, scalable resource for vision-language and language only tasks. Yet only pieces of webpages are kept in existing datasets: image-caption pairs, long text articles, or raw HTML, never all in one place. Webpage tasks have resultingly received little attention and structured image-text data left underused. To study multimodal webpage understanding, we introduce the Wikipedia Webpage suite (WikiWeb2M) containing 2M pages with all of the associated image, text, and structure data¹. We verify its utility on three generative tasks: page description generation, section summarization, and contextual image captioning. We design a novel attention mechanism Prefix Global, which selects the most relevant image and text content as global tokens to attend to the rest of the webpage for context. By using page structure to separate such tokens, it performs better than full attention with lower computational complexity. Extensive experiments show that the new data in WikiWeb2M improves task performance compared to prior work.

## 1 Introduction

Webpages are a source of multimodal, structured content which have been used for both pretraining and finetuning purposes. Large scale noisy text or multimodal datasets scraped from the web have been used to pretrain large language or contrastive models (Raffel et al., 2020; Jia et al., 2021; Radford et al., 2021; Aghajanyan et al., 2022). Downstream tasks built from webpages have included instruction following, image captioning, news captioning, image-sentence retrieval, and image-article retrieval (Shi et al., 2015; Li et al., 2020; Gur et al., 2022; Sharma et al., 2018; Biten et al., 2019; Liu et al., 2021; Srinivasan et al., 2021; Tan et al., 2022).

![](images/188136c8b31ad61ab5948c05826f3f74b41cbc5edca32c2833f54ae849941a5d.jpg)  
Figure 1: Tasks we study with WikiWeb2M. Our dataset provides a unified webpage sample that contains all text, image, and structure, enabling new tasks like page description generation. For image captioning and section summarization, remaining page text and images provide useful context, aiding task performance.

Yet limited prior work has studied tasks to evaluate multimodal webpage understanding itself.

Many classification and generation problems can be studied with webpages: taxonomic webpage classification, webpage retrieval, web image captioning, and page summarization. However, to date there is no open source, multimodal dataset that retains all webpage content. E.g., the Wikipedia Image Text (WIT) dataset (Srinivasan et al., 2021) does not retain HTML structure and misses out on text sections. Thus, we propose the new Wikipedia Webpage (WikiWeb2M) dataset of over 2M pages, which unifies webpage content to include all text, images, and their location (e.g., section index) in a single sample. Table 1 compares the statistics of WikiWeb2M to the existing English WIT dataset.

Figure 1 shows an example of our WikiWeb2M benchmark suite; we design a set of tasks that require webpage understanding at varying degrees of granularity. Specifically, we use page description generation, section summarization, and contextual image captioning to evaluate a model's ability to understand a webpage at a global, regional, and local level, respectively. For page description generation, the goal is to generate an overarching global description of the webpage. The task of section summarization generates a sentence that captures the key content of one section. Finally, contextual image captioning generates a caption for one image within the webpage.

<table><tr><td rowspan="2">Dataset</td><td colspan="6"># Webpage Sections</td><td colspan="2"># Images</td></tr><tr><td>Structural Heading</td><td></td><td>Text</td><td>Image</td><td>Both</td><td>Total</td><td>Unique</td><td>Total</td></tr><tr><td>WIT (En)</td><td></td><td></td><td></td><td></td><td>199,872 2,847,929</td><td>3,047,801</td><td>3,660,2114,955,835</td><td></td></tr><tr><td>WikiWeb2M</td><td>731,394</td><td>686,3766,817,950221,5233,236,2541</td><td></td><td></td><td></td><td>11,693,497</td><td>4,438,6425,940,431</td><td></td></tr></table>

Table 1: WikiWeb2M versus WIT (Srinivasan et al., 2021). WikiWeb2M re-introduces millions of text and multimodal webpage sections. We report counts over all splits; train, validation, and test are reported separately in Appendix A. WikiWeb2M and WIT (English subset) come from the same webpages.

WikiWeb2M's tasks will allow for general study of multimodal content understanding with manyto-many text and image relationships and can also specifically improve interaction with web content. For example, a webpage description may provide a user who is blind more agency by allowing them to preview content before listening to the entire body of image and text with a screen reader (Vtyurina et al., 2019). In addition to contextual captioning and section summarization aiding assistive technology, these tasks can be used for modern content generation, as there is growing interest in providing multimodal web snippets (Nkemelu et al., 2023). The study of webpages in a multimodal context has even been motivated from a sociological and anthropological perspective (Pauwels, 2012).

While we curate a new dataset with Wikipedia, we note it is just one of many domains that could be used to study multimodal webpage understanding. Instructional websites, news articles, recipes, blogs, and more have bodies of text and images interleaved by layout or HTML structure.

We utilize the T5 (Raffel et al., 2020) framework to address the WikiWeb2M tasks. One challenge in modeling webpage tasks is the length of the input data (i.e., a long sequence results from flattening webpage text and images). While the full attention originally used in T5 is performant, it results in a quadratic computational complexity with respect to the input sequence length. Thus, we define a new mixture of local-global attention, Prefix Global, which uses our structured webpage data to select the most salient text and images as global tokens in the prefix of our input sequence. Prefix Global is ultimately more efficient, meaning longer input sequences can be used to reach better task performance. Our results can be beneficial to the many structured image-text domains outside of webpages such as mobile apps, figures, posters, infographics, and documents.

We include ablations across multiple axes: the pretrained checkpoint we initialize from, the input sequence length, the feature inputs, and the attention mechanism. We importantly find that images improve performance for all tasks, while prior work on contextual image captioning claimed otherwise (Nguyen et al., 2022). We are also able to improve task performance now that we have access to the entire page's content. Still, there is plenty of room to improve upon our benchmark suite.

We summarize our contributions below:

• A new open source multimodal webpage dataset, WikiWeb2M, containing 2M pages curated from English Wikipedia articles. Each sample contains all text, images, and structure present per page.

• A suite of multimodal generation webpage tasks that reflect webpage understanding at three granularities: page description, section summarization, contextual image captioning.

• A new attention mechanism, Prefix Global, which is a mixture of local-global attention that separates a prefix of global tokens. By defining more salient content from structured pages, it can outperform full attention while requiring fewer attention computations.

• Ablations on attention, sequence length, input features, and model size. Images can help all tasks, notably by over 15% on contextual captioning, and page context boosts average performance by over 4% and 3% for section summarization and captioning, respectively.

<table><tr><td>WikiWeb2M</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td># Pages</td><td>1,803,225</td><td>100,475</td><td>100,833</td></tr><tr><td># Sections</td><td>10,519,294</td><td>585,651</td><td>588,552</td></tr><tr><td># Total Images</td><td>5,340,708</td><td>299,057</td><td>300,666</td></tr></table>

Table 2: Breakdown of the number of pages, sections, and images contained in each WikiWeb2M dataset split.

## 2 The WikiWeb2M Dataset

We create the Wikipedia Webpage (WikiWeb2M) dataset to have an all-in-one multimodal webpage dataset where all text, image, and structure content is retained. WikiWeb2M is built by starting with the Wikipedia Image Text (WIT; Srinivasan et al., 2021) English pages2. We re-scrape webpage samples and keep all text, image, and structure available, providing more contextual data which can be used to model existing tasks like contextual image captioning, as well as enable new webpage understanding tasks like page description generation. We start with WIT URLs to create a high quality multimodal webpage dataset that has already gone through extensive content and topic filtering.

Each webpage sample includes the page URL page title, section titles, section text, images and their captions, and indices for each section, their parent section, their children sections, and more. This differs from WIT, which defined individual samples as image-caption pairs with metadata (e.g., originating section title). Appendix A.3 includes a comparison of fields available in WikiWeb2M versus WIT. In Table 1, we report the number of sections and images compared to the English subset of WIT. We add nearly 1M total images to the dataset by keeping the images on a webpage regardless of whether they have image captions available.

We provide section counts by type: structural, heading, text only, image only, and both text and image. Structural and heading sections do not contain immediate text. The former has subsections. For heading sections, a section title was available, while the content linked to a different article, was empty, or only had tables. We only retain sections if they are content sections (e.g., not the “See Also” section). A significant 6.8M text sections are in WikiWeb2M, none of which were available in WIT. For image quality control, we keep JPEG and PNG image types³. We make a random 90/5/5 split and show the number of pages, sections, and images per split in Table 2. Note that Table 2 reflects statistics of WikiWeb2M, which is later refined to build our downstream tasks datasets. It can be repurposed for other webpage understanding tasks or reprocessed with different data filters.

<table><tr><td>Task</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>Page Desc.</td><td>1,435,263</td><td>80,103</td><td>80,339</td></tr><tr><td>Section Summ.</td><td>3,082,031</td><td>172,984</td><td>173,591</td></tr><tr><td>Image Caption.</td><td>2,222,814</td><td>124,703</td><td>124,188</td></tr></table>

Table 3: Number of samples for page description generation, section summarization, and image captioning task datasets after additional filtering of WikiWeb2M.

## 2.1 The WikiWeb2M Tasks

We apply WikiWeb2M to three tasks which reflect different granularities of webpage understanding: the page, section, or element level. Table 3 contains task sample counts which are achieved by further task-specific filtering; these data processing steps are included in Appendix A.1, with a discussion of potential dataset noise in Appendix A.2. We describe the tasks below.

Page Description Generation. In the task of page description generation, the goal is to generate a description of a page given the rest of the webpage's image, text, and structure. We use the Wikipediaprovided page description (not collecting annotations) and generate summaries from multimodal inputs, which differs from existing text-only article summarization work; this matters when we want to create a multimodal snippet from a webpage.

Section Summarization. The goal of section summarization is to generate a single sentence that highlights a particular section's content. The summary is generated given all images and (non-summary) text present in the target and context sections; see Figure 3 for a task example. Following the leading sentence bias, we use the first sentence of a section as a pseudo summary (which is removed from the model inputs). We also found that a majority of human annotators deemed the first sentence as a reasonable summary; these findings are later discussed in Appendix F.

Contextual Image Captioning. Nguyen et al. (2022) proposed contextual image captioning with WIT as the task of captioning an image given the image and its webpage context. Target images are those available in WIT to ensure they have quality captions that can be reconstructed. A Wikipedia image can have three caption types (not all are always available): the alt-text, reference, and attribution descriptions. Alt-text serves as a text description for accessibility purposes, the reference description comes directly below the image in the rendered webpage, and the attribution description contains captions unique to the image across all webpages it appears in. Prior work only input the image, attribution description and associated section text because that was all that was available.

![](images/fc586a543ef92579bc3957f601dccebfeb7f1b1c6cc940052bcf769e0b9f07cc.jpg)  
Figure 2: Local-global attention schemes. On the left we show Transient Global (TGlobal), which has local to local and local to global attention (Guo et al., 2022). We propose the new Prefix Global attention which additionally has global to global and global to local attention compared to TGlobal. We define global tokens as a fixed-length prefix of the input, unlike TGlobal which defines additional global tokens that are aggregates over the full input sequence.

## 3 Prefix Global Attention

When structured image-text data is available, we need not treat all images and text equally. With webpages, it may be more sensible to isolate certain parts as more important. E.g., in contextual image captioning, the model should focus on the target image and section it came from, while using the rest of the page as additional context. We can now isolate these inputs with the WikiWeb2M dataset because we have structural metadata signaling where each image and text element are located, as opposed to a bag of images and a single long body of text. I.e., the new structure available in our dataset can both serve as new inputs to the model and enable new attention mechanisms.

We thus propose Prefix Global, a local-global attention, to capitalize on this intuition. A mixture of local and global attention weights provides the means to designate certain inputs as “global" tokens which can specially attend to the rest of the input sequence, while others only have local attention to a radius of r tokens to the left and right. Not only is it desirable to prioritize more salient image and text content from the input data, but it can also reduce the computational complexity of the attention mechanism. While full attention is performant by allowing all input tokens to attend to each other, it results in a quadratic computational complexity $( O ( l ^ { 2 } )$ for a sequence of length l).

Figure 2 illustrates our Prefix Global and prior work's Transient Global attention schemes, where in each the ith row represents what the ith token can attend to. Guo et al. (2022) introduced LongT5 as an adaptation of the T5 model with Transient Global (TGlobal) attention to balance the efficiency of local attention, which allows for much longer input sequences to be held in memory, with the higher performance of full attention. TGlobal resulted in similar or better performance than full attention with much longer sequence lengths, while having a complexity of $O ( l ( r + k ) )$ for a variety of text summarization tasks (where $\begin{array} { r } { k = \frac { l } { 1 6 } ) } \end{array}$ .

In addition to the local attention of TGlobal (see the blue attention diagonal in Figure 2 (left)), “transient" global tokens are defined on the fly per layer. TGlobal defines k globals as the average of every 16 input tokens, which are additionally attended to by all other inputs. As a result, TGlobal has more global tokens as the sequence length increases. In contrast, shown in Figure 2 (right), Prefix Global uses a constant number of global tokens. Specifically, it takes a prefix of the input sequence. This is inspired by the leading sentence bias (Kedzie et al., 2018; Xing et al., 2021; Zhu et al., 2021), which shows that earlier content in a body of text is often of greater importance. We define different prefixes for each task in Section 4. While we use section structure to define our prefixes, Prefix Global can use structure from other sources: HTML/the Document Object Model, rendered webpage regions, PDF document layouts, or simply knowing a priori what task inputs are most salient.

![](images/4bfd104d3aa629fd18e520bd9c949eb38984202d7299b646b41dad0e979ec508.jpg)  
Figure 3: WikiWeb2M section summarization with Prefix Global. With WikiWeb2M, we can now use section structure to separate the most relevant webpage content. The global tokens of Prefix Global (in green) are the first 512 tokens of the target section to be summarized: the first x images of the section, the section index, title, body text. and captions. Then the remaining sections (in blue) from the webpage are input; these have local attention, while the prefix global tokens attend to every other token. We decode the summary (in orange) given the page inputs.

Prefix Global has a computational complexity of O((l − k) · r + k · l) for k global tokens, similar to local-global attention schemes ETC (Ainslie et al., 2020), Longformer (Beltagy et al., 2020), and Big-Bird (Zaheer et al., 2020). However, Prefix Global does not require any special pretraining and instead finetunes directly from full attention checkpoints (T5 in our case). This is distinct from LongT5, which also required pretraining with TGlobal attention to be effective. Thus, as we show in Section 5 with Prefix Global's higher performance, it is both a more flexible and performant attention. We also are the first to demonstrate using a local-global attention with multimodal inputs, and further show Prefix Global's ability to be performant in multimodal finetuning from a text-only checkpoint.

## 4 Experiments

We now detail the model variants used for experiments, parameter settings for reproducing our set up, the metrics used for evaluation, and key ablations we perform.

Model Architectures. We benchmark with the T5 (Raffel et al., 2020) encoder-decoder framework. T5 takes a sequence of image and text inputs and we embed images in our input sequence using a frozen ViT model (Dosovitskiy et al., 2021). We note that finetuning ViT may further improve performance. We compare three models defined by different encoder attention schemes: the original T5 which uses full attention, LongT5 with TGlobal attention by Guo et al. (2022) (checkpoints are publicly available), and our Prefix Global attention. We finetune all models from a T5 checkpoint pretrained with full attention on the text-only C4 dataset, and a ViT pretrained on either ImageNet (Deng et al., 2009) or JFT (Hinton et al., 2015).

Parameter Settings. We finetune each model for 218 steps as done by Raffel et al. (2020) with a 128 batch size. Each model is trained on 16 TPUs, with the base model taking between 24-32 hours to run⁴ (varies by task) with an input sequence length of 1024. We do not perform hyperparameter tuning: all models use the Adafactor optimizer with a constant learning rate of 1e-3, an Adafactor offset of 1M to account for pretraining steps, and loss normalizing factor of 218. For Prefix Global experiments, the default prefix size k is 512. For both Transient Global and Prefix Global, the local attention neighborhood r is set to 127, as done in LongT5 (Guo et al., 2022).

Metrics. For quantitative results, we report BLEU-4 (Papineni et al., 2002), ROUGE-L (Lin, 2004), and CIDEr (Vedantam et al., 2015) metrics from a single run. BLEURT (Pu et al., 2021), CLIP-Score and RefCLIPScore (Hessel et al., 2021) are additionally reported in Appendix C for all results in the main text. We include qualitative results in Appendix B; we perform two qualitative studies to (1) inspect the quality of generated text for all finetuning tasks and (2) discuss when and why images may help the more text-based tasks of page description generation and section summarization.

Ablations. We compare each attention at different input lengths. Our sequence length ablations also include experiments where Prefix Global and TGlobal have the same number of global tokens to strictly compare how they define global tokens. Then we ablate webpage inputs (section text, titles, structure, images, image captions) and use the best feature combinations for any remaining experiments. We run experiments with different model sizes (B16 or $\mathrm { L } 1 6 \mathrm { T } 5 + \mathrm { V i T } ^ { 5 } )$ for Prefix Global at a 1k input sequence length. Lastly, we verify that WikiWeb2M's new annotations improve performance over prior work. Specifically, we ablate if the target, description, or context sections are input and if sections only from WIT vs. WikiWeb2M are input (since many text and multimodal context sections were not originally kept in the WIT dataset).

## 4.1 Defining Prefix Global Attention Inputs

Each sample's images are always included as part of the input's prefix tokens. We ablated the number of images that contribute to each task's prefix and include ablations in Appendix D.3. We use six images for page description and one image input for section summarization and image captioning.

We describe each task's prefix below. Note that we remove the text that serves as the target summary or caption from our inputs to the model for each task; this ensures there is no model “cheating." $E . g .$ , for section summarization, since we utilize the first sentence of a section as its pseudo target summary, we remove it from the inputs to the model.

Page Description. We input the images, page URL, page title, and all sections (index, title, text, captions) in their structured page order. In addition to the images, URL, and page title participating in the prefix, we also include all section titles and section first sentences (up to 512 tokens). This outperformed keeping the section titles and text concatenated in order; see Appendix D.1.

Section Summarization. The target section to be summarized is prepended to each sample's input sequence. This means the target section's index, title, non-summary text, images, and captions contribute to the global tokens of Prefix Global. Then the page

URL, title, and remaining sections follow in order. Figure 3 illustrates how an input sequence is defined with Prefix Global for section summarization.

Contextual Image Captioning. Similar to section summarization, the target image and its originating section's content contribute to the prefix tokens (the index, title, text, and non-target captions), followed by the URL, page title, and context sections.

## 5 Results

We now detail experimental results, first evaluating performance and efficiency of each attention type at different sequence lengths. Then, we report input feature, model size, and annotation ablations.

## 5.1 Attention and Sequence Length

Performance Comparison. We begin by evaluating performance for each task (page description, section summarization, and contextual image captioning) when training T5 encoders with different attention types and input sequence lengths in Figure 4. Prefix Global always performs better than TGlobal. We include two Prefix Global settings: a fixed Prefix512 which sets 512 input tokens to the prefix (default used for all other experiments), as well as a $\mathrm { P r e f i x } _ { T G l o b a l }$ which assigns the same number of global tokens as TGlobal. Prefix $T G l o b a l$ uses $\frac { l } { 1 6 }$ globals, where l is the input sequence length (TGlobal aggregates every 16 input tokens as a global token). This allows us to compare the way both attention mechanisms define global tokens.

Despite TGlobal defining additional side inputs as global tokens, it consistently underperforms Prefix Global even with the same number of globals. This confirms that defining a special prefix from the input sequence is better than taking aggregates over the full sequence. In Appendix D.1, we also show that just using the prefix of the in-order page inputs for page description (as opposed to pulling out the section titles and first sentences) performs better than TGlobal. These results collectively show Prefix Global to be preferable to TGlobal. One key takeaway is that separating out more relevant inputs (via structure or other known biases like leading sentence bias) is a good idea.

Full attention and Prefix Global generally have higher performance at longer sequence lengths. It is impressive that Prefix Global scales or maintains performance with larger sequences even when its number of globals is fixed to 512 $( i . e .$ , the number of globals is not scaled with respect to input length). On the other hand, while TGlobal scales the number of globals to sequence length, its performance does not consistently scale. E.g., performance plateaus or even drops at 4k input sequence length for page description and section summarization, respectively. This may be because TGlobal defines globals as aggregates over the full input sequence, which could introduce more noise or less semantically rich text at longer sequence lengths.

![](images/2eba365dc985b1d3bcb0e057ce4f1a3d29f9a20b52a6b595bc01c55b2855a470.jpg)

![](images/8cef54358bdfa13c575a641cb4700c7eb5e90a7994684e60e1a31de82e1e6998.jpg)

![](images/1c03f14e67db38baa45627945ee04bb5553673fb4b0ab2b799e1848d419d5430.jpg)  
Figure 4: Encoder attention and sequence length experiments. We use Prefix Global, TGlobal, and full attention at 1k, 2k, and 4k sequence lengths. Our experiments verify that Prefix Global is more performant than prior local-global attention TGlobal, and can even be more performant than full attention at long sequence lengths. Note that full attention at the 4k sequence length does not fit into memory. ROUGE-L is plotted.

<table><tr><td rowspan="2">Input Length</td><td colspan="3">Attention Mechanism</td></tr><tr><td>TGlobal</td><td>Prefix Global</td><td>Full</td></tr><tr><td>1024</td><td>325,632</td><td>916,480</td><td>1,048,576</td></tr><tr><td>2048</td><td>782,336</td><td>2,225,152</td><td>4,194,304</td></tr><tr><td>4096</td><td>2,088,960</td><td>4,842,496</td><td>16,777,2166</td></tr></table>

Table 4: The approximate number of FLOPs for each attention ignoring the # of attention heads and embedding dimension (both are the same for each attention). As sequence length increases, Prefix Global requires much fewer computations than full attention.

One anomalous result occurs for image captioning: Prefix Global with 256 globals (PrefixTGlobal at 4k input length) outperforms the 512 variant; as we did not exhaustively ablate the number of global tokens, further performance gains could be reached by optimizing the number of globals per task.

Prefix Global outperforms full attention at all sequence lengths on image captioning, which may be due to the global tokens including the target image and most relevant section content. This should ease the process of learning the most relevant tokens by allowing full attention between the first k target section tokens with the rest of the input sequence, while contextual information from other sections has local attention. For section summarization and page description, Prefix Global outperforms full attention at the 4k sequence length, while full attention cannot fit in memory. Given that the entire page's content can be useful for generating a page level description, it is sensible that full attention may perform better for smaller sequence lengths as it allows for attention between all input tokens.

Efficiency Comparison. Prefix Global can outperform full attention, while only requiring O((l — k) · r + k · l) attention complexity for k global tokens. When implementing the Prefix Global attention, we manually created tensors representing block sparsity to avoid computing the full cross attention. We provide the approximate number of FLOPs for each attention mechanism in Table 4 when ignoring the number of attention heads and embedding dimension. At the 2k input sequence length Prefix Global requires about half the FLOPs of full attention, and experimentally takes about half the time to complete the same experiment with all other settings fixed. The number of FLOPs of Prefix Global at 4k is just over those of full attention at the 2k input length, and is able to fit into memory and maximize performance for each task.

Lastly, the full attention and Prefix Global FLOP difference grows with sequence length. This can sometimes be seen experimentally: performance gaps are larger between full and Prefix Global for page description at 2k vs. 1k (0.20 vs. 0.09).

## 5.2 Feature Ablations

We investigate the role of each input feature with Prefix Global attention and fix sequence length to 1k. Starting with just the text available from webpage sections, we incrementally add section titles, indices and special tokens defining section structure (the struct column of Table 5), the captions of images within each section, and the images. Each input boosts performance8 except section structure which has mixed results; for multimodal experiments we include these extra tokens if they helped in the text-only experiments. This may be due to these extra tokens consuming global tokens in the prefix that otherwise could have been more useful.

<table><tr><td colspan="5">Feature Inputs</td><td colspan="3">Page Desc.</td><td colspan="3">Section Summ.</td><td colspan="3">Image Caption.</td></tr><tr><td>Text</td><td>Title</td><td>Struct</td><td>Caption</td><td>Image</td><td>B</td><td>R</td><td>C</td><td>B R</td><td>C</td><td>B</td><td>R</td><td></td><td>C</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>13.60</td><td>37.75</td><td>77.12</td><td>9.48</td><td>28.35</td><td>65.75</td><td>9.83</td><td>33.00</td><td>133.70</td></tr><tr><td>V</td><td>V</td><td></td><td></td><td></td><td>13.63</td><td>37.88</td><td>77.97</td><td>9.78 29.14</td><td>68.90</td><td></td><td>9.84</td><td>33.40</td><td>135.30</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>14.07</td><td>37.96</td><td>77.88</td><td>8.70 29.24</td><td>69.19</td><td></td><td>10.15</td><td>33.38</td><td>135.10</td></tr><tr><td></td><td></td><td></td><td>V</td><td></td><td>13.12</td><td>38.43</td><td>81.19</td><td>10.08</td><td>29.23</td><td>69.45</td><td>9.90</td><td>33.57</td><td>136.03</td></tr><tr><td>V</td><td>V</td><td>レ</td><td>V</td><td></td><td>13.22</td><td>38.38</td><td>81.38</td><td>9.51</td><td>29.22</td><td>69.24</td><td>10.03</td><td>33.69</td><td>137.07</td></tr><tr><td>V</td><td>V</td><td>V7</td><td></td><td>V</td><td>13.16</td><td>37.96</td><td>78.39</td><td>9.31</td><td>29.20</td><td>69.19</td><td>11.74</td><td>37.46</td><td>156.34</td></tr><tr><td></td><td>V</td><td></td><td>レ</td><td>V</td><td>14.00</td><td>38.50</td><td>81.49</td><td>10.12</td><td>29.43</td><td>69.89</td><td>11.84</td><td>37.69</td><td>158.19</td></tr></table>

Table 5: Feature ablations with WikiWeb2M. We ablate over the section body text, title, structure, captions, and images. Utilizing multimodal inputs results in the best performance for all tasks. We report BLEU-4 (B), ROUGE-L (R) and CIDEr (C) metrics

<table><tr><td rowspan=2 colspan=1>Task</td><td rowspan=2 colspan=1>Model</td><td rowspan=2 colspan=1>ViTData</td><td rowspan=1 colspan=1>Metric</td></tr><tr><td rowspan=1 colspan=1>B     R      C</td></tr><tr><td rowspan=2 colspan=1>PageDesc.</td><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>im21kJFT</td><td rowspan=1 colspan=1>14.0038.50 81.4913.2538.49 82.02</td></tr><tr><td rowspan=1 colspan=1>Large</td><td rowspan=1 colspan=1>im21kJFT</td><td rowspan=1 colspan=1>14.6739.63 88.9014.5639.56 88.48</td></tr><tr><td rowspan=2 colspan=1>SectionSumm.</td><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>im21kJFT</td><td rowspan=1 colspan=1>10.1229.43 69.8910.1529.40 70.03</td></tr><tr><td rowspan=1 colspan=1>Large</td><td rowspan=1 colspan=1>im21kJFT</td><td rowspan=1 colspan=1>11.1030.61 76.8711.2430.54 76.92</td></tr><tr><td rowspan=2 colspan=1>ImageCap.</td><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>im21kJFT</td><td rowspan=1 colspan=1>11.8437.69158.1911.6637.35156.01</td></tr><tr><td rowspan=1 colspan=1>Large</td><td rowspan=1 colspan=1>im21kJFT</td><td rowspan=1 colspan=1>12.5138.05162.3112.0837.33158.81</td></tr></table>

Table 6: Pretrained model checkpoint ablations. We vary the size of the T5 and ViT models (Base means both T5 and ViT are base-sized models, Large means both are large models) and which image dataset the ViT model was pretrained with (ImageNet or JFT-300M).

Images and their captions both improve performance, but result in the highest performance for each task when used in tandem. This illustrates that even when text captions are available, having their visual counterpart is beneficial. In Table 5, when we include captions for the image captioning task, it refers to context captions from other images in the page that never serve as target images. Interestingly, this boosts performance. We suspect contextual captions help the model to learn the style of captions we aim to generate.

## 5.3 Pretrained Checkpoint and Model Size

In Table 6, we perform additional experiments with ViT pretrained on JFT and large T5/ViT models. Unsurprisingly, larger models result in better performance. For page description and section summarization, scaling the model size results in larger performance gains than the impact of any individual feature we ablated. On the other hand, model size has smaller gains for image captioning compared to the impact of our feature ablations; the worst to best performance gap changed by an average of 17.66% for feature ablations and only by 2.43% for model size, where we average the performance delta of BLEU-4, ROUGE-L, and CIDEr.

Preference to ViT representations pretrained on JFT or ImageNet varies by task: section summarization tends to prefer JFT, while page description and image captioning consistently perform best with large ImageNet trained representations.

## 5.4 Comparison to WIT Annotations

The proposed WikiWeb2M is a superset of WIT. For the same set of webpages, we unify all sections into a webpage sample and reintroduce millions of sections and images that were not kept in WIT. Table 7 contains runs when using the original WIT data, the WIT data reprocessed to join the page sections it originally contained, and our WikiWeb2M.

For section summarization, the page description is more important than the other context sections. The page description may be more generally relevant to all sections, while each section to be summarized contains a distinct topic compared to the context sections from other parts of the webpage. Lastly, we find WikiWeb2M's additional context sections improve captioning performance the most compared to those already available in WIT (comparing the last two rows of Table 7). This confirms the importance of the new annotations in Wiki-Web2M compared to those available in prior work.

<table><tr><td rowspan="2">Task</td><td colspan="2">Input Section Type</td><td rowspan="2">Section Source</td><td colspan="3">Metric</td></tr><tr><td>Target Description Context</td><td></td><td>BLEU-4</td><td>ROUGE-L</td><td>CIDEr</td></tr><tr><td rowspan="3">Section Summarization</td><td>V</td><td></td><td rowspan="3"></td><td>8.90</td><td>27.82</td><td>60.20</td></tr><tr><td>V V</td><td></td><td>WikiWeb2M</td><td>9.46 28.86</td><td>66.67</td></tr><tr><td>V V</td><td>V</td><td>10.12</td><td>29.43</td><td>69.89</td></tr><tr><td rowspan="4">Image Captioning</td><td>V</td><td></td><td>WIT</td><td>10.92</td><td>36.21</td><td>148.53</td></tr><tr><td>V V</td><td></td><td>WIT</td><td>11.21</td><td>36.63</td><td>150.98</td></tr><tr><td>V V</td><td>V</td><td>WIT</td><td>11.45</td><td>36.88</td><td>152.69</td></tr><tr><td>V V</td><td>V</td><td>WikiWeb2M</td><td>11.84</td><td>37.69</td><td>158.19</td></tr></table>

Table 7: Section input ablations. We vary using the target section, page description, and context sections. For image captioning, we also vary whether the sections come from the smaller WIT or our WikiWeb2M superset - we do not run this ablation for section summarization, as it would result in a different number of train/val/test samples. Results show that using all section types and the annotations made newly available with WikiWeb2M improve performance.

## 6 Related Work

Webpage tasks have been studied with text only HTML for web element classification, HTML description generation, and web navigation. Gur et al. (2022) proposed finetuning Large Language Models for these tasks. Reinforcement Learning methods have also trained agents to perform language commands in handcrafted web environments (Gur et al., 2019; Liu et al., 2018; Jia et al., 2019).

Wikipedia has previously been used to develop downstream tasks. For example, WIT (Srinivasan et al., 2021) released image-caption pairs from Wikipedia, in addition to some contextual section text. While WIT does not contain all of the page content, Nguyen et al. (2022) studied contextual image captioning with the available annotations. This is a webpage task and not strictly an imagetext problem, as additional section text is included to aid in Wikipedia image captioning, where captions often contain finer-grained, knowledge based information. AToMiC also studied ways to improve multimodal retrieval with Wikipedia by defining more realistic evaluation sets (Yang et al., 2023).

Aghajanyan et al. (2022) proposed CM3, a Transformer with a causally masked pretraining objective. While CM3 relied on pretraining data from the web containing the images and HTML of a webpage, this dataset was not open sourced. Their results illustrated that rich HTML data could be used to learn representations for tasks such as image generation, image in-filling, and entity disambiguation and linking. This demonstrates that webpage data can generalize to non-webpage tasks, but leaves webpage specific problems unexplored.

To our knowledge there is no open source multimodal webpage data that captures all modalities. C4 was recently extended to a multimodal version, MMC4 (Zhu et al., 2023). However, MMC4 does not retain structure, and instead uses CLIP scores to noisily match images to chunks of text that it could be aligned with. MMC4 has not yet been used for pretraining or downstream applications. In mobile apps, the closest domain to webpages, there are two open source datasets that contain all modalities (text, image, and structure): Rico (Deka et al., 2017) and MoTIF (Burns et al., 2022).

## 7 Conclusion

In this paper we study three generative tasks for multimodal webpage understanding: page description generation, section summarization, and contextual image captioning. To do so, we present the WikiWeb2M dataset, which retains all of the text, images, and structure from more than 2M pages. We propose a new attention, Prefix Global, which outperforms full attention by allowing the most salient text and images to specially attend to all inputs. Extensive ablations on attention mechanism, sequence length, model size and checkpoint, input features and section type reveal the most impactful factors on our benchmark suite and verify using WikiWeb2M to study webpage understanding.

## Limitations

The WikiWeb2M dataset reprocessed the webpages available in WIT. We begin with only the English subset of WIT, while it originally contained 108 languages. Our dataset is limited to English and does not cover the vast multilingual data on Wikipedia. We can extend our dataset to cover all languages in WIT, but acknowledge it is monolingual to date.

For page description generation and section summarization, we use pseudo summaries that are readily available from Wikipedia pages. While this is desirable from a scalability perspective and is practiced in other works, it can limit the evaluation quality of these tasks. However, we did perform a small scale pilot to collect human annotations for the section summarization task in which we asked the annotators if the first sentence sufficed; 94% of the time the majority vote out of five was yes. Pseudo summaries have also been used for other tasks like summarizing instructional videos (Narasimhan et al., 2022).

For the model settings we explore, we did not try all exhaustive combinations of features, attention mechanism, model configuration, and input length. We also only use T5 variants, but note T5 is stateof-the-art for generation style problems. Lastly, we design our set of fine-tuning tasks for generative tasks. Our work currently does not include tasks like webpage taxonomy classification or webpage retrieval, but additional tasks like topic classification could be performed with WikiWeb2M.

## Ethics Statement

While the Internet provides a vast and rich domain to collect data from, it also has potential risks. Wikipedia is a highly curated and monitored knowledge base of articles, but it can be edited by the public, which can create potential quality risks. Additionally, Wikipedia is a largely fact-based domain, where incorrectly summarizing an article could result in misinformation. We hope our dataset can be used as a new resource to improve the accuracy and factual correctness of text generation machine learning models. As we use Wikipedia data, there is no user data nor P.I.I. in the proposed WikiWeb2M dataset. Additionally, we ran analysis to remove a small subset of pages with potentially sensitive topics (e.g., natural disasters, funeral, blood).

## Acknowledgements

This work is supported, in part, by the Google Ph.D.   
Fellowship program.

## References

Armen Aghajanyan, Bernie Huang, Candace Ross, Vladimir Karpukhin, Hu Xu, Naman Goyal, Dmytro Okhonko, Mandar Joshi, Gargi Ghosh, Mike Lewis, and Luke Zettlemoyer. 2022. Cm3: A causal masked multimodal model of the internet. arXiv:2201.07520.

Joshua Ainslie, Santiago Ontanon, Chris Alberti, Vaclav Cvicek, Zachary Fisher, Philip Pham, Anirudh Ravula, Sumit Sanghai, Qifan Wang, and Li Yang 2020. ETC: Encoding long and structured inputs in transformers. In Empirical Methods in Natural Language Processing (EMNLP).

Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. arXiv:2004.05150.

Ali Furkan Biten, Lluís Gómez, Marçal Rusiñol, and Dimosthenis Karatzas. 2019. Good news, everyone! context driven entity-aware captioning for news images. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Andrea Burns, Deniz Arsan, Sanjna Agrawal, Ranjitha Kumar, Kate Saenko, and Bryan A. Plummer. 2022. A dataset for interactive vision language navigation with unknown command feasibility. In European Conference on Computer Vision (ECCV).

Biplab Deka, Zifeng Huang, Chad Franzen, Joshua Hibschman, Daniel Afergan, Yang Li, Jeffrey Nichols, and Ranjitha Kumar. 2017. Rico: A mobile app dataset for building data-driven design applications. In User Interface Software and Technology (UIST).

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. ImageNet: A Large-Scale Hierarchical Image Database. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR).

Mandy Guo, Joshua Ainslie, David Uthus, Santiago Ontanon, Jianmo Ni, Yun-Hsuan Sung, and Yinfei Yang. 2022. LongT5: Efficient text-to-text transformer for long sequences. In Findings of the Association for Computational Linguistics: NAACL.

Izzeddin Gur, Ofir Nachum, Yingjie Miao, Mustafa Safdari, Austin Huang, Aakanksha Chowdhery, Sharan Narang, Noah Fiedel, and Aleksandra Faust. 2022. Understanding html with large language models. arXiv:2210.03945.

Izzeddin Gur, Ulrich Rueckert, Aleksandra Faust, and Dilek Hakkani-Tur. 2019. Learning to navigate the web. In International Conference on Learning Representations (ICLR).

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. CLIPScore: a referencefree evaluation metric for image captioning. In Empirical Methods in Natural Language Processing (EMNLP).

Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. 2015. Distilling the knowledge in a neural network. arXiv:1503.02531.

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, and Tom Duerig. 2021. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning (ICML).

Sheng Jia, Jamie Kiros, and Jimmy Ba. 2019. Domq-net: Grounded rl on structured language. In International Conference on Learning Representations (ICLR).

Chris Kedzie, Kathleen McKeown, and Hal Daumé III. 2018. Content selection in deep learning models of summarization. In Empirical Methods in Natural Language Processing (EMNLP).

Yang Li, Jiacong He, Xin Zhou, Yuan Zhang, and Jason Baldridge. 2020. Mapping natural language instructions to mobile UI action sequences. In Association for Computational Linguistics (ACL).

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Association for Computational Linguistics (ACL).

Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. 2018. Reinforcement learning on web interfaces using workflow-guided exploration. In International Conference on Learning Representations (ICLR).

Fuxiao Liu, Yinghan Wang, Tianlu Wang, and Vicente Ordonez. 2021. Visualnews: Benchmark and challenges in entity-aware image captioning. In Empirical Methods in Natural Language Processing (EMNLP).

Medhini Narasimhan, Arsha Nagrani, Chen Sun, Michael Rubinstein, Trevor Darrell, Anna Rohrbach, and Cordelia Schmid. 2022. Tl; dw? summarizing instructional videos with task relevance and crossmodal saliency. In European Conference on Computer Vision (ECCV).

Khanh Nguyen, Ali Furkan Biten, Andres Mafla, Lluis Gomez, and Dimosthenis Karatzas. 2022. Show, interpret and tell: Entity-aware contextualised image captioning in wikipedia. arXiv:2209.10474.

Daniel Nkemelu, Peggy Chi, Daniel Castro Chin, Krishna Srinivasan, and Irfan Essa. 2023. Automatic multi-path web story creation from a structural article. arXiv:2310.02383.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. BLEU: A method for automatic evaluation of machine translation. In Association for Computational Linguistics (ACL).

Luc Pauwels. 2012. A Multimodal Framework for Analyzing Websites as Cultural Expressions. In Journal of Computer-Mediated Communication (JCMC)

Amy Pu, Hyung Won Chung, Ankur P Parikh, Sebastian Gehrmann, and Thibault Sellam. 2021. Learning Compact Metrics for MT. In Empirical Methods of Natural Language Processing (EMNLP).

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. arXiv:2103.00020.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. In Journal of Machine Learning Research (JMLR).

Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. 2018. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Association for Computational Linguistics (ACL).

Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. 2015. World of bits: An open-domain platform for web-based agents. In 34th International Conference on Machine Learning (ICML).

Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. 2021. Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning. In International ACM Conference on Special Interest Group on Information Retrieval (SIGIR).

Reuben Tan, Bryan A. Plummer, Kate Saenko, J. P. Lewis, Avneesh Sud, and Thomas Leung. 2022. Newsstories: Illustrating articles with visual summaries. In European Conference on Computer Vision (ECCV).

Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. 2015. Cider: Consensus-based image description evaluation. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Alexandra Vtyurina, Adam Fourney, Meredith Ringel Morris, Leah Findlater, and Ryen W. White. 2019. Bridging screen readers and voice assistants for enhanced eyes-free web search. In International ACM Conference on Computers and Accessibility (AS-SETS).

Linzi Xing, Wen Xiao, and Giuseppe Carenini. 2021. Demoting the lead bias in news summarization via alternating adversarial learning. In Association for Computational Linguistics (ACL).

Jheng-Hong Yang, Carlos Lassance, Rafael Sampaio de Rezende, Krishna Srinivasan, Miriam Redi, Stéphane Clinchant, and Jimmy Lin. 2023. Atomic: An image/text retrieval test collection to support multimedia content creation. In International ACM Conference on Special Interest Group on Information Retrieval (SIGIR).

Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. 2020. Big bird: Transformers for longer sequences. In Advances in Neural Information Processing Systems (NeurIPS).

Chenguang Zhu, Ziyi Yang, Robert Gmyr, Michael Zeng, and Xuedong Huang. 2021. Leveraging lead bias for zero-shot abstractive news summarization. In International ACM Conference on Special Interest Group on Information Retrieval (SIGIR).

Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. 2023. Multimodal c4: An open, billion-scale corpus of images interleaved with text.

## A Additional Dataset Details

## A.1 Dataset Processing

We now provide additional details on the filters and data processing steps used to convert WikiWeb2M into our three downstream task datasets.

For page description, we retain a page from Wiki-Web2M if it is not list-heavy and contains at least two sections with image or text content that do not contain a list or table. A small subset of Wikipedia pages are essentially lists9; we consider pages that explicitly have “list\_of" in their URL to be list heavy-pages and remove them. We also remove pages with fewer than two rich sections to ensure there is enough content for a page description task to be appropriate.

For a page section to serve as a target section for the task of section summarization, we require it to have at least five sentences, contain neither a table nor list, and not be the root section. We filter out the root because the root (first) section is often the page description, which we choose to keep as a distinct task.

Lastly for image captioning, we follow Nguyen et al. (2022) and use the reference description as the ground truth caption to be generated. However, unlike Nguyen et al. (2022), we do not input the attribution description for the target image to be captioned because it often heavily overlaps with the reference description (the reference description is used as the target text to generate). We further discuss this design choice in Appendix E. Again, we only consider images that were originally in WIT as target images to be captioned to ensure quality captions. We also only keep target images to be images which have a reference description of at least three words.

We additionally note that in WikiWeb2M we release all three types of captions for each image (the alternative-text, reference description, and attribution description), although not all three are always available for each image. The alternative text caption for images is used for accessibility purposes, and future work can focus on generating these descriptions, as opposed to the reference description.

## A.2 Dataset Noise

Our dataset is built from the web, being processed from raw HTML. Noise may exist in our dataset in the formatting of text, e.g., mathematical formulas may have additional formatting text around them. In building our task datasets, the only noise we may introduce to the best of our knowledge is the processing of first sentences. The first sentence is separated from each section for the section summarization pseudo summary. It is also used as part of the global inputs for page description generation.

Specifically, the code used to parse the first sentence may prematurely split a sentence from a period that does not signal the end of the sentence. We did manually inspect samples early on and found this to be rare (e.g., 96/100 random samples were split correctly). Additionally, the sentences which were split prematurely could still be valid standalone sentences. Our released dataset does include all raw text, so others can reprocess it as they see fit. Figure 5 includes a code snippet for the sentence preprocessor we use and Table 8 illustrates the 4/100 prematurely split sentences from our random sample.

## A.3 Dataset Analysis

We provide a side by side comparison of the fields we open source with the WikiWeb2M dataset compared to those pre-existing in WIT in Table 9. Note that in addition to the new fields we introduce, WikiWeb2M has different data processing which allows for a great number of sections and images to be retained, as seen in Tables 10-12. In the main text, Table 1 provided the aggregate counts over all splits for each section type and the number of images in our dataset versus prior work WIT. Tables 10, 11, and 12 provide the same statistics but now broken down for train, validation, and test sets, respectively.

In Figure 6, two WikiWeb2M dataset samples are visually illustrated. Specifically, the webpages on the topics of Succulent Plant and the Aguas Livres Aqueduct are shown on the left of the figure to visualize the original webpage for illustration purposes, and on the right we show a subset of the fields available in our WikiWeb2M samples for these same pages.

For additional data analysis we provide sequence length details. We include the median, average, 90th percentile, and maximum sequence length values for all input fields that make up a sample's input sequence in the train split. We define sequence length as the number of tokens after preprocessing and SentencePiece tokenization. See Table 13 for the sequence length of the page URL, title, description, section title and text, and image captions available (the alt-text, attribution description, and reference description).

Lastly, we provide aggregate high level statistics over the train split of WikiWeb2M in Table 14. This includes statistics on the number of sections per page (the number of total sections as well as the number of content sections, with the latter referring to sections which contain image or text content) and the number of images per page or section.

## B Qualitative Examples

To qualitatively evaluate our model's ability on our suite of webpage understanding tasks, we include two qualitative analyses. First, we report several random output samples for page description generation, section summarization, and contextual image captioning in Tables 15-17. In Appendix B.1, we discuss our findings from these randomly sampled outputs. Next, in Appendix B.2, we include sample outputs whose metrics improved the most from the text-only model to the multimodal model, to explore why images can be helpful for webpage understanding tasks. In this second setting we investigate samples for page description and section summarization, since these tasks do not obviously require images in the same manner as contextual image captioning.

## B.1 General Qualitative Examples

We start with our first analysis of randomly sampled outputs for all three fine-tuning tasks. Samples are selected from the test set.

## B.1.1 Page Description Generation

Beginning with page description in Table 15, the target and predicted output text are provided for three random pages on the topics of the Horahora Power Station, Hedevig Lund, and Cape Nome.

For the first article on the Horahora Power Station, the predicted output text is quite coherent and well formed, despite containing some inaccurate details that conflict with the content of the webpage. Our model correctly references the date it was opened (1913) and the date the power station was flooded (1947). It also correctly references Lake Karapiro, which was formed and ultimately led to the submerging of the power station. On the other hand, the name of the “Waikato" River was swapped with “Waihi." The model also referred to Horahora as a coal-fired station when it is actually a hydroelectric power station.

Next, for the shorter article on Hedevig Lund, we find the model prediction to be very close to the target page description, although the painter's last names are slightly incorrect. Upon inspecting the page text, it appears the model included additional last names from the painter's parents' names (Ole Wilhelm Erichsen and Abel Marie née Isaachsen). In future work, methods that use pointer networks or direct copying from input text can be used to ameliorate these named entity failures.

alphabets = "([A-Za-z])"   
2 prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"   
3 suffixes = "(Inc|Ltd|Jr|Sr|Co)"   
starters ="(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"   
acronyms = "([A-Z][.][À-z][.](?:[A-Z][.])?)"   
websites = "[.](com|net|org|io|gov|me|edu)"   
7 digits = "([0-9])"   
8   
9 def PreprocessText(og\_text):   
10 text = " " + og\_text + "   
11 text = text.replace("\n", " ")   
12 text = re.sub(prefixes, "\\1<prd>", text)   
13 text = re.sub(websites, "<prd>\\1", text)   
14 text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)   
15 text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)   
16 text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)   
17 text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)   
1819 text = re.sub(alphabsts+.".alphabets".]”,1<1r \2pd", text)   
20 text = re.sub(" " + suffixes + "[.]", "\\1<prd>", text)   
13A450 text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)   
text = re.sub(r"(\d+)[.](\d+) ", "\\1<prd>\\2 ", text) # decimal numbers   
if "»" in text:   
text = text.replace("."", "".")   
if "\"" in text:   
02793033 text = text.replace(".\"", "\".")   
if "!" in text:   
text = text.replace("!\"", "\"!")   
if "?" in text:   
text = text.replace("?\"", "\"?")   
if "e.g." in text:   
33 text = text.replace("e.g.", "e<prd>g<prd>")   
34 if "i.e." in text:   
3536 text = text.replace("i.e.", "i<prd>e<prd>")   
if "..." in text:   
37 text = text.replace("...", "<prd><prd><prd>")   
389 if "Ph.D" in text:   
text = text.replace("Ph.D.", "Ph<prd>D<prd>")   
40   
41 text = text.replace(".", ".<stop>")   
42 text = text.replace("?", "?<stop>")   
43 text = text.replace("!", "!<stop>")   
44 text = text.replace("<prd>", ".")   
45 return text   
46   
47 def GetFirstRestSentences(og\_text):   
48 """Splits a body of text into its first sentence and the remaining text.   
49   
50 Code modified from   
51 https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences   
52   
53 Args:   
54 og\_text: The input text string.   
55 Returns:   
56 first: The first sentence of a body of text.   
57 rest: The remaining, rejoined sentences from a body of text that follow the first sentence.   
8 " u "   
60 text = PreprocessText(og\_text)   
61 sentences = text.split("<stop>")   
62 sentences = sentences[:-1]   
63 sentences = [s.strip() for s in sentences]   
64   
65 first = sentences[0]   
66 rest = og\_text[len(first)+1:]   
67 return first, rest  
Figure 5: The code used for sentence splitting. This preprocessing is used to separate the first sentence from the rest of the section body for section summarization targets and for page description generation global input tokens.

<table><tr><td rowspan=1 colspan=1>Full First Sentence</td><td rowspan=1 colspan=1>Processed First Sentence</td></tr><tr><td rowspan=1 colspan=1>In 2007, Saade was a founding member of What&#x27;sUp!, a Swedish boy band which included RobinStjernberg, Ludwig &quot;hejLudde&quot; Keijser and JohanYngvesson.</td><td rowspan=1 colspan=1>In 2007, Saade was a founding member of What&#x27;sUp!</td></tr><tr><td rowspan=1 colspan=1>Unicamp offers over one thousand extension pro-grams to the community, with different levels ofminimum requirements (high school degree, un-dergraduate degree, etc.) and across all areas ofstudy, focusing mainly on specialization coursesand community outreach.</td><td rowspan=1 colspan=1>Unicamp offers over one thousand extension pro-grams to the community, with different levels ofminimum requirements (high school degree, under-graduate degree, etc.</td></tr><tr><td rowspan=1 colspan=1>The proposed biosynthesis of ascofuranone wasreported by Kita et al., as well as by Abe et al.</td><td rowspan=1 colspan=1>The proposed biosynthesis of ascofuranone wasreported by Kita et al.</td></tr><tr><td rowspan=1 colspan=1>1988 The choir sang the German premiere ofJoseph Jongen&#x27;s Mass for choir, brass ensembleand organ, Op. 130, which was not yet in printthen, both in the Stiftskirche of Aschaffenburg andin St. Bonifatius.</td><td rowspan=1 colspan=1>1988 The choir sang the German premiere ofJoseph Jongen&#x27;s Mass for choir, brass ensembleand organ, Op.</td></tr></table>

Table 8: Failures of our sentence processor. We found 4/100 randomly sampled sections had first sentences which were prematurely split. We share these captions here and find they still are fairly reasonable sentences. The “full' first sentence on the left is determined via manual inspection.

The Cape Nome article is another example with a slightly longer page description (four sentences). This sample strongly illustrates the model's ability to convey a factually accurate and topically relevant page description even when the output text does not entirely contain the same content as the target description. Both the target and predicted descriptions begin with an overall summary of the topic, followed by geographical information. Our model’s generated text also provides some historical context that is accurately summarized from the article, which the ground truth description does not It seems our model attempts to summarize each of the sections on the page to form a coherent page description, which may differ from the target page description on Wikipedia (i.e., the Wikipedia page description need not cover topics from the entire webpage and can vary in style page to page).

## B.1.2 Section Summarization

For the task of section summarization, we include links to the webpage and the target section to be summarized from the article, and the target and predicted text in Table 16. Starting with the historical section on imageboards, we find that the target section summary is slightly more section specific than the predicted summary. I.e., the model generated summary “Futallaby is a free and open-source imageboard script" could be in sections other than the historical section. That being said, the historical section does discuss that Futallaby is freely available, making the model predictions sensible, relevant, and factually correct.

In the second section summarization example on the topic of Jamaica's pirate economy, the target summary discusses Spanish resistance to English occupancy to provide context for the growing pirate economy. However, neither the target nor predicted section summary directly address the pirate economy. The model prediction is mostly accurate with correct references to English occupancy in 1655, but implicitly refers to Port Royal as a fort at the foot of the Blue Mountains, which geographically, is slightly questionable.

The third section summarization sample concerns the science fantasy novel “Thuvia, Maid of Mars," written by Edgar Rice Burroughs. Our trained model correctly references Burroughs finishing a novel by June 1914, but it was Thuvia, Maid of Mars he finished, not the book “Tarzan." The model seems to have confused multiple facts relating to this book: Thuvia, Maid of Mars was the fourth, not third, novel in the Barsoom series and Tarzan was not a part of this novel series (although Burroughs did also write Tarzan).

<table><tr><td>WikiWeb2M Field</td><td>WIT Field</td></tr><tr><td>page_title</td><td>page_title</td></tr><tr><td>page_url raw_page_description</td><td>page_url context_page_description</td></tr><tr><td>section_title</td><td>section_title</td></tr><tr><td>section_text</td><td>context_section_description</td></tr><tr><td>section_image_url</td><td>image_url</td></tr><tr><td>section_image_mime_type</td><td>image_mime_type</td></tr><tr><td>section_image_width</td><td>original_width</td></tr><tr><td>section_image_height</td><td>original_height</td></tr><tr><td>section_image_raw_ref_desc</td><td>caption_reference_description</td></tr><tr><td>section_image_raw_attr_desc</td><td>caption_attribution_description</td></tr><tr><td>section_image_alt_text_desc</td><td>caption_alt_text_description</td></tr><tr><td>clean_page_description</td><td></td></tr><tr><td>section_image_clean_ref_desc</td><td></td></tr><tr><td>section_image_clean_attr_desc</td><td></td></tr><tr><td>section_image_captions</td><td></td></tr><tr><td>section_index</td><td></td></tr><tr><td>section_raw_1st_sentence</td><td></td></tr><tr><td>section clean 1st sentence</td><td></td></tr><tr><td>section_rest_sentence</td><td></td></tr><tr><td>section_depth</td><td></td></tr><tr><td>section_heading_level</td><td></td></tr><tr><td>section_subsection_index</td><td></td></tr><tr><td></td><td></td></tr><tr><td>section_parent_index</td><td></td></tr><tr><td>page_contains_images</td><td></td></tr><tr><td>section_contains_images</td><td></td></tr><tr><td>section_image_in_wit</td><td></td></tr><tr><td>split is_page_description_sample</td><td></td></tr><tr><td>page_content_sections_without_table_list</td><td></td></tr><tr><td></td><td></td></tr><tr><td>section_contains_table_or_list</td><td></td></tr><tr><td></td><td></td></tr><tr><td>is_section_summarization_sample</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>is_image_caption_sample</td><td></td></tr></table>

Table 9: Dataset fields in our WikiWeb2M dataset versus the WIT dataset.

## B.1.3 Contextual Image Captioning

Lastly, in Table 17, we provide qualitative examples for contextual image captioning. We include links to the webpage and image (as well as illustrate the image within the table), plus the target and predicted image captions. In the first image from the Longpré-le-Sec commune in France, while the target caption describes the main road in the image, a church is also present at the end of the road. This is confirmed by another image on the webpage which shows the same church. Thus, while our model did not predict the same exact target caption, it is still visually and factually accurate.

The second image is a photo of a painting of a woman. This image has a more generic target caption, and it appears that our model tends to prefer generating detailed captions. As a result, it contains factually inaccurate information, stating the painting itself is of Rabindranath, the son of Maharshi Devendranath Tagore, who developed Santiniketan. Additionally, the generated caption states the mural is at the Ashram Complex in Santiniketan. While the painting is at Santiniketan, it is not confirmed to be at the Ashram Complex given the content of the article; while the article states “It [the Ashram Complex] has beautiful frescoes by Nandalal Bose," it remains ambiguous.

Then, for the third randomly selected image from the article on WWOR-TV, the model's generated caption is quite accurate to the image and also overlaps heavily with the content of the ground truth caption. The only subtle inaccuracy in the predicted text is that it states the TV logo was in use from the early 1970s to early 1980s, when it was actually used until the year 1987, which should be considered the late 1980s.

<table><tr><td rowspan="2">Dataset</td><td colspan="6"># Webpage Sections</td><td colspan="2"># Images</td></tr><tr><td>Structural</td><td>Heading</td><td>Text</td><td>Image</td><td>Both</td><td>Total</td><td>Unique</td><td>Total</td></tr><tr><td>WIT (En)</td><td></td><td></td><td></td><td>179,769</td><td>2,562,275</td><td>2,742,044</td><td>3,183,132</td><td>4,456,169</td></tr><tr><td>WikiWeb2M</td><td>658,241</td><td></td><td>616,5346,134,086</td><td>199,165</td><td>52,911,268</td><td>10,519,294</td><td>3,867,2775,340,708</td><td></td></tr></table>

Table 10: Comparison of WikiWeb2M and the WIT (Srinivasan et al., 2021) dataset. We report counts here for the train split of WikiWeb2M.
<table><tr><td rowspan="2">Dataset</td><td colspan="6"># Webpage Sections</td><td rowspan="2"># Images</td></tr><tr><td>Structural</td><td>Heading</td><td>Text</td><td>Image</td><td>Both</td><td>Total Unique</td></tr><tr><td>WIT (En)</td><td></td><td></td><td></td><td>10,198</td><td>142,737</td><td>152,935</td><td>Total 238,155 249,313</td></tr><tr><td>WikiWeb2M</td><td>36,410</td><td>34,274</td><td>341,310</td><td>11,276</td><td>162,381</td><td>585,651</td><td>284,975 299,057</td></tr></table>

Table 11: Comparison of WikiWeb2M and the WIT (Srinivasan et al., 2021) dataset. We report counts here for the val split of WikiWeb2M.

## B.2 Unimodal to Multimodal Qualitative Examples

We now select a random subset of test set outputs for page description and section summarization for the best performing text-only model and the best performing multimodal (image and text) model. These models are base size T5/ViT models, as that was the model size used to perform feature ablations. Specifically, we select samples which have a higher ROUGE-L score with the multimodal model. For a random subset of 1000 samples, we reverse sort by the change in ROUGE-L between the unimodal and multimodal models, looking to inspect the samples most positively impacted by the inclusion of images. We hope to understand the settings under which images can aid these tasks.

As noted previously, we include this analysis for page description and section summarization, since these tasks may not require images, while image captioning inherently uses the input image. These examples are included in Tables 18 and 19. We did find that some of the most improved samples were an artifact of the text-only model repeating text tokens many times (a common failure of text generation models) and do not include those in our examples.

## B.2.1 Page Description

Starting with page description generation, we include three examples in Table 18. For each page we link to the Wikipedia article and include the target page description, the page description generated by the text-only model (noted as text under the type column), and the page description generated by the multimodal model (noted as multi under the type column) for comparison.

Across these examples, we find one trend: images in the webpage can improve the generated description's specificity. In the page description task, we allow up to six images present on the page to be included. Starting with the first example from the webpage on Joan Carling, we see that the page description output from the multimodal model touches upon more topics and specifies details beyond a high-level sentence summary (the latter being more similar to the output from the textonly model). The multimodal model's generated text includes references to the Champions of the Earth Lifetime Achievement Award that was given to Joan Carling as well as greater detail about her relationship with the Philippines and being of the Igorot people. These events and concepts are both captured in the images in the page, which seem to help specify more detail from the page in the generated description.

Similarly, for the second example from the page on Alice Bemis Taylor, the text-only model generated a brief and high level summary about her. On the other hand, the images on the webpage portray numerous important locations (either her childhood home or community spaces she actively participated in or founded such as the Colorado Springs Arts Center and Day Nursery). As a result of including these images, the generated description now included more specific information regarding these places. By including these images, their corresponding captions and related textual concepts in the input sequence are more greatly attended to. This may be a byproduct of images always serving as global tokens with our Prefix Global attention mechanism.

<table><tr><td rowspan="2">Dataset</td><td colspan="6"># Webpage Sections</td><td rowspan="2"># Images</td></tr><tr><td>Structural</td><td>Heading</td><td>Text</td><td>Image</td><td>Both Total</td><td>Unique Total</td></tr><tr><td>WIT (En)</td><td></td><td></td><td></td><td>9,905</td><td>142,917</td><td>152,822</td><td>238,924 250,353</td></tr><tr><td>WikiWeb2M</td><td>36,743</td><td>35,568</td><td>342,554</td><td>11,082</td><td>162,605</td><td>588,552</td><td>286,390 300,666</td></tr></table>

Table 12: Comparison of WikiWeb2M and the WIT (Srinivasan et al., 2021) dataset. We report counts here for the test split of WikiWeb2M.

<table><tr><td>Sequence Length</td><td>Med Avg</td><td>Max</td><td> $P _ { 9 0 }$ </td></tr><tr><td>Page URL</td><td>16 17.40</td><td>89</td><td>23</td></tr><tr><td>Page Title</td><td>5 5.20</td><td>42</td><td>8</td></tr><tr><td>Page Description</td><td>110 122.27</td><td>695</td><td>289</td></tr><tr><td>Section Title</td><td>3 3.64</td><td>176</td><td>7</td></tr><tr><td>Section Text</td><td>119 212.87</td><td>62,418</td><td>523</td></tr><tr><td>Image Alt-Text</td><td>4 5.58</td><td>1,433</td><td>14</td></tr><tr><td>Image Attribution</td><td>17 29.76</td><td>30,121</td><td>59</td></tr><tr><td>Image Reference</td><td>9 8.20</td><td>3,640</td><td>27</td></tr></table>

Table 13: Sequence length statistics for various Wiki-Web2M data fields. Sequence length is defined by the number of SentencePiece Tokens. We report the median, average, maximum, and 90th percentile sequence length values for the train split.

<table><tr><td>Statistic</td><td>Med Avg</td><td>Max</td><td>P90</td></tr><tr><td>Total Sections / Page</td><td>4</td><td>5.83 987</td><td>12</td></tr><tr><td>Content Sections / Page</td><td>4</td><td>5.13 939</td><td>11</td></tr><tr><td>Images / Page</td><td>1</td><td>2.96 1,679</td><td>6</td></tr><tr><td>Images / Section</td><td>0</td><td>0.51 653</td><td>1</td></tr></table>

Table 14: Analysis for the number of sections and images per page or section in the train split of the Wiki-Web2M dataset. Here we consider content sections as a section with text or image content within it (as in not only containing lists, tables, or heading only).

Lastly, we include the descriptions generated for the webpage on the topic of the Franco-German Parliamentary Assembly. Again this example demonstrates that images can help maintain the correct level of specificity for the page description. In this case, the images help refocus the description to be more high-level than the description generated from the text model, which includes too much detail (unlike the prior examples which tended towards too little). The two images on this page are centered around the respective meeting locations of the parliamentary in France and Germany, which helps focus the topic of the page description.

## B.2.2 Section Summarization

In Table 19 we now include section summarization examples for several pages and include links to the webpage and specific section to be summarized. Slightly different from the trend found for page description generation, we see that images can improve the topical relevance of the generated section summary.

For example, with the webpage on Johann Joachim Quantz, the target section summary for the section on the Court of Frederick discusses how Quantz joined the court as a flute teacher to Frederick II. The image in this section illustrates Frederick the Great playing flute, which directs the model to focus on this instrument. The text-only model fails to mention anything about instruments in the summary, only discussing interpersonal relationships. On the other hand, the multimodal model mentions both the flute and specifically Frederick playing the flute, which is depicted in the included image and also explains the reason why Quantz joined the court.

Next, the section on the 1800s history with respect to the St. Peter Catholic Church (in Montgomery, Alabama) continues to illustrate how images can improve topic relevance. While the text model references the year the church was founded, the multimodal model references the location, which the target summary also describes (although they reference the location of the church at different degrees of granularity). If a specific section does not have images, an image is included from the images in the chronological order they appear on the webpage. This means that for this section, the included image is the first image on the page - a map of Alabama showing the location of the St. Peter Catholic Church.

A third example can be found with the P&T

<table><tr><td>Webpage</td><td>Target Text</td><td>Predicted Text</td></tr><tr><td>Horahora Power Station</td><td>Horahora Power Station was an early hydro- Horahora Power Station was a coal-fired electric power station on the Waikato River in New Zealand. It was the country&#x27;s first large- scale power station, completed in 1913. Ini- tially built to service a gold mine, the power station was expanded to supply a significant part of the North Island. The power station remained in use until was submerged by Lake Karapiro, which was formed to supply the</td><td>power station located on the banks of the Waihi River in the Waikato region of New Zealand. It was built in 1913 and flooded in 1947 when Lake Karapiro was formed behind the Karapiro Power Station dam.</td></tr><tr><td>Hedevig Lund</td><td>larger Karapiro Power Station. Hedevig Thorine Christine Erichsen Lund was a Norwegian painter.</td><td>Hedevig Lund née Erichsen was a Norwegian painter.</td></tr><tr><td>Cape Nome</td><td>Cape Nome is a headland on the Seward Peninsula in the U.S. state of Alaska. It is sit- uated on the northern shore of Norton Sound, 15 miles to the east of Nome also on Norton Sound. It is delimited by the Norton Sound to the south, Hastings Creek on the west, a lagoon on the east and an estuary formed by the Flambeau River and the Eldorado River. From the sea shore, Cape Nome extends in- land by about 4 miles, connected by road with Nome.</td><td>Cape Nome is a headland on the Seward Peninsula of Alaska, United States. It is lo- cated 129 miles to the south east of the Bering Strait. The Cape Nome mining precinct was established in 1898 immediately after the dis- covery of gold at Golovnin Bay.</td></tr></table>

Table 15: Page description qualitative examples. We include three random samples from the page description test set and compare the target page description and predicted model output text.

Group article and 1900s-1950s Growth and Expansion section. The target section summary discusses the opening of the Shanghai office. The summary generated from the text-only model discusses the P&T Group expanding its business, which while correct, does not discuss the connection between opening new offices and expanding business. On the other hand, with the help of the HSBC Shanghai office image input to the multimodal, the multimodal model generated a description which explicitly mentioned the office opening. However, due to both Shanghai and Hong Kong locations being mentioned in the section, the model confuses the two in the generated section summary.

## C Additional Metrics

For results reported in the main paper, we additionally report BLEURT, CLIPScore, and RefCLIP-Score (the latter two are only relevant for contextual image captioning). See Tables 20, 21, 22. All results trends stay the same except that BLEURT is insensitive to most of the feature ablation results for section summarization and page description

generation.

## D Additional Model Ablations

We now provide additional experiments that could not fit into the main text, which include data processing ablations, additional feature ablations, and further comparisons to prior work.

## D.1 Page Description

We first performed ablations on the data processing of WikiWeb2M for the page description generation task dataset. Specifically, we tried varying the number of content sections required for a particular page to be retained. See Table 23 for comparison of when we required two vs. three vs. four sections to contain image or text content without a table or list. We found the added samples improve performance consistently (i.e., the most performant setting is when the number of content sections required per page is set to two).

<table><tr><td rowspan=1 colspan=1>Webpage</td><td rowspan=1 colspan=1>Section</td><td rowspan=1 colspan=1>Target Text</td><td rowspan=1 colspan=1>Predicted Text</td></tr><tr><td rowspan=1 colspan=1>Imageboard</td><td rowspan=1 colspan=1>Historical</td><td rowspan=1 colspan=1>Futallaby is a PHP script based onthe Futaba script from Futaba Chan-nel.</td><td rowspan=1 colspan=1>Futallaby is a free and open-sourceimageboard script.</td></tr><tr><td rowspan=1 colspan=1>Colony ofJamaica</td><td rowspan=1 colspan=1>Jamaica&#x27;sPirateEconomy</td><td rowspan=1 colspan=1>Spanish resistance continued forsome years after the English con-quest, in some cases with the helpof the Jamaican Maroons, but Spainnever succeeded in retaking the is-land.</td><td rowspan=1 colspan=1>The English occupied the island ofJamaica in 1655, establishing a fortat the foot of the Blue Mountains.</td></tr><tr><td rowspan=1 colspan=1>Thuvia,Maid of Mars</td><td rowspan=1 colspan=1>Background</td><td rowspan=1 colspan=1>Burroughs began writing Thuvia,Maid of Mars, in April 1914, at thetime describing it as a ‘Carthoris’story.</td><td rowspan=1 colspan=1>By June 1914, Burroughs had com-pleted the third novel in the Bar-soom series, Tarzan.</td></tr></table>

Table 16: Section summarization qualitative examples. We include three random samples from the section summarization test set and compare the target section summary and predicted model output text.

We also allow for text only pages to be kept in the dataset, as there are a small subset (roughly 2% of pages) that do not have any images after our processing. This could be due to the Wikipedia pages changing since the original WIT dataset was released, or because we only allow JPEG and PNG images while WIT contained some other image types like SVG. We include additional ablations in Table 24 showing the effect of including or not including these unimodal pages; their effect is minimal given how few there are in the dataset.

In Table 25, we show ablations for our prefix design with the page description generation task. Including the section titles and first sentences of each section in the prefix as global tokens improved performance for a majority of metrics, and we kept this set up for the rest of our experiments. We note that even when not using a specially designed prefix (i.e., flattening the section inputs and allowing the first 512 tokens to serve in the prefix, not separating out section titles or first sentences), the Prefix Global attention mechanism still outperforms Transient Global. This follows the principal from leading sentence bias that earlier information in the input text is more important. Thus, if you have a priori knowledge that a particular part of the input is more important than others, separating it into the prefix of our attention mechanism can be effective.

## D.2 Contextual Image Captioning

As image captioning inherently requires images, we performed additional feature ablations on the text features while always including the image (see rows 5-9 in Table 26). We verify in row 5 that when inputting only the image and no contextual text, it is incredibly difficult to generate strong captions for these images which contain a lot of fine-grained information. However, in support of the importance of having both images and text inputs, we find that for every text-only to multimodal comparison (where all features are the same except images are included in the latter), the multimodal setting always results in substantial performance gains. For example, quantitatively comparing row 1 and row 6 in Table 26, where either only the section text is input versus the section text and image to be captioned are input, the performance differences are: BLEU-4 9.83 vs. 11.27, ROUGE-L 33.00 vs. 36.90, and CIDEr 133.70 vs. 153.44.

Again, this differs from the findings of Nguyen et al. (2022); their experimental design likely minimized the impact of images because they also feed in the attribution description as a textual input, which often is quite similar to the target caption. As a result, the model can “cheat" and utilize the attribution description while not relying on the visual input. We have more discussion regarding this in Appendix E.

## D.3 All Tasks

For each task in our WikiWeb2M suite, we also ablated the number of images input to the model. These additional ablations are shown in Table 27. Results in the main text use the 90th percentile number of images per page, six, for page description generation, and only one for section summarization and image captioning. Here we also try the average value for number of images per page which is three. We include these ablations for contextual image captioning as well, as we were curious whether having contextual (non-target) images input to the model would help at all. Ultimately it sometimes hurt performance, likely adding noise and making it more challenging for the model to discern which input image was the target image to be captioned. We only used the single target image as input for the rest of our experiments.

## E Contextual Image Captioning Task Design

We now provide additional discussion on the task of contextual image captioning and how our input design differs from prior work. Nguyen et al. (2022) recently introduced the task of contextual image captioning. We found our metrics were lower than those they reported for the task and investigated the causes. We ran additional experiments for contextual image captioning with the exact same sample inputs as Nguyen et al. (2022). Specifically, we tried only using the page description, target image, target image's section text, and the attribution description as a sample's inputs.

By including the attribution description (which often heavily overlaps with the target caption to be generated), our performance is much higher, nearly matching prior work even when using different data splits (the prior work's dataset splits are not released). We report these reproduced results for our splits in Table 28. As discussed earlier, for our contextual image captioning task, we chose not to input the attribution description of an image given how much overlap it has with the target caption (the reference description). In terms of other experimental differences, we also use ViT (Dosovitskiy et al., 2021) image representations while prior work used ResNet-152 (He et al., 2016), although both were pretrained on ImageNet (Deng et al., 2009).

## F Section Summarization Pseudo Summaries

We were motivated to study the task of section summarization as a subproblem of Webpage Story Generation, which is the task of converting a webpage to an Instagram Story-like format. It consists of one multimodal Story page or slide per section, containing a section summary and paired image (from the same webpage). Our section summarization task is a subpart of this problem and we proposed an improvement over the News, textonly CNN/DailyMail PEGASUS model used to generate summaries in the prior Wiki2Story work by Nkemelu et al. (2023). Specifically, our formulation of multimodal section summarization is desirable so that we can also take images as contextual input, as the goal is to generate multimodal content for a user to consume on the topic of a particular webpage (in this case, a Wikipedia article).

Originally, we attempted to collect human written section summaries ourselves. But when running an initial data collection pilot we found that when explaining the intended application of webpage stories, a majority of the annotators deemed the first sentence to almost always (94% of the time) be a good enough pseudo summary. In the other cases when a majority of annotators voted otherwise, we found the annotation quality was too poor to use the collected written summaries. It proved very difficult to collect free form summaries of Wikipedia content. For both of these reasons we continued modeling section summarization with our pseudo summaries (the first sentence of the section).

To perform this data collection pilot, we used an internal crowd sourcing platform to hire seven crowd workers. They were located in India and paid hourly wages competitive for their locale. They have standard rights as contractors. The last four pages of our paper include a PDF of our instructions to annotators. We also tried to collect labels for well suited images for each section but ultimately did not use these annotations.

Rendered Webpage  
WikiWeb2M Sample Fields  
![](images/87e5c46affe99e32dfde0d5bb3c0bc2c419cfe7c96a95c48c0bd38d924f5e8d9.jpg)  
Figure 6: Example samples from WikiWeb2M. Here we illustrate two Wikipedia articles: Succulent Plant and Aguas Livres Aqueduct. The rendered webpage is illustrated on the left. On the right, a sample subset of fields are shown; these are stored in TensorFlow Sequence Example format.

<table><tr><td>Webpage</td><td>Image</td><td>Target Text</td><td>Predicted Text</td></tr><tr><td>Longpré -le-Sec</td><td><img src="images/bfe2a807a38fd6019bc028975703ea363ccc07acd7bebe062264f3de5c2359d0.jpg"/> Link to Image</td><td>The main road in Longpré- le-Sec.</td><td>The church in Longpré-le- Sec.</td></tr><tr><td>Santiniketan</td><td><img src="images/b77ba7ab85dd447191fbfd528fdd53c4f153f73254b7732d9bc25c57440eed83.jpg"/></td><td>Mural by Nandalal Bose.</td><td>A painting of Rabindranath at the Santiniketan Ashram.</td></tr><tr><td>WWOR-TV</td><td>Link to Image <img src="images/1e2926c9f2bca1fc32308c879c34811f63affde7b552d76a09a1a960b297556d.jpg"/></td><td>1971 WOR-TV I.D. slide. This dotted 9’ logo was used from 1970 to 1987.</td><td>WOR-TV logo, used from the early 1970s until the early 1980s.</td></tr></table>

Table 17: Contextual image captioning examples. We include three random samples from the image captioning test set and compare the target image caption and predicted model output text.

<table><tr><td rowspan=1 colspan=1>Webpage</td><td rowspan=1 colspan=1>Type</td><td rowspan=1 colspan=1>Text</td></tr><tr><td rowspan=3 colspan=1>JoanCarling</td><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>Joan Carling is an indigenous Filipino human rights activist and environmentalistwho has defended the rights of native and marginalized peoples for over twodecades. She has served as Secretary General of the Asia Indigenous PeoplesPact and has chaired the Cordillera People&#x27;s Alliance in the Philippines. Carlinghas also contributed to the United Nations Framework Convention on ClimateChange and REDD+ activities and has served as a member of the United NationsPermanent Forum on Indigenous Issues. In September 2018 she received theChampions of the Earth Lifetime Achievement Award from the United NationsEnvironment Programme in recognition of her work as an environmentalist anda defender of human rights.</td></tr><tr><td rowspan=1 colspan=1>Text</td><td rowspan=1 colspan=1>Joan Carling is a Filipino environmentalist and human rights activist. [ROUGE-L 13.11]</td></tr><tr><td rowspan=1 colspan=1>Multi</td><td rowspan=1 colspan=1>Joan Carling is a Filipino indigenous human rights activist and a member of theIgorot people. She is the former Secretary General of the Cordillera PeoplesAlliance, a national Indigenous Peoples&#x27; organization. She was the Co-convenerof the Indigenous Peoples Major Group on the Sustainable Development Goals.In 2018, she was designated a terrorist by the Philippine authorities for analleged connection with the Communist Party of the Philippines and the NewPeople&#x27;s Army. In September 2018, she received the Champions of the EarthLifetime Achievement Award from the United Nations Environment Programme.[ROUGE-L 39.61]</td></tr><tr><td rowspan=3 colspan=1>AliceBemisTaylor</td><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>Alice Bemis Taylor was a philanthropist and was inducted into the ColoradoWomen&#x27;s Hall of Fame in 2010. For her significant contributions to ColoradoCollege, Colorado Springs Fine Arts Center and the Colorado Springs DayNursery and other organizations, she was named “Lady Bountiful&quot; by the press.</td></tr><tr><td rowspan=1 colspan=1>Text</td><td rowspan=1 colspan=1>Alice Bemis Taylor was an American philanthropist and philanthropist.[ROUGE-L 21.05]</td></tr><tr><td rowspan=1 colspan=1>Multi</td><td rowspan=1 colspan=1>Alice Cogswell Bemis Taylor was an American philanthropist and philanthropist.She was the founder of the Colorado Springs Day Nursery, the Colorado SpringsFine Arts Center, the Child Guidance Clinic and the Colorado Springs Commu-nity Chest. She was inducted into the Colorado Women&#x27;s Hall of Fame in 2010.[ROUGE-L 45.36]</td></tr><tr><td rowspan=3 colspan=1>Franco-GermanParliamentaryAssembly</td><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>The Franco-German Parliamentary Assembly is a joint body of the GermanBundestag and the French National Assembly formed to enable cooperationbetween both houses.</td></tr><tr><td rowspan=1 colspan=1>Text</td><td rowspan=1 colspan=1>The Franco-German Parliamentary Assembly is an inter-parliamentary organisa-tion between the French and German parliaments. It was established in March2019 following the Aachen Treaty, which was signed on 22 January 2019 byAngela Merkel and Emmanuel Macron. [ROUGE-L 28.13]</td></tr><tr><td rowspan=1 colspan=1>Multi</td><td rowspan=1 colspan=1>The Franco-German Parliamentary Assembly is a joint parliamentary assemblyof the French National Assembly and the German Bundestag. [ROUGE-L 59.09]</td></tr></table>

Table 18: Page description qualitative examples. We include random samples from the page description test set that had the largest metric improvement with images included, and compare the target page description and predicted model output text

<table><tr><td rowspan=1 colspan=1>Webpage</td><td rowspan=1 colspan=1>Section</td><td rowspan=1 colspan=1>Type</td><td rowspan=1 colspan=1>Text</td></tr><tr><td rowspan=3 colspan=1>JohannJoachimQuantz</td><td rowspan=3 colspan=1>CourtofFrederick</td><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>When Frederick II became King of Prussia in 1740, Quantz finallyaccepted a position as flute teacher, flute maker and composer.</td></tr><tr><td rowspan=1 colspan=1>Text</td><td rowspan=1 colspan=1>Quantz was a friend of Frederick the Great, who was a close friendof his father. [ROUGE-L 10.81]</td></tr><tr><td rowspan=1 colspan=1>Multi</td><td rowspan=1 colspan=1>Quantz was a friend of Frederick the Great, who was a great lover ofthe flute. [ROUGE-L 16.22]</td></tr><tr><td rowspan=3 colspan=1>St. PeterCatholicChurch</td><td rowspan=3 colspan=1>1800s</td><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>In 1833, arrangements were made to build the church on a property,donated by Edward Hanrick, on the corner of Lawrence Street andAdams Avenue.</td></tr><tr><td rowspan=1 colspan=1>Text</td><td rowspan=1 colspan=1>St. Peter Catholic Church was founded in 1834, but had no residentpastor until 1850. [ROUGE-L 5.00]</td></tr><tr><td rowspan=1 colspan=1>Multi</td><td rowspan=1 colspan=1>St. Peter Catholic Church is the third oldest Catholic church inMontgomery, Alabama. [ROUGE-L 10.53]</td></tr><tr><td rowspan=3 colspan=1>P&amp;TGroup</td><td rowspan=3 colspan=1>1900s-1950s:Growth andExpansion</td><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>In 1920s, the Shanghai office was opened.</td></tr><tr><td rowspan=1 colspan=1>Text</td><td rowspan=1 colspan=1>In the early 1900s, the firm expanded its business.[ROUGE-L 25.00]</td></tr><tr><td rowspan=1 colspan=1>Multi</td><td rowspan=1 colspan=1>In 1905, the Hong Kong office was opened. [ROUGE-L 66.67]</td></tr></table>

Table 19: Section summarization qualitative examples. We include random samples from the section summarization test set which had the largest metric improvement and compare the target section summary and predicted model output text.

<table><tr><td colspan="4">Feature Inputs</td><td>Page Desc.</td><td>Section Summ.</td><td colspan="3">Image Caption.</td></tr><tr><td></td><td>Text Title Struct Caption Image</td><td></td><td></td><td>BLEURT</td><td>BLEURT</td><td>BLEURT</td><td>CLIPScore</td><td>Ref CLIPScore</td></tr><tr><td>V</td><td></td><td></td><td></td><td>0.51</td><td>0.43</td><td>0.36</td><td>0.6850</td><td>0.7146</td></tr><tr><td>V</td><td>V</td><td></td><td></td><td>0.51</td><td>0.44</td><td>0.36</td><td>0.6845</td><td>0.7153</td></tr><tr><td>V</td><td>V</td><td>V</td><td></td><td>0.51</td><td>0.45</td><td>0.37</td><td>0.6865</td><td>0.7166</td></tr><tr><td>V</td><td>V</td><td>V</td><td></td><td>0.51</td><td>0.45</td><td>0.37</td><td>0.6851</td><td>0.7154</td></tr><tr><td>V</td><td>V</td><td>V V</td><td></td><td>0.51</td><td>0.45</td><td>0.37</td><td>0.6878</td><td>0.7177</td></tr><tr><td>V</td><td>V</td><td>10</td><td>V</td><td>0.51</td><td>0.45</td><td>0.41</td><td>0.7340</td><td>0.7575</td></tr><tr><td>V</td><td>V</td><td>V</td><td>V</td><td>0.51</td><td>0.45</td><td>0.41</td><td>0.7329</td><td>0.7576</td></tr></table>

Table 20: Feature ablations with WikiWeb2M. We ablate over the section body text, title, structure, captions, and images. We report BLEURT, CLIPScore, and RefCLIPScore metrics for the results in Table 5.

<table><tr><td rowspan="2">Task</td><td>Input Section Type</td><td rowspan="2">Section Source</td><td colspan="3">Metric</td></tr><tr><td>Target Description Context</td><td>BLEURT</td><td>CLIPScore</td><td>RefCLIPScore</td></tr><tr><td rowspan="2">Section Summarization</td><td>V</td><td rowspan="2">WikiWeb2M</td><td>0.43</td><td>一</td><td></td></tr><tr><td>V V V V</td><td>0.44 0.45</td><td>一</td><td></td></tr><tr><td rowspan="4">Image Captioning</td><td>V</td><td>V WIT</td><td>0.40</td><td>0.7287</td><td>0.7527</td></tr><tr><td>V V</td><td>WIT</td><td>0.40</td><td>0.7307</td><td>0.7537</td></tr><tr><td></td><td>V WIT</td><td>0.40</td><td>0.7325</td><td></td></tr><tr><td>V V V V</td><td>WikiWeb2M</td><td>0.41</td><td>0.7329</td><td>0.7558 0.7576</td></tr></table>

Table 21: Section input ablations. We try using only the target section, both the target section and page description and/or context section(s), and vary if the sections come from the smaller WIT or our WikiWeb2M superset. We report BLEURT, CLIPScore, and RefCLIPScore metrics for the results in Table 7.

<table><tr><td rowspan=2 colspan=1>Task</td><td rowspan=2 colspan=1>Model</td><td rowspan=2 colspan=1>ViT Data</td><td rowspan=1 colspan=1>Metric</td></tr><tr><td rowspan=1 colspan=1>BLEURT CLIPScoreRefCLIPScore</td></tr><tr><td rowspan=2 colspan=1>PageDescription</td><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>im21kJFT</td><td rowspan=1 colspan=1>0.51          一0.51          一</td></tr><tr><td rowspan=1 colspan=1>Large</td><td rowspan=1 colspan=1>im21kJFT</td><td rowspan=1 colspan=1>0.52          一0.52          一</td></tr><tr><td rowspan=2 colspan=1>SectionSummarization</td><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>im21kJFT</td><td rowspan=1 colspan=1>0.45          一0.45          一</td></tr><tr><td rowspan=1 colspan=1>Large</td><td rowspan=1 colspan=1>im21kJFT</td><td rowspan=1 colspan=1>0.46          一               一0.46</td></tr><tr><td rowspan=2 colspan=1>ImageCaptioning</td><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>im21kJFT</td><td rowspan=1 colspan=1>0.41      0.7329        0.75760.40      0.7291        0.7534</td></tr><tr><td rowspan=1 colspan=1>Large</td><td rowspan=1 colspan=1>im21kJFT</td><td rowspan=1 colspan=1>0.41       0.7374        0.76110.41      0.7263        0.7527</td></tr></table>

Table 22: Pretrained model checkpoint ablations. We report BLEURT, CLIPScore, and RefCLIPScore metrics for the results in Table 6.

<table><tr><td colspan="2"># Content Section Filter Threshold</td><td colspan="4">Metric</td></tr><tr><td>Train</td><td>Test</td><td>BLEU-4 ROUGE-1</td><td>ROUGE-2</td><td>ROUGE-L</td><td>CIDEr</td></tr><tr><td>2</td><td>2</td><td>13.79</td><td>45.43</td><td>27.72</td><td>38.34 81.16</td></tr><tr><td>3</td><td>2</td><td>12.38</td><td>44.35 27.05</td><td>37.63</td><td>75.35</td></tr><tr><td>4</td><td>2</td><td>12.79</td><td>44.10</td><td>26.08 36.91</td><td>69.52</td></tr><tr><td>2</td><td>3</td><td>13.42</td><td>44.31</td><td>26.07</td><td>36.64 69.69</td></tr><tr><td>3</td><td>3</td><td>12.32</td><td>43.52</td><td>25.81 36.28</td><td>67.56</td></tr><tr><td>4</td><td>3</td><td>12.77</td><td>43.56</td><td>25.13 35.82</td><td>64.04</td></tr><tr><td>2</td><td>4</td><td>12.98</td><td>43.11</td><td>24.45</td><td>34.90 59.69</td></tr><tr><td>3</td><td>4</td><td>12.01</td><td>42.41</td><td>24.28 34.63</td><td>58.03</td></tr><tr><td>4</td><td>4</td><td>12.54</td><td>42.65</td><td>23.92</td><td>34.41 57.19</td></tr></table>

Table 23: Page description performance across different filtering thresholds. We change the threshold for how many rich content sections a page must have to be included in our page description generation task dataset.

<table><tr><td rowspan="2">Task</td><td colspan="2">Sample Modalities</td><td colspan="5">Metric</td></tr><tr><td>Train</td><td>Test</td><td>BLEU-4 ROUGE-1</td><td>ROUGE-2</td><td></td><td>ROUGE-L</td><td>CIDEr</td></tr><tr><td rowspan="4">Page Description</td><td>Multimodal</td><td>Combined</td><td>12.27</td><td>44.75</td><td>27.71</td><td>38.16</td><td>80.07</td></tr><tr><td>Combined</td><td>Combined</td><td>13.79</td><td>45.43</td><td>27.72</td><td>38.34</td><td>81.16</td></tr><tr><td>Multimodal</td><td>Multimodal</td><td>12.30</td><td>44.67</td><td>27.58</td><td>38.03</td><td>79.00</td></tr><tr><td>Combined</td><td>Multimodal</td><td>13.77</td><td>45.30</td><td>27.55</td><td>38.16</td><td>79.77</td></tr><tr><td rowspan="4">Section Summarization</td><td>Multimodal</td><td>Combined</td><td>9.83</td><td>34.96</td><td>15.18</td><td>29.64</td><td>70.82</td></tr><tr><td>Combined</td><td>Combined</td><td>9.93</td><td>34.86</td><td>15.16</td><td>29.53</td><td>70.86</td></tr><tr><td>Multimodal</td><td>Multimodal</td><td>9.74</td><td>34.89</td><td>15.10</td><td>29.56</td><td>70.22</td></tr><tr><td>Combined</td><td>Multimodal</td><td>9.84</td><td>34.78</td><td>15.06</td><td>29.43</td><td>70.17</td></tr></table>

Table 24: Comparison of only retaining multimodal page samples versus also allowing for text only pages (combined).

<table><tr><td colspan="6">Prefix Inputs</td><td colspan="2">Metric</td></tr><tr><td>Images</td><td>URL</td><td>Title</td><td>Title</td><td>Sentence</td><td>Page Page Section Section 1st All Section Content</td><td>BLEU-4 ROUGE-L</td><td>CIDEr</td></tr><tr><td>V</td><td>V</td><td>V</td><td></td><td></td><td>V</td><td>13.79 38.34</td><td>81.16</td></tr><tr><td>V</td><td>V</td><td>V</td><td>V</td><td></td><td></td><td>13.00 38.33</td><td>81.02</td></tr><tr><td>V</td><td>V</td><td>V</td><td>V</td><td>V</td><td></td><td>12.59</td><td>38.47 81.68</td></tr></table>

Table 25: Prefix Global prefix ablations for page description generation. The input column “all section content" refers to when all section indices, titles, body text, and captions are concatenated in order and then tokens contribute to the prefix up to the first 512 tokens

<table><tr><td colspan="5">Feature Inputs</td><td colspan="2">Metric</td></tr><tr><td>Text</td><td>Title</td><td>Struct</td><td>Caption</td><td>Image</td><td>BLEU-4 ROUGE-L</td><td>CIDEr</td></tr><tr><td>V</td><td></td><td></td><td></td><td></td><td>9.83 9.84</td><td>33.00 133.70</td></tr><tr><td>V</td><td>V</td><td>V V</td><td rowspan="4"></td><td rowspan="4"></td><td>33.40</td><td>135.30</td></tr><tr><td>V</td><td></td><td></td><td>10.15 33.38</td><td>135.10</td></tr><tr><td>V</td><td>V</td><td></td><td>10.03 33.69</td><td>137.07</td></tr><tr><td></td><td></td><td>V V</td><td>3.18 14.55</td><td>17.43</td></tr><tr><td>V V</td><td></td><td></td><td></td><td>V</td><td>11.27 36.90</td><td>153.44</td></tr><tr><td>√</td><td>V</td><td>V</td><td></td><td>V</td><td>11.64</td><td>37.39 156.27</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>11.74 37.46</td><td>156.34</td></tr><tr><td>V</td><td>V</td><td>V</td><td>V</td><td>V</td><td>11.84 37.69</td><td>158.19</td></tr></table>

Table 26: Additional feature ablations with WikiWeb2M for contextual image captioning. We ablate over the section body text, title, structure, captions, and images. We report BLEU-4, ROUGE-L and CIDEr metrics. We include rows already reported in the main text for ease of side by side comparison across all feature ablations.

<table><tr><td rowspan="2">Task</td><td rowspan="2">Number of Input Images</td><td colspan="2">Metric</td></tr><tr><td>BLEU-4 ROUGE-L</td><td>CIDEr</td></tr><tr><td rowspan="2">Page Description</td><td>1</td><td>13.44 38.52</td><td>81.52</td></tr><tr><td>3 6</td><td>13.55 14.00</td><td>38.46 82.00 38.50 81.49</td></tr><tr><td rowspan="2">Section Summarization</td><td>1</td><td>10.12</td><td>29.43 69.89</td></tr><tr><td>3 6</td><td>10.10</td><td>29.35 70.29</td></tr><tr><td rowspan="2">Image Captioning</td><td>1</td><td>9.67 11.84</td><td>29.37 70.29 37.69 158.19</td></tr><tr><td>3 6</td><td>11.92</td><td>37.41 157.27</td></tr></table>

Table 27: Ablations varying the number of input images per task.

<table><tr><td rowspan="2">Contextual Image Captioning</td><td rowspan="2">Split</td><td rowspan="2">Image Input</td><td colspan="4">Text Input</td><td colspan="3">Metric</td></tr><tr><td>Desc.</td><td>Target Context Section</td><td>Section</td><td>Attribution Desc.</td><td>B</td><td>R</td><td>C</td></tr><tr><td>Nguyen et al.</td><td>Unknown</td><td>ResNet</td><td>V</td><td>V</td><td></td><td>V</td><td>23.83</td><td>48.80</td><td>276.60</td></tr><tr><td rowspan="2">Ours</td><td rowspan="2">WikiWeb2M</td><td rowspan="2">ViT</td><td>V</td><td>V</td><td>V</td><td></td><td>11.84</td><td>37.69</td><td>158.19</td></tr><tr><td>V</td><td>V</td><td></td><td>V</td><td>25.20</td><td>50.01</td><td>242.50</td></tr></table>

Table 28: Experimental results when reproducing the task set up of Nguyen et al. (2022). We do not use the same dataset splits since they were not released in prior work. Here our set up uses the same sample inputs as prior work, unlike our result in the main text which does not input the attribution description.

# INSTRUCTIONS FOR ANNOTATORS

We are collecting annotations for a task called “Wiki2Story." Wiki2Story has so far been defined as a data conversion process from Wikipedia webpages to Wikipedia “stories."The goal is to turn a Wikipedia webpage into an Instagram-like story which contains one story page per Wikipedia section. Each story page includes a section summary and paired image; see the below examples. We provide two examples of what an Introduction and History story page may look like for these sections of the AppleWikipedia article.

![](images/79c9c7bad56276e50a93578a115870a1d172c075331470f70add0f9942afb13d.jpg)

![](images/c3f3de3963e0192b9c9c6bac7b1671e6af2d815bef7992e10fe156bb3c38ca14.jpg)

![](images/d482f73b3ad0b7cba9083ff33ad3119d2f374bde57a4472cd5802d7dd23224f5.jpg)  
The apple is thought to have been domesticated 4000-10000 years ago in the Tian Shan Mountains.

![](images/be72ccc6ae128d9709f99b73b1bd6a4ce2c2b65e8f3001e8c0667f45b8ae9ae8.jpg)

We want to obtain annotations of these section summaries and have you select the most appropriate image to pair with it. In particular, the section summary should be a highlight of the section content. The highlight should: be self contained, condense the section's factual information into a sentence of ideally fewer than 30 words, and retain enough detail for the reader to learn something from the story page. The highlight is supposed to be an educational glimpse of the full section's content, remaining fully true to the original text.

We provide examples below of both strong and weak summaries for our use case. Note that we expect the summary to contain only factually correct information from what is provided in the original section text. We provide weak summary examples to demonstrate ways the summary style can be incorrect.

## Example 1: The Proverb section of the Apple Wikipedia article.

## SECTION TEXT

The proverb, "An apple a day keeps the doctor away", addressing the supposed health benefits of the fruit, has been traced to 19th-century Wales, where the original phrase was "Eat an apple on going to bed, and you'll keep the doctor from earning his bread". In the 19th century and early 20th, the phrase evolved to "an apple a day, no doctor to pay" and "an apple a day sends the doctor away"; the phrasing now commonly used was first recorded in 1922. Despite the proverb, a 2015 study found no evidence that eating an apple daily prevents visits to a physician.

## STRONG SUMMARIES

The proverb “An apple a day keeps the doctor away” has been traced back to 19th-century Wales, but has yet to be proven scientifically.

“An apple a day keeps the doctor away” originated in Wales in the 19th century with the phrasing “Eat an apple on going to bed, and you'll keep the doctor from earning his bread."

The proverb “An apple a day keeps the doctor away” has had multiple phrasings over the years, first being traced to 19th-century Wales.

WEAK SUMMARIESThese are poor summaries because they use words like “this” or “it”, have a dialogue-like style, or overly abstract the factual information from the Wikipedia section.

This section talks about the phrase “An apple a day keeps the doctor away” and all of the ways it has been said.

It talks about how apples don't actually prevent doctor visits.

The apple proverb has existed for centuries.

## Example 2: The Breeds section of the Dog Wikipedia article.

## SECTION TEXT

Dogs are the most variable mammal on earth with around 450 globally recognized dog breeds. In the Victorian era, directed human selection developed the modern dog breeds, which resulted in a vast range of phenotypes. Most breeds were derived from small numbers of founders within the last 200 years, and since then dogs have undergone rapid phenotypic change and were formed into today's modern breeds due to artificial selection imposed by humans. The skull, body, and limb proportions vary significantly between breeds, with dogs displaying more phenotypic diversity than can be found within the entire order of carnivores. These breeds possess distinct traits related to morphology, which include body size, skull shape, tail phenotype, fur type and color. Their behavioral traits include guarding, herding, and hunting, retrieving, and scent detection. Their personality traits include hypersocial behavior, boldness, and aggression, which demonstrates the functional and behavioral diversity of dogs. As a result, present day dogs are the most abundant carnivore species and are dispersed around the world. The most striking example of this dispersal is that of the numerous modern breeds of European lineage during the Victorian era.

## STRONG SUMMARIES

Around 450 dog breeds have been globally recognized, making dogs the most variable mammal with significant differences in behavioral traits and physical characteristics.

The breeding of dogs during the Victorian Era resulted in around 450 globally recognized dog breeds from a small number of founders.

Dogs are the most variable mammal on Earth, having significant variance in skull, body, and limb proportions, also differing personality traits like sociability and boldness.

Present day dogs are the most abundant carnivore with around 450 recognized dog breeds.

With around 450 globally recognized breeds and high variance in both physical and behavioral characteristics, dogs display more phenotypic diversity than all other carnivores combined.

WEAK SUMMARIESThese are poor summaries because they are either too brief, reduce and abstract the factual information too much, or use words like “this” and “it."

There are many types of dogs on Earth.

This section discusses the difference behavioral, physical, and personality traits dog breeds can have.

It describes how dog breeding was done to result in 450 breeds and mentions that dogs are the most variable mammal.

When selecting images, we want you to choose the image you see best fit to go with the section content and summary text. It should be topically relevant and the image you feel is most visually appealing.

## UI PLUGIN EXAMPLE (UPDATED)

Read the below section from a Wikipedia page. The first sentence is highlighted in yellow. Does the first sentence provide a strong and concise (fewer than 30 words) summary of the contents of the entire section?

If you need additional context on this section's text and or topic to answer this question, you can click here to see the description of the Wikipedia page. Otherwise, continue with the annotation task.

## Section Title: Etymology

Wikipedia Page Title: Apple

The word apple, formerly spelled æppel in Old English, is derived from the Proto-Germanic root \*ap(a)laz, which could also mean fruit in general. This is ultimately derived from Proto-Indo-European \*ab(e)l-, but the precise original meaning and the relationship between both words[clarification needed] is uncertain.

As late as the 17th century, the word also functioned as a generic term for all fruit other than berries but including nuts—such as the 14th century Middle English word appel of paradis, meaning a banana. This use is analogous to the French language use of pomme.

Yes, it well summarizes the whole section

No, it does not well summarize the whole section

IF “HERE” WAS CLICKED ON FOR ADDITIONAL CONTEXT ABOVE, SHOW THEFOLLOWING PAGE DESCRIPTION TEXT:

An apple is an edible fruit produced by an apple tree (Malus domestica). Apple trees are cultivated worldwide and are the most widely grown species in the genus Malus. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found today. Apples have been grown for thousands of years in Asia and Europe and were brought to North America by European colonists. Apples have religious and mythological significance in many cultures, including Norse, Greek, and European Christian tradition.

IF THE ANSWER TO THE ABOVE QUESTION WAS NO, SHOW THE FOLLOWING SUMMARIZATION TASK:

Please write a single sentence summarizing the Wikipedia section. Keep the summary factually accurate to the original text and summarize the content concisely; try to use 15-30 words at most. The goal is to provide an educational, interesting snippet for a story page of this section.

Section Title: Etymology

Wikipedia Page Title: Apple