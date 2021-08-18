Private Domain Topic Representation Training
============================================

## Introduction
In the enterprise private domain knowledge management setting, topic representation is critical for knowledge conflation, retrieval and related items discovery.
Topics are the main entities in the knowledge graph, under which there are attributes like definitions, related people and related documents.
Topic representation should accurately reflect the topic's relationship with others in semantic space even when they have ambiguous names. It is important to enforce topic representations are consistent with all other word token representations.

For topic conflation tasks, conflation algorithm could avoid topic over merge by comparing the similarity of their embeddings, while in retrieval tasks, K-NN algorithm can effectively discover topics by using query context.
When documents and people in the corpus also consistently encoded with the representation in the same topic representation space, they can attach to related topics based on the embedding distance as well.

### Related Works
There are in general 2 ways to generate topic embeddings - relation based and content based.

The relation based approach generates topic embeddings based on the topic relations with its neighbor entities and attributes.
For example, we can build bi-part graph for topic and people entities, and each topic can be represented by its neighbor people.
The representation can be learnt independent of the topic content or any other prior knowledge, but the accuracy of the learnt representation largely depends on the density and quality of its attributes, and it is especially challenge during cold start.
The content based approaches takes the definition and surrounding context of the topic for generating its embedding.
It is usually more flexible and information rich, but require focused language model training on the target domain.

This work based on the content representation learning, but requires much fewer training samples and no human labeling.

### Challenges
Private domain topic representation learning majorly poses following 3 challenges:
1. Large volume of topics with various length of tokens needs to be jointly optimized for generating fixed size embeddings, under the same embedding space of word tokens.
2. Content summarization if needed for generating topic embeddings from their related documents, given topic definition text may be not widely available on all topics. Large domain refined language model shall be used for quality summarization.
3. Topic representation training and language model refinement may be impossible on small tenant with very limited documents or user labels.

## Approaches
In our method, we use pre-trained language model (BERT) to generate topic representation in private domain by joint masked LM continues training and topic embedding training.
The masked LM with topic are predicating probabilities of following distribution:

$$ P(x_m | x_i, x_t, z_t) $$

where $x_m$ represents all masked tokens, $x_t$ represents unmasked topic tokens, $x_i$ represents all unmasked normal tokens, and $z_t$ represents the fixed length topic embedding associated with topic tokens $x_t$.
There could be multiple tokens $x_t$ maps to a single topic embedding $z_t$.

The fixed size topic embedding were assigned with a position embedding very close to the topic tokens $x_t$ to take advantage of the sub-word information, while the topic embedding $z_t$ itself carries over in-ambiguously in multiple documents.
Our observations in the preliminary experiment supports the claim of single token fixed-size topic embedding reduces topic ambiguities while also captures certain sub-word information.

There are also different design choices about how to encode the document context into the topic.
One obvious alternative is to use the "Sentence-BERT" to encode the topic definition or the summary of the topic related documents.
The idea is to capture the topic semantic information by encoding its definition or related documents, but less practical than our approach because:
1. topic definitions are not always available;
2. related documents summary may not directly support the topic but containing other information that may reduce the accuracy of the topic representation;
3. not be able to continue training the model to improve the topic representation.

### Prompt based Topic Token Embedded LM
The topic embeddings $z_t$ are initialized with prompt-assistant masked LM topic prediction from its related document contexts.
Prompt text " ( aka \[MASK\]) " is inserted immediately after the topic tokens in the context of the document.
For example, for topic "Who Knows What", we will generate following annotated sentence:

> Viva topics utilize multiple sources of topics from Alexandria, Yukon and Who Knows What **(aka \[MASK\])** for building the knowledge base.

The mask token after prompt "( aka )" pulls attentions from the multi-tokens topic words and its surrounding context text.
By averaging all the mask token predictions for the topic in different documents, it generates topic embedding initialization.

This method naturally initialize the topic embedding into the pre-trained word token embedding space, with such, continues domain specific topic infused masked LM fine-tuning become possible.
However, based on our experiments, this initialization approach is very effective even without continues fine tuning, which makes it possible to directly generate topic embeddings on small private domain settings, where personalized training is in-practical.

### Dynamic Topic Embedding Swap
While topic embeddings are not picked from a fixed close token set, it is still desirable to keep the model size (word embeddings and token decoders) fixed, to minimize the complexity impact brought by the fast increasing topic counts.
It is important for GPU based accelerations as VRAM size is limited.
We developed a technique called "dynamic embedding swap" that supports dynamically expanding the topic embeddings while keep the model fix-sized.
It assumes there are limited number of topics appears in each training epoch, though the total number of the topics across epochs are unbounded.

Our model reserves 1000 empty slots in word embedding and decoder layers to support dynamic embeddings loading in the middle of the training epochs.
During the data-loading stage of each epoch, system will dynamically discover all topics occurred in the sentences, and load previously initialized topic embeddings into reserved 1000 empty slots in the embedding matrix and decoder matrix.
At the end of each epoch, topic embeddings are again swapped back to external storage for future reference.
As the ADEM optimizer is stateful against parameters, it is necessary to re-initialize the optimizer before each new epoch of training.

Based on our observation, the dynamic embedding swap process does not cause training accuracy deterioration while managing the fixed size model parameters loaded in the VRAM.

## Experiment
We experimented the topic representation generation over small subset of the stack-overflow domain documents, and evaluated the performance on the related topic task.

### Experiment Setup
We downloaded and down-sampled 1% of the achieved stack-overflow data, with questions, answers as well as tags, and only the best answer for each question is kept.
We use tags occurred in more than 500 of the questions and answers as topics to extract semantic representations (embeddings).
Google news pretrained case-folded small BERT model is used to perform prompt assisted masked LM topic prediction as well as further joint embedding fine tuning based on the stack-overflow data.
To enable dynamic topic embedding swapping, we take advantage of the 1000 unused token in the BERT model, which saves additional effort of expanding the BERT model structure.

For masked topic prediction, we use the BERT mask token output right before token decoder layer as the single instance of raw topic embedding.
The embedding vector is then normalized into the average *norm2* size of all other tokens in the BERT vocabulary, and gets averaged over all instances of the documents.
We directly take the average bias of BERT vocabulary as the topic decoding bias.
During the fine tuning stage, we set the batch size to be 64 paragraphs, and we limits 5000 paragraph and 1000 unique topics in each epoch.

We experimented 2 different settings on the masked LM training:
1. set the prompt text and the artificial topic token as token type 1, and avoid predicting them in the final loss computation;
2. treat prompt text and artificial topic token as normal token type 0, and predict them in the final loss computation just like any other tokens.
The second approach gets slightly better result in our experiments.

We setup 3 different topic embedding generation approaches for comparison:
1. baseline: use domain fine-tuned BERT model directly generate embeddings for every tokens in each topic, and average them over the topic;
2. prediction: use pretrained BERT model to predict the topic embedding with prompt based approach (covering 3 epochs);
3. fine-tuned: use #2 as the initialization of the topic embeddings, and perform joint masked LM training (covering 3 epochs).

SBS Surplus score and P-value are computed between 1-2 and 2-3 comparison.

### Experiment Preliminary Result
We randomly pick 50 topics from the generated related topic list from each experiments, and assign SBS scores.
Win or Loss were assigned if right or left (respectively) side have equal or more than 2 topics in the top 5 positions are significantly better than the other side, and we assign 2 points.
WeakWin or WeakLoss were assigned if the right or left (respectively) side have less than 2 topics in the topic 5 positions are significantly better than the other side, and we assign 1 point.
For all other cases, we assign the result as neutral with 0 credit.

 Metrics  | Exp 1 - 2 | Exp 2 - 3
----------|-----------|-----------
Neutral   | 11        | 32
Win       | 20        | 1
Loss      | 1         | 0
WeakWin   | 12        | 11
WeakLoss  | 6         | 6
Surplus   | 0.88      | 0.14
P-Value   | < 0.0001  | 0.026

With the limitation of the computing resource during FHL, we only finished the experiment 3 on 3 epochs training (less than 0.03% of the stack overflow dataset).
Nevertheless, early observations already shown the trends of improvements over experiment 2 with prediction only approach.
To view the complete list of the result topic output result, please refer to the files in the result folder.

## Conclusion
We proposed a novel approach to generate topic embeddings in small private domains.
Topic embeddings are initialized by prompt assisted masked LM with a small documents set, and then jointly fine tuned in the domain with all other token embeddings.
Our preliminary experiments successfully demonstrated the effectiveness of the prompt based initialization, and the improvements of fine tuning even with extremely small set of the data.

For future explorations, it is interested to see the embedding performance under different fine tuning tasks, e.g. contrastive training based on topics occurs in the same or different documents.
More experiments with different NLP tasks are also needed to verify the performance of the topic embeddings.