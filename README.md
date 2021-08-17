# Private Domain Topic Representation Training
==============================================

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
The topic embeddings $z_t$ are initialized with prompt-assistant masked LM prediction from its related document contexts.
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
During the data-loading stage of each epoch, system will dynamically discover all topics occurred in the sentences, and assign topic embeddings into reserved 1000 empty slots.
When each epoch finished, topic embeddings are swapped back to external storage.

## Experiment


## Conclusion