---
author: Roman Castagné
title: "Artificial Pretraining of Masked Language Models"
date: 2023-02-14T00:00:00+02:00
math: true
---



According to Chinchilla [[1]](#references) scaling laws, data may soon be a bottleneck for training really large language models. In fact, it is already one for almost all languages except a few exception, including English.

Thus, during the last few months, we experimented with the following question : what if we could pretrain Masked Language Models (think BERT or RoBERTa) using only artificially created data, then continue pretraining on languages with very little resources e.g. Breton ? Our synthetic data should be cheap and fast to generate, and have properties that allow the models to “learn something”.

<img src="cover_image_artificial.png">

 

> Yes, but how will it know what “New York” is?
> 

Obviously, the intent behind this work is not to replace entirely data, but rather to see if we can teach the networks to **learn some simple operations** before having to cope with semantics and meaning. Similar work was done for automatic summarization [[2]](#references), showing that teaching a model to perform basic operations using synthetic data (e.g. copy one line) almost matches the performance of a fully pretrained model.

A secondary objective to this work is to understand what features are learnt from the training data by language models, and how much are these features used when transferring to another language. Previous works [[3]](#references) have shown that it was possible to pre-train a model on a different modality than text (for instance music) and still transfer with great performance. What exactly are the operations learnt that make this transfer possible ?


*This work was done as part of my PhD supervised by Benoît Sagot and Eric de la Clergerie, funded by the PRAIRIE chair of Benoît Sagot.*

# The Method

We first start by training our Masked Language Model on the synthetic data. For all our experiments, we worked with BERT-based models with 12 layers and hidden size 768 processing sequences of length maximum 128. We use batches containing 256 samples and train for 100k steps. This is a modest setup in terms of training but it enables us to experiment new ideas pretty quickly.

In a second part, we throw the embeddings (because our synthetic language has no vocabulary overlap with the target language) and retrain using the same setup on the target language. We usually stop training around 50k iterations because the models have already started overfitting before.

<img src="parts_trained.png">

We chose Breton as our low resource target language. Breton is one of the languages included in the Wikiann dataset which means that we have at least some finetuning data. At the same time, Breton is relatively low resource with around 5M tokens in the corresponding OSCAR subset.

We measure the models’ accuracy on an evaluation subset of the Breton training data and use compare it against the accuracy of a model trained from scratch on this data.

---

# What we tried

### The Baselines

Language model representations are surprisingly robust to new modalities and languages. Thus, previous works have for instance recycled models to create new ones in languages with less data [[4]](#references) or added languages to multilingual models after training [[5]](#references). 

Our baseline is similar. We use the same setting described earlier to train a model on English OSCAR data, then re-initialize its embeddings and retrain it on Breton data. This baseline performs particularly well, around ~ accuracy points above the model pretrained from scratch.

We use two other control baselines trained on different languages, Czech and Turkish. They belong to language families further from Breton, which should allow us to test how much language closeness accounts for in the transfer.

To our surprise, the three models (English, Czech and Turkish) transfer to Breton with the same accuracy.  

### Our methods

We try to identify a few key characteristics in natural data that should appear in our artificial language as well.

- **Structure**; natural data has a structure, whether it be images, text or sound. Since word embeddings are context independent, we hypothesize that the structure is mainly modelled with the Transformer layers and the positional embeddings and that we should be able to transfer it to the target language.
- **Co-occurrence**; this is a very vague term encompassing a lot of higher concepts such as meaning for instance. However, we think that it is important that the model learns to take context dependent decisions.

For the structure, we used dependency trees from the French GSD treebank. We used these trees to learn a Probabilistic Context Free Grammar (PCFG) from which we then sampled new trees, that we linearized to form sequences. This process yields sequences of tags corresponding each to a dependency relation. We then relexicalize these tags with items from a vocabulary. Although it may not look like it, our goal is rather to construct data generated from an underlying structure than “natural sounding sentences”. 

<img src="tree_generation.png">

Our first method (which we will call **PCFG+Zipf**) for relexicalization does not account for co-occurrences. We assign to each dependency relation a set of integers (whose size is computed from the real data) corresponding to tokens. We then relexicalize sequences of tags by drawing a token for each dependency relation from its set, using a Zipf law.


Inspired from Ri and Tsuruoka [[6]](#references) we also try another method (**PCFG+loglinear**) for linearization which aims at enforcing a notion of context. For each sequence of tags, we randomly draw a “topic vector” $v_t\in \mathbb{R}^d$. Every token from each set is assigned a random embedding vector $v_s \in \mathbb{R}^d$. Then, for each dependency relation, we sample a vocabulary element in its set $S$ of tokens according to the following distribution :

$$
p(w|t) = \frac{\exp(v_w \cdot v_t)}{\sum_{s\in S}\exp(v_s \cdot v_t)}
$$

Intuitively, the words that are similar according to their (random) embeddings will be often sampled together.

For both methods, we generate 1 million sentences, corresponding roughly to 15 million tokens.

---

# Results

We present in the following graph the masked language modeling evaluation accuracies obtained by each of the models on Breton. The x-axis corresponds to the source pretraining data. From scratch was initialized randomly before the target pretraining.

The models trained on synthetic data fail to significantly improve on a model trained from scratch on Breton. The models pretrained on natural language however gain a consistant advantage of ~3 accuracy points compared to pretraining from scratch. Adding a level of co-occurrence (the PCFG+loglinear model) did not seem to help the transfer.

<img src="accuracies.png">

In the following section, we look at a few questions we had during this work.

---

# Analysis

### How important are position embeddings in the transfer?

Two types of modules are transferred when pretraining on the target language, Breton : the Transformer layers and the positional embeddings. Although we reason mostly around the Transformer layers as encoding structure in our sentences, we may be diminishing the importance of good positional embeddings, trained on natural data.

To test this importance, we reset the positional embeddings at the beginning of pretraining on the target language, both for the models trained on natural language and the models trained on artificial data. 


<img src="pos_vs_no_pos_embeddings.png">



We can see on the previous graph that re-initializing the positional embeddings only delays training for a few steps, but does not impact the final performance. The Transformer layers are indeed responsible for most of the transfer.

### How different are the models trained with artificial data and natural language data?

Studying the standard deviations of parameters reveals an interesting phenomenon : despite being trained on the same number of steps, the models trained on artificial data have much lower standard deviations than the models trained on natural language data. 

### Is the artificial task learnt only in the embeddings?

After training a model on the artificial dataset, we take a look at the word embeddings obtained using a PCA to project them on a 2D space. Each point has a color corresponding to its dependency relation tag.

<img src="embeddings_pca.png">

As we can see, the embeddings are pretty well clustered by dependency relations. This is expected in part, because knowing precisely what relations surround a masked token is the only way to predict correctly the masked dependency relation. However, we want the model to rely on the Transformer layers rather than the embeddings to solve the task since we throw away the embeddings at the end of the artificial pretraining.

To enforce this behaviour, we retrain our model on artificial data but freeze the embeddings during pretraining, forcing the model to rely entirely on the position embeddings, Transformer layers and language modeling head. 

However, the transfer accuracy remains the same on Breton.

### Can we scale the size of our artificial dataset?

The synthetic dataset is relatively small compared to the language datasets we use, which may explain why we fail to transfer. However, increasing the dataset size from 15 million tokens to 744 million tokens did not improve the performance, neither during the artificial pretraining nor on the target pretraining.

This may be a clue towards what is missing from our dataset, since scaling a natural language dataset should result (until a certain limit) on improved performance. The task may be either too easy or too hard to solve, resulting in a plateau in the model performance.

---
<!-- 
# The Literature
This part should be very short, this is not a paper. Just a few pointers to important works.
--- -->

# Conclusion

Despite our best efforts to incorporate some key characteristics of natural language in our synthetic dataset (a structured way of generating sequences and a notion of co-occurrence of tokens), we were unable to induce biases in the Transformer network that would transfer to the pretraining on a target language with smaller resources. However, we believe this is rather due to our artificial language being ill-defined than the impossibility of transferring useful capabilities to Language Models without using real data, as previous works have shown such transfer for other tasks.

---

# References

[1] Training Compute-Optimal Large Language Models, *Hoffmann et al., 2022*. [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)

[2] Does Pretraining for Summarization Require Knowledge Transfer?, *Krishna et al., 2021*. [https://arxiv.org/abs/2109.04953](https://arxiv.org/abs/2109.04953)

[3] Learning Music Helps You Read: Using Transfer to Study Linguistic Structure in Language Models, *Papadimitriou and Jurafsky, 2020*. [https://arxiv.org/abs/2004.14601](https://arxiv.org/abs/2004.14601)

[4] Embedding Recycling for Language Models, *Saad-Falcon et al., 2022*. [https://arxiv.org/abs/2207.04993](https://arxiv.org/abs/2207.04993)

[5] When Being Unseen from mBERT is just the Beginning: Handling New Languages With Multilingual Language Models, *Muller et al., 2021*. [https://arxiv.org/abs/2010.12858](https://arxiv.org/abs/2010.12858)

[6] Pretraining with Artificial Language: Studying Transferable Knowledge in Language Models, *Ri and Tsuruoka, 2022*. [https://arxiv.org/abs/2203.10326](https://arxiv.org/abs/2203.10326)