---
title: Word2vec and Negative Sampling
date: "2019-09-01T00:00:00.000Z"
template: "post"
draft: false
slug: "/posts/word2vec-and-negative-sampling/"
category: "Paper Summary"
tags:
  - "Paper Summary"
  - "Neural Networks"
  - "NLP"
description: "An overview of the popular word2vec model and methods of optimizing its training."
socialImage: ""
---

- [Introduction](#introduction)
- [Skip-gram Model](#skip-gram-model)
- [Negative Sampling](#negative-sampling)
- [Subsampling](#subsampling)
- [Evaluating Embedding Quality](#evaluating-embedding-quality)
- [Implementation](#implementation)
- [Applications](#applications)

## Introduction
Word2vec is a simple and elegant model for learning vector representations of words from text (sequences of words). At the core is the **distributional hypothesis**, which hypothesizes that words that frequently appear close to each other in text share similar meanings. For example, "apple" is more similar to "banana" than "boy" because ("apple", "banana") occur in the same sentence more frequently than ("apple", "boy"). Thus the word2vec vector representation of "apple" will be closer in distance to the vector representation of "banana" than the vector representation of "boy".

Given a text, we can represent this text as a sequence of words: $w_1, w_2, w_3,..., w_T$. Here, $T$ represents the total number of words. If we select an arbitrary target word $w_t$, we can define a context of size $c$. The $c$ words before ($w_{t-1}, w_{t-2},..., w_{t-c}$) and the $c$ words after ($w_{t+1}, w_{t+2},..., w_{t+c}$) $w_t$ are considered the context of nearby words.

<figure>
    <img src="/media/word2vec/Context Explanation.svg" />
    <figcaption>The words "restocked", "the", "and", and "pears" are in the context of the target word "apple"</figcaption>
</figure>

For the rest of this article, let $V$ be the size of the vocabulary (distinct words) in the text and 300 be the size of the vector representations (embeddings) we are trying to learn. Vector representations are considered "close" if their cosine distance is small.

## Skip-gram Model
For a single target word $w_t$ and a context of size $c$, we have $2c$ (input, output) pairs where the input is $w_t$ and the output is a word in the context. In the example above, the (input, output) pairs are:
- (apples, restocked)
- (apples, the)
- (apples, and)
- (apples, pears)

During training, for each (input, output) pair, we try to maximize the log likelihood as our objective function:

$$ 
J_{ML}=\log p(w_{o}|w_{i})
$$

<figure>
    <img src="/media/word2vec/Skip-gram Model.svg" />
    <figcaption></figcaption>
</figure>

In the model above, for each (input, output) pair, the input and output words are each represented as 1-hot vectors of size $1xV$. For example, if "apple" is the third word in the vocabulary, it's 1-hot vector is $[0, 0, 1, 0, 0, ..., 0, 0]$. The model has two sets of embeddings for each word, an input embedding and an output embedding (the input embeddings are usually taken to be the word2vec embeddings after training takes place). $E_i$ is a $Vx300$ weight matrix of input embeddings where row $k$ contains the input embedding for the $k$th word in the vocabulary. Similarly, $E_o$ is a $300xV$ weight matrix of output embeddings where column $k$ contains the output embedding for the $k$th word in the vocabulary.

The input 1-hot vector is first multiplied by $E_i$ to produce $e_i$, the 300 dimensional embedding for the input word (this operation essentially just selects the relevant embedding row from $E_i$):

$$
w_{i}E_{i}=e_{i}
$$

Next, the embedding $e_i$ is multiplied by $E_o$ to create $s_i$, a $V$ dimensional vector where each entry $k$ is the dot product between the input word's input embedding $e_i$ and the $k$th word in the vocabulary's output vector. Since dot product of closer vectors are higher, this value should be higher when the kth word is more similar to the input word. We now can calculate the probability of the output word given the input word by applying the softmax function to the output word's entry in $s_i$:

$$
p(w_{o}|w_{i})=\frac{\text{exp}(s_{i}[o])}{\sum_{k=1}^{V}\text{exp}(s_{i}[k])}
$$

$$
J_{ML}=\log p(w_{o}|w_{i})=\log \frac{\text{exp}(s_{i}[o])}{\sum_{k=1}^{V}\text{exp}(s_{i}[k])}
$$

Computing the gradient of the objective function for a single training example is computationally expensive because we need to do a number of calculations that is proportional to the vocabulary size $V$ (we need to update every word's output embedding). This brings us to a more clever and efficient model formulation, negative sampling.

## Negative Sampling
The idea of negative sampling is for each (input, output) pair, we sample $k$ negative (input, random) pairs from the unigram distribution (distribution of all words in the vocabulary). So now, given the same text, we suddenly have $k+1$ times as many input pairs as before. Continuing our last example and taking $k=2$, for the pair (apples, pears), we now have 3 training examples:
- (apples, pears) — real pair
- (apples, random word 1) — negative pair
- (apples, random word 2) — negative pair

During training, for each pair, we try to maximize an objective function that tries to differentiate real pairs from noise using logistic regression:

$$
J_{NS}=\log \sigma (e_{i}^{T}e_{o}) \text{ if positive pair}
$$

$$
J_{NS}=\log \sigma (-e_{i}^{T}e_{o})\text{ if negative pair}
$$

where $e_i$ is the input word's input embedding and $e_o$ is the output/random word's output embedding. Notice that in the objective function, the sign of the dot product between $e_i$ and $e_o$ is negative for negative pairs. The objective function for a negative pair achieves a higher value when the embeddings are very different (have a very negative dot product). This is because when we randomly sample a word to generate a negative (input, random) pair, we expect the random word not to be similar to the input word.

<figure>
    <img src="/media/word2vec/Negative Sampling.svg" />
    <figcaption></figcaption>
</figure>

The word2vec paper found that empirically, drawing negative samples from the unigram distribution raised to the $\frac{3}{4}$ power $U(w)^{\frac{3}{4}}/Z$ outperformed the unigram distribution $U(w)$. Intuitively, this flattens the $U(w)$ so that more frequent words are slightly underweighted compared to before and less frequent words are slightly overweighted compared to before.

## Subsampling
Subsampling of frequent words speeds up training and improves the vector representations of less frequent words. This is because we observe common words such as "the" so many times that we learn their embeddings very well and further training examples don't change their embeddings significantly. So it would be a better use of training time to prioritize the training examples of the less frequent words, so that we can learn good embeddings for them as well. The subsampling method used in the word2vec paper discards a training pair with target word $w_i$ (is this correct?) probability

$$
P(w_{i})=1-\sqrt{\frac{t}{f(w_{i})}}
$$

where $t$ is a chosen threshold and $f(w_i)$ is the frequency of word $i$. So 
$f(w_i) = \frac{\text{count of word i}}{\text{total number of words in text}}$.

## Evaluating Embedding Quality
How can we tell how good our vector representations are? Well for one, we can manually inspect them — words that are related or have similar meanings (e.g. apple and pear) should have vectors that are close in distance. In the word2vec paper, the word embeddings are evaluated using an analogical reasoning task. Analogies such as "Germany" : "Berlin" :: "France": ? are solved by finding the word having the vector closest to vec("Berlin") - vec("Germany") + vec("France"). This analogy would be successfully solved if the resulting word vector was vec("Paris").

## Implementation
[TensorFlow implementation](https://github.com/K-Niu/word2vec)

## Applications
Word2vec can be a way to featurize text that is more informative than simply word counts because it captures a degree of semantic meaning and relationship between words as well. The vectors produced by word2vec can be then be utilized as features for downstream natural language processing modeling tasks.

Another interesting and less obvious application of word2vec is in recommendating items to users based on interaction history. For example, we can consider a listening session of songs for a user to be a "text", where each "word" is a song. By applying word2vec, we can learn "song" vectors and recommend users new songs that are similar to (songs with vectors that are close) the ones they listen to.

## Resources
- Word2vec [paper](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (Mikolov et al.)
- Tensorflow [tutorial](https://www.tensorflow.org/tutorials/representation/word2vec)