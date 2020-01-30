---
title: 2019 Data Science Bowl
date: "2020-01-26T00:00:00.000Z"
template: "post"
draft: false
slug: "/posts/2019-data-science-bowl/"
category: "Kaggle Summary"
tags:
  - "Kaggle Summary"
  - "Ordinal Regression"
  - "RNN"
  - "Gradient Boosting"
description: "Predicting a child's performance on an assessment based on an educational app's gameplay data - an ordinal regression problem."
socialImage: ""
---

- [Introduction](#introduction)
- [Evaluation Metric: Quadratic Weighted Kappa (QWK)](#evaluation-metric-quadratic-weighted-kappa-qwk)
- [Cross Validation](#cross-validation)
- [Ordinal Regression Models and Losses](#ordinal-regression-models-and-losses)
- [Models](#models)
  - [Gradient Boosting](#gradient-boosting)
  - [RNN](#rnn)
- [Results/Takeaways](#resultstakeaways)
- [Code](#code)
- [Final Ranking](#final-ranking)
- [Resources](#resources)


## Introduction
This year, Kaggle's annual Data Science Bowl focused in the area of education, with the challenge being to predict a child's performance on an assessment based on his/her interaction with *PBS KIDS Measure Up!*, a math game-based learning app on the iPad.

A child's journey through the app consists of a progression of multiple titles (**sessions**) that can be broadly classified into 4 **types**:
- Clips: short educational videos presenting mathematical concepts
- Activities: interactive ways for children to familiarize themselves with concepts
- Games: modules for children to put concepts into practice and that may contain feedback for correctness
- Assessments: modules containing a measure of success

Within each of these **sessions**, finer grained individual actions (**events**) such as clicks are also tracked.

<figure>
    <img src="/media/2019-data-science-bowl/Gameplay Data.svg" />
    <figcaption>Example of a child's gameplay data on the <i>PBS KIDS Measure Up!</i> app. At the most granular level, we have "events" (individual actions such as a click). Groups of consecutive events form high level aggregations called "sessions".</figcaption>
</figure>

Given a child's gameplay data, we would like to predict the performance of the child on a randomly selected assessment, where performance falls into one of the following 4 ordered categories:
- Group 3: the assessment was solved on the first attempt
- Group 2: the assessment was solved on the second attempt
- Group 1: the assessment was solved after 3 or more attempts
- Group 0: the assessment was never solved

Thus, this is an ordinal regression problem as we are trying to predict categories, but there is an ordering among the categories (Group 3 > Group 2 > Group 1 > Group 0).

## Evaluation Metric: Quadratic Weighted Kappa (QWK)
The quadratic weighted kappa score measures the agreement between two outcomes. Here, we have $N=4$ accuracy groups.

$$
\kappa = 1 - \frac{\Sigma_{i,j}w_{i,j}O_{i,j}}{\Sigma_{i,j}w_{i,j}E_{i,j}}
$$
where 
* $O_{i,j}$ (observed outcomes) is the number of users that are actually in accuracy group $i$ but are predicted to be in accuracy group $j$.
* $E_{i,j}$ (expected outcomes) is equal to the total number of users that are actually in accuracy group $i$ multiplied by the total number of users that are predicted to be in accuracy group $j$.
* $w_{i,j}=\frac{(i-j)^2}{(N-1)^2}$ is the penalty for incorrectly predicting a user, who is actually in accuracy group $i$, is in accuracy group $j$. We see that for correct predictions $i=j$, there is no penalty ($w_{i,i}=0$) and for further off predictions, there is a increasing quadratic penalty $w_{0,2}= \frac{4}{9} > w_{0,1} = \frac{1}{9}$. 

## Cross Validation
A significant component to this competition was constructing a robust cross validation scheme as the training data and the test data had different preprocessing methods applied to them. The training data and test data contained disjoint sets of users. However, the training data contained the entire gameplay history of users while the test data contained gameplay history that was truncated at the start event of a randomly selected assessment. Thus, the test data contained shorter gameplay histories.

<figure>
    <img src="/media/2019-data-science-bowl/Train vs Test.svg" />
    <figcaption>Example of a user gameplay history with 3 assessments (red). If the user appears in the training set, the whole history is included. However, if the user appears in the test set, there is an equal probability (1/3) that the user's truncated gameplay data will only include events up to the first, second, and third assessment start events.</figcaption>
</figure>

A lot of people found that their local cross validation QWK score was significantly off from their public test set QWK score. 

To combat this, a popular strategy for cross validation was to apply the same random truncating technique to the user gameplay data in the local validation set. This improved the difference between local cross validation QWK score and public test set QWK score, but the difference still tended not to be completely stable, perhaps due to the small size of typically used local validation sets. 

I opted to include all possible versions of a user's truncated gameplay history (so in the example image above, 3 versions) in my validation set and but weight each gameplay history's QWK contribution based on the number of assessments the user took (1/3 for the example above). This provided similar stability as the random truncated technique but was computationally more efficient.

## Ordinal Regression Models and Losses
Finding a model to deal with ordinal regression was another important choice. In the beginning, many people treated this problem as a classification problem, not taking into account the inate ordering of the accuracy groups and using a categorical cross entropy loss for their models. 

A big performance improvement happened when someone suggested to treat the problem as a regression problem, using a mean squared error loss (mirroring the quadratic weight penalty of QWK), and then finding thresholds based on the data distribution to separate the accuracy groups. 

Another proposal was to use a loss function that would optimize the QWK more directly, introduced in the paper *Weighted kappa loss function for multi-class classification of ordinal data in deep learning* (de la Torre et al.). 

After trying all of these, I found that the regression method with thresholding consistently gave the best performance.

## Models
### Gradient Boosting
A popular approach on Kaggle is to use gradient boosted models, so I tried LightGBM. These models aren't set up naturally to take into account temporal features, so many people crafted features that incorporated temporal information, such as average session time or past accuracy group counts. 

People went wild with engineering hundreds and hundreds of features. I did the same but attempted to reduce the number of feautures in my final model by applying feature selection techniques -- I removed one of each pair of highly correlated features and also features with high entropy.

### RNN
Because of the sequential/temporal nature of the gameplay data, I also tried a basic RNN with a couple sequence features and a nonsequence feature. Since the sequential nature of the data is hierarchical, an RNN can be trained on the high level sessions, or the lower level events. I only experimented with a session-based RNN.

<figure>
    <img src="/media/2019-data-science-bowl/Session RNN Architecture.svg" />
    <figcaption>Architecture for a basic session-based RNN.</figcaption>
</figure>

## Results/Takeaways
My LightGBM model performed better than my RNN. However, it also used many more features. If I had more time, I would like to have explored the RNN approach more deeply, as to me, it seems like a more elegant way to capture the temporal nature of the data. 

At the end of the competition, it was clear feature engineering was key -- many of the top finishers used some ensemble of gradient boosting and DNNs with a carefully selected but large number of features. In fact, the 1st place winner used a single LightGBM model, generated 20,000 candidate features, and used 500 of them. However, one of the top finishers did use a RNN with only a handful of features.

Also, just to emphasize how important a stable cross validation setup was for this competition, here is a plot of the shake-up in public leaderboard ranking to private leaderboard ranking at the end of the competition.

<figure>
    <img src="/media/2019-data-science-bowl/Leaderboard Shakeup.svg" />
</figure>

Many people had huge falls due to overfitting to the public leaderboard. But others had huge rises due to a stable cross validation setup.

## Code
A cleaned up version of the code I used is available [here](https://github.com/K-Niu/kaggle/dsb_bowl_2019). The gradient boosting (LightGBM) code is pretty generic, but the RNN code is a good starter of how to implement a RNN in Keras/Tensorflow using `tf.feature_column` and `tf.keras.experimental.SequenceFeatures` (the documentation was sparse and it took me a while to get it working!).


## Final Ranking
Top 17% (593/3497)

## Resources
- 2019 Data Science Bowl Competition [overview](https://www.kaggle.com/c/data-science-bowl-2019/overview) and [discussion](https://www.kaggle.com/c/data-science-bowl-2019/discussion)
- QWK loss for deep learning [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666) (de la Torre et al.)