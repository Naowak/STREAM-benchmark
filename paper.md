# Sequential Tasks Review to Evaluate Artificiel Memory 

The ability to maintain and manipulate information over time is a key aspect of Artificial Intelligence. Sequential models, such as Recurrent Neural Networks (RNNs) and Transformers [REF], have been widely used for tasks requiring working memory, such as natural language processing and time series prediction. However, evaluating the capacity of these models to maintain and manipulate information over long periods remains a challenge. Existing methods often focus on one or two specific tasks (e.g. SequentialMNIST [REF], copy task [REF], etc.), which may not be representative of the model's generalization capabilities.

In this paper, we propose a new benchmark called STREAM (Sequential Tasks Review to Evaluate Artificiel Memory) that aims to address this limitation by providing a diverse set of 12 tasks that test different aspects of working memory. By providing a standardized framework for evaluation, STREAM allows researchers to easily compare the performance of different model architectures.

In addition, we evaluate the performance of four different model architectures on the STREAM benchmark to serve as a baseline: Long Short-Term Memory (LSTM) [REF], Transformers [REF], Transformer-Decoder [REF], and Echo State Networks (ESN) [REF]. We compare the models' performance on each task and analyze their strengths and weaknesses.

## Introduction

Since the release of Transformers [REF] in 2017, no major breakthrought has been made in the field of neural networks architectures for Sequential Tasks. Even though Transformers have been proven to be very efficient for a wide range of tasks, they are not perfect and suffers from some limitations. For instance, Transformers aren't RNNs : they do not rely on previous internal state to compute the next one, they need to look at the whole input sequence to compute the next word at each timestep, and the number of computation grows quadratically with the length of the sequence. Which is a major limitation for long sequences and long-term dependencies.

For those reasons, a certain amount of researchers (like Yann LeCun [REF]) are convinced that the next step in Artificial Intelligence won't be made with current architectures like Transformers, but with new ones that might be inspired by existing architectures like LSTMs or ESNs or by our brain.

But, such models often needs to be scaled to be efficiant on complex problems (e.g. Natural Language Processing) and to be able to generalize on a wide range of tasks. And, it is not always easy to train and evaluate those scaled models : ressources like GPUs are expensives. Further, there can be a lot of different variation from a unique model (e.g. hyperparameters, initialization, etc.) that can lead to very different results, one can not afford to train and evaluate countless of scaled models.

We tried to tackle this problem by providing a set of 12 small but complex sequential tasks so one can train, evaluate and compare its new archiectures to existing ones without the need to scale them first. Each task is designed to be adjustable in difficulty and complexity, so that we can create different level of difficulty for the benchmark.

## Tasks

## Evaluated Models : Baseline

## Experimental Results

## Discussion

## Conclusion

## Annexes

## References
Transformers
LSTM
ESN
https://www.youtube.com/watch?v=5t1vTLU7s40

## Acknowledgements
JeanZay et Plafrim