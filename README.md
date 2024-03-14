# Bayesian-ML-Averaging-Weights-Leads-to-Wider-Optima-and-Better-Generalization


## Overview
This repository contains our implementation and analysis of the Stochastic Weight Averaging (SWA) method, an innovative optimization strategy that moves away from the traditional paradigm of minimizing the loss function through a straightforward path. By averaging the weights obtained through Stochastic Gradient Descent (SGD), SWA facilitates highly efficient fine-tuning of model parameters. This approach is likened to fast ensembling techniques known for their effectiveness in various contexts, such as Kaggle competitions, without the drawback of requiring excessive weight storage.

## Motivation
Our goal is to explore SWA's capabilities, understand the authors' intuitions, and assess its effectiveness through practical implementation and comparison with its recent iteration, Periodic SWA.

## Implementation
We have reimplemented the SWA algorithm in a controlled setting, allowing for a precise examination of its behavior and performance implications. This section of the repository documents the code and the steps taken in our implementation process, including modifications made to adapt the algorithm for our specific case study.



## REFERENCES

1. Guo, L., Jin, S. (2023). "Stochastic Weight Averaging Revisited." Applied Sciences. Ed. Sangtae Ahn.

2. Keskar, N.S., Mudigere, D., Nocedal, J., Smelyanskiy, M., Tang, P.T.P. "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.".

3. Wilson, A.G., Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D. "Averaging Weights Leads to Wider Optima and Better Generalization." 
