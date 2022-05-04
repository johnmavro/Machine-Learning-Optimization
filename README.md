# Machine-Learning-Optimization

## Reference papers:

"Deep Neural Network Training with Frank-Wolfe", https://arxiv.org/pdf/2010.07243.pdf

"Deep Frank-Wolfe for Neural Network Optimization", https://arxiv.org/pdf/1811.07591.pdf

## Requirements

Torch, TorchVision, Numpy, Scipy, Pandas
A file requirements.txt will be added soon.

## Files

DFW.py contains the implementation of the Deep Frank Wolfe optimizer compatible with the Pytorch library, as presented in Algorithm 1 in  https://arxiv.org/pdf/1811.07591.pdf

SFW.py contains the implementation of the Stochastic Frank Wolfe optimizer compatible with the Pytorch library, as presented in Algorithm 1 in https://arxiv.org/pdf/2010.07243.pdf

The folder constraints contains constraints.py which implements the standard projection algorithm and the solutions to the oracle problem for SFW.

The folder utils contains support functions for the training of the models and loading of the datasets for the baseline tests.

## Results

The results are obtained on a MLP with 3 hidden layers and are used as a baseline

|  Opt. Method | Epochs | Test accuracy  
| ----- | ----- | ----- |
| SGD   |   10    | 65 %  | 
| Adam  |   10    | 95 %  | 
| DFW   |   10    | 96 %  |
