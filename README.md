# Optimization for Machine Learning Project
This repository contains the implementation of the Block Coordinate Descent and of the Deep Frank Wolfe algorithm in Pytorch. The work is inspired by the two papers [Deep Frank-Wolfe for Neural Network Optimization](https://arxiv.org/pdf/1811.07591.pdf) and [Global Convergence of Block Coordinate Descent in Deep Learning](https://arxiv.org/pdf/1803.00225.pdf).

## Structure
* `Frank_Wolfe` - Folder containing the implementation of the Deep Frank-Wolfe Algorithm
  * `utils` - Contains utilities to load data-sets, normalize batches and other tasks
  * `architectures.py` - Contains the implementation of architectures used for the empirical experiments
  * `MultiClassHingeLoss.py` - Contains the implementation of the multi-class Hinge Loss as described in the original DFW paper
  * `DFW.py` - Implementation of the DFW optimizer in Pytorch
* `results` - Contains the Python dictionaries with the results
*  `DFW.ipynb` - Notebook for reproducibility of the results concerning Deep Frank-Wolfe 

## Requirements

## Reproducibility
- [Colab Notebook for Deep Frank-Wolfe](https://colab.research.google.com/drive/1mpsunyV-11yDXPhZLznryLxJoMx4Zqxd)
- [Colab Notebook for Block Coordinate Descent](https://colab.research.google.com/drive/1mpsunyV-11yDXPhZLznryLxJoMx4Zqxd) TODO

## Example of usage
* Deep Frank-Wolfe Algorithm
```python
eta = 0.1  # proximal coefficient
momentum = 0.9  # momentum parameter
optimizer = DFW(model.parameters(), eta=eta, momentum=momentum,
                prox_steps=2)  # define optimizer

optimizer.zero_grad()
output = model.train()(x)
loss = loss_criterion(y, output)
loss.backward()
optimizer.step(lambda: float(loss), model, x, y)  # needs to have access to the loss and the model
```

## Results

### Coordinate Descent

### Frank Wolfe
| GoogLeNet |      |
| ----- | ----- |
|  | Test accuracy (%) |
| SGD (with schedule) | 92.79 | 
| DFW | 90.89 |
| DFW multistep |  92.13 | 
| Adam  | 91.45 |

## Authors
- Federico Betti
- Ioannis Mavrothalassitis
- Luca Rossi
