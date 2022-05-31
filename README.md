# Optimization for Machine Learning Project
This repository contains the implementation of the Block Coordinate Descent and of the Deep Frank Wolfe algorithm in Pytorch. The work is inspired by the two papers  [Deep Frank-Wolfe for Neural Network Optimization](https://arxiv.org/pdf/1811.07591.pdf) and [Accelerated Coordinate Descent with Adaptive Coordinate Frequencies](http://proceedings.mlr.press/v29/Glasmachers13.pdf).

## TODO
* Plot of the learning rate difference between SGD and DFW
* Test DFW on the five architectures in architectures.py with CIFAR10 and CIFAR100
* Test also SGD with and without momentum, Adam and Adagrad on the same five architectures with CIFAR10 and CIFAR100
* Notebook to do well
* For all plots, average over 5 runs
* Investigate possibility of solving more steps of the proximal in the end
* Adjust Utilities.categorical_accuracy in the code
* Explain well why another proximal step may help asymptotically

## Structure

## Instructions

## Requirements

## Reproducibility
- [Colab Notebook for Deep Frank Wolfe](https://colab.research.google.com/drive/1mpsunyV-11yDXPhZLznryLxJoMx4Zqxd)
- [Colab Notebook for Block Coordinate Descent](https://colab.research.google.com/drive/1mpsunyV-11yDXPhZLznryLxJoMx4Zqxd) NOTE: TODO

## Coordinate Descent

### Results

## Frank Wolfe

### Results

## Authors
- Federico Betti
- Ioannis Mavrothalassitis
- Luca Rossi
