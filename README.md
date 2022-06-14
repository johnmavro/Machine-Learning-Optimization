# Optimization for Machine Learning Project
This repository contains the implementation of the Block Coordinate Descent and of the Deep Frank Wolfe algorithm in Pytorch. The work is inspired by the two papers [Deep Frank-Wolfe for Neural Network Optimization](https://arxiv.org/pdf/1811.07591.pdf) and [Global Convergence of Block Coordinate Descent in Deep Learning](https://arxiv.org/pdf/1803.00225.pdf).

## Structure
* `Frank_Wolfe` - Folder containing the implementation of the Deep Frank-Wolfe Algorithm
  * `figures` - Contains the figures which are shown in the report for Frank-Wolfe
  * `results` - Training results for DFW algorithm.
  * `utils` - Contains utilities for training with DFW algorithm.
  * `architectures.py` - Contains the implementation of the architectures used for the empirical experiments.
  * `DFW.py` - Implementation of the DFW optimizer in Pytorch.
  * `MultiClassHingeLoss.py` - Contains the implementation of the multi-class Hinge Loss as described in the original DFW paper.
* `Block_Coordinate_Descent` - Folder containing the implementations for the Block Coordinate Descent Algorithm.
  * `results` - Training results for BCD Algorithm.
  * `CD_utilities.py` - Contains utilities for training with BCD algorithm.
  * `Convolution_BCD.ipynb` - Notebook for the comparison between a CNN and an MLP trained with the BCD algorithm.
  * `layers.py` - Contains the implementation of the activations and the update functions, for the three basic layers (Fully Connected, Convolution, Average pooling)
  * `Torch_architectures.py` - This file contains a 4 layer MLP for pytorch, so that we can do comparisons between BCD and traditional optimizers.
  * `Train_functions.py` - This file contains two training functions, one for our architecture and one for pytorch.
  * `utilities.py` - This file contains some basic utility functions.
* `Report` - This folder contains the report of the obtained results
  * `report.pdf` - Report pdf file
*  `BCD.ipynb` - Notebook for reproducibility of the results concerning Block Coordinate Descent
*  `DFW.ipynb` - Notebook for reproducibility of the results concerning Deep Frank-Wolfe
* `requirements.txt` - Requirements text file

## Installation
To clone the following repository, please run:\
`git clone https://github.com/johnmavro/Machine-Learning-Optimization.git`

## Requirements
Requirements for the needed packages are available in requirements.txt. To install the needed packages, please run:\
`pip install -r requirements.txt`

## Reproducibility of the results
We provide two notebooks `DFW.ipynb` and `BCD.ipynb` for reproducibility of the obtained results. We recommend runnning the notebooks in Google Colab for your facilitation, since we provide a simple interface.\
Each notebook is organized as follows: you can perform training, with a selected architecture and a selected optimizer, by running the third to last cell, and the second to last cell will show a plot of the obtained training trends. Consequently, the results you obtained can be saved in dictionaries different from the ones in which our results are stored to produce complete plots of all optimizers. For further instructions, please refer to the single notebooks. Otherwise, running the very last cell of each notebook will load directly our results from the dictionaries, thus showing the plots which are presented in the report.\
Finally, in both notebooks, you find the hyper-parameters we used for training at the very bottom.

## Example of usage
* Deep Frank-Wolfe
```python
eta = 0.1  # proximal coefficient
momentum = 0.9  # momentum parameter
optimizer = DFW(model.parameters(), eta=eta, momentum=momentum,
                prox_steps=2)  # define optimizer with multistep

# given x a sample and y its target output
optimizer.zero_grad()
output = model.train()(x)
loss = loss_criterion(output, y)
loss.backward()
optimizer.step(lambda: float(loss))  # the step needs to have access to the loss
```

* Block Coordinate Descent
```python

# The optimizer is not compatible with pytorch
# To perform training, please run the following function

row_size = 40  # row_size = number of rows (of the image) in the output
column_size = 40  # column_size = number of columns (of the image) in the output
Layer_list = [["Perceptron", column_size, row_size]]
input_size = 784 # for MNIST
hidden_size = 2*input_size #size of the first hidden layer
output_size = 10 # for MNIST

# x_train = input features for train
# x_test = input features for test
# y_train = labels for train
# y_test = labels for test
# y_train_one_hot = one_hot_representation of training labels
# y_test_one_hot = one_hot_representation of testing labels
GD_update = False  # True if we use our Block Coordinate descent + Gradient VN Update
linear_extension = False # True if we use prox-linear strategy for the VN update
I1 = hidden_size # unless you use convolution
I2 = 1 # unless convolution
niter = 50 # number of epochs

# hyper-parameters
gamma = 0.1
alpha = 4

train_losses,test_losses, accuracy_train, accuracy_test, epochs_times, Ws, bs =
                   execute_training(Layer_list, input_size, hidden_size, output_size, x_train, x_test,
                   y_train, y_test, y_train_one_hot, y_test_one_hot, GD_Update, linear_extension,
                   I1 = I1, I2=I2, niter = niter, gamma = gamma, alpha = alpha)

```

## Report
The report in pdf format can be found in the folder `Report`.

## Acknowledgements
We took as a baseline for our work the GitHub repositories linked to the papers which inspired the work, namely
[Block Coordinate Descent](https://github.com/IssamLaradji/BlockCoordinateDescent) and [Deep Frank Wolfe](https://github.com/oval-group/dfw#acknowledgments).

## Authors
- Federico Betti
- Ioannis Mavrothalassitis
- Luca Rossi
