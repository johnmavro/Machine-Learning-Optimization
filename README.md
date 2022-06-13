# Optimization for Machine Learning Project
This repository contains the implementation of the Block Coordinate Descent and of the Deep Frank Wolfe algorithm in Pytorch. The work is inspired by the two papers [Deep Frank-Wolfe for Neural Network Optimization](https://arxiv.org/pdf/1811.07591.pdf) and [Global Convergence of Block Coordinate Descent in Deep Learning](https://arxiv.org/pdf/1803.00225.pdf).

## Structure
* `Frank_Wolfe` - Folder containing the implementation of the Deep Frank-Wolfe Algorithm
  * `figures` - Contains the figures which are shown in the report for Frank-Wolfe
  * `results` - Contains the Python dictionaries with the results for the experiments regarding the Frank Wolfe Algorithm
  * `utils` - Contains utilities to load data-sets, normalize batches and other tasks
  * `architectures.py` - Contains the implementation of the architectures used for the empirical experiments
  * `DFW.py` - Implementation of the DFW optimizer in Pytorch
  * `MultiClassHingeLoss.py` - Contains the implementation of the multi-class Hinge Loss as described in the original DFW paper
* `Block_Coordinate_Descent` - Folder containing the implementations for the Block Coordinate Descent Algorithm
  * `results` - Contains the results of the Block Coordinate Descent Algorithm experiments, the figures and the dictionaries
  * `CD_utilities.py` - Contains the functions which are necessary to run the updates for BCD
  * `Convolution_BCD.ipynb` - This notebook contains a comparison between a CNN and an MLP trained with the BCD algorithm as suggested in the original paper.
  * `layers.py` - This file contains the implementation of the activations and the update functions, for the three basic layers Fully Connected, Convolution, Average pooling
  * `Torch_architectures.py` - This file contains a 4 layer MLP for pytorch, so that we can do comparisons between BCD and traditional optimizers.
  * `Train_functions.py` - This file contains two training functions, one for our architecture and one for pytorch.
  * `utilities.py` - This file contains some basic utility functions.
* `Report` - This folder contains the report of the obtained results
  * `report.pdf` - Report pdf file
*  `BCD.ipynb` - Notebook for reproducibility of the results concerning Block Coordinate Descent
*  `DFW.ipynb` - Notebook for reproducibility of the results concerning Deep Frank-Wolfe
* `requirements.txt` - Requirements text file

## Requirements
Requirements for the needed packages are available in requirements.txt. To install the needed packages, please run:\
`pip install -r requirements.txt`

## Reproducibility
We provide two notebooks `DFW.ipynb` and `BCD.ipynb` for reproducibility of the obtained results.
We recommend to run the notebooks in Google Colab for your facilitation since we provide a simple interface.

## Example of usage
* Deep Frank-Wolfe
```python
eta = 0.1  # proximal coefficient
momentum = 0.9  # momentum parameter
optimizer = DFW(model.parameters(), eta=eta, momentum=momentum,
                prox_steps=2)  # define optimizer with multistep

optimizer.zero_grad()
output = model.train()(x)
loss = loss_criterion(y, output)
loss.backward()
optimizer.step(lambda: float(loss), model, x, y)  # needs to have access to the loss and the model
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

# Hyper-parameters of Block coordinate descent (refer to the report)
gamma = 0.1
alpha = 4

train_losses,test_losses, accuracy_train, accuracy_test, epochs_times, Ws, bs =
                   execute_training(Layer_list, input_size, hidden_size, output_size, x_train, x_test,
                   y_train, y_test, y_train_one_hot, y_test_one_hot, GD_Update, linear_extension,
                   I1 = I1, I2=I2, niter = niter, gamma = gamma, alpha = alpha)

```

## Results

### Coordinate Descent

#### MNIST

| 4-layer perceptron   |  Test accuracy (%)|  Time per epoch
|:---------------------|-------------------|-------------------|
| BCD                  | 94.54             |      1.62         |
| Adam                 | 96.46             |      16.21        |
| Prox Linear          | 94.56             |      2.03         |
| BCD(GD update)       | 93.68             |      1.84         |
| SGD (with schedule)  | 94.52             |      9.9          |
| BCD + Adam           | 95.85             |      7.1          |
| BCD + SGD            | 95.56             |      5.1          |

##### Hyper-parameters used

* Block Coordinate Descent: γ = 0.1, α = 4.
* Adam: γ = 0.001, β1 = 0.9, β2 = 0.999.
* SGD: γ = 0.01, μ = 0.9, scheduler StepLR(stepsize=15,
gamma=0.2).
* Block Coordinate Descent with Prox Linear VN update:
γ = 0.1, α = 4.
* Block Coordinate Descent with GD update (4) for VN :
γ = 0.1, α = 4, T = 250, ηt = O (1/T).
* Block Coordinate Descent: γ = 0.1, α = 4

### Frank Wolfe

#### CIFAR10

| GoogLeNet            |  Test accuracy (%)|
|:---------------------|-------------------|
| SGD (with schedule)  | 92.79             |
| DFW                  | 90.89             |
| DFW multistep        | 92.13             |
| Adam                 | 91.45             |


##### Hyper-parameters used

* DFW and DFW multistep: η = 0.1, μ = 0.9.
* Adam: γ = 0.001, β1 = 0.9, β2 = 0.999.
* SGD: γ = 0.01, μ = 0.9, scheduler StepLR(stepsize=20,
gamma=0.2).

## Report
The report in pdf format can be found in the folder `report`.

## Authors
- Federico Betti
- Ioannis Mavrothalassitis
- Luca Rossi
