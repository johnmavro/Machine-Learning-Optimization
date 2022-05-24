# Machine-Learning-Optimization - Coordinate descent for Deep Neural Networks

## Reference papers:

"Accelerated Coordinate Descent with Adaptive Coordinate Frequencies", http://proceedings.mlr.press/v29/Glasmachers13.pdf <br />
"Global Convergence of Block Coordinate Descent in Deep Learning", https://proceedings.mlr.press/v97/zeng19a.html <br />


<!--- ## CoordinateDescent.ipynb
For the time being this file only contains the following:

1.
```python
class MultiLayerPerceptron(torch.nn.Module):
def __init__(self):

def forward(self,x):
```
This class is a definition of 3 layer perceptron with Relu activation functions and an output layer of size 10 (1 output node corresponds to 1 of the 10 digits of the mnist dataset). The class contains two functions the ***init*** function which initializes the class object and the ***forward*** function which takes an ***input x*** and produces an ***output*** by passing the input through the perceptron.

2. 
```python
def accuracy(predicted, reference):
```
The definition of the accuracy metric usable with ***Pytorch tensors*** with the following parameters:<br>
***predicted*** the predicted labels from the model<br>
***reference*** the true value of samples.

3.
```python
def train_model(model,dataset_train,dataset_test,optimizer,criterion,epochs):
```
The function used for training the model and its parameters are the following:<br>
***model*** the model we wish to train (must be implemented in pytorch)<br>
***dataset_train*** the training dataset (use pytorch dataset loader to create this see mnist example in notebook)<br>
***dataset_test*** the testing dataset (use pytorch dataset loader to create this see mnist example in notebook)<br>
***optimizer*** the optimizer (must be compatible with pytorch optim library)<br>
***criterion*** the loss metric that the optimizer uses<br>
***epochs*** the number of epochs that the optimizer should iterate over the dataset
)
--->

## Files

**baseline_Adam.ipynb**: contains the implementation of the baseline model using Adam optimizer. The results will be compared with the notebook block_coordinate_descent.

**baseline_SGD.ipynb**: contains the implementation of the baseline model using SGD optimizer. The results will be compared with the notebook block_coordinate_descent.

**Block_coordinate_descent.ipynb**: contains the implementation of the block coordinate descent approach to optimize Deep Neural Networks.

**utilities.py**: contains useful functions shared between notebooks.

The folder **block_coordinate_figures**: contains the figures obtained by plotting the evolution of test accuracies and train losses when the neural network is optimized using the different optimizers.


## Results


-MNIST with linearization of the loss:
1 hidden layer with 1500 neurons each. 

|  Opt. Method | Epochs | Test accuracy | Training time |
| ----- | ----- | ----- | ----- |
| SGD   |   50    | 91 % | 282 s |
| Adam  |   20    | 98 % | 115 s |
| Block Coordinate Descent | 39 | 96.08 % | 47.83 s |




-Fashion-MNIST with linearization of the loss:
1 hidden layer with 1500 neurons each.
(gamma=1, alpha=2)

|  Opt. Method | Epochs | Test accuracy | Training time |
| ----- | ----- | ----- | ----- |
| SGD   | 100  | 84.63 % | 536.4345s  |
| Adam  | 20  | 89.73 %    | 106.0462s    |
| Block Coordinate Descent | 29 | 86.83 % | 39.26 s |


-CIFAR10 with linearization of the loss:
1 hidden layer with 1500 neurons each.
(alpha =2, gamma = 0.2)

|  Opt. Method | Epochs | Test accuracy | Training time |
| ----- | ----- | ----- | ----- |
| SGD   | 100  | 45.52 % | 692.9886s    |
| Adam  |  20  | 53.41 % | 141.1729s |
| Block Coordinate Descent | 5 | 41.37% | 9.21 s |


-MNIST Dataset without linearization of the loss:
1 hidden layer with 1500 neurons each.

|  Opt. Method | Epochs | Test accuracy | Training time |
| ----- | ----- | ----- | ----- |
| SGD   |   50    | 91 % | 282 s |
| Adam  |   20    | 98 % | 115 s |
| Block Coordinate Descent | 38 | 96.15% | 52s |

-FASHION-MNIST without linearization of the loss:
1 hidden layer with 1500 neurons each.
(alpha =2, gamma = 1.8)

|  Opt. Method | Epochs | Test accuracy | Training time |
| ----- | ----- | ----- | ----- |
| SGD   |       |  |  |
| Adam  |       |  |  |
| Block Coordinate Descent | 28 | 86.63% | 38.75s |



-CIFAR10 without linearization of the loss:
1 hidden layer with 1500 neurons each.
( <img src="https://render.githubusercontent.com/render/math?math=\alpha=7"/>,  <img src="https://render.githubusercontent.com/render/math?math=\gamma=0.2"/>)


|  Opt. Method | Epochs | Test accuracy | Training time |
| ----- | ----- | ----- | ----- |
| SGD   |       |  |  |
| Adam  |       |  |  |
| Block Coordinate Descent | 42 | 43.80% | 155.97s |

# Remark
1. From the first update rule of the paper we see that the size of <img src="https://render.githubusercontent.com/render/math?math=\alpha"/> cannot be negligable otherswise the constraint that enforces the non-linear function <img src="https://render.githubusercontent.com/render/math?math=V = \sigma(U)" /> will be neglected. This probably allows the network to go in a bizarre direction and creates problems.
2. From the second update of the paper we see that the relative size of <img src="https://render.githubusercontent.com/render/math?math=\alpha"/> and <img src="https://render.githubusercontent.com/render/math?math=\gamma"/> needs to be large enough for the same reason.
