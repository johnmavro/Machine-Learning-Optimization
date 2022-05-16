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

The following results are obtained on a multi-layer-perceptron with 1 hidden layer with 1500 neurons each and using the MNIST Dataset + linearization of the loss. 

|  Opt. Method | Epochs | Test accuracy | Training time |
| ----- | ----- | ----- | ----- |
| SGD   |   50    | 91 % | 282 s |
| Adam  |   20    | 98 % | 115 s |
| Block Coordinate Descent | 39 | 96.08 % | 47.83 s |

The following results are obtained on a multi-layer-perceptron with 1 hidden layer with 1500 neurons each and using the Fashion-MNIST Dataset + linearization of the loss. 

|  Opt. Method | Epochs | Test accuracy | Training time |
| ----- | ----- | ----- | ----- |
| SGD   |      |     |     |
| Adam  |      |     |     |
| Block Coordinate Descent | 29 | 86.83 % | 39.26 s |

(gamma=1, alpha=2)

The following results are obtained on a multi-layer-perceptron with 1 hidden layer with 1500 neurons each and using the Cifar10 dataset + linearization of the loss. 

|  Opt. Method | Epochs | Test accuracy | Training time |
| ----- | ----- | ----- | ----- |
| SGD   |      |     |     |
| Adam  |      |     |     |
| Block Coordinate Descent | 5 | 41.37% | 9.21 s |

(alpha =2, gamma =0.2)


The following results are obtained on a multi-layer-perceptron with 1 hidden layer with 1500 neurons each and using the MNIST Dataset without linearization of the loss. 

|  Opt. Method | Epochs | Test accuracy | Training time |
| ----- | ----- | ----- | ----- |
| SGD   |   50    | 91 % | 282 s |
| Adam  |   20    | 98 % | 115 s |
| Block Coordinate Descent | 38 | 96.15% | 52s |

The following results are obtained on a multi-layer-perceptron with 1 hidden layer with 1500 neurons each and using the F-MNIST Dataset without linearization of the loss. 

|  Opt. Method | Epochs | Test accuracy | Training time |
| ----- | ----- | ----- | ----- |
| SGD   |       |  |  |
| Adam  |       |  |  |
| Block Coordinate Descent | 28 | 86.63% | 38.75s |

(alpha =2, gamma = 1.8)

The following results are obtained on a multi-layer-perceptron with 1 hidden layer with 1500 neurons each and using the CIFAR10 Dataset without linearization of the loss. 

|  Opt. Method | Epochs | Test accuracy | Training time |
| ----- | ----- | ----- | ----- |
| SGD   |       |  |  |
| Adam  |       |  |  |
| Block Coordinate Descent | 42 | 43.80% | 155.97s |

(alpha =7, gamma = 0.2)
