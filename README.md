# Machine-Learning-Optimization - Coordinate descent for Deep Neural Networks

# Reference papers:

"Accelerated Coordinate Descent with Adaptive Coordinate Frequencies", http://proceedings.mlr.press/v29/Glasmachers13.pdf <br />
"Global Convergence of Block Coordinate Descent in Deep Learning", https://proceedings.mlr.press/v97/zeng19a.html <br />

# Files

[comment]: <> (
## CoordinateDescent.ipynb

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


# Baseline tests

Tested on MNIST dataset using a 3 layer MLP as a baseline model

## Results

|      | epochs | accuracy | time    |
|------|--------|----------|---------|
| Adam | 10     | 0.97920  | 120.039 |
| SGD  | 50     | 0.90920  | 545.086 |
