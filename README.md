# Machine-Learning-Optimization

# Files

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
The definition of the accuracy metric usable with ***Pytorch tensors*** with the following parameters:
***predicted*** the predicted labels from the model
***reference*** the true value of samples.

3.
```python
def train_model(model,dataset_train,dataset_test,optimizer,criterion,epochs):
```
The function used for training the model and its parameters are the following:
***model*** the model we wish to train (must be implemented in pytorch)
***dataset_train*** the training dataset (use pytorch dataset loader to create this see mnist example in notebook)
***dataset_test*** the testing dataset (use pytorch dataset loader to create this see mnist example in notebook)
***optimizer*** the optimizer (must be compatible with pytorch optim library)
***criterion*** the loss metric that the optimizer uses
***epochs*** the number of epochs that the optimizer should iterate over the dataset

# Reference papers:

"Accelerated Coordinate Descent with Adaptive Coordinate Frequencies", http://proceedings.mlr.press/v29/Glasmachers13.pdf <br />
"Global Convergence of Block Coordinate Descent in Deep Learning", https://proceedings.mlr.press/v97/zeng19a.html <br />

