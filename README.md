# Machine-Learning-Optimization

# Files

## CoordinateDescent.ipynb

For the time being this file only contains the following:

1.
```python
class MultiLayerPerceptron(torch.nn.Module):
```
This class is a definition of 3 layer perceptron with Relu activation functions and an output layer of size 10 (1 output node corresponds to 1 of the 10 digits of the mnist dataset). The class contains the following two functions:

2. 
```python
def accuracy(predicted_logits, reference):
```
The definition of the accuracy metric usable with ***Pytorch tensors*** with the following parameters:

3.
```python
def train_model(model,dataset_train,dataset_test,optimizer,criterion,epochs):
```
The function used for training the model and its parameters are the following:

# Reference papers:

"Accelerated Coordinate Descent with Adaptive Coordinate Frequencies", http://proceedings.mlr.press/v29/Glasmachers13.pdf <br />
"Global Convergence of Block Coordinate Descent in Deep Learning", https://proceedings.mlr.press/v97/zeng19a.html <br />

