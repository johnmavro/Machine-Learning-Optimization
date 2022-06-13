# Coordinate Descent Pytorch Optimizers

1.
```python
class CyclicCoordinateDescent(optim.Optimizer):
def __init__(self,params,lr):

def step(self,closure = None):
```

This class implements a basic Coordinate Descent Optimizer.

2.
```python
class ExpectationCoordinateDescent(optim.Optimizer):
def __init__(self,params,lr):

def step(self,closure = None):
```

This class implements a coordinate descent algorithm that updates each coordinate based on a probability proportional to its gradient.

3.
```python
class RevExpectationCoordinateDescent(optim.Optimizer):
def __init__(self,params,lr):

def step(self,closure = None):
```

This class implements a coordinate descent algorithm that updates each coordinate with a probability the is proportional to the inverse of the probability.

4.
```python
class KLargestCoordinateDescent(optim.Optimizer):
def __init__(self,params,lr):

def step(self,closure = None):
```

This class implements a coordinate descent algorithm that updates the K-coordinates with the largest gradients.