import numpy as np
import torch
import torch.nn as nn
import torchvision
import math

from CD_utilities import *


def shift_right(l):
  """
  Shifts right a python list by one element.
  """
  return l[-1:]+l[:-1]

def filter_conv(W,I1,I2,size = 2):
  """
  This function filters the entries of the matrix W so that it behaves like a Convolution.
  :param W: The weight matrix W that contains weights from one layer to the next.
  :param I1: The first dimension of the original 2D matrix
  :param I2: The first dimension of the original 2D matrix
  :return: The filtered weight matrix W
  """
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  mask_list = []
  for i in range(size):
    mask_list += [1]*size+[0]*(I2-size)
  mask_list +=[0]*(I1-size)*I2
  full_mask = [mask_list]
  counter = I2-size
  for i in range((I2-size+1)*(I1-size+1)-1):
    next_mask=shift_right(full_mask[-1])
    #print(counter)
    if(counter==0):
      counter = I2-size
      for j in range(size-1):
        next_mask=shift_right(next_mask)
    else:
      counter -=1
    full_mask.append(next_mask)
  if(torch.tensor(full_mask).shape[0]!=W.shape[0]):
    print(torch.tensor(full_mask).shape[0],W.shape[0])
  return torch.mul(torch.tensor(full_mask).to(device),W)

class Layer():
  """
  A simple layer class for the three different types of layers of our network,
  Perceptron, Convolution, Average Pooling this class is not equivalent to a
  pytorch default layer since the update is done by the function that calculates
  the closed form solution for the optimal weights. This class is also
  constructed in such a way to be able to process input of up to 2 dimensions.
  """
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def __init__(self,col_size,row_size,col_out,row_out,layer_type=["Perceptron"]):
    """
    The init function initializes the basic parameters of the layer:
    :param col_size: the column size of the input
    :param row_size: the row size of the input
    :param col_out: the column size of the output
    :param row_out: the row size of the output
    :param layer_type: the type of the layer,
                       if the layer is a Perceptron we expect a single string
                       that says that the layer is a perceptron
                       if the layer is a Convolution we expect a s string that
                       specifies it as a convolution and 1 parameter which is
                       the size of the convolution
                       if the layer is an Average Pooling layer then we expect
                       analogous paramaters as the convolution layer
    Note that apart from the average pooling, for the other types of layers the initial weight is assigned a random value.
    """
    self.col_size = col_size
    self.row_size = row_size
    self.col_out = col_out
    self.row_out = row_out
    self.layer_type = layer_type
    std = math.sqrt(1/(row_size*col_size))
    if(self.layer_type[0] =="Average Pooling"):
      self.weights = torch.add(torch.FloatTensor(self.row_out*self.col_out,self.row_size*self.col_size),1).to(self.device)
      self.weights = filter_conv(self.weights,self.col_size,self.row_size,self.layer_type[1])
      self.weights = torch.mul(self.weights,1/(self.layer_type[1]*self.layer_type[1]))
      self.bias = torch.FloatTensor(row_out*col_out,1).uniform_(-std, std).to(self.device)
    elif(self.layer_type[0]=="Convolution"):
      self.weights = torch.FloatTensor(self.row_out*self.col_out,self.row_size*self.col_size).uniform_(-std, std).to(self.device)
      self.weights = filter_conv(self.weights,self.col_size,self.row_size,self.layer_type[1])
    else:
      self.weights = torch.FloatTensor(row_out*col_out,row_size*col_size).uniform_(-std, std).to(self.device)
    self.bias = torch.FloatTensor(row_out*col_out,1).uniform_(-std, std).to(self.device)

  def forward_pass(self,input,N):
    """
    A function that does a forward pass from the layer, is capable of doing it
    for N samples simultaniously.
    :param input: The input that we do the forward pass on, should have dimension
                  input_dimension_layer x number_of_samples
    :param N: The number of samples that are in the input
    """
    return torch.addmm(self.bias.repeat(1, N), self.weights, input)

  def update_layer(self,output,input,alpha,rho):
    """
    A function that applies the closed form solution, for all different types of layers
    :param output: The output of the layer with respect to which we should compute the closed form solution
    :param input: The input of the layer with respect to which we should compute the closed form solution
    :param alpha: The parameter alpha of the algorithm that is described on the paper
    :param rho: The parameter rho of the algorithm that is described on the paper
    """
    if(self.layer_type[0]!="Average Pooling"):
      self.weights,self.bias = update_wb_js(output,input,self.weights,self.bias,alpha,rho)
      if(self.layer_type[0]=="Convolution"):
        self.weights = filter_conv(self.weights,self.col_size,self.row_size,self.layer_type[1])

  def get_weights(self):
    """
    This function returns the current weights of the layer
    """
    return self.weights

  def get_bias(self):
    """
    This function returns the bias of the layer
    """
    return self.bias

  def get_type(self):
    """
    This function returns the type of the layer
    """
    return self.layer_type

#############################################################
#      Average Pooling Functions                            #
#      (Not Used)                                           #
#############################################################

def avg_pool(W,I1,I2,size = 2):
  mask_list = []
  for i in range(size):
    mask_list += [1/size**2]*size+[0]*(I2-size)
  mask_list +=[0]*(I1-size)*I2
  full_mask = [mask_list]
  counter = I2-size
  for i in range((I2-size+1)*(I1-size+1)-1):
    next_mask=shift_right(full_mask[-1])
    if(counter==0):
      counter = I2-size
      for j in range(size-1):
        next_mask=shift_right(next_mask)
    else:
      counter -=1
    full_mask.append(next_mask)
  return torch.mul(torch.tensor(full_mask).to(device),W)
