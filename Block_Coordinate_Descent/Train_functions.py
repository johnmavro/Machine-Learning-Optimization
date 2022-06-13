import numpy as np
import pandas as pd
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy

from utilities import *
from Torch_architectures import *
from CD_utilities import *
from layers import *

#Train function for Block Coordinate Descent
def execute_training(layers, input_size, hidden_size, output_size, train_set, val_set,
                     train_labels, val_labels, y_train_one_hot, y_test_one_hot, use_gradient, linear_prox, I1 = 40, I2 = 40,
                     niter = 100, gamma = 1, alpha = 5):
  """
  The function takes the following arguements and produces a list of weights and biases with which
  you can use the make_pred function to get a list of predictions
  :param layers: A list that has layers as inputs e.g. ["Perceptron",size1,size2]
  :param input_size: The total size of the input layer
  :param hidden_size: The size of the hidden layer
  :param output_size: The size of the output layer (usefull for multiclass classification)
  :param train_set: The training set
  :param val_set: The validation set
  :param train_labels: The training labels
  :param val labels: The validation labels
  :param y_train_one_hot: The one hot encoding of the train set
  :param y_test_one_hot: The one hot encoding of the test set
  :param use_gradient: True if the first update of V is carried out without linearization but using the gradient
  :param linear_prox: Use the linear extensiion for the output layer
  :param I1: The row size of the first hidden layer
  :param I2: The column size of the first hidden layer
  :param niter: The default number of epochs to train the network
  :param gamma: The gamma parameter of the algorithm
  :param alpha: The alpha parameter of the algorithm
  :return train_loss,test_loss,accuracy_train,accuracy_test,time,Ws,bs: Returns
    The list of losses for the train set through all iterations.
    The list of losses for the test set through all iterations.
    The accuracy on the train set through all iterations.
    The accuracy on the test set through all iterations.
    The time spent at each iteration.
    Two lists that go in order from the input to the output layer of the weights and the biases of each layer.
  """

  N = len(train_labels)
  N_test = len(val_labels)

  # weight initialization (we replicate pytorch weight initialization)

  std = math.sqrt(1/input_size)
  Layer1 = Layer(input_size,1,hidden_size,1,["Perceptron"])
  W = Layer1.get_weights()
  b = Layer1.get_bias()

  U = torch.addmm(b.repeat(1, N), W, train_set) # equivalent to W1@train_set+b1.repeat(1,N)
  V = nn.ReLU()(U)

  Ws = [W]
  bs = [b]
  Us = [U]
  Vs = [V]
  Layers = [Layer1]
  row = [I1]
  col = [I2]

  cr_row_size = I1
  cr_col_size = I2
  size = 4
  avg_size = 2
  for cr_layer in layers:
    std = math.sqrt(1/hidden_size)
    if(cr_layer[0] !="Perceptron"):
      Layer_i = Layer(cr_col_size,cr_row_size,cr_col_size-cr_layer[1]+1,cr_row_size-cr_layer[1]+1,cr_layer)
      W = Layer_i.get_weights()
      b = Layer_i.get_bias()
      Layers.append(Layer_i)
      row.append(cr_row_size)
      col.append(cr_col_size)
      cr_row_size = cr_row_size - cr_layer[1]+1
      cr_col_size = cr_col_size - cr_layer[1]+1
    else:
      Layer_i = Layer(cr_col_size,cr_row_size,cr_layer[1],cr_layer[2],[cr_layer[0]])
      W = Layer_i.get_weights()
      b = Layer_i.get_bias()
      Layers.append(Layer_i)
      row.append(cr_row_size)
      col.append(cr_col_size)
      cr_row_size = cr_layer[1]
      cr_col_size = cr_layer[2]
    if(cr_layer[0] != "Average Pooling"):
      #print(W.shape,Vs[-1].shape)
      U = torch.addmm(b.repeat(1, N), W, Vs[-1])
      V = nn.ReLU()(U)
    else:
      U = torch.addmm(b.repeat(1, N), W, Vs[-1])
      V = U
    Ws.append(W)
    bs.append(b)
    Us.append(U)
    Vs.append(V)

  row.append(cr_row_size)
  col.append(cr_col_size)
  std = math.sqrt(1/hidden_size)
  Layer_out = Layer(cr_col_size,cr_row_size,output_size,1,["Perceptron",10])
  W = Layer_out.get_weights()
  b = Layer_out.get_bias()

  Layers.append(Layer_out)

  U = torch.addmm(b.repeat(1, N), W, Vs[-1])
  V = U
  Ws.append(W)
  bs.append(b)
  Us.append(U)
  Vs.append(V)

  # constant initialization

  gamma1 = gamma2 = gamma3 = gamma4 = gamma

  rho = gamma
  rho1 = rho2 = rho3 = rho4 = rho

  alpha1 = alpha2 = alpha3 = alpha4 = alpha5 = alpha6 = alpha7 \
  = alpha8 = alpha9 = alpha10 = alpha

  # vector of performance initialization

  loss1 = np.empty(niter)
  loss2 = np.empty(niter)
  loss_class = np.empty(niter)
  train_class = np.empty(niter)
  accuracy_train = np.empty(niter)
  accuracy_test = np.empty(niter)
  time1 = np.empty(niter)

  opt_accuracy = 0
  early_Ws = Ws
  early_bs = bs
  for k in range(niter):

    start = time.time()
    Last_layer = Layers[-1]
    W = Last_layer.get_weights()
    b = Last_layer.get_bias()
    # update V3
    if use_gradient == True:
      if (k == 1):
        Vs[-1] = (y_train_one_hot + gamma3*Us[-1] + alpha1*Vs[-1])/(1+ gamma3 + alpha1)
      else:
        for i in range(250):
          Vs[-1] = Vs[-1] - (gamma3*(Vs[-1]-Us[-1])+torch.exp(Vs[-1])/torch.sum(torch.exp(Vs[-1]),dim=0)-y_train_one_hot) * 0.01/(i+1)
    else:
        if(linear_prox):
            Vs[-1] = (gamma*Vs[-1]+alpha*Us[-1]+ y_train_one_hot - Vs[-1])/(gamma+alpha)
        else:
            Vs[-1] = (y_train_one_hot + gamma3*Us[-1] + alpha1*Vs[-1])/(1+ gamma3 + alpha1)

    # update U3
    Us[-1] = (gamma3*Vs[-1] + rho3*(torch.mm(W,Vs[-2]) + b.repeat(1,N)))/(gamma3 + rho3)

    # update W3 and b3
    W, b = update_wb_js(Us[-1],Vs[-2],Ws[-1],bs[-1],alpha1, rho3)
    Ws[-1] = W
    bs[-1] = b
    Layers[-1].update_layer(Us[-1],Vs[-2],alpha1, rho3)

    for i in range(len(Vs)-2,0,-1):
      Layer_next = Layers[i+1]
      Layer_cur = Layers[i]
      L_next_type = Layer_next.get_type()
      W_next = Layer_next.get_weights()
      W_cur = Layer_cur.get_weights()
      b_next = Layer_next.get_bias()
      b_cur = Layer_cur.get_bias()
      if(L_next_type[0]=="Average Pooling"):
        Vs[i] = update_no_activation(Us[i],Us[i+1],W_next,b_next,rho3,gamma2)
        Us[i] = Vs[i]
      else:
        Vs[i] = update_v_js(Us[i],Us[i+1],W_next,b_next,rho3,gamma2)
        Us[i] = relu_prox(Vs[i],(rho2*torch.addmm(b_cur.repeat(1,N), W_cur, Vs[i-1]) +
                                alpha2*Us[i])/(rho2 + alpha2),(rho2 + alpha2)/gamma2, row[i+1]*col[i+1], N)
        W,b = update_wb_js(Us[i],Vs[i-1],W_cur,b_cur,alpha3,rho2)
        Layers[i].update_layer(Us[i],Vs[i-1],alpha3,rho2)

    # update V1
    Vs[0] = update_v_js(Us[0],Us[1],Ws[1],bs[1],rho2,gamma1)

    # update U1
    Us[0] = relu_prox(Vs[0],(rho1*torch.addmm(bs[0].repeat(1,N), Ws[0], train_set) +
                             alpha7*Us[0])/(rho1 + alpha7),(rho1 + alpha7)/gamma1, hidden_size, N)

    # update W1 and b1
    W, b = update_wb_js(Us[0],train_set,Ws[0],bs[0],alpha8,rho1)
    Ws[0] = W
    bs[0] = b
    Layers[0].update_layer(Us[0],train_set,alpha8,rho1)

    pred_Ws = [l.get_weights() for l in Layers]
    pred_bs = [l.get_bias() for l in Layers]
    pred,prob_train = make_pred(pred_Ws,pred_bs,train_set,N)

    pred_test, prob_test = make_pred(pred_Ws,pred_bs,val_set,N_test)

    loss_class[k] = torch.sum(- y_test_one_hot * torch.log(prob_test))
    train_class[k] = torch.sum(- y_train_one_hot * torch.log(prob_train))
    loss1[k] = gamma/2*torch.pow(torch.dist(Vs[-1],y_train_one_hot,2),2).cpu().numpy()
    loss2[k] = loss1[k] + gamma/2 * torch.pow(torch.dist(torch.addmm(bs[0].repeat(1,N), Ws[0], train_set),Us[0],2),2).cpu().numpy()

    for i in range(1,len(layers)):
      loss2[k] = loss2[k] + gamma/2 * torch.pow(torch.dist(torch.addmm(bs[i].repeat(1,N), Ws[i], Vs[i-1]),Us[i],2),2).cpu().numpy()

    # compute training accuracy
    correct_train = pred == train_labels
    accuracy_train[k] = np.mean(correct_train.cpu().numpy())

    # compute validation accuracy
    correct_test = pred_test == val_labels
    accuracy_test[k] = np.mean(correct_test.cpu().numpy())

    # compute training time
    stop = time.time()
    duration = stop - start
    time1[k] = duration

    # print results
    if(use_gradient):
        print('Epoch', k + 1, '/', niter, '\n',
          '-', 'time:', time1[k], '-', 'sq_loss:', train_class[k], '-', 'tot_loss:',
          loss2[k], '-', 'loss_class:', loss_class[k], '-', 'acc:',
          accuracy_train[k], '-', 'val_acc:', accuracy_test[k])
    else:
        print('Epoch', k + 1, '/', niter, '\n',
          '-', 'time:', time1[k], '-', 'sq_loss:', loss1[k], '-', 'tot_loss:',
          loss2[k], '-', 'loss_class:', loss_class[k], '-', 'acc:',
          accuracy_train[k], '-', 'val_acc:', accuracy_test[k])
    if(accuracy_test[k]>opt_accuracy):
      early_Ws = Ws
      early_bs = bs
      opt_accuracy = accuracy_test[k]

  print('The total time spent is:', np.sum(time1), 's')
  print('\n\n')
  print('Early stopping accuracy:',opt_accuracy)
  if(use_gradient):
      return train_class,loss_class,accuracy_train,accuracy_test,time1,early_Ws,early_bs
  else:
      return loss1,loss_class,accuracy_train,accuracy_test,time1,early_Ws,early_bs

#Train function for pytorch models
def train_model(model, dataset_train, dataset_test, optimizer, criterion, epochs,scheduler,optimizer_name="SGD"):
    """
    The function is used to train the neural network
    :param model: <class '__main__.MultiLayerPerceptron'>
                    The model we wish to train
    :param dataset_train: <class 'torch.utils.data.dataloader.DataLoader'>
                    The train pytorch dataloader
    :param dataset_test: <class 'torch.utils.data.dataloader.DataLoader'>
                    The test pytorch dataloader
    :param optimizer: <class 'torch.optim.sgd.SGD'>
                    The used pytorch optimizer
    :param criterion: <class 'torch.nn.modules.loss.CrossEntropyLoss'>
                    The loss used during the training
    :param epochs: int
                    The number of epochs
    :return: train losses, accuracies: lists of training losses and test accuracies respectively
    """
    train_losses = []
    test_losses = []
    accuracies = []
    train_accuracies = []
    epoch_time = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):  # loop over the dataset multiple times
        start = time.time()
        epoch_loss = 0.0
        model.train()
        running_loss = 0
        n_steps = 0
        acc_tmp = []
        for batch_x, batch_y in dataset_train:
            n_steps = n_steps + 1
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Get output and evaluate with loss function
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            running_loss += loss.item() * len(batch_y)
            acc_tmp.append(accuracy(predictions, batch_y))
            # Initialize optimizer
            optimizer.zero_grad()
            loss.backward()

            # Update the network
            optimizer.step()

        running_loss = running_loss / n_steps
        train_losses.append(running_loss)
        train_accuracies.append(sum(acc_tmp).item() / len(acc_tmp))
        if(optimizer_name == "SGD" or optimizer_name == "Coordinate-Descent+SGD"):
          scheduler.step()
        # Test the quality on the test set
        model.eval()
        accuracies_test = []

        for batch_x, batch_y in dataset_test:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            loss_test = criterion(prediction, batch_y)
            running_loss += loss.item() * len(batch_y)
            accuracies_test.append(accuracy(prediction, batch_y))

        running_loss = running_loss / n_steps
        test_losses.append(running_loss)
        print("Epoch {} | Test accuracy: {:.5f}".format(epoch, sum(accuracies_test).item() / len(accuracies_test)))
        elapsed_time = time.time()-start
        epoch_time.append(elapsed_time)
        accuracies.append(sum(accuracies_test).item() / len(accuracies_test))
    return train_losses, test_losses, train_accuracies, accuracies, epoch_time
