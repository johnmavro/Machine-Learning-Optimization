from matplotlib import pyplot as plt
import numpy as np
import torch


def plot_train_losses(num_epochs, losses, optimizer):
    """
    The function plots the evolution of the train losses with the number of epochs
    :param num_epochs: int
                    The number of epochs
    :param losses: list
                    A list containing the training losses
    :param optimizer: string
                    A string containing the name of the optimizer used to train the model
    :return: None
    """
    fig = plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('train losses')
    plt.plot(np.arange(0, num_epochs), losses)
    plt.title('train losses')
    fig.savefig('block_coordinates_figures/' + 'train losses_' + optimizer)
    plt.show()


def plot_test_accuracy(num_epochs, accuracies, optimizer):
    """
    The function plots the evolution of the test accuracies with the number of epochs
    :param num_epochs: int
                    The number of epochs
    :param accuracies: list
                    A list containing the accuracies
    :param optimizer: string
                    A string containing the name of the optimizer used to train the model
    :return: None
    """
    fig = plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('test accuracies')
    plt.title('test accuracies')
    plt.plot(np.arange(0, num_epochs), accuracies)
    fig.savefig('block_coordinates_figures/' + 'test accuracies_' + optimizer)
    plt.show()


def accuracy(predicted_logits, reference):
    """
    Compute the ratio of correctly predicted labels
    :param predicted_logits: float32 tensor of shape (batch size, num classes)
                        The logits predicted by the model
    :param reference: int64 tensor of shape (batch_size) with the class number
                        Ground-truth labels
    :return: accuracy: float
    """

    labels = torch.argmax(predicted_logits, 1)
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()


def load_dataset(train, test, n_classes):
  """
  This function is used to preprocess the datasets and to split them in train and test
  :param train: The training datasets
  :param test: The testing datasets
  :return x_train, y_train, x_test, y_test
  """

  #train-set initialization
  x_d0 = train[0][0].size()[0]
  x_d1 = test[0][0].size()[1]
  x_d2 = train[0][0].size()[2]
  
  N = x_d3 = len(train)

  x_train = torch.empty(N,x_d0*x_d1*x_d2)
  y_train = torch.empty(N, dtype=torch.long)
  for i in range(N): 
    x_train[i,:] = torch.reshape(train[i][0], (1, x_d0*x_d1*x_d2))
    y_train[i] = train[i][1]
  x_train = torch.t(x_train)

  # we perform one-hot encoding of the train labels
  y_train_one_hot = torch.zeros(N, n_classes).scatter_(1, torch.reshape(y_train, (N, 1)), 1)
  y_train_one_hot = torch.t(y_train_one_hot)
  y_train = y_train

  #test-set initialization
  N_test = x_d3_test = len(test)
  x_test = torch.empty(N_test,x_d0*x_d1*x_d2)
  y_test = torch.empty(N_test, dtype=torch.long)
  for i in range(N_test): 
    x_test[i,:] = torch.reshape(test[i][0], (1, x_d0*x_d1*x_d2))
    y_test[i] = test[i][1]
  x_test = torch.t(x_test)

  # we perform one-hot encoding of the test labels
  y_test_one_hot = torch.zeros(N_test, n_classes).scatter_(1, torch.reshape(y_test, (N_test, 1)), 1)
  y_test_one_hot = torch.t(y_test_one_hot)

  return x_train, y_train, x_test, y_test, y_train_one_hot, y_test_one_hot, x_d1, x_d2



