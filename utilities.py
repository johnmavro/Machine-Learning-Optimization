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
    plt.xlabel('epochs')
    plt.ylabel('train losses')
    plt.plot(np.arange(0, num_epochs), losses)
    plt.title('train losses')
    plt.savefig('block_coordinates_figures/' + 'train losses_' + optimizer)
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
    plt.xlabel('epochs')
    plt.ylabel('test accuracies')
    plt.title('test accuracies')
    plt.plot(np.arange(0, num_epochs), accuracies)
    plt.savefig('block_coordinates_figures/' + 'test accuracies_' + optimizer)
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


def train_test_split(train, test):
    """
    The functions splits the train and test set in x_train, x_test, y_train, y_test
    :param train: <class: 'torchvision.datasets.mnist.MNIST'>
                        The train set
    :param test: <class: 'torchvision.datasets.mnist.MNIST'>
                        The test set
    :return: x_train, y_train, x_test, y_test: The train and test set separated in x and y
    """
    # train-set initialization

    x_d0 = train[0][0].size()[0]  # first dimension
    x_d1 = train[0][0].size()[1]  # second dimension
    x_d2 = train[0][0].size()[2]  # third dimension
    N = len(train)
    K = 10  # number of classes

    # we initialize empty arrays which will contains x_train and x_test
    x_train = torch.empty((N, x_d0 * x_d1 * x_d2))
    y_train = torch.empty(N, dtype=torch.long)

    for i in range(N):
        x_train[i, :] = torch.reshape(train[i][0], (1, x_d0 * x_d1 * x_d2))
        y_train[i] = train[i][1]

    x_train = torch.t(x_train)

    # y_one_hot = torch.zeros(N, K).scatter_(1, torch.reshape(y_train, (N, 1)), 1)
    # y_one_hot = torch.t(y_one_hot).to(device=device)

    # test-set initialization
    N_test = len(test)
    x_test = torch.empty((N_test, x_d0 * x_d1 * x_d2))
    y_test = torch.empty(N_test, dtype=torch.long)
    for i in range(N_test):
        x_test[i, :] = torch.reshape(test[i][0], (1, x_d0 * x_d1 * x_d2))
        y_test[i] = test[i][1]

    x_test = torch.t(x_test)

    # y_test_one_hot = torch.zeros(N_test, K).scatter_(1, torch.reshape(y_test, (N_test, 1)), 1)
    # y_test_one_hot = torch.t(y_test_one_hot).to(device=device)

    return x_train, y_train, x_test, y_test



