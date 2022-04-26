from matplotlib import pyplot as plt
import numpy as np


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
    plt.ylabel('train losses')
    plt.plot(np.arange(0, num_epochs), accuracies)
    plt.savefig('block_coordinates_figures/' + 'test accuracies_' + optimizer)
    plt.show()


