import torch
from tqdm import tqdm
from constraints.constraints import *
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import pickle


def is_cuda_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def initialize_train_test(trainset, testset):
    """
    Initialization of training and testing set for the case of IMAGE classification (namely 3D objects)
    :param trainset: training set
    :param testset: testing set
    :return: x_train, y_train, x_test, y_test
    """
    # cuda availability
    device = is_cuda_available()

    # train-set initialization
    x_d0 = trainset[0][0].size()[0]
    x_d1 = trainset[0][0].size()[1]
    x_d2 = trainset[0][0].size()[2]
    N = len(trainset)
    x_train = torch.empty((N, x_d0 * x_d1 * x_d2), device=device)
    y_train = torch.empty(N, dtype=torch.long)
    for i in range(N):
        x_train[i, :] = torch.reshape(trainset[i][0], (1, x_d0 * x_d1 * x_d2))
        y_train[i] = trainset[i][1]
    x_train = torch.t(x_train)
    y_train = y_train.to(device=device)

    # test-set initialization
    N_test = len(testset)
    x_test = torch.empty((N_test, x_d0 * x_d1 * x_d2), device=device)
    y_test = torch.empty(N_test, dtype=torch.long)
    for i in range(N_test):
        x_test[i, :] = torch.reshape(testset[i][0], (1, x_d0 * x_d1 * x_d2))
        y_test[i] = testset[i][1]
    x_test = torch.t(x_test)
    y_test = y_test.to(device=device)

    return x_train, y_train, x_test, y_test


def accuracy(predicted_logits, reference):
    """
    Compute the ratio of correctly predicted labels

    :param predicted_logits: float32 tensor of shape (batch size, num classes)
    :param reference: int64 tensor of shape (batch_size) with the class number
    """
    labels = torch.argmax(predicted_logits, 1)
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()


def train_model(model, dataset_train, dataset_test, optimizer, criterion, epochs, constraint_type=None,
                constraints=None, unconstrained=False):
    """
    Training a model with loss function criterion with algorithm optimizer
    :param constraint_type: type of constraint for constrained optimization
    :param unconstrained: true for unconstrained optimization
    :param constraints: a dictionary for the constraints in constrained optimization
    :param model: usually MLP perceptron
    :param dataset_train: training set
    :param dataset_test: testing set
    :param optimizer: must be passed as a string
    :param criterion: loss function
    :param epochs: number of epochs
    :return: printing accuracy after each epoch
    """
    device = is_cuda_available()

    dict_stats = {}
    losses = []
    test_accuracies = []

    for epoch in tqdm(range(epochs)):
        # loop over the dataset multiple times
        epoch_loss = 0.0
        model.train()
        iteration = 0
        closure = None
        for batch_x, batch_y in dataset_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Get output and evaluate with loss function
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            losses.append(loss)

            # Initialize optimizer
            optimizer.zero_grad()
            loss.backward()

            # Update the network
            optimizer.step(lambda: float(loss))

        # Test the quality on the test set
        model.eval()
        accuracies_test = []
        for batch_x, batch_y in dataset_test:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            accuracies_test.append(accuracy(prediction, batch_y))
            test_accuracies.append(accuracy(prediction, batch_y))
        print("Epoch {} | Test accuracy: {:.5f}".format(epoch, torch.mean(torch.tensor(accuracies_test))))

    dict_stats = {
        "train losses": losses,
        "test accuracies": test_accuracies,
    }

    return dict_stats


def list_optimizers():
    """
    Return the list of optimizers trained
    :return:
    """
    optimizers = ["SGD", "Adam", "SFW", "DFW"]
    return optimizers


def plot_stats(dict_stats, save=True, optimizer=None):
    """

    :param optimizer: name of the optimizer to save figures
    :param dict_stats: dictionary containing stats
    :param save: True to save figures
    :return:
    """
    fig, ax = plt.subplots(1, len(dict_stats.keys))

    optimizers = list_optimizers()
    assert optimizer in optimizers

    for (el, ind) in enumerate(dict_stats.keys()):
        ax[ind].plot(torch.arange(dict_stats[el].size(dim=0)), dict_stats[el], label=el)

    if save:
        output_folder = os.path.join(os.getcwd(), 'figures')  # set the output folder
        os.makedirs(output_folder, exist_ok=True)
        # saving figures in "png" format
        fig.savefig(output_folder + '/train_losses_and_accuracies' + optimizer + '.png', bbox_inches='tight')
        # saving figures in "eps" format
        fig.savefig(output_folder + '/train_losses_and_accuracies' + optimizer + '.eps',
                    format='eps', bbox_inches='tight')
