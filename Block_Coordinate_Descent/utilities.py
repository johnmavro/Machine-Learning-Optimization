from matplotlib import pyplot as plt
import numpy as np
import torch
import pickle
import os

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

def plot_convergence_rate_losses(num_epochs, losses, optimizer):
    """
    The function plots the evolution of the train losses with the number of epochs
    The loss is plotted in loglog and in semilogy to explore the rate of convergence
    :param num_epochs: int
                    The number of epochs
    :param losses: list
                    A list containing the training losses
    :param optimizer: string
                    A string containing the name of the optimizer used to train the model
    :return: None
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle('Training loss rate of convergence')
    axes[0].semilogy(np.arange(losses.shape[0]), losses)
    axes[0].set(xlabel='epochs', ylabel='train_loss')
    axes[0].set_title('semilogy plot')
    axes[1].loglog(np.arange(losses.shape[0]), losses)
    axes[1].set(xlabel='epochs', ylabel='train_loss')
    axes[1].set_title('loglog plot')
    fig.savefig('block_coordinates_figures/' + 'train losses_' + optimizer + 'rate_conv')
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


def pickle_results(name, train_loss, loss_class, accuracy_train, accuracy_val,
                  weights, biases, alpha, gamma):
  """
  The function save the model in a pickle file.
  :param name: The name of the file in which we save the model
  :param train_loss: A list containing the training loss each epoch
  :param loss_class: A list containing the cross-entropy loss for each epoch
  :param accuracy_train: A list containing the accuracy for the training set at each epoch
  :param accuracy_val: A list containing the accuracy for the test set at each epoch
  :param weights: The weights of the best model
  :param biases: The biases of the best model
  :param alpha: The best found alpha
  :param gamma: The best found gamma
  :return None
  """
  dictionary_save = {"Weights": weights,"Biases":biases, "train_loss": train_loss,
  "loss_class":loss_class,"accuracy_train":accuracy_train,"accuracy_test":accuracy_val,
  "alpha": alpha, "gamma": gamma}

  results_name = "results_"+name
  a_file = open(results_name,"wb")
  pickle.dump(dictionary_save,a_file)
  a_file.close()

    
def plot_stats(dataset_name, opt_list, opt_list_tilda, model_type):
    """
    Plots test accuracy and training loss for all the tested optimizers
    :param dataset_name: name of the dataset (CIFAR10, MNIST, FMNIST)
    :param opt_list: list of the names of the optimizers to reproduce
    :param opt_list_tilda: list of the type of the optimizers
    :param model_type: architecture
    :return:
    """
    stats_dict_list = []
    list_optimizers = opt_list
    for optimizer in list_optimizers:
        output_folder = os.path.join(os.getcwd(), dataset_name)
        os.makedirs(output_folder, exist_ok=True)
        fname = output_folder + '/stats_dict_Multilayer-Perceptron_' + optimizer + '.pkl'
        with open(fname, 'rb') as handle:
            stats_dict = pickle.load(handle)
        stats_dict_list.append(stats_dict)

    list_optimizers_tilda = opt_list_tilda
    nepochs = stats_dict_list[0][list_optimizers[0]]['epochs']
    
    average_time = np.array([stats_dict_list[i][list_optimizers_tilda[i]]['epochs']
                         for i in range(len(list_optimizers_tilda))])
    
    tmp = [stats_dict_list[i][list_optimizers_tilda[i]]['test_acc']
                         for i in range(len(list_optimizers_tilda))]
    for i in tmp:
        i.insert(0,0.1)
    test_acc = np.array(tmp)
    tmp2 = [stats_dict_list[i][list_optimizers_tilda[i]]['train_losses']
                             for i in range(len(list_optimizers_tilda))]
    maximum = 0
    for i in tmp2:
        if(int(3*i[0]/2)>maximum):
          maximum = int(3*i[0]/2)
    for i in tmp2:
      i.insert(0,maximum)
    train_losses = np.array([stats_dict_list[i][list_optimizers_tilda[i]]['train_losses']
                             for i in range(len(list_optimizers_tilda))])
    
    fig, ax = plt.subplots(1, 2, figsize=(13.4, 4.8), squeeze=False)
    fig.tight_layout(pad=7.)
    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
    for i in range(len(test_acc)):
        ax[0, 0].plot(np.arange(nepochs + 1), test_acc[i])
        ax[0, 0].set_ylim([0, 1])
        ax[0, 0].set_xlabel('Epoch', fontsize='x-large')
        ax[0, 0].set_ylabel('Test accuracy', fontsize='xx-large')
        ax[0, 0].set_title('Test accuracy', fontsize='xx-large')
        ax[0, 0].legend(list_optimizers)
        ax[0, 1].plot(np.arange(nepochs + 1), train_losses[i])
        ax[0, 1].set_xlabel('Epoch', fontsize='x-large')
        ax[0, 1].set_ylabel('Training loss', fontsize='xx-large')
        ax[0, 1].set_title('Training loss', fontsize='xx-large')
        ax[0, 1].legend(list_optimizers)
    plt.show()
