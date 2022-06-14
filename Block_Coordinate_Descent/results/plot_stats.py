import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def plot_stats(dataset_name, model_type):
    """
    Plots test accuracy and training loss for all the tested optimizers (Adam, SGD, DFW, DFW multistep)
    :param dataset_name: name of the dataset (CIFAR10, MNIST, FMNIST)
    :param model_type: architecture
    :return:
    """
    stats_dict_list = []
    list_optimizers = ["Adam", "Coordinate-Descent", "Coordinate-Descent+Adam", "SGD"]
    for optimizer in list_optimizers:
        output_folder = os.path.join(os.getcwd(),'Block_Coordinate_Descent/results/' + dataset_name)
        os.makedirs(output_folder, exist_ok=True)
        fname = output_folder + '/stats_dict_Multilayer-Perceptron_' + optimizer + '.pkl'
        with open(fname, 'rb') as handle:
            stats_dict = pickle.load(handle)
        stats_dict_list.append(stats_dict)

    list_optimizers_tilda = ["Adam", "Coordinate-Descent", "Coordinate-Descent+Adam", "SGD"]
    nepochs = 50#stats_dict_list[0]["Adam"]['epochs']
    print(dataset_name)
    average_time = np.array([stats_dict_list[i][list_optimizers_tilda[i]]['epochs']
                         for i in range(len(list_optimizers_tilda))])
    #print(stats_dict_list)
    test_acc = np.array([stats_dict_list[i][list_optimizers_tilda[i]]['test_acc']
                         for i in range(len(list_optimizers_tilda))])
    train_losses = np.array([stats_dict_list[i][list_optimizers_tilda[i]]['train_losses']
                             for i in range(len(list_optimizers_tilda))])
    print("Average Epoch Times:")
    for i in range(len(average_time)):
        print(list_optimizers[i],":",np.average(average_time[i]))
    print("Test Accuracy:")
    for i in range(len(test_acc)):
        print(list_optimizers[i],":",np.amax(test_acc[i]))
    #print(test_acc)
    fig, ax = plt.subplots(1, 2, figsize=(13.4, 4.8), squeeze=False)
    fig.tight_layout(pad=7.)
    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
    for i in range(len(test_acc)):
        #print(test_acc[i])
        #print(nepochs)
        ax[0, 0].plot(np.arange(nepochs + 1), [0.1]+test_acc[i])
        ax[0, 0].set_ylim([0, 1])
        ax[0, 0].set_xlabel('Epoch', fontsize='x-large')
        ax[0, 0].set_ylabel('Test accuracy', fontsize='xx-large')
        ax[0, 0].set_title('Test accuracy', fontsize='xx-large')
        ax[0, 0].legend(list_optimizers)
        ax[0, 1].plot(np.arange(nepochs + 1), [3000]+train_losses[i])
        ax[0, 1].set_xlabel('Epoch', fontsize='x-large')
        ax[0, 1].set_ylabel('Training loss', fontsize='xx-large')
        ax[0, 1].set_title('Training loss', fontsize='xx-large')
        ax[0, 1].legend(list_optimizers)
    plt.show()
