import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import pickle
import math
import sys
import warnings
import torch
import time
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


def is_cuda_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def list_optimizers():
    """
    Return the list of optimizers trained
    :return:
    """
    optimizers = ["SGD", "Adam", "SFW", "BCD"]
    return optimizers


def list_datasets():
    datasets = ["CIFAR10", "CIFAR100", "IMAGENET"]
    return datasets


class Utilities:

    @staticmethod
    @torch.no_grad()
    def categorical_accuracy(y_true, output, topk=1):
        """Computes the precision@k for the specified values of k"""
        prediction = output.topk(topk, dim=1, largest=True, sorted=False).indices.t()
        n_labels = float(len(y_true))
        return prediction.eq(y_true.expand_as(prediction)).sum().item() / n_labels


class RetractionLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Retracts the learning rate as follows. Two running averages are kept, one of length n_close, one of n_far. Adjust
    the learning_rate depending on the relation of far_average and close_average. Decrease by 1-retraction_factor.
    Increase by 1/(1 - retraction_factor*growth_factor)
    """
    def __init__(self, optimizer, retraction_factor=0.3, n_close=5, n_far=10, lowerBound=1e-5, upperBound=1, growth_factor=0.2, last_epoch=-1):
        self.retraction_factor = retraction_factor
        self.n_close = n_close
        self.n_far = n_far
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.growth_factor = growth_factor

        assert (0 <= self.retraction_factor < 1), "Retraction factor must be in [0, 1[."
        assert (0 <= self.lowerBound < self.upperBound <= 1), "Bounds must be in [0, 1]"
        assert (0 < self.growth_factor <= 1), "Growth factor must be in ]0, 1]"

        self.closeAverage = RunningAverage(self.n_close)
        self.farAverage = RunningAverage(self.n_far)

        super(RetractionLR, self).__init__(optimizer, last_epoch)

    def update_averages(self, loss):
        self.closeAverage(loss)
        self.farAverage(loss)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        factor = 1
        if self.farAverage.is_complete() and self.closeAverage.is_complete():
            if self.closeAverage.result() > self.farAverage.result():
                # Decrease the learning rate
                factor = 1 - self.retraction_factor
            elif self.farAverage.result() > self.closeAverage.result():
                # Increase the learning rate
                factor = 1./(1 - self.retraction_factor*self.growth_factor)

        return [max(self.lowerBound, min(factor * group['lr'], self.upperBound)) for group in self.optimizer.param_groups]


class RunningAverage(object):
    """Tracks the running average of n numbers"""

    def __init__(self, n):
        self.n = n
        self.reset()

    def reset(self):
        self.sum = 0
        self.avg = 0
        self.entries = []

    def result(self):
        return self.avg

    def get_count(self):
        return len(self.entries)

    def is_complete(self):
        return len(self.entries) == self.n

    def __call__(self, val):
        if len(self.entries) == self.n:
            l = self.entries.pop(0)
            self.sum -= l
        self.entries.append(val)
        self.sum += val
        self.avg = self.sum / len(self.entries)

    def __str__(self):
        return str(self.avg)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def result(self):
        return self.avg

    def __call__(self, val, n=1):
        """val is an average over n samples. To compute the overall average, add val*n to sum and increase count by n"""
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return str(self.avg)


def setDatasetAttributes(dataset):
    assert dataset in list_datasets()
    if str(dataset) == "CIFAR10":
        means = (0.4914, 0.4822, 0.4465)
        stds = (0.2023, 0.1994, 0.2010)
        datasetDict = getattr(torchvision.datasets, 'CIFAR10')
    elif str(dataset) == "CIFAR100":
        means = (0.5071, 0.4867, 0.4408)
        stds = (0.2675, 0.2565, 0.2761)
        datasetDict = getattr(torchvision.datasets, 'CIFAR100')
    elif str(dataset) == "IMAGENET":
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
        datasetDict = getattr(torchvision.datasets, 'IMAGENET')
    return {'mean': means, 'std': stds, 'datasetDict': datasetDict}


def setTrainAndTest(dataset):

    dict = setDatasetAttributes(dataset)

    trainTransformDict = {
        dataset: transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=dict['mean'], std=dict['std']), ]),
    }

    testTransformDict = {  # Links dataset names to test dataset transformers
        dataset: transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=dict['mean'], std=dict['std']), ]),
    }

    return trainTransformDict, testTransformDict


# def train_network(nepochs, batch_size, model, constraints, trainData, testData, optimizer,
#                   lr_step_size, lr_decrease_factor, lr_scheduler_active=True, retraction=True):
#     """
#
#     :param lr_decrease_factor:
#     :param lr_step_size:
#     :param retraction:
#     :param lr_scheduler_active:
#     :param optimizer:
#     :param nepochs:
#     :param batch_size:
#     :param model:
#     :param constraints:
#     :param trainData:
#     :param testData:
#     :return:
#     """
#
#     # initialize lists for results
#     train_losses = []
#     test_losses = []
#     train_accuracies = []
#     test_accuracies = []
#
#     # check cuda availability
#     device = is_cuda_available()
#
#     make_feasible(model, constraints)
#
#     # define the loss object
#     loss_criterion = torch.nn.CrossEntropyLoss().to(device=device)
#     model = model.to(device=device)
#
#     # Loaders
#     trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True,
#                                           pin_memory=torch.cuda.is_available(), num_workers=2)
#     testLoader = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False,
#                                          pin_memory=torch.cuda.is_available(), num_workers=2)
#
#     # initialize some necessary metrics objects
#     train_loss, train_accuracy = AverageMeter(), AverageMeter()
#     test_loss, test_accuracy = AverageMeter(), AverageMeter()
#
#     if lr_scheduler_active:
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step_size,
#                                                     gamma=lr_decrease_factor)
#
#     if retraction:
#         retractionScheduler = RetractionLR(optimizer=optimizer)
#
#     # function to reset metrics
#     def reset_metrics():
#         train_loss.reset()
#         train_accuracy.reset()
#         test_loss.reset()
#         test_accuracy.reset()
#
#     @torch.no_grad()
#     def evaluate_model(data="train"):
#         if data == "train":
#             loader = trainLoader
#             mean_loss, mean_accuracy = train_loss, train_accuracy
#         elif data == "test":
#             loader = testLoader
#             mean_loss, mean_accuracy = test_loss, test_accuracy
#
#         sys.stdout.write(f"Evaluation of {data} data:\n")
#         for x_input, y_target in Bar(loader):
#             x_input, y_target = x_input.to(device), y_target.to(device)  # Move to CUDA if possible
#             output = model.eval()(x_input)
#             loss = loss_criterion(output, y_target)
#             mean_loss(loss.item(), len(y_target))
#             mean_accuracy(Utilities.categorical_accuracy(y_true=y_target, output=output), len(y_target))
#
#     start = time.time()
#     for epoch in range(nepochs + 1):
#         reset_metrics()
#         sys.stdout.write(f"\n\nEpoch {epoch}/{nepochs}\n")
#         if epoch == 0:
#             # Just evaluate the model once to get the metrics
#             evaluate_model(data='train')
#         else:
#             # Train
#             sys.stdout.write(f"Training:\n")
#             for x_input, y_target in Bar(trainLoader):
#                 x_input, y_target = x_input.to(device), y_target.to(device)  # Move to CUDA if possible
#                 optimizer.zero_grad()  # Zero the gradient buffers
#                 output = model.train()(x_input)
#                 loss = loss_criterion(output, y_target)
#                 loss.backward()  # Backpropagation
#                 optimizer.step(constraints=constraints)
#                 train_loss(loss.item(), len(y_target))
#                 train_accuracy(Utilities.categorical_accuracy(y_true=y_target, output=output), len(y_target))
#
#             if lr_scheduler_active:
#                 scheduler.step()
#             if retraction:
#                 # Learning rate retraction
#                 retractionScheduler.update_averages(train_loss.result())
#                 retractionScheduler.step()
#
#         evaluate_model(data='test')
#         sys.stdout.write(f"\n Finished epoch {epoch}/{nepochs}: Train Loss {train_loss.result()} | Test Loss {test_loss.result()} | Train Acc {train_accuracy.result()} | Test Acc {test_accuracy.result()}\n")
#
#     elapsed_time = time.time()-start
#     sys.stdout.write(f"Time elapsed for the current epoch {elapsed_time}")
#
#     return train_losses, test_losses, train_accuracies, test_accuracies, elapsed_time
