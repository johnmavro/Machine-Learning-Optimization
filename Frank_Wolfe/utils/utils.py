import torch
import torchvision
from torchvision import transforms


def is_cuda_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


class Utilities:

    @staticmethod
    @torch.no_grad()
    def categorical_accuracy(y_true, output, topk=1):
        """
        Computes the precision@k for the specified values of k
        :param y_true: target
        :param output: output of the current model
        :param topk: topk percentage for the accuracy
        :return:
            - the topk accuracy computed by comparing output and y_true
        """
        prediction = output.topk(topk, dim=1, largest=True, sorted=False).indices.t()
        n_labels = float(len(y_true))
        return prediction.eq(y_true.expand_as(prediction)).sum().item() / n_labels


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
    """
    Sets the dataset attributes (mean, stds, ...)
    :param dataset: the string of the dataset
        - Note: only CIFAR10, CIFAR100 and IMAGENET are valid strings
    :return:
        - a dictionary with mean, standard deviation and the attribute of the dataset
    """
    assert dataset in ('CIFAR10', 'CIFAR100')
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
    """
    Divides into training and testing after the previous function setDatasetAttributes is called to set attributes
    :param dataset: the string of the dataset
        - Note: only CIFAR10, CIFAR100 and IMAGENET are valid strings
    :return:
        - a dictionary for the training and one for the testing sets
    """
    # set attributes
    dataset_dict = setDatasetAttributes(dataset)

    # Links dataset names to test dataset transformers
    trainTransformDict = {
        dataset: transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_dict['mean'], std=dataset_dict['std']), ]),
    }

    # Links dataset names to test dataset transformers
    testTransformDict = {
        dataset: transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_dict['mean'], std=dataset_dict['std']), ]),
    }

    return trainTransformDict, testTransformDict
