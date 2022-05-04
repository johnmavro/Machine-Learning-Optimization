from torchvision import datasets, transforms


def list_benchmark_datasets():
    avail_datasets = ["CIFAR10", "CIFAR100", "MNIST"]
    return avail_datasets


def load_data(filename):
    """
    Getter for the benchmark dataset
    :param filename: benchmark dataset
    :return: train test and test set from filename
    """
    assert filename in list_benchmark_datasets()
    ts = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    if str(filename) == "MNIST":
        mnist_trainset = datasets.MNIST('../data', train=True, download=True, transform=ts)
        mnist_testset = datasets.MNIST(root='../data', train=False, download=True, transform=ts)
        return mnist_trainset, mnist_testset
    elif str(filename) == "CIFAR10":
        cifar10_trainset = datasets.CIFAR10('../data', train=True, download=True, transform=ts)
        cifar10_testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=ts)
        return cifar10_trainset, cifar10_testset
    elif str(filename) == "CIFAR100":
        cifar100_trainset = datasets.CIFAR100('../data', train=True, download=True, transform=ts)
        cifar100_testset = datasets.CIFAR100(root='../data', train=False, download=True, transform=ts)
        return cifar100_trainset, cifar100_testset
