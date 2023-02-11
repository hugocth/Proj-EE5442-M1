# File to download the 4 datasets: MNIST, FashionMNIST, CIFAR10 and CIFAR100

import torchvision
import torchvision.transforms as transforms


def download_datasets():

    ## For MNIST datasets

    mnist_data = torchvision.datasets.MNIST(root="./data", download=True)
    fashion_mnist_data = torchvision.datasets.FashionMNIST(root="./data", download=True)

    ## For CIFAR datasets

    cifar10_data = torchvision.datasets.CIFAR10(root='./data', download=True)
    cifar100_data = torchvision.datasets.CIFAR100(root='./data', download=True)

    print("Datasets successfully downloaded")
    return 


if __name__ == "__main__":
    download_datasets()