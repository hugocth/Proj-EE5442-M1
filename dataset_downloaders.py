# File to download the 4 datasets: MNIST, FashionMNIST, CIFAR10 and CIFAR100

import torch
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


def load_dataset(dataset, model_is_LeNet=False):

    ## Setup transformation pipeline for MNIST and CIFAR
    transform_MNIST = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_CIFAR = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if model_is_LeNet: # No resizing because input for LeNet is 32x32
        transform_MNIST = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_CIFAR = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    ## Load datasets
    if dataset == "MNIST":
        trainset = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=transform_MNIST)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_MNIST) 
        in_channels = 1
        num_classes = 10


    elif dataset == "FashionMNIST":
        trainset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=False, transform=transform_MNIST)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_MNIST)   
        in_channels = 1
        num_classes = 10

    elif dataset == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_CIFAR)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_CIFAR) 
        in_channels = 3
        num_classes = 10

    elif dataset == "CIFAR100":
        trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=False, transform=transform_CIFAR)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_CIFAR)      
        in_channels = 3
        num_classes = 100

    else:
        print("Dataset not recognized. Possible datasets: MNIST, FashionMNIST, CIFAR10, CIFAR100")
        return 

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)

    return trainloader, testloader, in_channels, num_classes


if __name__ == "__main__":
    download_datasets()