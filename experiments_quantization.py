# Main file to run the experiments

#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import pandas as pd
import time
from datetime import datetime

from networks import LeNet, AlexNet, VGG11, BasicBlock, ResNet

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


def train(model, trainloader, is_double_experiment=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Begin training")
    t0 = time.time()
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if is_double_experiment:
                inputs = inputs.double()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    t1 = time.time()
    training_time = t1 - t0
    print('Finished Training')
    print(f'Elapsed time for training: {int(training_time)}s')
    return model, training_time

def test(model, testloader, is_double_experiment=False):
    t0 = time.time()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if is_double_experiment:
                images = images.double()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    t1 = time.time()

    accuracy = 100 * correct / total
    inference_time = t1 - t0

    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return round(size_all_mb, 2)
    
    model_size = get_model_size(model)

    print('Accuracy of the network on the 10000 test images: %d %%' % (accuracy))
    print(f'Elapsed time for inference: {int(inference_time)}s')

    return accuracy, inference_time, model_size


def main(datasets):

    res_df = pd.DataFrame(columns=["Model", "Dataset", "Quantization", "Accuracy", "Training Time", "Inference Time", "Model Size"])

    for dataset in datasets:
        ## Load dataset
        trainloader, testloader, in_channels, num_classes = load_dataset(dataset)

        ## Create instances of models
        leNet = LeNet(in_channels=in_channels, num_classes=num_classes)
        alexNet = AlexNet(in_channels=in_channels, num_classes=num_classes)
        vgg11 = VGG11(in_channels=in_channels, num_classes=num_classes)
        resNet18 = ResNet(in_channels=in_channels, num_layers=18, block=BasicBlock, num_classes=num_classes)
        models = [leNet, alexNet, vgg11, resNet18]
        models_labels = ["leNet", "alexNet", "vgg11", "resNet18"]
        
        ## For local test
        models = [leNet]
        models_labels = ["leNet"]
        

        for i, model in enumerate(models):

            if model == leNet:
                trainloader, testloader, in_channels, num_classes = load_dataset(dataset, model_is_LeNet=True)

            ## fp32
            model, training_time = train(model=model, trainloader=trainloader) ## Simple precision
            accuracy, inference_time, model_size = test(model=model, testloader=testloader)
            res_df.loc[res_df.shape[0]] = [models_labels[i], dataset, torch.float, accuracy, training_time, inference_time, model_size]

            ## float16 and int8 --> doesn't work on macOS ...
            for quantization in [torch.float16, torch.qint8]:
                q_model = torch.quantization.quantize_dynamic(
                    model,                                  # the original model
                    {torch.nn.Conv2d, torch.nn.Linear},     # a set of layers to dynamically quantize
                    dtype=quantization                      # the target dtype for quantized weights
                )
                accuracy, inference_time, model_size = test(model=q_model, testloader=testloader)
                res_df.loc[res_df.shape[0]] = [models_labels[i], dataset, quantization, accuracy, training_time, inference_time, model_size]

            ## fp64
            accuracy, training_time, model_size = test(model=model.double(), testloader=testloader, is_double_experiment=True)
            res_df.loc[res_df.shape[0]] = [models_labels[i], dataset, torch.double, accuracy, training_time, inference_time, model_size]

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    res_df.to_csv(f"results/experiment_{dt_string}.csv")
    return


if __name__ == "__main__":
    datasets = ["MNIST"]
    main(datasets)