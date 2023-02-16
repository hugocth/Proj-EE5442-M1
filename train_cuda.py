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
from dataset_downloaders import load_dataset


if torch.cuda.is_available():
  device = torch.device("cuda:0")
else:
  device = torch.device("cpu")
print(device)


def train(model, model_label, trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print(f"Begin training for {model_label}")
    t0 = time.time()
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            outputs = outputs.to(device)
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


def main(datasets): # TODO : parser

    for dataset in datasets:
        ## Load dataset
        trainloader, _, in_channels, num_classes = load_dataset(dataset)

        ## Create instances of models
        leNet = LeNet(in_channels=in_channels, num_classes=num_classes)
        alexNet = AlexNet(in_channels=in_channels, num_classes=num_classes)
        vgg11 = VGG11(in_channels=in_channels, num_classes=num_classes)
        resNet18 = ResNet(in_channels=in_channels, num_layers=18, block=BasicBlock, num_classes=num_classes)
        models = [leNet, alexNet, vgg11, resNet18]
        models_labels = ["leNet", "alexNet", "vgg11", "resNet18"]


        for i, model in enumerate(models):
            if models_labels[i] == "leNet":
                trainloader, _, in_channels, num_classes = load_dataset(dataset, model_is_LeNet=True)
            else:
                trainloader, _, in_channels, num_classes = load_dataset(dataset)
            model = model.to(device)
            model, training_time = train(model=model, model_label=models_labels[i], trainloader=trainloader)
            torch.save(model.state_dict(), f"models/{models_labels[i]}_on_{dataset}")


if __name__ == "__main__":
    datasets = ["MNIST"]
    main(datasets)