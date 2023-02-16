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


def test(model, testloader, is_double_experiment=False): # TODO: modify to quantisize model AND inputs, and many quantization modes (dynamic, static, QAT ...)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if is_double_experiment:
                images = images.double()
            t0 = time.time()
            outputs = model(images)
            t1 = time.time()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
    print(f"Model Size: {model_size} MB")

    return accuracy, inference_time, model_size


def main():

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
            print("For fp32")
            accuracy, inference_time, model_size = test(model=model, testloader=testloader)
            res_df.loc[res_df.shape[0]] = [models_labels[i], dataset, torch.float, accuracy, training_time, inference_time, model_size]

            ## float16 and int8 --> doesn't work on macOS ...
            for quantization in [torch.float16, torch.qint8]:
                q_model = torch.quantization.quantize_dynamic(
                    model,                                  # the original model
                    {torch.nn.Conv2d, torch.nn.Linear},     # a set of layers to dynamically quantize
                    dtype=quantization                      # the target dtype for quantized weights
                )
                print(f"For {quantization}")
                accuracy, inference_time, model_size = test(model=q_model, testloader=testloader)
                res_df.loc[res_df.shape[0]] = [models_labels[i], dataset, quantization, accuracy, training_time, inference_time, model_size]

            ## fp64
            print(f"For fp64")
            accuracy, training_time, model_size = test(model=model.double(), testloader=testloader, is_double_experiment=True)
            res_df.loc[res_df.shape[0]] = [models_labels[i], dataset, torch.double, accuracy, training_time, inference_time, model_size]

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    res_df.to_csv(f"results/experiment_{dt_string}.csv")
    return


if __name__ == "__main__":
    datasets = ["MNIST"]
    main(datasets)