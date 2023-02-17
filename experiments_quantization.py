# Main file to run the experiments

#%matplotlib inline

import torch
import torch.nn as nn

import pandas as pd
import time
from datetime import datetime
import os
from copy import deepcopy

from networks import LeNet, AlexNet, VGG11, BasicBlock, ResNet
from dataset_downloaders import load_dataset

backend = 'qnnpack'
torch.backends.quantized.engine = backend

def quantization_experiment(model, model_label, dataset, test_loader):
    data_types = ["float32", torch.qint8, "float64"]
    res = []
    for data_type in data_types:
        print(f"Begin for {model_label}, {data_type}")
        if data_type == "float32":
            model = model
            accuracy, mag, inference_time = test(model=model, testloader=test_loader)
            model_size = get_size_of_model(model)
        elif data_type == "float64":
            d_model = model.double()
            accuracy, mag, inference_time = test(model=d_model, testloader=test_loader, is_double_experiment=True)
            model_size = get_size_of_model(model)            
        else:
            q_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Conv2d, nn.Linear}, 
                dtype=data_type) 
            accuracy, mag, inference_time = test(model=q_model, testloader=test_loader)
            model_size = get_size_of_model(model)                  
        res.append([model_label, dataset, data_type, accuracy, mag, inference_time, model_size])
    return res


def get_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size


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
    mag = torch.mean(abs(outputs)).item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (accuracy))
    print(f'Elapsed time for inference: {inference_time}s')

    return accuracy, mag, inference_time


def main():

    res_df = pd.DataFrame(columns=["Model", "Dataset", "Quantization", "Prediction Accuracy", "Mean Abs Value of Output", "Inference Time", "Model Size"])

    trained_model_filenames= [model_filename for model_filename in os.listdir(path="models") if not model_filename[0]=="."]
    for trained_model_filename in trained_model_filenames:

        PATH = f"models/{trained_model_filename}"
        model_label = trained_model_filename.split("_")[0]
        dataset = trained_model_filename.split("_")[2]
        
        # # To test on only 1 model and 1 dataset
        # if not((model_label == "alexNet") & (dataset =="MNIST")):
        #     continue

        flag = (model_label == "leNet")
        _, test_loader, in_channels, num_classes = load_dataset(dataset=dataset, model_is_LeNet=flag)

        device = torch.device('cpu')
        assert model_label in ["leNet", "alexNet", "vgg11", "resNet18"]
        if model_label == "leNet":
            model = LeNet(in_channels=in_channels, num_classes=num_classes)
        elif model_label == "alexNet":
            model = AlexNet(in_channels=in_channels, num_classes=num_classes)
        elif model_label == "vgg11":
            model = VGG11(in_channels=in_channels, num_classes=num_classes)
        elif model_label == "resNet18":
            model = ResNet(in_channels=in_channels, num_layers=18, block=BasicBlock, num_classes=num_classes)
        
        model.load_state_dict(torch.load(PATH, map_location=device))
        print(f"Begin for {trained_model_filename}...")
        # experiment_results = quantization_experiment(model=model, model_label=model_label, dataset=dataset, test_loader=test_loader)
        # for result in experiment_results:
        #     res_df.loc[res_df.shape[0]] = result
        print(get_size_of_model(model, model_label))
        
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    res_df.to_csv(f"results/experiment_{dt_string}.csv")
    return


if __name__ == "__main__":
    main()