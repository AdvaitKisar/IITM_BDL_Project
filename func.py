# Importing libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
torch.manual_seed(0)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def display_images(images, labels):
    figure = plt.figure(figsize=(10, 10))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        idx = torch.randint(len(labels), size=(1,)).item()
        img = images[idx]
        label_1 = chr(65+labels[idx])
        label_2 = chr(97+labels[idx])
        figure.add_subplot(rows, cols, i)
        plt.title(f'Image: {label_1} or {label_2}')
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

def load_data(dataset):
    file_name = f'emnist-letters-{dataset}.csv'
    data = pd.read_csv(file_name, header=None)
    labels = list(data.iloc[:, 0].astype(int))
    labels = np.array([l-1 for l in labels])
    images = data.iloc[:, 1:].values.reshape(-1, 28, 28)
    images = images.astype(float)

    # display_images(images, labels)
    images_flat = images.reshape(-1, images.shape[1]*images.shape[2])
    mean = np.mean(images_flat)
    std = np.std(images_flat)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[mean], std=[std]),  # Normalize pixel values
    ])
    images = transform(images)
    images = images.permute(1, 2, 0)
    images = images.view(-1, 1, 28, 28)
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    return images, labels

class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
    

def create_model(model_no, model_config):
    model_hyperparams = model_config[model_no]
    if model_hyperparams['Activation Function'] == 'Tanh':
        AF = nn.Tanh()
    elif model_hyperparams['Activation Function'] == 'ReLU':
        AF = nn.ReLU()
    elif model_hyperparams['Activation Function'] == 'Sigmoid':
        AF = nn.Sigmoid()

    J1 = model_hyperparams['J1']
    J2 = model_hyperparams['J2']
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(num_features=6),
        AF,
        nn.AvgPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
        nn.BatchNorm2d(num_features=16),
        AF,
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=16, out_channels=J1, kernel_size=5, stride=1),
        nn.BatchNorm2d(num_features=J1),
        AF,
        
        nn.Flatten(),
        nn.Linear(in_features=J1, out_features=J2),
        nn.Linear(in_features=J2, out_features=26),
    )
    return model

def accuracy(out, labels): # Function for calculating accuracy
    '''
    out: output logits with 
    labels: labels with dimension (n_samples,)
    '''
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

def evaluate_model(model, dataloader): # Evaluates the model on test data
    total_loss = 0
    correct = 0
    total_samples = 0

    total_tp = 0  # True positives
    total_tn = 0  # True negatives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for images, labels in dataloader:
            images = images.float()
            labels = torch.tensor(labels, dtype=torch.long)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            correct += accuracy(outputs, labels)
            outputs.detach()
            total_samples += labels.size(0)

            for i in range(len(labels)):
                if predicted[i] == labels[i]:  # Correct prediction
                    if labels[i] == 1:  # True positive
                        total_tp += 1
                    else:
                        total_tn += 1  # True negative
                else:  # Incorrect prediction
                    if predicted[i] == 1:  # False positive
                        total_fp += 1
                    else:
                        total_fn += 1  # False negative
    acc = (correct/total_samples)*100
    avg_loss = total_loss/total_samples

    print("Evaluation Metrics for Test Set:- ")
    print(f"Accuracy: {acc:.2f} %")
    print(f"Average Loss: {avg_loss:.4f}")
    
    # Metrics across classes
    macro_precision = precision_score(labels, predicted, average='macro')
    macro_recall = recall_score(labels, predicted, average='macro')
    macro_f1 = f1_score(labels, predicted, average='macro')

    print(f"Macro-averaged Precision: {macro_precision:.4f}")
    print(f"Macro-averaged Recall: {macro_recall:.4f}")
    print(f"Macro-averaged F1-Score: {macro_f1:.4f}")