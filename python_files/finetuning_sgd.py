import torch
import copy
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50 = models.resnet50(pretrained=True) # Load a pretrained ResNet18
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, 10) # CIFAR-10 has 100 classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet50
model.load_state_dict(torch.load('model_weights.pth'))
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

transform = transforms.Compose([
    transforms.Resize(224), # Resize images to fit ResNet input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # CIFAR-100 normalization
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

c = len(trainloader)
criterion = nn.CrossEntropyLoss()

n_models = 0

metrics_df = pd.DataFrame(columns=['Epoch', 'Running_Model_Loss', 
                                   'Running_Model_Accuracy'])

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

best_test_accuracy = 0.
for epoch in range(60): 
    running_loss = 0.
    for i, data in enumerate(tqdm(trainloader), start=1):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    with torch.no_grad():
        avg_train_loss = running_loss / len(trainloader)
        avg_test_loss, test_accuracy = test(model, testloader, device)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model_SGD_finetuning.pth')

        new_row = pd.DataFrame({'Epoch': [epoch], 'SGD_Model_Loss': [avg_test_loss], 'SGD_Model_Accuracy': [test_accuracy]})
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

        print(f"Epoch {epoch} - SGD Model Accuracy: {test_accuracy:.2f}%")
        metrics_df.to_csv('model_performance_metric_sgd.csv', index=False)
