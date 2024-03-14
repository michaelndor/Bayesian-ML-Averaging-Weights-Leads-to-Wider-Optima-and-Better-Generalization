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

model = torch.load('model_weights.pth')
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

writer = SummaryWriter('runs/cifar100_resnet50_finetuning_SWA')

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Example optimizer
transform = transforms.Compose([
    transforms.Resize(224), # Resize images to fit ResNet input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # CIFAR-100 normalization
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

alpha1, alpha2 = 0.001, 0.001  # LR bounds (for constant LR, alpha1 = alpha2)
c = len(trainloader)
criterion = nn.CrossEntropyLoss()

w_swa = copy.deepcopy(model.state_dict())
n_models = 0

metrics_df = pd.DataFrame(columns=['Epoch', 'Running_Model_Loss', 'SWA_Model_Loss', 
                                   'Running_Model_Accuracy', 'SWA_Model_Accuracy'])


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
for epoch in range(40): 
    running_loss = 0.
    swa_loss = 0.
    model.train()
    for i, data in enumerate(tqdm(trainloader), start=1):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    n_models += 1
    with torch.no_grad():
        if i % 10 == 0 :
            model.load_state_dict(w_swa)
            n_models = 1
            
        # Update SWA parameters
        for name, w_model_param in model.state_dict().items():
            if w_model_param.is_floating_point():
                w_swa_param = w_swa[name]
                w_swa_param.mul_(n_models - 1).add_(w_model_param.data).div_(n_models)

        swa_model = copy.deepcopy(model)
        swa_model.load_state_dict(w_swa)

        # Evaluate running model
        avg_train_loss = running_loss / len(trainloader)
        avg_test_loss, test_accuracy = test(model, testloader, device)

        swa_test_loss, swa_test_accuracy = test(swa_model, testloader, device)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model_p_SWA_finetuning.pth')

        new_row = pd.DataFrame({'Epoch': [epoch], 'Running_Model_Loss': [avg_test_loss],
                                'SWA_Model_Loss': [swa_test_loss], 'Running_Model_Accuracy': [test_accuracy],
                                'SWA_Model_Accuracy': [swa_test_accuracy]})
        
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

        print(f"Epoch {epoch} - Running Model Accuracy: {test_accuracy:.2f}%, SWA Model Accuracy: {swa_test_accuracy:.2f}%")
        metrics_df.to_csv('model_performance_metric_p_swa.csv', index=False)

