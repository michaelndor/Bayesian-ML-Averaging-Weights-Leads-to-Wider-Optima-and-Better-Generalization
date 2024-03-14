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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load('model_trained.pth')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

w_swa = copy.deepcopy(model.state_dict())
n_models = 0

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
for epoch in range(30):  
    model.train()
    running_loss = 0.
    for i, data in enumerate(tqdm(trainloader), start=1):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Logging training loss
    avg_train_loss = running_loss / len(trainloader)
    # Evaluate on test set
    avg_test_loss, test_accuracy = test(model, testloader, device)
    avg_train_loss, train_accuracy = test(model, trainloader, device)
    print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%', f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        print('saving...')
        torch.save(model.state_dict(), 'model_weights.pth')

print('Finished Training')
