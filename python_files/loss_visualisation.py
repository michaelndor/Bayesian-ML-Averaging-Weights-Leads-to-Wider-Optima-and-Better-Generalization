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

resnet50 = models.resnet50(pretrained=True)
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, 10)  # Ajuster pour les 10 classes de CIFAR-10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50 = resnet50.to(device)
model = resnet50
model.load_state_dict(torch.load('model_weights.pth'))

model_initial = models.resnet50(pretrained=True)
model_initial.fc = nn.Linear(num_ftrs, 10)
model_initial = model_initial.to(device)
model_initial.load_state_dict(torch.load('best_model_SGD_finetuning.pth'))

direction = [param1.data - param2.data for param1, param2 in zip(model.parameters(), model_initial.parameters()) if param1.requires_grad]
params = [param.data for param in model.parameters() if param.requires_grad]
# direction = [torch.randn_like(p) for p in params]

transform = transforms.Compose([
    transforms.Resize(224), # Resize images to fit ResNet input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # CIFAR-100 normalization
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

def evaluate_loss(model, dataloader, criterion, alpha, direction):
    original_params = [p.data.clone() for p in model.parameters()]
    with torch.no_grad():
        for p, d in zip(model.parameters(), direction):
            p.data.add_(alpha * d)   

    model.eval()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
    
    # Restauration des paramètres originaux
    with torch.no_grad():
        for p, original in zip(model.parameters(), original_params):
            p.data.copy_(original)
    
    return total_loss / len(dataloader)

alpha_range = np.linspace(-3, 3, 30)
loss_surface_alpha = np.zeros(len(alpha_range))
criterion = nn.CrossEntropyLoss()

for i, alpha in tqdm(enumerate(alpha_range), total=len(alpha_range)):
    loss_surface_alpha[i] = evaluate_loss(model, testloader, criterion, alpha, direction)
    print(loss_surface_alpha[i])

np.save('loss_surface_sgd_model.npy', loss_surface_alpha)
plt.plot(alpha_range, loss_surface_alpha, label='Loss in Alpha Direction')
plt.xlabel('Alpha')
plt.ylabel('Loss')
plt.title('Loss Surface along Alpha Direction')
plt.legend()
plt.show()
