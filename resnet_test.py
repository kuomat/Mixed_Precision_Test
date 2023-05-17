import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.cuda.amp import autocast, GradScaler

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4096, shuffle=True, num_workers=2)
print("Finished loading dataset")

num_classes = 10
model = resnet50(pretrained=True)
model.fc = nn.Linear(2048, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

device = torch.device('cuda:4' if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)

scaler = GradScaler()

print("Start training")
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Enable autocasting for forward pass
        with autocast():
          outputs = model(inputs)
          loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Print training progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
