import torch
from torchvision import datasets, transforms
from torch import nn, optim
from sklearn.metrics import classification_report

# 1. GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando:", device)

# 2. Transformações (COM DATA AUGMENTATION)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 3. Dados
train_data = datasets.ImageFolder('archive/chest_xray/train', transform=train_transform)
test_data = datasets.ImageFolder('archive/chest_xray/test', transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

classes = train_data.classes

# 4. Modelo MELHORADO
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = CNN().to(device)

# 5. Loss melhor (sem sigmoid)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Treinamento
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().view(-1, 1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Acc: {100*correct/total:.2f}%")

print("Treinamento finalizado!")

# 7. Avaliação
model.eval()
correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.float().view(-1, 1).to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

accuracy = 100 * correct / total
print(f"\nAcurácia no teste: {accuracy:.2f}%")

# 8. Exemplos
print("\nExemplos de previsões:")

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).int()

        for i in range(min(5, len(images))):
            print(f"Previsto: {classes[preds[i].item()]} | Real: {classes[labels[i].item()]}")

        break

# 9. Relatório
print("\nRelatório de Classificação:")
print(classification_report(all_labels, all_preds, target_names=classes))