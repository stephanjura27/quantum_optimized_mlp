import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

data = load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

encoder = OneHotEncoder(sparse_output=False)
y_train_1hot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_1hot = encoder.transform(y_test.reshape(-1, 1))

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_1hot, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_1hot, dtype=torch.float32)

class NormalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(NormalCrossEntropyLoss, self).__init__()
    def forward(self, outputs, targets):
        epsilon = 1e-9
        ce_loss = -torch.sum(targets * torch.log(outputs + epsilon), dim=1)
        return torch.mean(ce_loss)

class ClassicMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ClassicMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        self.softmax = nn.Softmax(dim=1)
        nn.init.uniform_(self.fc1.weight, a=-1.0, b=1.0)
        nn.init.uniform_(self.fc2.weight, a=-1.0, b=1.0)
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

input_size = X_train.shape[1]
hidden_size = 32
output_size = y_train_1hot.shape[1]
model = ClassicMLP(input_size, hidden_size, output_size)

criterion = NormalCrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 16

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    _, true_labels = torch.max(targets.data, 1)
    correct = (predicted == true_labels).sum().item()
    accuracy = 100 * correct / targets.size(0)
    return accuracy

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
        correct_train += (torch.argmax(outputs, dim=1) == torch.argmax(batch_y, dim=1)).sum().item()
        total_train += batch_y.size(0)
    epoch_loss = running_loss / total_train
    epoch_acc = 100 * correct_train / total_train
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_X.size(0)
            correct_test += (torch.argmax(outputs, dim=1) == torch.argmax(batch_y, dim=1)).sum().item()
            total_test += batch_y.size(0)
    epoch_test_loss = test_loss / total_test
    epoch_test_acc = 100 * correct_test / total_test
    test_losses.append(epoch_test_loss)
    test_accuracies.append(epoch_test_acc)
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
    print(f"  Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.2f}%")

epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-', label='Train Loss')
plt.plot(epochs, test_losses, 'r-', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Cross Entropy Loss over Epochs')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
