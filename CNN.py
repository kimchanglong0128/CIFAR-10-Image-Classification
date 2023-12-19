# 在使用MLP调参时, 发现正则化会使 Accuracy降低(Dropout, 梯度衰减)
# epoch 的变化对 Accuracy 影响不大
# 激活函数也是同样影响不大
# 所以在CNN中不调此类参数

# 同时优化器分别使用 Adam , SGD, RMSprop, AdamW, Nadam
# 隐藏层为2~7
# learn_rate为0.001,0.07,0.05,0.01 
# momentum为 0.9, 0.8, 0.7, 0.6, 0.5
# 在使用MLP调参时使用的时CPU, CNN使用GPU加速


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SimpleCNN(nn.Module):
    def __init__(self, num_hidden_layers):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        flattened_size = 64 * 16 * 16  # 根据卷积层的输出更新这个值
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Sequential(nn.Linear(flattened_size, 512), nn.ReLU()))
            flattened_size = 512  # 更新为隐藏层的输出尺寸

        self.output_layer = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv_layers(x)
        # print(x.size())
        x = x.view(x.size(0), -1)  # Flatten the tensor
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def train_and_evaluate(model, train_loader, test_loader, optimizer, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}: Accuracy: {accuracy}%')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

learning_rates = [0.001, 0.07, 0.05, 0.01]
momentums = [0.9, 0.8, 0.7, 0.6, 0.5]
hidden_layers_counts = [2, 3, 4, 5, 6, 7]

for hidden_layers in hidden_layers_counts:
    for lr in learning_rates:
        for momentum in momentums:
            model = SimpleCNN(hidden_layers).to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            print(f"Training with SGD - Hidden Layers: {hidden_layers}, LR: {lr}, Momentum: {momentum}")
            train_and_evaluate(model, train_loader, test_loader, optimizer, device)

# 对于其他优化器，不需要使用动量参数
other_optimizers = [optim.Adam, optim.RMSprop, optim.AdamW, optim.NAdam]
for hidden_layers in hidden_layers_counts:
    for opt in other_optimizers:
        for lr in learning_rates:
            model = SimpleCNN(hidden_layers).to(device)
            optimizer = opt(model.parameters(), lr=lr)
            print(f"Training with {opt.__name__} - Hidden Layers: {hidden_layers}, LR: {lr}")
            train_and_evaluate(model, train_loader, test_loader, optimizer, device)

# Training with AdamW - Hidden Layers: 3, LR: 0.001
# MAX -> Epoch 3: Accuracy: 71.70%




# 优化器和隐藏层配置的影响:
# Adam 优化器：在大多数隐藏层配置下，Adam 优化器在低学习率（0.001）时表现最佳
# 随着隐藏层数量的增加，性能呈现波动趋势，但在较低的学习率下性能相对稳定

# SGD 优化器：SGD优化器在较低动量（如0.6）和中等学习率（0.01）下表现较好
# 随着隐藏层数量的增加，性能有所下降，这表明对于更深层次的网络，SGD优化器可能不是最优选择

# RMSprop 优化器：RMSprop 在低学习率（0.001）下与中等隐藏层数量（如3-4层）结合时，表现较好
# 然而，当隐藏层数量增加到6-7层时，其性能开始下降

# AdamW和NAdam 优化器：这两种优化器在低学习率下表现相似，尤其是在中等隐藏层配置（如3-4层）时
# 然而，随着隐藏层数量的增加，性能也开始下降

# 学习率和隐藏层数量的影响:
# 高学习率的影响：在所有优化器中，较高的学习率（如 0.05, 0.07）普遍导致性能严重下降，通常只达到约10%的准确率
# 这表明高学习率可能导致模型无法有效学习

# 隐藏层数量的影响：在大多数情况下，2-4个隐藏层的配置提供了较好的结果
# 然而，随着隐藏层数量的增加到6-7层，多数优化器的性能开始下降
# 这可能是由于过多的隐藏层导致了过拟合或训练难度增加

# 总结:

# 在CNN中，选择合适的优化器和学习率组合对于提高模型性能至关重要
# 低学习率通常更适合于较深的网络结构

# 隐藏层数量应根据任务的复杂性和可用数据量来选择
# 过多的隐藏层可能会导致过拟合或训练效率下降




# Epoch 1: Accuracy: 62.91%
# Epoch 2: Accuracy: 66.71%
# Epoch 3: Accuracy: 67.55%
# Epoch 4: Accuracy: 68.86%
# Epoch 5: Accuracy: 67.02%
# Epoch 6: Accuracy: 68.12%
# Epoch 7: Accuracy: 66.21%
# Epoch 8: Accuracy: 68.43%
# Epoch 9: Accuracy: 68.63%
# Epoch 10: Accuracy: 65.42%
# Training with RMSprop - Hidden Layers: 2, LR: 0.07
# Epoch 1: Accuracy: 14.53%
# Epoch 2: Accuracy: 9.97%
# Epoch 3: Accuracy: 10.02%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 2, LR: 0.05
# Epoch 1: Accuracy: 10.01%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 2, LR: 0.01
# Epoch 1: Accuracy: 25.08%
# Epoch 2: Accuracy: 13.3%
# Epoch 3: Accuracy: 9.99%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 2, LR: 0.001
# Epoch 1: Accuracy: 65.47%
# Epoch 2: Accuracy: 71.34%
# Epoch 3: Accuracy: 71.5%
# Epoch 4: Accuracy: 71.09%
# Epoch 5: Accuracy: 70.54%
# Epoch 6: Accuracy: 70.59%
# Epoch 7: Accuracy: 70.45%
# Epoch 8: Accuracy: 69.49%
# Epoch 9: Accuracy: 69.79%
# Epoch 10: Accuracy: 69.13%
# Training with AdamW - Hidden Layers: 2, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 2, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 2, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 2, LR: 0.001
# Epoch 1: Accuracy: 62.96%
# Epoch 2: Accuracy: 70.56%
# Epoch 3: Accuracy: 71.44%
# Epoch 4: Accuracy: 70.7%
# Epoch 5: Accuracy: 70.08%
# Epoch 6: Accuracy: 70.7%
# Epoch 7: Accuracy: 70.46%
# Epoch 8: Accuracy: 69.64%
# Epoch 9: Accuracy: 69.72%
# Epoch 10: Accuracy: 69.63%
# Training with NAdam - Hidden Layers: 2, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 2, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 2, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with Adam - Hidden Layers: 3, LR: 0.001
# Epoch 1: Accuracy: 66.98%
# Epoch 2: Accuracy: 69.97%
# Epoch 3: Accuracy: 71.14%
# Epoch 4: Accuracy: 70.32%
# Epoch 5: Accuracy: 70.41%
# Epoch 6: Accuracy: 69.08%
# Epoch 7: Accuracy: 69.38%
# Epoch 8: Accuracy: 68.74%
# Epoch 9: Accuracy: 69.73%
# Epoch 10: Accuracy: 69.1%
# Training with Adam - Hidden Layers: 3, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with Adam - Hidden Layers: 3, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with Adam - Hidden Layers: 3, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 3, LR: 0.001
# Epoch 1: Accuracy: 56.88%
# Epoch 2: Accuracy: 59.15%
# Epoch 3: Accuracy: 67.82%
# Epoch 4: Accuracy: 67.87%
# Epoch 5: Accuracy: 66.87%
# Epoch 6: Accuracy: 65.95%
# Epoch 7: Accuracy: 65.34%
# Epoch 8: Accuracy: 62.88%
# Epoch 9: Accuracy: 66.43%
# Epoch 10: Accuracy: 64.78%
# Training with RMSprop - Hidden Layers: 3, LR: 0.07
# Epoch 1: Accuracy: 9.99%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 3, LR: 0.05
# Epoch 1: Accuracy: 10.07%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.01%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 3, LR: 0.01
# Epoch 1: Accuracy: 33.55%
# Epoch 2: Accuracy: 35.69%
# Epoch 3: Accuracy: 29.54%
# Epoch 4: Accuracy: 38.39%
# Epoch 5: Accuracy: 45.33%
# Epoch 6: Accuracy: 45.34%
# Epoch 7: Accuracy: 45.25%
# Epoch 8: Accuracy: 56.85%
# Epoch 9: Accuracy: 56.29%
# Epoch 10: Accuracy: 54.6%
# Training with AdamW - Hidden Layers: 3, LR: 0.001
# Epoch 1: Accuracy: 66.51%
# Epoch 2: Accuracy: 70.36%
# Epoch 3: Accuracy: 71.7%
# Epoch 4: Accuracy: 70.53%
# Epoch 5: Accuracy: 69.5%
# Epoch 6: Accuracy: 68.58%
# Epoch 7: Accuracy: 70.0%
# Epoch 8: Accuracy: 69.43%
# Epoch 9: Accuracy: 69.08%
# Epoch 10: Accuracy: 68.81%
# Training with AdamW - Hidden Layers: 3, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 3, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 3, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 3, LR: 0.001
# Epoch 1: Accuracy: 59.96%
# Epoch 2: Accuracy: 69.14%
# Epoch 3: Accuracy: 67.51%
# Epoch 4: Accuracy: 69.73%
# Epoch 5: Accuracy: 68.45%
# Epoch 6: Accuracy: 69.99%
# Epoch 7: Accuracy: 69.81%
# Epoch 8: Accuracy: 70.09%
# Epoch 9: Accuracy: 69.22%
# Epoch 10: Accuracy: 68.41%
# Training with NAdam - Hidden Layers: 3, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 3, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 3, LR: 0.01
# Epoch 1: Accuracy: 36.95%
# Epoch 2: Accuracy: 38.9%
# Epoch 3: Accuracy: 47.02%
# Epoch 4: Accuracy: 49.55%
# Epoch 5: Accuracy: 53.93%
# Epoch 6: Accuracy: 57.07%
# Epoch 7: Accuracy: 56.5%
# Epoch 8: Accuracy: 57.62%
# Epoch 9: Accuracy: 59.69%
# Epoch 10: Accuracy: 58.62%
# Training with Adam - Hidden Layers: 4, LR: 0.001
# Epoch 1: Accuracy: 61.81%
# Epoch 2: Accuracy: 68.93%
# Epoch 3: Accuracy: 70.38%
# Epoch 4: Accuracy: 69.82%
# Epoch 5: Accuracy: 69.97%
# Epoch 6: Accuracy: 68.79%
# Epoch 7: Accuracy: 67.99%
# Epoch 8: Accuracy: 69.24%
# Epoch 9: Accuracy: 69.22%
# Epoch 10: Accuracy: 70.05%
# Training with Adam - Hidden Layers: 4, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with Adam - Hidden Layers: 4, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with Adam - Hidden Layers: 4, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 4, LR: 0.001
# Epoch 1: Accuracy: 50.18%
# Epoch 2: Accuracy: 56.02%
# Epoch 3: Accuracy: 63.21%
# Epoch 4: Accuracy: 64.67%
# Epoch 5: Accuracy: 65.3%
# Epoch 6: Accuracy: 64.63%
# Epoch 7: Accuracy: 67.25%
# Epoch 8: Accuracy: 66.17%
# Epoch 9: Accuracy: 66.89%
# Epoch 10: Accuracy: 66.57%
# Training with RMSprop - Hidden Layers: 4, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 4, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 4, LR: 0.01
# Epoch 1: Accuracy: 28.13%
# Epoch 2: Accuracy: 31.25%
# Epoch 3: Accuracy: 31.88%
# Epoch 4: Accuracy: 12.48%
# Epoch 5: Accuracy: 40.9%
# Epoch 6: Accuracy: 43.7%
# Epoch 7: Accuracy: 9.94%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 4, LR: 0.001
# Epoch 1: Accuracy: 61.93%
# Epoch 2: Accuracy: 67.97%
# Epoch 3: Accuracy: 70.17%
# Epoch 4: Accuracy: 69.76%
# Epoch 5: Accuracy: 69.35%
# Epoch 6: Accuracy: 69.0%
# Epoch 7: Accuracy: 69.71%
# Epoch 8: Accuracy: 69.31%
# Epoch 9: Accuracy: 69.5%
# Epoch 10: Accuracy: 69.29%
# Training with AdamW - Hidden Layers: 4, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 4, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 4, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 4, LR: 0.001
# Epoch 1: Accuracy: 53.12%
# Epoch 2: Accuracy: 67.06%
# Epoch 3: Accuracy: 69.35%
# Epoch 4: Accuracy: 68.48%
# Epoch 5: Accuracy: 69.32%
# Epoch 6: Accuracy: 68.85%
# Epoch 7: Accuracy: 68.67%
# Epoch 8: Accuracy: 67.84%
# Epoch 9: Accuracy: 69.01%
# Epoch 10: Accuracy: 69.25%
# Training with NAdam - Hidden Layers: 4, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 4, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 4, LR: 0.01
# Epoch 1: Accuracy: 22.51%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with Adam - Hidden Layers: 5, LR: 0.001
# Epoch 1: Accuracy: 59.92%
# Epoch 2: Accuracy: 67.22%
# Epoch 3: Accuracy: 69.6%
# Epoch 4: Accuracy: 70.13%
# Epoch 5: Accuracy: 68.86%
# Epoch 6: Accuracy: 69.62%
# Epoch 7: Accuracy: 69.3%
# Epoch 8: Accuracy: 68.43%
# Epoch 9: Accuracy: 68.63%
# Epoch 10: Accuracy: 68.33%
# Training with Adam - Hidden Layers: 5, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with Adam - Hidden Layers: 5, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with Adam - Hidden Layers: 5, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 5, LR: 0.001
# Epoch 1: Accuracy: 42.42%
# Epoch 2: Accuracy: 60.68%
# Epoch 3: Accuracy: 59.67%
# Epoch 4: Accuracy: 65.78%
# Epoch 5: Accuracy: 67.55%
# Epoch 6: Accuracy: 67.18%
# Epoch 7: Accuracy: 65.93%
# Epoch 8: Accuracy: 66.23%
# Epoch 9: Accuracy: 66.7%
# Epoch 10: Accuracy: 65.95%
# Training with RMSprop - Hidden Layers: 5, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 9.99%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 9.99%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 5, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 5, LR: 0.01
# Epoch 1: Accuracy: 15.62%
# Epoch 2: Accuracy: 21.02%
# Epoch 3: Accuracy: 16.33%
# Epoch 4: Accuracy: 19.65%
# Epoch 5: Accuracy: 22.99%
# Epoch 6: Accuracy: 10.23%
# Epoch 7: Accuracy: 15.98%
# Epoch 8: Accuracy: 10.05%
# Epoch 9: Accuracy: 14.38%
# Epoch 10: Accuracy: 16.43%
# Training with AdamW - Hidden Layers: 5, LR: 0.001
# Epoch 1: Accuracy: 59.47%
# Epoch 2: Accuracy: 67.33%
# Epoch 3: Accuracy: 68.68%
# Epoch 4: Accuracy: 69.93%
# Epoch 5: Accuracy: 70.77%
# Epoch 6: Accuracy: 69.51%
# Epoch 7: Accuracy: 69.17%
# Epoch 8: Accuracy: 67.91%
# Epoch 9: Accuracy: 68.5%
# Epoch 10: Accuracy: 68.79%
# Training with AdamW - Hidden Layers: 5, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 5, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 5, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 5, LR: 0.001
# Epoch 1: Accuracy: 59.23%
# Epoch 2: Accuracy: 66.64%
# Epoch 3: Accuracy: 66.85%
# Epoch 4: Accuracy: 69.39%
# Epoch 5: Accuracy: 69.0%
# Epoch 6: Accuracy: 68.81%
# Epoch 7: Accuracy: 68.07%
# Epoch 8: Accuracy: 68.77%
# Epoch 9: Accuracy: 69.48%
# Epoch 10: Accuracy: 69.22%
# Training with NAdam - Hidden Layers: 5, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 5, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 5, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with Adam - Hidden Layers: 6, LR: 0.001
# Epoch 1: Accuracy: 53.35%
# Epoch 2: Accuracy: 65.76%
# Epoch 3: Accuracy: 68.7%
# Epoch 4: Accuracy: 67.27%
# Epoch 5: Accuracy: 69.76%
# Epoch 6: Accuracy: 69.21%
# Epoch 7: Accuracy: 68.65%
# Epoch 8: Accuracy: 68.94%
# Epoch 9: Accuracy: 68.4%
# Epoch 10: Accuracy: 69.1%
# Training with Adam - Hidden Layers: 6, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with Adam - Hidden Layers: 6, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with Adam - Hidden Layers: 6, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 6, LR: 0.001
# Epoch 1: Accuracy: 43.3%
# Epoch 2: Accuracy: 56.79%
# Epoch 3: Accuracy: 62.0%
# Epoch 4: Accuracy: 65.88%
# Epoch 5: Accuracy: 68.39%
# Epoch 6: Accuracy: 68.49%
# Epoch 7: Accuracy: 68.12%
# Epoch 8: Accuracy: 65.69%
# Epoch 9: Accuracy: 67.7%
# Epoch 10: Accuracy: 65.61%
# Training with RMSprop - Hidden Layers: 6, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 6, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 6, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 6, LR: 0.001
# Epoch 1: Accuracy: 56.99%
# Epoch 2: Accuracy: 64.19%
# Epoch 3: Accuracy: 68.41%
# Epoch 4: Accuracy: 69.2%
# Epoch 5: Accuracy: 69.65%
# Epoch 6: Accuracy: 68.96%
# Epoch 7: Accuracy: 68.52%
# Epoch 8: Accuracy: 69.25%
# Epoch 9: Accuracy: 69.11%
# Epoch 10: Accuracy: 68.82%
# Training with AdamW - Hidden Layers: 6, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 6, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 6, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 6, LR: 0.001
# Epoch 1: Accuracy: 55.95%
# Epoch 2: Accuracy: 55.44%
# Epoch 3: Accuracy: 66.07%
# Epoch 4: Accuracy: 65.43%
# Epoch 5: Accuracy: 68.65%
# Epoch 6: Accuracy: 67.32%
# Epoch 7: Accuracy: 68.66%
# Epoch 8: Accuracy: 67.63%
# Epoch 9: Accuracy: 67.86%
# Epoch 10: Accuracy: 68.35%
# Training with NAdam - Hidden Layers: 6, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 6, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 6, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with Adam - Hidden Layers: 7, LR: 0.001
# Epoch 1: Accuracy: 43.43%
# Epoch 2: Accuracy: 60.38%
# Epoch 3: Accuracy: 67.65%
# Epoch 4: Accuracy: 68.52%
# Epoch 5: Accuracy: 69.22%
# Epoch 6: Accuracy: 67.43%
# Epoch 7: Accuracy: 68.46%
# Epoch 8: Accuracy: 69.7%
# Epoch 9: Accuracy: 68.66%
# Epoch 10: Accuracy: 68.42%
# Training with Adam - Hidden Layers: 7, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with Adam - Hidden Layers: 7, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with Adam - Hidden Layers: 7, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 7, LR: 0.001
# Epoch 1: Accuracy: 33.11%
# Epoch 2: Accuracy: 55.7%
# Epoch 3: Accuracy: 60.02%
# Epoch 4: Accuracy: 66.18%
# Epoch 5: Accuracy: 66.98%
# Epoch 6: Accuracy: 65.64%
# Epoch 7: Accuracy: 67.43%
# Epoch 8: Accuracy: 67.43%
# Epoch 9: Accuracy: 65.99%
# Epoch 10: Accuracy: 68.78%
# Training with RMSprop - Hidden Layers: 7, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 7, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with RMSprop - Hidden Layers: 7, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 7, LR: 0.001
# Epoch 1: Accuracy: 46.62%
# Epoch 2: Accuracy: 60.63%
# Epoch 3: Accuracy: 65.49%
# Epoch 4: Accuracy: 67.3%
# Epoch 5: Accuracy: 68.0%
# Epoch 6: Accuracy: 66.96%
# Epoch 7: Accuracy: 68.04%
# Epoch 8: Accuracy: 67.46%
# Epoch 9: Accuracy: 67.06%
# Epoch 10: Accuracy: 67.4%
# Training with AdamW - Hidden Layers: 7, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 7, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with AdamW - Hidden Layers: 7, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 7, LR: 0.001
# Epoch 1: Accuracy: 47.05%
# Epoch 2: Accuracy: 58.36%
# Epoch 3: Accuracy: 65.48%
# Epoch 4: Accuracy: 68.6%
# Epoch 5: Accuracy: 68.86%
# Epoch 6: Accuracy: 69.63%
# Epoch 7: Accuracy: 66.97%
# Epoch 8: Accuracy: 69.46%
# Epoch 9: Accuracy: 68.67%
# Epoch 10: Accuracy: 67.26%
# Training with NAdam - Hidden Layers: 7, LR: 0.07
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 7, LR: 0.05
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%
# Training with NAdam - Hidden Layers: 7, LR: 0.01
# Epoch 1: Accuracy: 10.0%
# Epoch 2: Accuracy: 10.0%
# Epoch 3: Accuracy: 10.0%
# Epoch 4: Accuracy: 10.0%
# Epoch 5: Accuracy: 10.0%
# Epoch 6: Accuracy: 10.0%
# Epoch 7: Accuracy: 10.0%
# Epoch 8: Accuracy: 10.0%
# Epoch 9: Accuracy: 10.0%
# Epoch 10: Accuracy: 10.0%