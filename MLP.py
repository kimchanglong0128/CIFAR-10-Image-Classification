import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32*32*3 , 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)  # 压平图像
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
# 加载DIFAR10数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# 初始化模型 , 损失函数, 优化器
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)   # 学习率0.001


# 训练模型
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for data, target in train_loader:
        optimizer_adam.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer_adam.step()

# 测试模型
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy : {100 * correct / total}%')
# 优化器Adam: 53.91%

# --------------------------------------------------------------------------------------------------------

### 优化器
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer_SGD = optim.SGD(model.parameters(), lr=0.001)   # 学习率0.001

# 训练模型
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for data, target in train_loader:
        optimizer_SGD.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer_SGD.step()

# 测试模型
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy : {100 * correct / total}%')
# 优化器SDG: 35.56%

# --------------------------------------------------------------------------------------------------------

# 造成准确率如此差异的原因:

# 1. 优化器的差异
# Adam: 结合了动量和自适应学习率的优化器，通常能更快地适应训练数据，特别是在初期阶段。对初始学习率的选择不太敏感
# SGD: 标准的随机梯度下降，可能需要更长的时间来收敛，并且对学习率和其他超参数的选择更为敏感
# --------------------------------------------------------------------------------------------------------
# 2. 学习率设定
# 两种优化器的学习率都设置为了 0.001

# 针对Adam

model = MLP()
learning_rates = [0.001, 0.07, 0.05, 0.01]

for lr in learning_rates:
    print(f"(Adam) Training with learning rate: {lr}")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for data, target in train_loader:
            optimizer.zero_grad() 
            output = model(data) 
            loss = criterion(output, target) 
            loss.backward()  
            optimizer.step() 

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')


# (Adam)Training with learning rate: 0.001
# Epoch 1, Loss: 1.3067758083343506
# Epoch 2, Loss: 2.0453577041625977
# Epoch 3, Loss: 1.3213486671447754
# Epoch 4, Loss: 1.5622050762176514
# Epoch 5, Loss: 1.0327119827270508
# Epoch 6, Loss: 1.375564694404602
# Epoch 7, Loss: 1.0017966032028198
# Epoch 8, Loss: 1.3338642120361328
# Epoch 9, Loss: 0.7528111338615417
# Epoch 10, Loss: 0.9362533688545227
# (Adam)Training with learning rate: 0.07
# Epoch 1, Loss: 2.282205104827881
# Epoch 2, Loss: 2.2858383655548096
# Epoch 3, Loss: 2.3051414489746094
# Epoch 4, Loss: 2.3706634044647217
# Epoch 5, Loss: 2.2924795150756836
# Epoch 6, Loss: 2.326995372772217
# Epoch 7, Loss: 2.299683094024658
# Epoch 8, Loss: 2.325462818145752
# Epoch 9, Loss: 2.2993574142456055
# Epoch 10, Loss: 2.2690093517303467
# (Adam)Training with learning rate: 0.05
# Epoch 1, Loss: 2.329481363296509
# Epoch 2, Loss: 2.3233699798583984
# Epoch 3, Loss: 2.328845739364624
# Epoch 4, Loss: 2.2643649578094482
# Epoch 5, Loss: 2.2964015007019043
# Epoch 6, Loss: 2.323727607727051
# Epoch 7, Loss: 2.312704563140869
# Epoch 8, Loss: 2.3250341415405273
# Epoch 9, Loss: 2.3244714736938477
# Epoch 10, Loss: 2.3290178775787354
# (Adam)Training with learning rate: 0.01
# Epoch 1, Loss: 2.299947738647461
# Epoch 2, Loss: 2.3263771533966064
# Epoch 3, Loss: 2.2918975353240967
# Epoch 4, Loss: 2.3149948120117188
# Epoch 5, Loss: 2.3051693439483643
# Epoch 6, Loss: 2.308928966522217
# Epoch 7, Loss: 2.3128104209899902
# Epoch 8, Loss: 2.3095226287841797
# Epoch 9, Loss: 2.3058979511260986
# Epoch 10, Loss: 2.3032801151275635


# Adam 优化器
# 学习率 0.001：开始时损失较高，但随着训练的进行逐渐降低。这表明学习率可能设置得较为合适，模型正在学习
# 学习率 0.07, 0.05, 0.01：损失值在整个训练过程中保持相对较高和不稳定，这可能表明学习率过高，导致模型未能有效收敛

# --------------------------------------------------------------------------------------------------------

# SGD 针对不同的学习率所表现的情况
model = MLP()

learning_rates = [0.001, 0.07, 0.05, 0.01]

for lr in learning_rates:
    print(f"(SGD)Training with learning rate: {lr}")

    ## 使用 SGD 优化器并设置学习率
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for data, target in train_loader:
            optimizer.zero_grad()  
            output = model(data)  
            loss = criterion(output, target)  
            loss.backward()  
            optimizer.step()  

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')


# (SGD)Training with learning rate: 0.001
# Epoch 1, Loss: 2.2136359214782715
# Epoch 2, Loss: 2.1773197650909424
# Epoch 3, Loss: 2.2085626125335693
# Epoch 4, Loss: 1.9154356718063354
# Epoch 5, Loss: 1.9829708337783813
# Epoch 6, Loss: 1.9546219110488892
# Epoch 7, Loss: 2.0249335765838623
# Epoch 8, Loss: 1.6976277828216553
# Epoch 9, Loss: 1.688340425491333
# Epoch 10, Loss: 1.686211109161377
# (SGD)Training with learning rate: 0.07
# Epoch 1, Loss: 1.3891711235046387
# Epoch 2, Loss: 1.8403393030166626
# Epoch 3, Loss: 1.260148048400879
# Epoch 4, Loss: 0.9171316623687744
# Epoch 5, Loss: 1.083350658416748
# Epoch 6, Loss: 1.0029075145721436
# Epoch 7, Loss: 1.5473055839538574
# Epoch 8, Loss: 0.9807155728340149
# Epoch 9, Loss: 1.1742918491363525
# Epoch 10, Loss: 0.7865316271781921
# (SGD)Training with learning rate: 0.05
# Epoch 1, Loss: 0.7814550399780273
# Epoch 2, Loss: 0.6766301989555359
# Epoch 3, Loss: 0.26684027910232544
# Epoch 4, Loss: 0.6342592835426331
# Epoch 5, Loss: 0.4543738067150116
# Epoch 6, Loss: 0.41346946358680725
# Epoch 7, Loss: 0.16102857887744904
# Epoch 8, Loss: 0.6235482096672058
# Epoch 9, Loss: 0.46076324582099915
# Epoch 10, Loss: 0.4094746708869934
# (SGD)Training with learning rate: 0.01
# Epoch 1, Loss: 0.07576204836368561
# Epoch 2, Loss: 0.09030319005250931
# Epoch 3, Loss: 0.038117796182632446
# Epoch 4, Loss: 0.05483808368444443
# Epoch 5, Loss: 0.08883199840784073
# Epoch 6, Loss: 0.0991552472114563
# Epoch 7, Loss: 0.07003789395093918
# Epoch 8, Loss: 0.09551642835140228
# Epoch 9, Loss: 0.07037242501974106
# Epoch 10, Loss: 0.08626367896795273





# SGD 优化器的结果
# 学习率 0.001：损失逐渐降低，但整体下降趋势不如学习率为 0.01 时明显。这可能表明该学习率对于 SGD 来说偏低
# 学习率 0.07：损失值波动较大，这通常是学习率设置过高的迹象
# 学习率 0.05, 0.01：损失随着训练的进行而逐渐降低，且在学习率为 0.01 时表现更为稳定和有效

# 综合分析
# Adam vs. SGD：Adam 通常对学习率的选择不太敏感，而 SGD 对学习率更为敏感
# 这可能解释了为什么在使用 Adam 时，即使学习率变化，损失也能逐渐降低

# 适宜学习率的选择：对于 SGD 来说，学习率 0.01 似乎是一个较为合适的选择，因为它显示出了稳定的损失下降
# 对于 Adam，学习率 0.001 可能是一个更好的起点。

# --------------------------------------------------------------------------------------------------------

# 3. 缺乏动量（Momentum）
# 在 SGD 的应用中，通常会引入动量来帮助优化器跨越局部最小值并加速学习过程。在此 SGD 配置中，没有使用动量，这可能也是性能较低的一个因素

model = MLP()

# 定义动量
momentums = [0.9, 0.8, 0.7, 0.6, 0.5]

for momentum in momentums:
    print(f"(SGD, moentum) Training with moentum: {momentum}")

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=momentum)

    for epoch in range(num_epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# (SGD, moentum) Training with moentum: 0.9
# Epoch 1, Loss: 1.6882946491241455
# Epoch 2, Loss: 1.270872950553894
# Epoch 3, Loss: 1.3693727254867554
# Epoch 4, Loss: 0.8768354654312134
# Epoch 5, Loss: 1.2565217018127441
# Epoch 6, Loss: 1.4576574563980103
# Epoch 7, Loss: 1.7720766067504883
# Epoch 8, Loss: 0.6429325342178345
# Epoch 9, Loss: 1.1200039386749268
# Epoch 10, Loss: 0.5947933197021484
# (SGD, moentum) Training with moentum: 0.8
# Epoch 1, Loss: 0.9529250860214233
# Epoch 2, Loss: 0.6443812847137451
# Epoch 3, Loss: 0.4973174035549164
# Epoch 4, Loss: 0.47032463550567627
# Epoch 5, Loss: 0.38557395339012146
# Epoch 6, Loss: 0.38257676362991333
# Epoch 7, Loss: 0.5398959517478943
# Epoch 8, Loss: 0.33147531747817993
# Epoch 9, Loss: 0.29381418228149414
# Epoch 10, Loss: 0.0972357988357544
# (SGD, moentum) Training with moentum: 0.7
# Epoch 1, Loss: 0.08109087496995926
# Epoch 2, Loss: 0.28291648626327515
# Epoch 3, Loss: 0.123150534927845
# Epoch 4, Loss: 0.03332338109612465
# Epoch 5, Loss: 0.036272257566452026
# Epoch 6, Loss: 0.08525687456130981
# Epoch 7, Loss: 0.04676353558897972
# Epoch 8, Loss: 0.03829516842961311
# Epoch 9, Loss: 0.033884186297655106
# Epoch 10, Loss: 0.02605707198381424
# (SGD, moentum) Training with moentum: 0.6
# Epoch 1, Loss: 0.013717588968575
# Epoch 2, Loss: 0.00548269459977746
# Epoch 3, Loss: 0.017847854644060135
# Epoch 4, Loss: 0.02030044235289097
# Epoch 5, Loss: 0.00686743576079607
# Epoch 6, Loss: 0.009557297453284264
# Epoch 7, Loss: 0.0035443666856735945
# Epoch 8, Loss: 0.024388432502746582
# Epoch 9, Loss: 0.0064000329002738
# Epoch 10, Loss: 0.004529775585979223
# (SGD, moentum) Training with moentum: 0.5
# Epoch 1, Loss: 0.0103107625618577
# Epoch 2, Loss: 0.004610531963407993
# Epoch 3, Loss: 0.002604928333312273
# Epoch 4, Loss: 0.005813467316329479
# Epoch 5, Loss: 0.00252331281080842
# Epoch 6, Loss: 0.006754270754754543
# Epoch 7, Loss: 0.0066643450409173965
# Epoch 8, Loss: 0.004298363346606493
# Epoch 9, Loss: 0.018283244222402573
# Epoch 10, Loss: 0.006300807930529118


# 高动量值（0.9, 0.8）:
# 初期损失下降较快，但随着训练进行，损失波动较大
# 这可能表明在高动量值下，优化器可能在最小值附近overshoot/过冲，即使它能够快速接近最小值，但也容易因为动量太大而跳过最优点

# 中等动量值（0.7, 0.6）:
# 损失下降更稳定，并在训练过程中持续减少
# 这表明动量在这个范围内能够有效地帮助优化器更快地收敛，同时减少了过冲的风险
# 在动量为 0.7的情况下，损失值在训练过程中逐渐下降，并在整个过程中保持相对稳定
# 最终损失在最后一个 Epoch 达到了约 0.026
# 在动量为 0.6的情况下，损失值也表现出稳定的下降趋势，并在训练的最后阶段达到了较低的水平，最后一个 Epoch 的损失约为 0.0045

# 较低动量值（0.5）:
# 损失稳步下降，并保持较低的波动
# 较低的动量值可能导致训练速度稍慢，但可以提供更精细的更新，有助于更稳定地收敛到最小值


# --------------------------------------------------------------------------------------------------------

# 调整学习率/动量后的Accuracy

model = MLP()
num_epochs = 10 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.6)

for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

model.eval()
total = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

# Accuracy : 53.83%
# 虽然只调整了学习率和动量, 但Accuracy从35.56%提升到了53.83%, 与Adam相差不多

# --------------------------------------------------------------------------------------------------------

# 调整隐藏层的数量

class MLP_m(nn.Module):
    def __init__(self, num_hidden_layers):
        super(MLP_m, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        # 输入层到第一个隐藏层
        self.hidden_layers.append(nn.Linear(32*32*3, 512))
        
        # 添加 num_hidden_layers 个隐藏层
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(512, 512))
        
        # 输出层
        self.output_layer = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)  # 压平图像
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)


def train_and_evaluate_model(hidden_layers, optimizer_type="adam"):
    # 在函数内部创建模型实例
    model = MLP_m(hidden_layers)
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.6)
    else:
        raise ValueError("Unsupported optimizer type")

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
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
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# 对不同数量的隐藏层进行训练和评估
hidden_layers_list = [2, 3, 4, 5, 6, 7, 8, 9]
for hidden_layers in hidden_layers_list:
    accuracy_adam = train_and_evaluate_model(hidden_layers, "adam")
    accuracy_sgd = train_and_evaluate_model(hidden_layers, "sgd")
    print(f"Hidden Layers: {hidden_layers}, Adam Accuracy: {accuracy_adam}%, SGD Accuracy: {accuracy_sgd}%")


# Hidden Layers: 2, Adam Accuracy: 53.58%, SGD Accuracy: 54.26%
# Hidden Layers: 3, Adam Accuracy: 54.03%, SGD Accuracy: 53.58%
# Hidden Layers: 4, Adam Accuracy: 54.37%, SGD Accuracy: 50.37%
# Hidden Layers: 5, Adam Accuracy: 54.02%, SGD Accuracy: 50.99%
# Hidden Layers: 6, Adam Accuracy: 54.36%, SGD Accuracy: 44.97%
# Hidden Layers: 7, Adam Accuracy: 53.09%, SGD Accuracy: 20.63%
# Hidden Layers: 8, Adam Accuracy: 52.24%, SGD Accuracy: 10.0%
# Hidden Layers: 9, Adam Accuracy: 50.22%, SGD Accuracy: 10.0%

# 2-5层 ：准确率相对较高，
# 且 Adam 和 SGD 的表现相似。这表明对于 CIFAR10 数据集，这个范围内的隐藏层数量可能是一个比较合理的选择

# （6层及以上）： SGD 的准确率显著下降，特别是在7层以上时
# 这可能是过拟合的迹象，表明模型变得过于复杂，无法泛化到测试数据上
# 而 Adam 优化器似乎对增加的隐藏层数量更为稳健，尽管准确率也有所下降

# Adam优化器：Adam 在所有配置中都显示出较为稳定的性能，尤其是在具有更多隐藏层的模型中
# 可能是由于 Adam 优化器的自适应学习率特性，使其在处理复杂模型时更为有效

# SGD优化器：SGD 在少量隐藏层时表现良好，但在隐藏层数量增加时性能下降明显，尤其是在7层及以上的配置中
# 这表明 SGD 在处理较为复杂的网络结构时可能需要更加精细的参数调整和正则化策略

# --------------------------------------------------------------------------------------------------------

# 对于多类分类，使用 Softmax 激活函数是标准做法 , 不进行修改


# --------------------------------------------------------------------------------------------------------
# 不同的激活参数ReLU, Leaky ReLU, ELU, Swish 对准确率的影响

class MLP(nn.Module):
    def __init__(self, num_hidden_layers, activation_func='relu'):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(32*32*3, 512)])
        
        # 添加隐藏层
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(512, 512))
        
        self.output_layer = nn.Linear(512, 10)
        self.activation_func = activation_func

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.apply_activation(x)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def apply_activation(self, x):
        if self.activation_func == 'relu':
            return F.relu(x)
        elif self.activation_func == 'leaky_relu':
            return F.leaky_relu(x)
        elif self.activation_func == 'elu':
            return F.elu(x)
        elif self.activation_func == 'swish':
            return x * torch.sigmoid(x)  # Swish implementation
        else:
            raise ValueError("Unsupported activation function")

def train_and_evaluate_model(hidden_layers, optimizer_type="adam"):
    # 在函数内部创建模型实例
    model = MLP_m(hidden_layers)
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.6)
    else:
        raise ValueError("Unsupported optimizer type")

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
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
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy

activation_funcs = ['relu', 'leaky_relu', 'elu', 'swish']
for func in activation_funcs:
    accuracy_adam = train_and_evaluate_model(4, func, "adam")
    accuracy_sgd = train_and_evaluate_model(2, func, "sgd")
    print(f"Activation Function: {func}, Adam Accuracy: {accuracy_adam}%, SGD Accuracy: {accuracy_sgd}%")


# Activation Function: relu, Adam Accuracy: 53.31%, SGD Accuracy: 52.99%
# Activation Function: leaky_relu, Adam Accuracy: 54.61%, SGD Accuracy: 52.72%
# Activation Function: elu, Adam Accuracy: 53.7%, SGD Accuracy: 53.62%
# Activation Function: swish, Adam Accuracy: 53.8%, SGD Accuracy: 54.57%


# ReLU：ReLU 激活函数在两种优化器下都表现出相对一致的准确率
# Leaky ReLU：Leaky ReLU在 Adam优化器下表现稍好，这表明它可能在处理激活值接近零的情况时提供了一些优势
# ELU：ELU 激活函数的表现与 ReLU 相似，但在 SGD 下稍好一些。ELU有助于减少梯度消失问题，这可能对 SGD 优化器更有利
# Swish：Swish 激活函数在SGD下的表现最好, 结合了线性和非线性特性，可能提供了更好的泛化能力

# Adam 优化器：在所有激活函数中，Adam优化器的表现较为均衡，这与其自适应学习率特性相符
# SGD 优化器：SGD优化器在不同的激活函数下表现有所波动，但在使用Swish激活函数时表现最佳

# --------------------------------------------------------------------------------------------------------

# 实现正则化
# Dropout在隐藏层中添加 Dropout 层可以帮助减少过拟合。Dropout 通过在训练过程中随机“关闭”一些神经元，迫使网络学习更加鲁棒的特征
# 权重衰减：在优化器中添加权重衰减可以帮助限制模型复杂度，从而减少过拟合

class MLP_D(nn.Module):
    def __init__(self, num_hidden_layers, activation_func='relu', dropout_rate=0.5):
        super(MLP_D, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(32*32*3, 512)])
        self.dropout = nn.Dropout(dropout_rate)
        
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(512, 512))
        
        self.output_layer = nn.Linear(512, 10)
        self.activation_func = activation_func

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.apply_activation(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def apply_activation(self, x):
        if self.activation_func == 'relu':
            return F.relu(x)
        elif self.activation_func == 'leaky_relu':
            return F.leaky_relu(x)
        elif self.activation_func == 'elu':
            return F.elu(x)
        elif self.activation_func == 'swish':
            return x * torch.sigmoid(x)  # Swish implementation
        else:
            raise ValueError("Unsupported activation function")

def train_and_evaluate_model(hidden_layers, activation_func, optimizer_type="adam"):
    model = MLP_D(hidden_layers, activation_func, dropout_rate=0.5)
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 正则化
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.06, weight_decay=1e-5)
    else:
        raise ValueError("Unsupported optimizer type")

    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
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
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy

accuracy_adam = train_and_evaluate_model(4, "leaky_relu", "adam")
accuracy_sgd = train_and_evaluate_model(2, "swish", "sgd")

print(f"Adam with Leaky ReLU Accuracy: {accuracy_adam}%")
print(f"SGD with Swish Accuracy: {accuracy_sgd}%")

# Adam with Leaky ReLU Accuracy: 41.91%
# SGD with Swish Accuracy: 42.46%

# 在使用 Adam 优化器与 Leaky ReLU激活函数、以及SGD优化器与Swish激活函数时的准确率
# 这两种配置的准确率均有所下降

# --------------------------------------------------------------------------------------------------------
# 调整正则化参数：尝试使用不同的 Dropout率和权重衰减系数
# 降低 Dropout率 -> 0.2 , 减小权重衰减值 weight_decay=1e-6

# 代码部分相似, 只展示结果

# Adam with Leaky ReLU Accuracy: 43.61%
# SGD with Swish Accuracy: 43.06%

# 在降低 Dropout 率和减小权重衰减后，模型的准确率有所提高，但提升幅度有限

# --------------------------------------------------------------------------------------------------------
# 去掉Dropout和权重衰减
# 调试epoch -> [32, 64, 128]

class MLP_D(nn.Module):
    def __init__(self, num_hidden_layers, activation_func='relu'):
        super(MLP_D, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(32*32*3, 512)])
        
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(512, 512))
        
        self.output_layer = nn.Linear(512, 10)
        self.activation_func = activation_func

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.apply_activation(x)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def apply_activation(self, x):
        # 激活函数的实现
        if self.activation_func == 'relu':
            return F.relu(x)
        elif self.activation_func == 'leaky_relu':
            return F.leaky_relu(x)
        elif self.activation_func == 'elu':
            return F.elu(x)
        elif self.activation_func == 'swish':
            return x * torch.sigmoid(x)  
        else:
            raise ValueError("Unsupported activation function")


def train_and_evaluate_model(hidden_layers, activation_func, num_epochs, optimizer_type="adam"):
    model = MLP_D(hidden_layers, activation_func)
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.06)
    else:
        raise ValueError("Unsupported optimizer type")

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
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
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy

epochs_list = [32, 64, 128]
for num_epochs in epochs_list:
    accuracy_adam = train_and_evaluate_model(4, "leaky_relu", num_epochs, "adam")
    accuracy_sgd = train_and_evaluate_model(2, "swish", num_epochs, "sgd")

    print(f"Epochs: {num_epochs}, Adam with Leaky ReLU Accuracy: {accuracy_adam}%, SGD with Swish Accuracy: {accuracy_sgd}%")

# Epochs: 32, Adam with Leaky ReLU Accuracy: 53.71%, SGD with Swish Accuracy: 53.0%
# Epochs: 64, Adam with Leaky ReLU Accuracy: 52.54%, SGD with Swish Accuracy: 51.49%
# Epochs: 128, Adam with Leaky ReLU Accuracy: 53.47%, SGD with Swish Accuracy: 53.34%

# 准确率变化：在增加训练 epoch 数量的情况下
# Adam和SGD优化器的表现相对稳定
# 对于32, 64和128个 epoch，准确率有轻微波动，但整体变化不大

# 过拟合的可能性：通常随着训练 epoch 的增加，模型有更多机会学习数据，但也存在过拟合的风险
# 由于准确率没有显著提高，这表明模型可能未能充分从更长时间的训练中受益，或者数据集本身的限制导致了性能的上限

# Adam与SGD的表现：在这三种设置中，Adam优化器和SGD优化器的表现相近，这表明两种优化器都适用

# --------------------------------------------------------------------------------------------------------

# Activation Function: leaky_relu, learning_rate=0.001, epoch默认10的情况比32, 64, 128都好, 不使用dropout及梯度衰减
# Adam Accuracy: 54.61%


# Activation Function: swish, learning_rate=0.01, momentum=0.6, 不使用dropout及梯度衰减
# 对于SGD优化器来说, learning_rate, momentum影响较为明显
# SGD Accuracy: 54.57%



# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------