import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(2, 2)  # 输入维度为2，输出维度为2（二分类）

    def forward(self, x):
        return self.fc(x)

# 准确率计算函数
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):
    """计算准确率"""
    y_hat_classes = torch.argmax(y_hat, dim=1)  # 获取预测的类别
    return (y_hat_classes == y).sum().item()  # 返回正确预测的数量

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 创建示例数据集
X = torch.tensor([[0.1, 0.2], [0.2, 0.1], [0.4, 0.4], [0.8, 0.8]], dtype=torch.float32)
y = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # 真实标签

# 创建数据加载器
dataset = TensorDataset(X, y)
data_iter = DataLoader(dataset, batch_size=2)

# 创建模型、定义损失函数和优化器
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 模型训练（简单演示，通常需要更多轮次）
for epoch in range(10):
    model.train()
    for batch_X, batch_y in data_iter:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# 评估模型
accuracy_score = evaluate_accuracy(model, data_iter)
print(f'Model Accuracy: {accuracy_score:.2f}')