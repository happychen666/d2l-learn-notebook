import torch
import torch.nn as nn
import torch.optim as optim

# 2. 创建一个简单的线性模型
# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入特征1，输出特征1

    def forward(self, x):
        return self.linear(x)
    
# 3. 准备数据
# 生成一些简单的训练数据：
# 生成训练数据
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=False)  # 特征
y_train = torch.tensor([[2.0], [3.0], [4.0], [5.0]], requires_grad=False)  # 标签

# 4. 实例化模型和优化器
# 实例化模型
model = LinearRegressionModel()
# 实例化优化器
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 学习率为0.1

# 5. 定义损失函数
# 使用均方误差作为损失函数：
# 定义损失函数
criterion = nn.MSELoss()


# 6. 训练模型
# 现在我们可以进行模型训练，计算损失，反向传播并更新参数：
# 训练过程
num_epochs = 100
for epoch in range(num_epochs):
    # 1. 前向传播
    y_pred = model(x_train)  # 模型预测

    # 2. 计算损失
    loss = criterion(y_pred, y_train)  # 计算损失
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')  # 输出损失

    # 3. 清零梯度
    optimizer.zero_grad()

    # 4. 反向传播
    loss.backward()  # 计算梯度

    # 5. 更新参数
    optimizer.step()  # 更新参数


# 7. 获取优化后的权重和偏置
# 获取优化后的权重和偏置
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} = {param.data.numpy()}")
