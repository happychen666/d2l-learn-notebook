import torch
from d2l import torch as d2l

# 创建一些线性数据
def generate_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 生成噪声时使用 X 的形状
    noise = torch.normal(0, 0.01, (num_examples, 1))  # 创建与 y 相同形状的噪声
    y = torch.matmul(X, w) + b + noise  # 添加噪声
    return X, y

true_w = torch.tensor([2.0, -3.4])
true_b = 4.2

# 创造数据
X, y = generate_data(true_w, true_b, 1000)
# print(X)
# 初始化权重和偏置
W = torch.normal(0, 0.01, (2, 1), requires_grad=True) # 学习的权重
b = torch.zeros(1, requires_grad=True)                # 学习的偏置
# print(W)
# print(b)
def mse_loss(y_hat, y):
    return ((y_hat - y) ** 2).mean()

lr = 0.1  # 学习率
num_epochs = 10
batch_size = 10

for epoch in range(num_epochs):
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]

        # 前向传播
        y_hat = torch.matmul(X_batch, W) + b
        
        # 计算损失
        loss = mse_loss(y_hat, y_batch)

        # 反向传播
        loss.backward()

        # 使用 d2l.sgd 更新参数
        d2l.sgd([W, b], lr, batch_size)

        # 手动清零梯度
        W.grad.data.zero_()
        b.grad.data.zero_()

    print(f'epoch {epoch + 1}, loss {loss.item():.4f}')