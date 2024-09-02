好的，我们用 Sigmoid 激活函数来举一个简单的神经网络例子，计算其偏导数。

### 示例：单层神经网络

#### 1. 网络结构

- **输入**：\( x = 1 \)
- **权重**：\( w = 0.5 \)
- **偏置**：\( b = 0.2 \)
- **激活函数**：Sigmoid，定义为：
  \[
  f(z) = \frac{1}{1 + e^{-z}}
  \]
- **真实值**：\( y_{true} = 1 \)

#### 2. 前向传播

首先，计算网络的输出：

1. **计算线性组合**：
   \[
   z = wx + b = 0.5 \times 1 + 0.2 = 0.7
   \]

2. **激活函数（Sigmoid）**：
   \[
   y = f(z) = \frac{1}{1 + e^{-0.7}} \approx 0.668
   \]

#### 3. 损失函数

使用均方误差作为损失函数：
\[
L = \frac{1}{2}(y_{true} - y)^2 = \frac{1}{2}(1 - 0.668)^2 \approx 0.049
\]

#### 4. 计算偏导数

现在，我们计算损失函数 \( L \) 对权重 \( w \) 和偏置 \( b \) 的偏导数。

1. **计算损失函数对输出的偏导数**：
   \[
   \frac{\partial L}{\partial y} = -(y_{true} - y) = -(1 - 0.668) = -0.332
   \]

2. **计算 Sigmoid 函数的导数**：
   Sigmoid 函数的导数为：
   \[
   f'(z) = f(z)(1 - f(z)) = y(1 - y) \approx 0.668(1 - 0.668) \approx 0.222
   \]

3. **计算线性组合对权重的偏导数**：
   \[
   \frac{\partial z}{\partial w} = x = 1
   \]

4. **计算线性组合对偏置的偏导数**：
   \[
   \frac{\partial z}{\partial b} = 1
   \]

#### 5. 应用链式法则

- **对权重 \( w \)**：
\[
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w}
\]
将每一部分代入：
\[
\frac{\partial L}{\partial w} = -0.332 \cdot 0.222 \cdot 1 \approx -0.074
\]

- **对偏置 \( b \)**：
\[
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial b}
\]
将每一部分代入：
\[
\frac{\partial L}{\partial b} = -0.332 \cdot 0.222 \cdot 1 \approx -0.074
\]

### 总结结果

- 对于权重 \( w \) 的偏导数为 \( \frac{\partial L}{\partial w} \approx -0.074 \)
- 对于偏置 \( b \) 的偏导数为 \( \frac{\partial L}{\partial b} \approx -0.074 \)

### 解释复杂性

在这个例子中，虽然网络结构简单，但由于 Sigmoid 激活函数的非线性特性，计算偏导数变得稍微复杂。此外，在多层神经网络中，激活函数和链式法则的应用会使得偏导数的计算变得更加复杂。每一层都需要计算其导数，这会导致参数更新的计算变得非常繁琐。