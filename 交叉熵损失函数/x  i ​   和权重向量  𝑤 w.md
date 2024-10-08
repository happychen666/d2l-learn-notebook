公式 \( z_i = w^T x_i = \sum_{k} w_k x_{i,k} \) 是线性模型（例如线性回归或逻辑回归）中常用的计算公式，它描述了输入样本 \( x_i \) 和权重向量 \( w \) 的关系。下面我们来详细解释这个公式的含义和推导过程。

### 1. 向量表示和矩阵乘法：

- **\( w \)** 是一个权重向量，长度等于特征的数量。假设有 \( m \) 个特征，那么 \( w = [w_1, w_2, \dots, w_m] \)。
- **\( x_i \)** 是第 \( i \) 个样本的特征向量，长度也等于特征的数量。对于样本 \( i \)，\( x_i = [x_{i,1}, x_{i,2}, \dots, x_{i,m}] \)。

### 2. 线性组合：

在机器学习模型（如线性回归或逻辑回归）中，模型的预测值 \( z_i \) 通常是输入特征的线性组合。这种线性组合可以用向量的点积来表示。

- **向量点积**：对于两个相同长度的向量 \( w \) 和 \( x_i \)，它们的点积定义为：
  \[
  w^T x_i = \sum_{k=1}^{m} w_k x_{i,k}
  \]
  其中：
  - \( w_k \) 是权重向量 \( w \) 中的第 \( k \) 个分量。
  - \( x_{i,k} \) 是样本 \( x_i \) 中的第 \( k \) 个特征值。

  点积的结果是一个标量（即单一的数值），表示输入特征和权重的加权和。

### 3. 公式推导：

对于第 \( i \) 个样本，计算模型的预测值 \( z_i \)：
\[
z_i = w^T x_i = w_1 x_{i,1} + w_2 x_{i,2} + \dots + w_m x_{i,m}
\]
简写为：
\[
z_i = \sum_{k=1}^{m} w_k x_{i,k}
\]

### 4. 解释：

- **线性模型**：这个公式表示的是输入特征 \( x_i \) 的线性组合，用于模型预测。每个特征 \( x_{i,k} \) 都有一个对应的权重 \( w_k \)，它决定了这个特征在模型预测中的重要性。
- **预测过程**：模型的预测值 \( z_i \) 是所有特征值乘以其对应权重的加权和。这个结果再经过一个激活函数（如逻辑回归中的 Sigmoid 函数）就得到最终的预测概率。

### 5. 例子：

假设一个简单的模型有两个特征 \( x_1 \) 和 \( x_2 \)，对应的权重为 \( w_1 \) 和 \( w_2 \)，那么对于某个样本 \( x = [x_1, x_2] \)，线性组合 \( z \) 计算为：
\[
z = w_1 \cdot x_1 + w_2 \cdot x_2
\]

### 总结：

- **\( z_i = w^T x_i \)** 表示输入特征向量 \( x_i \) 和权重向量 \( w \) 的点积，这是模型的线性组合部分。
- **\( \sum_{k=1}^{m} w_k x_{i,k} \)** 是点积的展开形式，表示特征和权重的加权和。

这个公式广泛应用于线性回归、逻辑回归和神经网络等模型中，是机器学习中预测步骤的基础。