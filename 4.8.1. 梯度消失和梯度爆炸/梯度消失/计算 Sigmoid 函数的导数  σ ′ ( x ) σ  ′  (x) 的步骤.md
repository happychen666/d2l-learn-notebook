计算 Sigmoid 函数的导数 \(\sigma'(x)\) 的步骤如下：

### 1. 定义 Sigmoid 函数

Sigmoid 函数的定义为：
\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

### 2. 使用链式法则

为了求导，我们可以使用链式法则。Sigmoid 函数可以重写为 \(y = \sigma(x)\)，我们需要计算 \(y\) 关于 \(x\) 的导数，即 \(\frac{dy}{dx}\)。

### 3. 计算导数

首先，对 Sigmoid 函数求导：
\[
\sigma'(x) = \frac{d}{dx} \left( \frac{1}{1 + e^{-x}} \right)
\]

使用商法则（\(\frac{d}{dx} \left( \frac{u}{v} \right) = \frac{u'v - uv'}{v^2}\)），其中：

- \(u = 1\)
- \(v = 1 + e^{-x}\)

### 4. 计算 \(u'\) 和 \(v'\)

- \(u' = 0\)（因 \(u\) 是常数）
- \(v' = \frac{d}{dx} (1 + e^{-x}) = 0 - e^{-x} \cdot (-1) = e^{-x}\)

### 5. 应用商法则

将 \(u\) 和 \(v\) 代入商法则：
\[
\sigma'(x) = \frac{0 \cdot (1 + e^{-x}) - 1 \cdot e^{-x}}{(1 + e^{-x})^2}
\]
\[
= \frac{-e^{-x}}{(1 + e^{-x})^2}
\]

### 6. 化简

为了进一步简化，我们可以使用 \(\sigma(x)\) 的定义：
\[
\sigma(x) = \frac{1}{1 + e^{-x}} \quad \Rightarrow \quad 1 - \sigma(x) = \frac{e^{-x}}{1 + e^{-x}}
\]

因此，可以将导数表示为：
\[
\sigma'(x) = \frac{e^{-x}}{(1 + e^{-x})^2} = \sigma(x) \cdot (1 - \sigma(x))
\]

### 最终结果

所以，Sigmoid 函数的导数为：
\[
\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
\]

这个结果显示了 Sigmoid 函数在某一点的斜率与该点的输出值 \(\sigma(x)\) 之间的关系，也是梯度消失问题的根源之一。
 