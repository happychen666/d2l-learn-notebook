import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # 100 个样本，1 个特征
y = 2.5 * X + np.random.randn(100, 1) * 2  # 加入噪声

# 可视化数据
# plt.scatter(X, y)
# plt.xlabel("Feature")
# plt.ylabel("Target")
# plt.title("Generated Data")
# plt.show()




from sklearn.linear_model import LinearRegression

# 创建模型
model_no_reg = LinearRegression()
model_no_reg.fit(X, y)

# 预测
y_pred_no_reg = model_no_reg.predict(X)

# 可视化结果
plt.scatter(X, y)
plt.plot(X, y_pred_no_reg, color='red', label='No Regularization')
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression without Weight Decay")
plt.legend()
plt.show()