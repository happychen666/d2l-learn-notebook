import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # 100 个样本，1 个特征
y = 2.5 * X + np.random.randn(100, 1) * 2  # 加入噪声

# 可视化数据
plt.scatter(X, y)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Generated Data")
plt.show()




from sklearn.linear_model import Ridge

# 创建带有权重衰减的模型
model_with_reg = Ridge(alpha=1.0)  # alpha 是正则化参数
model_with_reg.fit(X, y)

# 预测
y_pred_with_reg = model_with_reg.predict(X)

# 可视化结果
plt.scatter(X, y)
plt.plot(X, y_pred_with_reg, color='green', label='With Weight Decay')
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression with Weight Decay")
plt.legend()
plt.show()