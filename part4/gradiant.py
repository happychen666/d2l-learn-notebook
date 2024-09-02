# inputs = [1, 2, 3]
# weights = [[0.2, 0.4, 0.5], [0.3, 0.1, 0.8]]
# biases = [0.1, -0.2]

# layer_outputs = []

# for neuron_weights, neuron_bias in zip(weights, biases):
#     print('neuron_weights, neuron_bias===',neuron_weights, neuron_bias)
#     neuron_output = 0
#     for n_ineuron_output += neuron_nput, weight in zip(inputs, neuron_weights):
#         print(n_input, weight)
#         neuron_output += n_input * weight
#     bias
#     print(neuron_bias,'-----')
#     layer_outputs.append(neuron_output)

# print(layer_outputs)

import numpy as np

inputs = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

# # 转置权重矩阵   
transposed_weights = np.array(weights).T

print(weights,transposed_weights)

# # 使用 numpy 的 matmul 函数计算神经元输出
layer_outputs = np.dot(inputs, transposed_weights) + biases


print(layer_outputs)