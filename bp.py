import numpy as np

lr = 0.25
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def sigmoid_derivative(s):
    return s * (1 - s)

def set_w(gd):
    global w
    w = w - lr * gd

def set_v(gd):
    global v
    v = v - lr * gd

# 第一层：隐层具有3个神经元，输入是2个，所以该层的权重系数v形状为:[2,3]
v = np.asarray([
    [0.1, 0.2, 0.3],
    [0.15, 0.25, 0.35]
])
b1 = np.asarray([0.35])
# 第二层：输出层具有2个神经元，输入是3个，所以该层的权重系数w的形状为:[3,2]
w = np.asarray([
    [0.4, 0.45],
    [0.5, 0.55],
    [0.6, 0.65]
])
b2 = np.asarray([0.65])

# 当前输入1个样本，每个样本2个特征属性，就相当于输入层的神经元是2个
x = np.asarray([
    [5.0, 10.0],
    [2.0, 8.0],
    [3.0, 12.0],
    [3.0, 11.0],
    [16.0, 2.0]
])
# 实际值
d = np.asarray([
    [0.95, 0.12],
    [0.93, 0.01],
    [0.23, 0.77],
    [0.53, 0.45],
    [0.01, 0.99]
])

def training():
    global b1, b2
    # 第一个隐藏的操作输出
    net_h = np.dot(x, v) + b1  # [N,3] N表示样本数量，3表示每个样本有3个特征
    out_h = sigmoid(net_h)
    # 输出层的操作输出
    net_o = np.dot(out_h, w) + b2  # [N,2] N表示样本数目，2表示每个样本有2个特征/2个输出
    out_o = sigmoid(net_o)
    loss = 0.5 * np.sum(np.power((out_o - d), 2))
    # print(loss)
    # print(net_h)
    # print(out_h)
    # print(net_o)
    # print(out_o)
    # print(x)
    # print("=" * 50)

    # TODO: 基于矩阵的反向传播 --> 基于Numpy实现全连接神经网络
    delta_o = sigmoid_derivative(out_o) * (out_o - d)
    delta_h = sigmoid_derivative(out_h) * np.dot(delta_o, w.T)
    
    set_w(gd=np.dot(out_h.T, delta_o))
    set_v(gd=np.dot(x.T, delta_h))
    b2 = b2 - lr * np.mean(delta_o, axis=0)
    b1 = b1 - lr * np.mean(delta_h, axis=0)
    
    return loss

for epoch in range(10000):
    loss = training()
    if epoch % 1000 == 0:
        print(f"训练次数 {epoch}, 损失: {loss}")

print(f"最后损失: {loss}")



