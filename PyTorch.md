机器学习/深度学习框架
[开源仓库](https://github.com/pytorch/pytorch)
[官方文档](https://pytorch.org/docs/stable/index.html)

使用了 [[PyTorch#^19414b|动态计算图]]
#### 计算图
[补充参考](https://www.geeksforgeeks.org/computational-graphs-in-deep-learning/)
Computational Graph， 表示计算过程的图形结构：
- Node：数据或操作
- Edge：数据流，控制流
e.g. computational graph for $Y=(a+b)*(b-c)$![[Pasted image 20241009150326.png|300]]
计算图的结构遵循了 链式法则，这让求导/梯度很容易，实现**自动微分**

计算图类型
- 静态 static 先定义再运行；创建后不能修改； 高效运行（生产）
- 动态 dynamic 运行过程中构建（根据需要生成node,edge）；牺牲性能但灵活（研究；开发） ^19414b
## Tensors
[参考](https://pytorch.org/tutorials/beginner/basics/tensor_tutorial.html)
- 类似矩阵或ndarray的数据结构
- 是PyTorch中计算的基本单位，可以在GPU上并行计算以加速
```python fold file=创建Tensor
>>> import torch 
>>> import numpy as np
>>> data = [[1, 2], [3, 4]]
>>> x_data = torch.tensor(data) # 可以从列表创建
>>> print(x_data) 
tensor([[1, 2],
        [3, 4]])
>>> np_array = np.array(data) 
>>> print(np_array)
[[1 2]
 [3 4]]
>>> x_np = torch.from_numpy(np_array) # 可以从ndarray创建
>>> print(x_np)
tensor([[1, 2],
        [3, 4]])
```
