# 《动手学习深度学习》PyTorch版笔记

## 文章内容
- [d2lzh_pytorch.ipynb](https://kolibreath.github.io/learning-pytorch/d2lzh_pytorch.html)
- [线性回归模型](https://kolibreath.github.io/learning-pytorch/线性回归模型.html)
- [线性回归模型的简洁实现](https://kolibreath.github.io/learning-pytorch/线性回归模型的简洁实现.html)
- [Softmax模型](https://kolibreath.github.io/learning-pytorch/softmax模型.html)
- [Softmax模型的简洁实现](https://kolibreath.github.io/learning-pytorch/Softmax的简洁实现.html)
- [多层感知机的从零开始实现](https://kolibreath.github.io/learning-pytorch/多层感知机的从零开始实现.html)
- [多层感知机的从零开始实现](https://kolibreath.github.io/learning-pytorch/多层感知机的简洁实现.html)
- [多项式函数的拟合实验](https://kolibreath.github.io/learning-pytorch/多项式函数的拟合实验.html)

## 说明和错误报告

### 3.2.4 定义模型
```
def linreg(X, w, b):
#     X = torch.tensor(X, dtype=torch.float32)
    X = X.clone().detach().float()
    return torch.mm(X, w) + b
```
需要改成这样的代码 貌似是X的类型有问题

### d2lzh_pytorch
这个库大家可以使用文章中链接的jupyter notebook代码生成。
1. 首先将代码复制到本地的jupyter notebook
2. 命名需要为d2lzh_pytorch.ipynb
3. 运行它就可以生成d2lzh_pytorch.py 之后就以使用了

### 3.6.8 Softmax回归的从零开始
代码
````
X, y = iter(test_iter).next()
````
会报错，这个地方参考了DataLoader的文档并且结合代码上下文的大意可知，需要获取一个test_iter（本质为DataLoader）的迭代器,改成如下代码即可
````
X, y = test_iter.__iter__().next()
````
