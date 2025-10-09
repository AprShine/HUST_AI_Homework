# 华中科技大学人工智能作业

此为华中科技大学 2025 年研究生课程——人工智能作业

## 题目

以 Python 作为主要编程语言，使用 `Pytorch/TensorFlow/MindSpore` 框架，构建卷积神经网络模型，实现对动物图像的分类任务。

## 数据集

使用的训练集/测试集如下：

[Animal-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

包含 10 种动物类型：

```python
translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}
```

## ResNet

是 CNN 中经典的网络架构，常用于图像识别，目前最新的网络架构或多或少有参考 ResNet 的思想。

### idea

