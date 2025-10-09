import torch
import torch.nn as nn
import torchvision
import numpy as np
import os, PIL, pathlib
import matplotlib.pylab as plt

from PIL import Image
from torchvision import transforms, datasets 
from torchvision import models

def import_data(data_path_name):
    # 数据统一格式
    img_height = 224
    img_width = 224 

    data_tranforms = transforms.Compose([
        transforms.Resize([img_height, img_width]),
        transforms.ToTensor(),
        transforms.Normalize(   # 归一化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225] 
        )
    ])

    return datasets.ImageFolder(root=data_path_name, transform=data_tranforms)

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    batch_size = len(dataloader)
    
    train_acc, train_loss = 0, 0 
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        # 训练
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # 梯度下降法
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录
        train_loss += loss.item()
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        
    train_acc /= size
    train_loss /= batch_size
    
    return train_acc, train_loss

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    batch_size = len(dataloader)
    
    test_acc, test_loss = 0, 0 
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
        
            pred = model(X)
            loss = loss_fn(pred, y)
        
            test_loss += loss.item()
            test_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        
    test_acc /= size
    test_loss /= batch_size
    
    return test_acc, test_loss

def show_data_expl(data_path_name):
    # 展示数据
    print(data_path_name)
    data_path_list = [f for f in os.listdir(data_path_name) if f.endswith(('jpeg'))]
    # 创建画板
    fig, axes = plt.subplots(2, 8, figsize=(16, 6))
    for ax, img_file in zip(axes.flat, data_path_list):
        path_name = os.path.join(data_path_name, img_file)
        img = Image.open(path_name) # 打开
        img=img.resize((224,224))
    # 显示
        ax.imshow(img)
        ax.axis('off')
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    dataset_path="./data/alessiocorrado99/animals10/versions/2/raw-img"
    data_dir = pathlib.Path(dataset_path)

    # 类别数量
    folder_names = [str(path).split('/')[0] for path in os.listdir(data_dir)]
    translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "ragno": "spider", 
                  "dog": "cane", "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}
    class_names=[translate[folder_name] for folder_name in folder_names]
    print(class_names)
    
    # 数据导入
    total_data=import_data(dataset_path)
    print(len(total_data))
    
    # 数据划分
    train_size = int(len(total_data) * 0.8)
    test_size = len(total_data) - train_size 

    train_data, test_data = torch.utils.data.random_split(total_data, [train_size, test_size])
    print(len(train_data),len(test_data))

    batch_size = 32 
    train_dl = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    test_dl = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )
    # 查看数据维度
    for data, labels in train_dl:
        print("data shape[N, C, H, W]: ", data.shape)
        print("labels: ", labels)
        break

    # resnet网络
    resnet_18=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs=resnet_18.fc.in_features
    for param in resnet_18.parameters():
        param.requires_grad = False
    resnet_18.fc=nn.Linear(num_ftrs,10)
    model=resnet_18.cuda() if device=="cuda" else resnet_18

    # 超参数
    learning_rate=1e-3
    epochs = 1

    # Loss and opt
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    for i in range(epochs):
        model.train()
        epoch_train_acc, epoch_train_loss = train(train_dl, model, loss_fn, optimizer, device=device)
        model.eval()
        epoch_test_acc, epoch_test_loss = test(test_dl, model, loss_fn, device=device)
    
        train_acc.append(epoch_train_acc)
        train_loss.append(epoch_train_loss)
        test_acc.append(epoch_test_acc)
        test_loss.append(epoch_test_loss)
    
        # 输出
        template = ('Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss:{:.3f}, Test_acc:{:.1f}%, Test_loss:{:.3f}')
        print(template.format(i + 1, epoch_train_acc*100, epoch_train_loss, epoch_test_acc*100, epoch_test_loss))
    
    print("Done")
if __name__ == "__main__":
    main()
