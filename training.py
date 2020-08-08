import time
from Moudel import Classifier
import torch.nn as nn
import torch
import numpy as np
from read_images import train_x, train_y, val_x, val_y
from Dataset import train_set,val_set,train_loader,val_loader,train_transform,batch_size,ImgDataset
from torch.utils.data import DataLoader


model = Classifier().cuda()     # 训练模型   # .cuda():用GPU加速
loss = nn.CrossEntropyLoss()  # 因为是分类任务，所以使用交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器  #模型参数即网络层参数
num_epoch = 30   # 迭代次数

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # 确保 model 是在 训练 model (开启 Dropout 等...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # 用 optimizer 将模型参数的梯度 gradient 归零
        train_pred = model(data[0].cuda())  # 利用 model 得到预测的概率分布，这边实际上是调用模型的 forward 函数
        batch_loss = loss(train_pred, data[1].cuda())  # 计算 loss （注意 prediction(data[0])跟 label(data[1]) 必须同时在 CPU 或是 GPU 上）
        batch_loss.backward()  # 反向传播：利用 back propagation 算出每个参数的 gradient
        optimizer.step()  # 以 optimizer 用 gradient 更新参数

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    # 验证集val
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # 將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time, \
               train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))



train_val_x = np.concatenate((train_x, val_x), axis=0)   # 将train_x和val_x拼接起来
train_val_y = np.concatenate((train_y, val_y), axis=0)   # 将train_y和val_y拼接起来
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)


model_best = Classifier().cuda()   # cuda加速
loss = nn.CrossEntropyLoss() # 因为是分类任务，所以使用交叉熵损失
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)    # optimizer 使用 Adam
num_epoch = 30

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        # 将结果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      (epoch + 1, num_epoch, time.time()-epoch_start_time, \
      train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))