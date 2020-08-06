import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from read_images import train_x, train_y, val_x, val_y

# training 时，通过随机旋转、水平翻转图片来进行数据增强（data augmentation）
train_transform = transforms.Compose([
    transforms.ToPILImage(),   # 将tensor数据转换为PIL图片
    transforms.RandomHorizontalFlip(),  # 0.5的概率水平翻转给定的PIL图像
    transforms.RandomRotation(15),  # 随机旋转图片
    transforms.ToTensor(),  # 将PIL图片变成 Tensor，并且把数值normalize到[0,1]
])
# testing 时，不需要进行数据增强（data augmentation）
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


class ImgDataset(Dataset):      # 继承Dataset类
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    # 必须要传回dataset的大小
    def __len__(self):
        return len(self.x)

    # 当函数利用[]取值时，dataset应该要怎么传回数据
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)   # shuffle为是否打乱顺序
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
