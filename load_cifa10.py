from os import path
import pickle
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from PIL import Image

from train import train


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label,transform):
        self.data = data_root
        self.label = data_label
        self.transform=transform
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        data = Image.fromarray(data)
        data = self.transform(data)
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

#1.加载cifar10数据集，返回的是train_loader,test_loader
def get_loader(args):

    #设置数据加载时的变换形式，包括撞转成tensor,裁剪，归一化
    transform_train=transforms.Compose([
        transforms.RandomResizedCrop((args.img_size,args.img_size),scale=(0.05,1.0)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    #默认使用cifar10数据集
    # if args.dataset=="mydata":
        # trainset=datasets.CIFAR10(root='./data',train=True,download=True,transform=transform_train)
        # testset=datasets.CIFAR10(root='./data',train=False,download=False,transform=transform_train)
    # else:
    #     trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    #     testset = datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_train)
    train_filepath='./data/train_batch_0'
    test_filepath='./data/test_batch_0'
    
    f=open(train_filepath, 'rb')
    entry=pickle.load(f,encoding='latin1')
    train_data=entry['data']
    train_labels=entry['labels']
    f.close()
    
    f=open(test_filepath, 'rb')
    entry=pickle.load(f,encoding='latin1')
    test_data=entry['data']
    test_labels=entry['labels']
    f.close()
    
    trainset=GetLoader(train_data,train_labels,transform_train)
    testset=GetLoader(test_data,test_labels,transform_test)
    
    print("train number:",len(trainset))
    print("test number:",len(testset))

    train_loader=DataLoader(trainset,batch_size=args.train_batch_size,shuffle=True)
    test_loader=DataLoader(testset,batch_size=args.eval_batch_size,shuffle=False)
    print("train_loader:",len(train_loader))
    print("test_loader:",len(test_loader))

    return train_loader,test_loader


# #定义一个实例配置文件
# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", default="mydata")
# parser.add_argument("--img_size", type=int, default=224,)
# parser.add_argument("--train_batch-size", default=16, type=int,)
# parser.add_argument("--eval_batch-size", default=16, type=int,)
#
# args = parser.parse_args()
# get_loader(args)