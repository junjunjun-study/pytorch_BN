import numpy as np
from MyBN import * #ZJK的BN函数
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms  #对minist做预处理
import datetime #时间
from visdom import Visdom #显示接口


#训练集采用minist数据集，每一张图片是28*28的灰度图片，即通道数为1
#训练中的另一个重要概念是epoch。每学一遍数据集，就称为1个epoch。
#注意每一个epoch都需打乱数据的顺序，以使网络受到的调整更具有多样性
#每次只选取1个样本，然后根据运行结果调整参数，这就是著名的随机梯度下降（SGD），而且可称为批大小（batch size）为1的SGD。
#批大小，就是每次调整参数前所选取的样本（称为mini-batch或batch）数量

#举例，若数据集中有1000个样本，批大小为10，那么将全部样本训练1遍后，网络会被调整1000/10=100次。但这并不意味着网络已达到最优，我们可重复这个过程，让网络再学1遍、2遍、3遍数据集。

class ConvNet_BN(nn.Module): #用的是zjk自己写的BN的CNN
    def __init__(self):
        super().__init__()
        # 1,28x28  这些结构里的参数都是不断变化的 不断学习的（有输入参数） 而像sigmod relu 池化 是没有变化的参数的 是没有学习的概念的 即__init__这里都是定义的可学习的
        self.conv1 = nn.Conv2d(1, 10, 5)   # 输入通道数1 输出通道数10,（即filter个数是10,提取了10个特征，特征图是10个）卷积核尺寸是5*5 24x24
        self.bn1 = BatchNorm(10)           #输入通道10，输出通道10
        self.conv2 = nn.Conv2d(10, 20, 3)  # 输入通道数10 输出通道数20 卷积核尺寸是3*3 128, 10x10
        self.bn2 = BatchNorm(20)           #输入通道20，输出通道20
        self.fc1 = nn.Linear(20 * 10 * 10, 500) # 输入向量大小20*10*10，输出向量维数500，即该全连接层神经元个数是500
        self.bn3 = BatchNorm(500)          #输入通道500，输出通道500
        self.fc2 = nn.Linear(500, 10) #输入向量维数500，输出向量维数10，即该全连接层神经元个数是10，即10个输出:0 1 2 3 4 5 6 7 8 9
        self.bn4 = BatchNorm(10)           #输入通道10，输出通道10

    def forward(self, x):
        in_size = x.size(0)  #即batch_size
        #第一层卷积
        out = self.conv1(x)  # 输出变成了 10*24*24
        out = self.bn1(out)  #10*24*24
        out = torch.sigmoid(out) #10*24*24
        out = F.max_pool2d(out, 2, 2)  # 10*12*12
        #第二层卷积
        out = self.conv2(out)  # 50*10*10
        out = self.bn2(out)    # 50*10*10
        out = torch.sigmoid(out) #50*10*10
        #进入全连接网络
        out = out.view(in_size, -1)#将输入变成一维
        out = self.fc1(out)      #全连接层神经元隐藏层个数500
        out = self.bn3(out)      #进行BN处理
        out = torch.sigmoid(out) #通过激励函数
        out = self.fc2(out)      #到达输出层，10个神经元
        out = self.bn4(out)      #进行BN处理
        out = F.log_softmax(out, dim=1) #获得输出，log使得梯度平缓
        return out


class ConvNet(nn.Module): #无BN的CNN
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1 = nn.Conv2d(1, 10, 5)  ## 输入通道数1 输出通道数10,卷积核尺寸是5*5 24x24
        self.conv2 = nn.Conv2d(10, 20, 3)  # 输入通道数10 输出通道数20 卷积核尺寸是3*3 128, 10x10
        self.fc1 = nn.Linear(20 * 10 * 10, 500) #输入通道数20*10*10，输出通道数500，即该全连接层神经元个数是500
        self.fc2 = nn.Linear(500, 10)  ##输入通道数500，输出通道数10，即该全连接层神经元个数是10,即10个输出：0 1 2 3 4 5 6 7 8 9

    def forward(self, x):
        in_size = x.size(0)#batch_size
        out = self.conv1(x)  # 输出变成了10*24*24
        out = torch.sigmoid(out)
        out = F.max_pool2d(out, 2, 2)  # 输出变成了10*12*12
        out = self.conv2(out)  # 输出变成了20*10*10
        out = torch.sigmoid(out)
        out = out.view(in_size, -1) #转换为1维
        out = self.fc1(out)   #500个神经元
        out = torch.sigmoid(out)
        out = self.fc2(out)   #10个神经元 10个输出
        out = F.log_softmax(out, dim=1) #log使得梯度平滑，取最大概率作为结果
        return out



class ConvNet_pyBN(nn.Module):#用的是pytorch的BN
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1 = nn.Conv2d(1, 10, 5)  # 输入通道数1 输出通道数10,（即filter个数是10,提取了10个特征）卷积核尺寸是5*5 24x24
        self.bn1 = nn.BatchNorm2d(10)     #pytorch的BN
        self.conv2 = nn.Conv2d(10, 20, 3)  # 输入通道数10 输出通道数20 卷积核尺寸是3*3 128, 10x10
        self.bn2 = nn.BatchNorm2d(20)     #pytorch的BN
        self.fc1 = nn.Linear(20 * 10 * 10, 500) # 输入向量维数20*10*10，输出向量维数500，即该全连接层神经元个数是500
        self.bn3 = nn.BatchNorm1d(500)    #pytorch的BN
        self.fc2 = nn.Linear(500, 10) #输入向量维数500，输出向量维数10，即该全连接层神经元个数是10，即10个输出:0 1 2 3 4 5 6 7 8 9
        self.bn4 = nn.BatchNorm1d(10)

    def forward(self, x):
        in_size = x.size(0) #batch_size
        out = self.conv1(x)  # 输出变成了 10*24*24
        out = self.bn1(out)  #10*24*24
        out = torch.sigmoid(out) #10*24*24
        out = F.max_pool2d(out, 2, 2)  # 10*12*12

        out = self.conv2(out)  # 50*10*10
        out = self.bn2(out)    # 50*10*10
        out = torch.sigmoid(out) #50*10*10

        out = out.view(in_size, -1) #要把二维的向量变成1维的 即开始进入全连接
        out = self.fc1(out)
        out = self.bn3(out)
        out = torch.sigmoid(out)
        out = self.fc2(out)
        out = self.bn4(out)
        out = F.log_softmax(out, dim=1)
        return out


def train(model, device, train_loader, optimizer, epoch):
    model.train() #设置训练
    loss_total = 0.0 #总偏差，用于计算平均偏差 MSE  mean squared error 均方误差
    counter = 0 #计数，用于计算平均偏差
    for batch_idx, (data, target) in enumerate(train_loader): #不断遍历这个train_loader这个对象，返回得到batch_id,data,target等数据
        data, target = data.to(device), target.to(device)     #当从train_loader取出这个data和target时，data和targrt还是张量，需要从CPU放到DEVICE里面，DEVICE可以是CPU，也可以是GPU
        optimizer.zero_grad() #梯度值清零
        output = model(data)#送入data，得到输出 此时model就是整个神经网络，可以看成黑盒子
        loss = F.nll_loss(output, target)#计算output和target的偏差
        loss_total += loss #loss加起来
        loss.backward()#求导
        counter += 1 #计数
        optimizer.step()#进行model对参数都更新
        if (batch_idx + 1) % 30 == 0:#一共118次，31，62,93,次的时候会打印一下
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    average_loss = loss_total / counter #mse mean squared error均方误差的结果
    return average_loss.item()  # 返回一个标量


def test(model, device, test_loader):
    model.eval() #设置训练
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


if __name__ == '__main__':
    # 启动visdom：python -m visdom.server
    BATCH_SIZE = 512  # 每一组批数据的大小 最好是2的倍数，这样能最大限度地利用显存 大概需要2G的显存
    EPOCHS = 20  # 总共训练批次

    # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多 这里ZJK直接使用，没有深究 （但是最后发现还是CPU运行···）
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    # 启动visdom 这里ZJK直接使用，没有深究
    vis = Visdom(env='MNIST with BN')
    #画图，ZJK没有深究
    loss_win = vis.line(X=np.column_stack((np.array(0), np.array(0))),
                        Y=np.column_stack((np.array(0), np.array(0))),
                        opts=dict(legend=['Train loss', 'Test loss'],
                                  title='loss without BN'),
                        )

    #导入数据集 通过pytorch自带的DataLoader模块直接加载了minist数据集
    train_loader = torch.utils.data.DataLoader(   #返回的是某一个对象，而不是某一个具体的图片 当遍历这个对象的时候，就会返回图片的ID，各个像素点，label 具体怎么返回在test和train里写了
        datasets.MNIST('data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),#下载的minist格式是numpy格式，转换为tensor张量
                           transforms.Normalize((0.1307,), (0.3081,)) #归一化
                       ])),
        batch_size=BATCH_SIZE, shuffle=True) #shuffle决定是否打乱顺序，shuffle = TRUE时打乱顺序

    test_loader = torch.utils.data.DataLoader( #导入测试数据
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=BATCH_SIZE, shuffle=True)

    #定义model
    model = ConvNet().to(DEVICE)# 模块化式的写法：我们model是Convnet_BN,并且把它送到DEVICE里面去(CPU/CUDA)
    optimizer = optim.SGD(model.parameters(), lr=1e-3) #学习步长/学习速率 优化方法，随机梯度下降，我们要把model里面的参数放进去，让优化器知道优化的是这些参数
    starTime = datetime.datetime.now()

    for epoch in range(1, EPOCHS + 1):#每个EPOCH训练一次60000张图片
        average_train_loss = train(model, DEVICE, train_loader, optimizer, epoch) #计算训练损失
        average_test_loss = test(model, DEVICE, test_loader)                      #计算测试损失
        vis.line(Y=np.column_stack((np.array([average_train_loss]), np.array([average_test_loss]))), #visdom画图 ZJK没有深究
                 X=np.column_stack((np.array([epoch]), np.array([epoch]))),
                 win=loss_win,
                 opts=dict(legend=['Train loss', 'Test loss'],
                           title='loss without BN'),
                 update='replace' if epoch == 1 else 'append'
                 )
