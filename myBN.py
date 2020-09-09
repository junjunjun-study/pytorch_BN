import torch.nn as nn
import torch

#y = ( x-mean(x) )/( sqrt(Var(x)+eps)  )*gamma + beta
#pytorch里面网络的层次或者结构都要继承nn.module
#然后要重构init和forward这两个方法
#对于全连接来说，num_feature对应神经元的个数
#对于一个BN网络结构来说，它面对的不是一个神经元，而是一层隐藏层
#对于之后的全连接网络而言，它只能接受一维的数据！！！！！！！！
class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):  #添加一些变量
        super(BatchNorm, self).__init__() #这里为固定写法，ZJK没有深究
        self.num_features = num_features #通道数
        self.eps = eps                  #分母中添加的一个值，目的是为了计算的稳定性
        self.momentum = momentum        #衰减率，用于更新mean和deviation的数值
        # 可学习的参数
        # 注意：这两个参数的改变并没有经过ZJK写的batch_normalization实现，而是自动更新的，一旦它是nn.Parameter设置的，那么它就会进入网络中，最后在反向传播时自动更新
        self.gamma = nn.Parameter(torch.Tensor(self.num_features), #nn.Parameter设置一个可学习（可修改）的参数 注意这里的gamma不是一个值，而是一个张量！！！有多少个num_features即有多少个神经元就有多少个gamma参数  torch.Tensor是把self.num_features转换为一个张量  requires_grad 需要求导
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(self.num_features), #同理这里的beta也是对应一个张量
                                 requires_grad=True)
        # mean and deviation 方差
        self.changing_mean = torch.zeros(self.num_features)  #返回一个全为0的张量
        self.changing_deviation = torch.ones(self.num_features)    #返回一个全为1的张量
        self.reset_parameters()  # 对参数进行初始化

    def reset_parameters(self):
        # 初始化参数
        # gamma为0~1之间的随机数，beta初始化为0
        nn.init.uniform_(self.gamma)  #使得self.gamma均匀分布 服从~U(0,1)  即gamma的张量里每个初值为1/num_features
        nn.init.zeros_(self.beta)     #全0分布 使得beta的张量里的每个初值为0
        # 均值和方差初始化为0和1
        nn.init.zeros_(self.changing_mean)  #全0分布，使得changing_mean这个张量里的每一个初值为0
        nn.init.ones_(self.changing_deviation)    #全1分布，使得changing_deviation这个张量里的每一个初值为1

    def forward(self, X):  #真正最后执行的函数 X是什么输入变量？其含义是什么？
        assert len(X.shape) in (2, 4)    #???????????????assert?啥意思？
        if X.device.type != 'cpu':  #判断是CPU运行还是GPU运行，ZJK选择GPU运行
            self.changing_mean = self.changing_mean.cuda()  #如果是GPU就把这个东西转换为cuda
            self.changing_deviation = self.changing_deviation.cuda()
        Y = self.batch_norm(X)  #之前为数据处理，这里为核心函数
        return Y

    def batch_norm(self, X): #输入的每个批数据
        #基本上面对全连接网络执行
        if len(X.shape) == 2:  # 一维的BN   为啥这就是一维的了？
            #一维的话，就是取行平均
            mean = torch.mean(X, dim=0)   #dim的不同值表示维度。特别的在dim = 0表示二维中的行，dim = 1在二维矩阵中表示列，不管一个矩阵是几维的，比如一个矩阵维度如下:(d0,d1,``````,dn-1),阿么dim = 0就表示对应到d0也就是第一个维度,dim = 1表示对应到第二个维度，依次类推
            #一维的话，也是对行计算
            deviation = torch.mean((X-mean) * (X-mean), dim=0)  #计算方差
            if self.training:  # 这个属性是继承了module的!!!!!!!!!!!!!!是mudule里self__init__里面的！这个training的意思是，如果training为真，那么现在运行的状态是训练状态，否则是测试状态，通过modul.eval()，modul.train()来更改
                X_hat = (X - mean) / torch.sqrt(deviation + self.eps)  #X_hat的含义是啥？
                self.changing_mean = self.momentum * self.changing_mean + (1.0 - self.momentum) * mean  #更新mean值 在训练时利用权重滑动平均法更新而得 待更新的值 = momentum(衰减率) * 旧值 + （1 - momentum） * 最新批次的均值； 衰减率越大，结果越依赖于之前的计算 导致每个批次的均值和方差差不多
                                                                                                        #也就是说这里的权重滑动平均的话，历史的占比会一直衰减，作用也越来越小
                self.changing_deviation = self.momentum * self.changing_deviation + (1.0 - self.momentum) * deviation #更新deviation值 方法如上
            else:#如果是训练状态，则直接进行计算
                X_hat = (X - mean) / torch.sqrt(deviation + self.eps)
            out = self.gamma * X_hat + self.beta
        # 基本上面对全连接网络之前的卷积层和池化层
        if len(X.shape) == 4:  # 二维的BN    (Batch_size,通道数，图片高度，图片宽度)
            shape_2d = (1, X.shape[1], 1, 1)   #这是一个元祖 (Batch_size,通道数，图片高度，图片宽度) 保留其通道数
            mean = torch.mean(X, dim=(0, 2, 3)).view(shape_2d) #把元祖里取平均再转换为张量传递给mean，保留通道数
            deviation = torch.mean((X - mean) * (X - mean), dim=(0, 2, 3)).view(shape_2d)#把元祖里取方差再转换为张量传递给deviation,保留通道数
            if self.training: #如果是训练，就更新mean和deviation的数值
                X_hat = (X - mean) / torch.sqrt(deviation + self.eps) #计算中间变量
                # 同一维的权重滑动平均,更新平均值
                self.changing_mean = self.momentum * self.changing_mean.view(shape_2d)\
                                   + (1.0 - self.momentum) * mean
                # 同一维的权重滑动平均，更新方差
                self.changing_deviation = self.momentum * self.changing_deviation.view(shape_2d)\
                                  + (1.0 - self.momentum) * deviation
            else: #如果不是训练，那就直接计算
                X_hat = (X - self.changing_mean.view(shape_2d)) / torch.sqrt(self.changing_deviation.view(shape_2d) + self.eps) #计算X_hat
            out = self.gamma.view(shape_2d) * X_hat + self.beta.view(shape_2d) #输出out

        return out
