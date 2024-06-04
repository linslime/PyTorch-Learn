import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from  torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.functional import  F
class Coupling(nn.Module):
    def __init__(self,input_dim,hidden_dim,hidden_layer,odd_flag):
        '''
        加性耦合层
        @param input_dim: 输入维度
        @param hidden_dim: 隐藏层维度
        @param hidden_layer: 隐藏层个数
        @param odd_flag: 当前耦合层是否是在整个模型中属于奇数（用作调换切分顺序）
        '''
        super().__init__()
        #用作判断是否需要互换切分的位置
        self.odd_flag=odd_flag%2

        #五层隐藏层，神经元1000
        self.input_transform=nn.Sequential(
            nn.Linear(input_dim//2,hidden_dim),
            nn.ReLU()
        )
        self.m=nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim),
                nn.ReLU(),
            ) for _ in range(hidden_layer-1)
        ])
        #输出为原始维度
        self.out_transform=nn.Sequential(
            nn.Linear(hidden_dim,input_dim//2),
        )
    def forward(self,x,reverse):
        '''
        @param x:  数据 [ batch_size , 784 ]
        @param reverse: 是否是反向推导（z->x）
        @return:
        '''
        batch_size,W=x.shape #取出维度
        #重构维度（为了切分）
        x=x.reshape(batch_size,W//2,2)

        #按奇偶性划分
        if self.odd_flag:
            x1, x2 = x[:, :, 0], x[:, :, 1]
        else:
            x2, x1 = x[:, :, 0], x[:, :, 1]

        #将x2输入神经网络
        input_transfrom=self.input_transform(x2)
        for i in self.m:
            input_transfrom=i(input_transfrom)
        out_transform=self.out_transform(input_transfrom)

        #是否是反向推导
        if reverse:
            x1=x1-out_transform #反函数
        else:
            x1=x1+out_transform

        #将数据组合回来
        if self.odd_flag:
            x=torch.stack((x1,x2),dim=2)
        else:
            x=torch.stack((x2,x1),dim=2)

        return x.reshape(-1,784)

class Scale(nn.Module):
    def __init__(self,input_dim):
        '''
        缩放层
        @param input_dim: 输入数据维度
        '''
        super().__init__()

        #构造与数据同维度的s
        self.s=nn.Parameter(torch.zeros(1,input_dim))
    def forward(self,x,reverse):
        '''
        @param x: 输入数据
        @param reverse: 是否是反向推导
        @return:
        '''
        if reverse:
            result=torch.exp(-self.s)*x #反函数
        else:
            result=torch.exp(self.s)*x
        return result,self.s
class NICE(nn.Module):
    def __init__(self,couping_num):
        '''
        @param couping_num: 耦合层个数
        '''
        super().__init__()
        #初始化耦合层
        self.couping=nn.ModuleList([
            Coupling(784,1000,5,odd_flag=i+1)
            for i in torch.arange(couping_num)
        ])
        #初始化缩放层
        self.scale=Scale(784)

    def forward(self,x,reverse):

        '''
        前向推导
        @param x: 输入数据
        @param reverse: #是否是反向
        @return:
        '''
        for i in self.couping:
            x=i(x,reverse)
        h,s=self.scale(x,reverse)
        return h,s
    def likeihood(self,h,s):
        #计算极大似然估计
        loss_s = torch.sum(s) #s的log雅可比行列式损失
        log_prob = proior.log_prob(h) #logictic分布极大似然
        loss_prob = torch.sum(log_prob, dim=1) #案列求和
        loss = loss_s + loss_prob #总损失
        #由于pytorch是最小值优化，故取反
        return -loss
    def generate(self,h):
        '''
        @param h: logistic分布采样所得
        @return:
        '''
        z,s=self.scale(h,True)
        for i in reversed(self.couping):
            z=i(z,True)
        return z
def train():
    #归一化
    transformer = transforms.Compose([
        transforms.ToTensor()
    ])
    #载入数据
    data = MNIST("data", transform=transformer, download=True)
    #存入写入器
    dataloader = DataLoader(data, batch_size=200, shuffle=True,num_workers=4)
    #初始化模型
    nice = NICE(4).to(device)
    #优化器
    optimer = torch.optim.Adam(params=nice.parameters(), lr=1e-3,eps=1e-4,betas=(0.9,0.999))

    #开始训练
    epochs = 1000

    for epoch in torch.arange(epochs):
        loss_all = 0
        dataloader_len = len(dataloader)
        for i in tqdm(dataloader, desc="第{}轮次".format(epoch)):
            sample, label = i
            sample = sample.reshape(-1, 784).to(device)

            h, s = nice(sample, False) #预测
            loss = nice.likeihood(h, s).mean() #计算损失

            optimer.zero_grad() #归零
            loss.backward() #反向传播
            optimer.step() #更新

            with torch.no_grad():
                loss_all += loss
        print("损失为{}".format(loss_all / dataloader_len))
        torch.save(nice, "nice.pth")

class Logistic(torch.distributions.Distribution):
    '''
    Logistic 分布
    '''
    def __init__(self):
        super(Logistic, self).__init__()

    def log_prob(self, x):

        return -(F.softplus(x) + F.softplus(-x))

    def sample(self, size):

        z = torch.distributions.Uniform(0., 1.).sample(size)
        return torch.log(z) - torch.log(1. - z)
if __name__ == '__main__':
    # 是否有闲置GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #先验分布
    proior = Logistic()

    #训练
    train()

    #预测
    x=proior.sample((10,784))#采样

    #载入模型
    nice=torch.load("nice.pth",map_location=device)
    #生成数据
    result=nice.generate(x)
    result=result.reshape(-1,28,28)
    for i in range(10):
        plt.subplot(2,5,i+1)
        img=result[i].detach().numpy()
        plt.imshow(img)
        plt.gray()
    plt.show()
