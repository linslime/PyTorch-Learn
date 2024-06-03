import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from  torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
class VAE(nn.Module):
    def __init__(self,input_dim,hidden_dim,gaussian_dim):
        super().__init__()
        #编码器
        #隐藏层
        self.fc1=nn.Sequential(
            nn.Linear(in_features=input_dim,out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=256),
            nn.Tanh(),
        )
        #μ和logσ^2
        self.mu=nn.Linear(in_features=256,out_features=gaussian_dim)
        self.log_sigma=nn.Linear(in_features=256,out_features=gaussian_dim)



        #解码（重构）
        self.fc2=nn.Sequential(
            nn.Linear(in_features=gaussian_dim,out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=512),
            nn.Tanh(),
            nn.Linear(in_features=512,out_features=input_dim),
            nn.Sigmoid() #图片被转为为0，1的值了，故用此函数
        )
    def forward(self,x):
        #隐藏层
        h=self.fc1(x)

        #计算期望和log方差
        mu=self.mu(h)
        log_sigma=self.log_sigma(h)

        #重参数化
        h_sample=self.reparameterization(mu,log_sigma)

        #重构
        reconsitution=self.fc2(h_sample)

        return reconsitution,mu,log_sigma

    def reparameterization(self,mu,log_sigma):
        #重参数化
        sigma=torch.exp(log_sigma*0.5) #计算σ
        e=torch.randn_like(input=sigma,device=device)

        result=mu+e*sigma #依据重参数化技巧可得

        return result
    def predict(self,new_x): #预测
        reconsitution=self.fc2(new_x)

        return reconsitution
def train():

    transformer = transforms.Compose([
        transforms.ToTensor(),
    ]) #归一化
    data = MNIST("./data", transform=transformer,download=True) #载入数据

    dataloader = DataLoader(data, batch_size=128, shuffle=True) #写入加载器

    model = VAE(784, 512, 20).to(device) #初始化模型

    optimer = torch.optim.Adam(model.parameters(), lr=1e-3) #初始化优化器

    loss_fn = nn.MSELoss(reduction="sum") #均方差损失
    epochs = 100 #训练100轮

    for epoch in torch.arange(epochs):
        all_loss = 0
        dataloader_len = len(dataloader.dataset)

        for data in tqdm(dataloader, desc="第{}轮梯度下降".format(epoch)):
            sample, label = data
            sample = sample.to(device)
            sample = sample.reshape(-1, 784) #重塑
            result, mu, log_sigma = model(sample) #预测

            loss_likelihood = loss_fn(sample, result) #计算似然损失

            #计算KL损失
            loss_KL = torch.pow(mu, 2) + torch.exp(log_sigma) - log_sigma - 1

            #总损失
            loss = loss_likelihood + 0.5 * torch.sum(loss_KL)

            #梯度归0并反向传播和更新
            optimer.zero_grad()

            loss.backward()

            optimer.step()
            with torch.no_grad():
                all_loss += loss.item()
        print("函数损失为：{}".format(all_loss / dataloader_len))
        torch.save(model, "./model/VAE.pth")
if __name__ == '__main__':
    #是否有闲置GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #训练
    train()

    #载入模型，预测
    model=torch.load("./model/VAE.pth",map_location="cpu")
    #预测20个样本
    x=torch.randn(size=(20,20))
    result=model.predict(x).detach().numpy()
    result=result.reshape(-1,28,28)
    #绘图
    for i in range(20):
        plt.subplot(4,5,i+1)
        plt.imshow(result[i])
        plt.gray()
    plt.show()

