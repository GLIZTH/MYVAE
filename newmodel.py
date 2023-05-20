

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class VAE(nn.Module):
    def __init__(self, image_size=2058, h_dim=500, z_dim=300):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim) # 均值 向量
        self.fc3 = nn.Linear(h_dim, z_dim) # 保准方差 向量
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
        
    # 编码过程
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    # 随机生成隐含向量
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 解码过程
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))
    
    # 整个前向传播过程：编码-》解码
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
def loss_fn(recon, x, mu, std):
    """
    :param recon: output of the decoder
    :param x: encoder input
    :param mu: mean
    :param std: standard deviation
    :return:"""
 
    recon_loss = nn.MSELoss()(recon, x)
    kl_loss=torch.mean(-0.5 * torch.sum(1 + std - mu ** 2 - std.exp(), dim = 1), dim = 0)
    loss = recon_loss+0.00025*kl_loss
    return loss
def train_loop(epoch,model,feature,optimizer,batch_size,device):
  model=model.to(device)
  model.train()
  for j in range(epoch):
    train_loss=0
    for i in range(0,len(feature),batch_size):
      y_pred,mu,sigma= model(feature[i:i+batch_size].view(-1,2058))
      loss=loss_fn(y_pred, feature[i:i+batch_size].view(-1,2058),mu,sigma)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      print(loss)
      train_loss+=loss
    print('epoch',j,train_loss)
feature=torch.load('feature.pt')
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
autoencoder = VAE().to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(),lr=0.001)
epoch=3000
batch_size=len(feature)
train_loop(epoch,autoencoder,feature.to(device),optimizer,batch_size,device)
recon=torch.zeros(feature.size(0),feature.size(1),feature.size(2))
pred=torch.zeros(feature.size(0),feature.size(1),feature.size(2))
for i in range(len(feature)):
    x,mu,sigma=autoencoder(feature[i].view(-1,2058).to(device))
    x=x.view(feature.size(1),feature.size(2))
    pred[i]=x
    for j in range(feature.size(1)):
        for k in range(feature.size(2)):
            if x[j][k]>=0.8:
                recon[i][j][k]=1