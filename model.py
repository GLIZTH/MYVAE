import torch.nn as nn
import torch
import torch.nn.functional as F
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims,device):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(2058, 400)
        self.linear2 = nn.Linear(400, latent_dims)
        self.linear3 = nn.Linear(400, latent_dims)

      
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))


        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*torch.rand_like(mu)

        return z,mu,sigma
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 400)
        self.linear2 = nn.Linear(400, 2058)
    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.view(-1,42,49)
class Autoencoder(nn.Module):
    def __init__(self, latent_dims,device):
        super(Autoencoder, self).__init__()
        self.VariationalEncoder = VariationalEncoder(latent_dims,device)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z,mu,sigma = self.VariationalEncoder(x)
        return self.decoder(z),mu,sigma
def loss_fn(recon, x, mu, std):
    """
    :param recon: output of the decoder
    :param x: encoder input
    :param mu: mean
    :param std: standard deviation
    :return:"""
    loss_fn=nn.MSELoss()
    recon_loss = loss_fn(recon,x)
    kl_loss=torch.mean(-0.5 * torch.sum(1 + std - mu ** 2 - std.exp(), dim = 1), dim = 0)
    loss = recon_loss+0.00025*kl_loss
    return loss
def train_loop(epoch,model,feature,optimizer,batch_size,device):
  model=model.to(device)
  model.train()
  for j in range(epoch):
    train_loss=0
    for i in range(0,len(feature),batch_size):
      y_pred,mu,sigma= model(feature[i:i+batch_size])
      loss=loss_fn(y_pred, feature[i:i+batch_size],mu,sigma)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      print(loss)
      train_loss+=loss
    print('epoch',j,train_loss)
feature=torch.load('feature.pt')
print(feature.size())
print(5)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
autoencoder = Autoencoder(500,device).to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(),lr=0.00001)
epoch=2000
batch_size=len(feature)
train_loop(epoch,autoencoder,feature.to(device),optimizer,batch_size,device)
recon=torch.zeros(feature.size(0),feature.size(1),feature.size(2))
pred=torch.zeros(feature.size(0),feature.size(1),feature.size(2))
for i in range(len(feature)):
    x,mu,sigma=autoencoder(feature[i].view(-1,feature.size(1),feature.size(2)).to(device))
    x=x.view(feature.size(1),feature.size(2))
    pred[i]=x
    for j in range(feature.size(1)):
        for k in range(feature.size(2)):
            if x[j][k]>=0.8:
                recon[i][j][k]=1