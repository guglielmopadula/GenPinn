import torch
from torch import nn

BATCH_SIZE=1

class Encoder(nn.Module):
    def __init__(self,latent_dim):
        super(Encoder, self).__init__()
        self.nn1=nn.Sequential(nn.Linear(6,100),nn.LayerNorm(100),nn.ReLU(),nn.Linear(100,100),nn.LayerNorm(100),nn.ReLU(),nn.Linear(100,100))
        self.nn2=nn.Sequential(nn.Linear(100,100),nn.LayerNorm(100),nn.ReLU(),nn.Linear(100,100),nn.LayerNorm(100),nn.ReLU(),nn.Linear(100,5))

    def forward(self,x,pos):
        x=torch.cat((x,pos),dim=2)
        x=x.reshape(-1,6)
        x=self.nn1(x)
        x=x.reshape(BATCH_SIZE,-1,100)
        x=torch.mean(x,dim=1)
        x=self.nn2(x)
        return x



class Decoder(nn.Module):
    def __init__(self,latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim=latent_dim
        self.model=nn.Sequential(nn.Linear(3+latent_dim,100),nn.LayerNorm(100),nn.ReLU(),nn.Linear(100,100),nn.LayerNorm(100),nn.ReLU(),nn.Linear(100,3))


    def forward(self,latent,pos):
        pos=pos.reshape(BATCH_SIZE,-1,3)
        latent=latent.reshape(-1,1,self.latent_dim).repeat(1,pos.shape[1],1)
        x=torch.cat((latent,pos),dim=2)
        x=x.reshape(-1,3+self.latent_dim)
        x=self.model(x)
        x=x.reshape(BATCH_SIZE,-1,3)
        return x

class AutoEncoder(nn.Module):
    def __init__(self,latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder=Encoder(latent_dim)
        self.decoder=Decoder(latent_dim)

    def forward(self,batch):
        x,pos=batch
        latent=self.encoder(x,pos)
        x=self.decoder(latent,pos)
        return x,latent
