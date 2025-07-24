from torch import nn
import torch.autograd as autograd
import torch
import numpy as np
from tqdm import trange
from torch import nn



class LBR(nn.Module):
    def __init__(self,in_features,out_features,drop_prob):
        super().__init__()
        self.lin=nn.Linear(in_features, out_features)
        self.batch=nn.BatchNorm1d(out_features)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(drop_prob)
    
    def forward(self,x):
        return self.dropout(self.relu(self.batch(self.lin(x))))
    


class HiddenNN(nn.Module):
    def __init__(self, in_dim,out_dim,drop_prob):
        super().__init__()
        self.fc1_interior = LBR(in_dim,in_dim,drop_prob)
        self.fc2_interior = LBR(in_dim,in_dim,drop_prob)
        self.fc3_interior = LBR(in_dim,out_dim,drop_prob)
        self.fc4_interior = LBR(out_dim,out_dim,drop_prob)
        self.fc5_interior = LBR(out_dim,out_dim,drop_prob)
        self.fc6_interior = LBR(out_dim,out_dim,drop_prob)
        self.fc7_interior = nn.Linear(out_dim,out_dim)
        
    def forward(self,x):
        x_hat=self.fc7_interior(self.fc6_interior(self.fc5_interior(self.fc4_interior(self.fc3_interior(self.fc2_interior(self.fc1_interior(x)))))))
        return x_hat

class NVP(nn.Module):
    def __init__(self, latent_dim,drop_prob):
        super().__init__()
        self.latent_dim=latent_dim
        perm=torch.randperm(latent_dim)
        self.pres=perm[:len(perm)//2]
        self.notpres=perm[len(perm)//2:]
        self.s=HiddenNN(len(self.pres),len(self.notpres),drop_prob)
        self.m=HiddenNN(len(self.pres),len(self.notpres),drop_prob)    
    
    def forward(self,x):
        x1=x.reshape(-1,self.latent_dim)[:,self.pres]
        x2=x.reshape(-1,self.latent_dim)[:,self.notpres]
        z1=x1
        z2=torch.exp(self.s(x1))*x2+self.m(x1)
        z=torch.zeros(z1.shape[0],self.latent_dim).to(z2.device)
        z[:,self.pres]=z1
        z[:,self.notpres]=z2
        return z,self.s(x1).sum(1)
    
    def my_backward(self,z):
        z1=z.reshape(-1,self.latent_dim)[:,self.pres]
        z2=z.reshape(-1,self.latent_dim)[:,self.notpres]
        x1=z1
        x2=torch.exp(-self.s(z1))*(z2-self.m(z1))
        x=torch.zeros(z1.shape[0],self.latent_dim).to(x1.device)
        x[:,self.pres]=x1
        x[:,self.notpres]=x2
        return x,-self.s(z1).sum(1)
    
class NF(nn.Module):
    def __init__(self, latent_dim,drop_prob):
        super().__init__()
        self.modulelist=nn.ModuleList([NVP(latent_dim,drop_prob) for i in range(5)])
        self.dist=torch.distributions.Normal(loc=0.,scale=1.)
        self.latent_dim=latent_dim

    def forward(self,x):
        det=0
        for flow in self.modulelist:
            x,det_tmp=flow(x)
            det=det+det_tmp

        return x,det
    
    def my_backward(self,z):
        det=0
        for flow in reversed(self.modulelist):
            z,det_tmp=flow.my_backward(z)
            det=det+det_tmp
        return z,det





def compute_loss(model,batch):
    x=batch
    z,lk=model.forward(x)
    lnorm=model.dist.log_prob(z).sum(1)
    loss=-torch.mean(lk+lnorm)
    return loss

def sample(n, model):
    var=torch.ones(n,model.latent_dim)
    z=torch.randn(var.shape[0],model.latent_dim)
    z,_=model.my_backward(z)
    return z

latent=torch.tensor(np.load("latent.npy")).float()
std=torch.std(latent,axis=0)
mean=torch.mean(latent,axis=0)

latent=(latent-mean)/std
model=NF(latent.shape[1],0.05)
optimizer=torch.optim.Adam(model.parameters(),1e-05)
for epochs in trange(509):
    optimizer.zero_grad()
    loss=compute_loss(model,latent)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        print(loss)


z=torch.randn(latent.shape[0],latent.shape[1])
z=sample(50, model)
z=z*std+mean
#np.save("nf_latent.npy",z.detach().numpy())
