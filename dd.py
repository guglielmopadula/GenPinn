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

class DD(nn.Module):

    
    def __init__(self,latent_dim,drop_prob,hidden_dim: int= 500,T=10,beta_min=0.002,beta_max=0.2,**kwargs):
        super().__init__()
        self.drop_prob=drop_prob
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim
        self.hidden_nn = HiddenNN(in_dim=self.latent_dim,out_dim=self.latent_dim,drop_prob=self.drop_prob)
        self.T=T
        self.beta_min=beta_min
        self.beta_max=beta_max
        self.train_losses=[]
        self.eval_losses=[]

    def beta(self, t):
        return self.beta_min + (t / self.T) * (self.beta_max - self.beta_min)

    def alpha(self, t):
        return 1 - self.beta(t)

    def bar_alpha(self,t):
        return torch.prod(self.alpha(torch.arange(t)))
    
    def sigma(self,t):
        return self.beta(t)






def compute_loss(model,batch):
    x=batch
    t=int(torch.floor(torch.rand(1)*model.T)+1)
    eps=torch.randn(x.shape).to(x.device)
    input=torch.sqrt(model.bar_alpha(t)).to(x.device)*batch+(torch.sqrt(1-model.bar_alpha(t))*eps).to(x.device)
    loss=torch.linalg.norm(eps-model.hidden_nn(input))
    return loss

def sample(n, model):
    var=torch.ones(n,model.latent_dim)
    x=torch.randn(var.shape[0],model.latent_dim)


    for t in reversed(range(1,model.T+1)):
        z=torch.randn(var.shape[0],model.latent_dim)
        x=1/torch.sqrt(model.bar_alpha(t))*(x-(1-model.alpha(t))/(1-model.bar_alpha(t))*model.hidden_nn(x))+model.sigma(t)*z
    return x

latent=torch.tensor(np.load("latent.npy")).float()
std=torch.std(latent,axis=0)
mean=torch.mean(latent,axis=0)

latent=(latent-mean)/std
model=DD(latent.shape[1],0.05)
optimizer=torch.optim.Adam(model.parameters(),1e-02)
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
#np.save("dd_latent.npy",z.detach().numpy())
