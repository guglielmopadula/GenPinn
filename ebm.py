from torch import nn
import torch.autograd as autograd
import torch
import numpy as np
from tqdm import trange
class LBR(nn.Module):
    def __init__(self,in_features,out_features,drop_prob):
        super().__init__()
        self.lin=nn.Linear(in_features, out_features)
        self.batch=nn.BatchNorm1d(out_features)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(drop_prob)
    
    def forward(self,x):
        return self.dropout(self.relu(self.batch(self.lin(x))))
    
class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim,drop_prob):
        super().__init__()
        self.fc1_interior = LBR(latent_dim,hidden_dim,drop_prob)
        self.fc2_interior = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc3_interior = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc4_interior = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc5_interior = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc6_interior = LBR(hidden_dim,hidden_dim,drop_prob)
        self.fc7_interior = nn.Linear(hidden_dim,1)
        
    def forward(self,x):
        x_hat=self.fc7_interior(self.fc6_interior(self.fc5_interior(self.fc4_interior(self.fc3_interior(self.fc2_interior(self.fc1_interior(x)))))))
        return x_hat
    

def compute_loss(model,batch):
    pos_x=batch
    neg_x = torch.randn_like(pos_x)
    neg_x = sample_langevin(neg_x, model, 0.01, 50)
    pos_out = model(pos_x)
    neg_out = model(neg_x)
    loss = (pos_out - neg_out) + 10 * (pos_out ** 2 + neg_out ** 2)
    loss = loss.mean()
    return loss

def sample_langevin(x, model, stepsize, n_steps):
    l_samples = []
    l_dynamics = []
    x.requires_grad = True
    noise_scale = np.sqrt(stepsize * 2)
    for _ in range(n_steps):
        l_samples.append(x.detach())
        noise = torch.randn_like(x) * noise_scale
        out = model(x)
        if out.requires_grad==False:
            out.requires_grad=True
        grad = autograd.grad(out.sum(), x, only_inputs=True)[0]
        dynamics = stepsize * grad + noise
        x = x + dynamics
        l_samples.append(x.detach())
        l_dynamics.append(dynamics.detach())
    return l_samples[-1]

latent=torch.tensor(np.load("latent.npy")).float()
std=torch.std(latent,axis=0)
mean=torch.mean(latent,axis=0)

latent=(latent-mean)/std
model=Generator(latent.shape[1],500,0.05)
optimizer=torch.optim.Adam(model.parameters(),1e-03)
for epochs in trange(50):
    optimizer.zero_grad()
    loss=compute_loss(model,latent)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        print(loss)


z=torch.randn(latent.shape[0],latent.shape[1])
z=sample_langevin(z, model, 0.01, 50)
z=z*std+mean
np.save("ebm_latent.npy",z.detach().numpy())