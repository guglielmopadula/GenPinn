import numpy as np
import meshio
import torch
from tqdm import trange
from torch import nn
from torchffd import TorchFFD3D

reference=meshio.read("data/Stanford_Bunny_3d.mesh")
points=reference.points
triangles=reference.cells_dict["triangle"]
boundary_indices=np.sort(np.unique(triangles.reshape(-1)))
num_points=len(reference.points)
inner_indices=np.sort(np.setdiff1d(np.arange(num_points),boundary_indices))

theta=np.load("latent_ffd.npy")

theta=torch.tensor(theta,dtype=torch.float32)

boundary_points=torch.tensor(points[boundary_indices],dtype=torch.float32)
inner_points=torch.tensor(points[inner_indices],dtype=torch.float32)

time=torch.linspace(0,1,100)

class Sin(nn.Module):

    def forward(self,x):
        return torch.sin(x)


def g(t):
    return 0.5+t*0.5

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(76,100),nn.Tanh(),nn.Linear(100,100),nn.Tanh(),nn.Linear(100,1))

    def forward(self,x):
        t=x[75]
        return 0.5+0.5*t+t*(self.model(x))

model=Model()

optimizer=torch.optim.Adam(model.parameters(),1e-03)




for epochs in range(1):
    for i in trange(600):
        optimizer.zero_grad()
        parameter=theta[i]
        torchffd=TorchFFD3D([3,3,3])
        parameter_x=parameter[:27]
        parameter_y=parameter[27:54]
        parameter_z=parameter[54:]
        torchffd.array_mu_x=parameter_x.reshape(3,3,3)
        torchffd.array_mu_y=parameter_y.reshape(3,3,3)
        torchffd.array_mu_z[:,:,1:]=parameter_z.reshape(3,3,2)
        def_point_inner=torchffd(inner_points)
        def_point_bound=torchffd(boundary_points)
        def_point_inner=torch.concatenate((parameter.reshape(1,-1).repeat(def_point_inner.shape[0],1),def_point_inner),axis=1)
        def_point_bound=torch.concatenate((parameter.reshape(1,-1).repeat(def_point_bound.shape[0],1),def_point_bound),axis=1)
        def_point_bound_0=torch.concatenate((def_point_bound,torch.zeros(def_point_bound.shape[0],1)),axis=1)
        def_point_inner_0=torch.concatenate((def_point_inner,torch.zeros(def_point_inner.shape[0],1)),axis=1)
        time=torch.linspace(0,1,100)
        def_point_inner_t=torch.concatenate((
            def_point_inner.reshape(-1,1,75).repeat(1,100,1),
            time.reshape(1,100,1).repeat(def_point_inner.shape[0],1,1)
        ),axis=2)

        def_point_bound_t=torch.concatenate((
            def_point_bound.reshape(-1,1,75).repeat(1,100,1),
            time.reshape(1,100,1).repeat(def_point_bound.shape[0],1,1)
        ),axis=2)

        def_point_bound_t=def_point_bound_t.reshape(-1,76)
        def_point_inner_t=def_point_inner_t.reshape(-1,76)
        def time_derivative(x):
            return torch.func.jacrev(model)(x)[:,75]

        def x_derivative(x):
            return torch.func.jacrev(model)(x)[:,72].reshape(-1)

        def xx_derivative(x):
            return torch.func.jacrev(x_derivative)(x)[:,72]

        def y_derivative(x):
            return torch.func.jacrev(model)(x)[:,73].reshape(-1)

        def yy_derivative(x):
            return torch.func.jacrev(y_derivative)(x)[:,73]


        def z_derivative(x):
            return torch.func.jacrev(model)(x)[:,74].reshape(-1)

        def zz_derivative(x):
            return torch.func.jacrev(z_derivative)(x)[:,74]

        u_t=torch.vmap(time_derivative)(def_point_inner_t).reshape(-1,1)
        u_xx=torch.vmap(xx_derivative)(def_point_inner_t).reshape(-1,1)
        u_yy=torch.vmap(yy_derivative)(def_point_inner_t).reshape(-1,1)
        u_zz=torch.vmap(zz_derivative)(def_point_inner_t).reshape(-1,1)
        loss_inner_t=torch.linalg.norm(u_t-u_xx-u_yy-u_zz)/len(def_point_inner_t)
        loss_bound_t=torch.linalg.norm(torch.vmap(model)(def_point_bound_t)-g(time).reshape(1,100,1).repeat(def_point_bound.shape[0],1,1).reshape(-1,1))/len(def_point_bound_t)
        loss=loss_inner_t+loss_bound_t
        loss.backward()
        optimizer.step()
        with torch.no_grad():
           if epochs%100==0:
               print(loss_inner_t,loss_bound_t)

torch.save(model,"pinn.pt")