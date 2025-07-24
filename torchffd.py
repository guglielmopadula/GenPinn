import torch
from torch import nn

def ranged_binom(n):
    seq=torch.arange(1,n+1)
    seq_pos=torch.concatenate((torch.tensor([1.]),torch.arange(1,n+1)))        
    return torch.prod(seq)/(torch.cumprod(seq_pos,0)*(torch.flip(torch.cumprod(seq_pos,0),(0,))))

class TorchFFD3D(nn.Module):
    def __init__(self,n_control_points):
        super().__init__()
        self.n_control_points_x=n_control_points[0]
        self.n_control_points_y=n_control_points[1]
        self.n_control_points_z=n_control_points[2]
        self.array_mu_x=(torch.zeros(n_control_points))
        self.array_mu_y=(torch.zeros(n_control_points))
        self.array_mu_z=(torch.zeros(n_control_points))

    def bernstein_mesh(self,points):
        px=points[:,0].reshape(-1,1,1,1)
        py=points[:,1].reshape(-1,1,1,1)
        pz=points[:,2].reshape(-1,1,1,1)
        seq_x=torch.arange(self.n_control_points_x).reshape(1,-1,1,1)
        seq_y=torch.arange(self.n_control_points_y).reshape(1,1,-1,1)
        seq_z=torch.arange(self.n_control_points_z).reshape(1,1,1,-1) 
        ran_x=ranged_binom(self.n_control_points_x-1).reshape(1,-1,1,1)
        ran_y=ranged_binom(self.n_control_points_y-1).reshape(1,1,-1,1)
        ran_z=ranged_binom(self.n_control_points_z-1).reshape(1,1,1,-1)
        return ran_x*ran_y*ran_z*(px**(seq_x))*((1-px)**(self.n_control_points_x-1-seq_x))*(py**(seq_y))*((1-py)**(self.n_control_points_y-1-seq_y))*(pz**(seq_z))*((1-pz)**(self.n_control_points_z-1-seq_z))


    def control_points(self):
        x = torch.linspace(0, 1, self.n_control_points_x)
        y = torch.linspace(0, 1, self.n_control_points_y)
        z = torch.linspace(0, 1, self.n_control_points_z)
        x_coords,y_coords,z_coords=torch.meshgrid(x,y,z,indexing='ij')
        x_coords=x_coords+self.array_mu_x
        y_coords=y_coords+self.array_mu_y
        z_coords=z_coords+self.array_mu_z
        return torch.concatenate((x_coords.unsqueeze(0),y_coords.unsqueeze(0),z_coords.unsqueeze(0)),axis=0)
    
    def forward(self,x):
        cp=self.control_points().unsqueeze(0)
        bm=self.bernstein_mesh(x).unsqueeze(1)
        return torch.sum(bm*cp,dim=(2,3,4))

    def forward_fun(self,x,cp):
        bm=self.bernstein_mesh(x).unsqueeze(1)
        return torch.sum(bm*cp.unsqueeze(0),dim=(2,3,4))
