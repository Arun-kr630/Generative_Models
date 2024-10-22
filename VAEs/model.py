import torch
import torch.nn as nn
from torch.nn import functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self,input_size,h_dim=200,z_dim=20):
        super().__init__()
        self.img_2hid=nn.Linear(input_size,h_dim)
        self.hid_2mu=nn.Linear(h_dim,z_dim)
        self.hid_2sigma=nn.Linear(h_dim,z_dim)

        self.z_2hid=nn.Linear(z_dim,h_dim)
        self.hid_2img=nn.Linear(h_dim,input_size)
        self.relu=nn.ReLU()

    def encode(self,x):
        h=self.relu(self.img_2hid(x))
        mu,sigma=self.hid_2mu(h),self.hid_2sigma(h)
        return mu,sigma
    def decode(self,z):
        h=self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))
        
    def forward(self,x):
        mu,sigma=self.encode(x)
        epsilon=torch.rand_like(sigma)
        z_reparametrized = mu + sigma*epsilon
        x_reconstructed=self.decode(z_reparametrized)
        return x_reconstructed,mu,sigma
if __name__=="__main__":
    x=torch.randn(4,28*28)
    vae=VariationalAutoEncoder(input_size=784)
    x,m,s=vae(x)
    print(x.shape)
    print(m.shape)
