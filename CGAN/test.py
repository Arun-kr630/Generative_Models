import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class Generator(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.linear=nn.Sequential(nn.Linear(input_size,128),nn.ReLU(),
                                  nn.Linear(128,256),nn.Tanh(),
                                  nn.Linear(256,512),nn.Tanh(),
                                  nn.Linear(512,28*28),nn.Sigmoid()
                                  )
    def forward(self,x):
        x=self.linear(x)
        x=x.view(-1,1,28,28)
        return x

z_dim=64
image_dim=28*28
gen=Generator(z_dim+10)
checkpoint=torch.load('checks_99.pth')
gen.load_state_dict(checkpoint['gen_state'])
device='cuda:4' if torch.cuda.is_available() else 'cpu'
gen=gen.to(device)
batch_size=4
fixed_noise=torch.randn((batch_size,z_dim)).to(device).to(device)
digits=torch.randint(0,10,(batch_size,),dtype=torch.long).to(device)
one_hot=F.one_hot(digits,num_classes=10).to(device)
fixed_noise=torch.cat((fixed_noise,one_hot),dim=1)
fixed_noise=fixed_noise.to(device)
generated_image=gen(fixed_noise).reshape(-1, 1, 28, 28)
print(digits)
for i in range(generated_image.shape[0]):
    torchvision.utils.save_image(generated_image[0], f'CGAN/image_{i}.jpg')
