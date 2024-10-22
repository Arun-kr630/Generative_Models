from torchvision import datasets
from tqdm import tqdm
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Generator(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.linear=nn.Sequential(nn.Linear(input_size,128),nn.ReLU(),
                                  nn.Linear(128,256),nn.Tanh(),
                                  nn.Linear(256,512),nn.Tanh(),
                                  nn.Linear(512,28*28),nn.Sigmoid()
                                  )
    def forward(self,x): # x--> [B,64+10]
        x=self.linear(x)
        x=x.view(-1,1,28,28)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Sequential(nn.Linear(11*28*28,512),nn.ReLU(),
                                  nn.Linear(512,256),nn.ReLU(),
                                  nn.Linear(256,128),nn.Tanh(),
                                  nn.Linear(128,32),nn.Tanh(),
                                  nn.Linear(32,1),nn.Sigmoid(),
                                  )
    def forward(self,x):# x--> [B,11*28*28]
        return self.linear(x)
n_classes=10
z_dim=64
batch_size=32
input_size=z_dim+n_classes
fixed_noise=torch.randn((batch_size,z_dim))
transform=transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.5,),(0.5))
])

dataset=datasets.MNIST(root='dataset/',transform=transform,download=True)
loader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

n_epochs=200
device='cuda:4' if torch.cuda.is_available() else 'cpu'
print(f"using device {device}")

gen=Generator(input_size).to(device)
disc=Discriminator().to(device)

loss=nn.BCELoss()

gen_opt=torch.optim.Adam(gen.parameters(),lr=3e-4)
disc_opt=torch.optim.Adam(disc.parameters(),lr=3e-4)

for epoch in range(n_epochs):
    loop=tqdm(loader,leave=False,total=len(loader),desc=f"Processing Epoch {epoch:02d}")
    for x,y in loop:
        x=x.to(device)
        y=y.to(device)
        noise=torch.randn((batch_size,z_dim)).to(device)
        one_hot_label=F.one_hot(y,num_classes=n_classes).to(device)
        input_noise= torch.cat((noise,one_hot_label),dim=1)

        fake=gen(input_noise)
        

        one_hot_label_img=one_hot_label.view(batch_size,-1,1,1)
        one_hot_label_img=one_hot_label_img.repeat(1,1,28,28)
        input_image=torch.cat((x,one_hot_label_img),dim=1)
        disc_real=disc(input_image.view(batch_size,-1)).view(-1)
        lossD_real=loss(disc_real,torch.ones_like(disc_real))

        fake_image=torch.cat((fake,one_hot_label_img),dim=1)
        disc_fake=disc(fake_image.view(batch_size,-1)).view(-1)
        lossD_fake=loss(disc_fake,torch.zeros_like(disc_fake))

        lossD=(lossD_fake+lossD_real)/2
        disc_opt.zero_grad()
        lossD.backward(retain_graph=True)
        disc_opt.step()


        disc_fake=disc(fake_image.view(batch_size,-1)).view(-1)
        loosG=loss(disc_fake,torch.ones_like(disc_fake))
        gen_opt.zero_grad()
        loosG.backward()
        gen_opt.step()
        
    if (epoch+1%100) == 0:
        checkpoint={'gen_state':gen.state_dict()}
        torch.save(checkpoint,f"checks_{epoch}.pth")








