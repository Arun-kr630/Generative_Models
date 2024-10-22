import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models import Discriminator,Generator


        
device='cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device {device}")
lr=3e-4
z_dim=64
image_dim=28*28
batch_size=32
num_epochs=50

disc=Discriminator(image_dim).to(device)
gen=Generator(z_dim,image_dim).to(device)
fixed_noise=torch.randn((batch_size,z_dim)).to(device)
transform=transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.5,),(0.5))
])

dataset=datasets.MNIST(root='dataset/',transform=transform,download=True)
loader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

opt_disc=optim.Adam(disc.parameters(),lr=lr)
opt_gen=optim.Adam(gen.parameters(),lr=lr)

criterian=nn.BCELoss()

writer_fake=SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real=SummaryWriter(f"runs/GAN_MNIST/real")
step=0

for epoch in range(num_epochs):
    loop=tqdm(enumerate(loader),total=len(loader),leave=False)
    for idx,(real,_) in loop:
        real=real.view(-1,28*28).to(device)
        batch_size=real.shape[0]

        noise=torch.randn(batch_size,z_dim).to(device)
        fake=gen(noise)
        disc_real=disc(real).view(-1)
        lossD_real=criterian(disc_real,torch.ones_like(disc_real))
        disc_fake=disc(fake).view(-1)
        lossD_fake=criterian(disc_fake,torch.zeros_like(disc_fake))
        lossD=(lossD_fake+lossD_real)/2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()


        output=disc(fake).view(-1)
        lossG=criterian(output,torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
        
        loop.set_description(f"Epoch=[{epoch}/{num_epochs}]")
