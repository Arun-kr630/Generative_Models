import torch
from tqdm import tqdm
import torchvision.datasets as datasets
from torch import nn
from model import VariationalAutoEncoder
from torchvision import transforms
from torch.utils.data import DataLoader

DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM=784
H_DIM=200
Z_DIM=0
NUM_EPOCHS=100
BATCH_SIZE=32
LR=3e-4

dataset=datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(),download=True)

train_loader=DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)
model=VariationalAutoEncoder(INPUT_DIM,H_DIM,Z_DIM).to(DEVICE)
optimizer=torch.optim.Adam(model.parameters(),lr=LR)

loss_fn=nn.BCELoss(reduction="sum")


for epoch in range(NUM_EPOCHS):
    loop=tqdm(enumerate(train_loader),leave=False,total=len(train_loader))
    for i,(x,_) in loop:
        x=x.to(DEVICE).view(x.shape[0],INPUT_DIM)
        x_reconstructed,mu,sigma=model(x)

        reconstruction_loss=loss_fn(x_reconstructed,x)
        kl_divergence=torch.sum(1+torch.log(sigma.pow(2))-mu.pow(2)-sigma.pow(2))

        loss=reconstruction_loss + kl_divergence
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_description(f"Epoch = [{epoch}/{NUM_EPOCHS}]")
    if (epoch+1)%1 == 0:
        checkpoint={
            'epoch':epoch,
            'model_state':model.state_dict(),
        }
        torch.save(checkpoint,f'VAES_checks/checkpoint_{epoch}.pth')


