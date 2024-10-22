import sys
import torch
from model import VariationalAutoEncoder
from torchvision.utils import save_image
import torchvision.datasets as datasets
from torchvision import transforms
INPUT_DIM=784
H_DIM=200
Z_DIM=0
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
dataset=datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(),download=True)
model=VariationalAutoEncoder(input_size=INPUT_DIM,h_dim=H_DIM,z_dim=Z_DIM).to(DEVICE)
checkpoint=torch.load("VAES_checks/checkpoint_99.pth",weights_only=True,map_location=torch.device('cpu'))#,
model.load_state_dict(checkpoint['model_state'])
model.eval()
def inference(digit, num_examples=1):
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            a=images[d].view(1, 784).to(DEVICE)
            mu, sigma = model.encode(a)
            mu=mu.to(DEVICE)
            sigma=sigma.to(DEVICE)
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma).to(DEVICE)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"generated_{digit}_ex{example}.png")

for idx in range(10):
    inference(idx, num_examples=1)
