import torch
import torchvision
from models import Generator
z_dim=64
image_dim=28*28
gen=Generator(z_dim=z_dim,img_dim=image_dim)
device='cuda' if torch.cuda.is_available() else 'cpu'
#TODO: load the model_state
batch_size=1
fixed_noise=torch.randn((batch_size,z_dim)).to(device)
generated_image=gen(fixed_noise).reshape(-1, 1, 28, 28)
for i in range(generated_image.shape[0]):
    torchvision.utils.save_image(generated_image[0], f'image_{i}.jpg')
