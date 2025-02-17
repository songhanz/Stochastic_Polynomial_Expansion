"""
This code is based on the implementation by NoviceStone
https://github.com/NoviceStone/VAE/tree/master
"""

import os
import math
import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from models import VAE
from datasets import FreyFaceDataset
from utils import produce_z_values, visualize_latentspace
import numpy as np


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

""" ======================================= PART 1: EXPERIMENTS ON MNIST DATASET ========================================== """

# Build the data input pipeline
batch_size = 100
test_dataset = torchvision.datasets.MNIST(root='./data/MNIST', train=False, transform=transforms.ToTensor(), download=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Specify the directory stored the trained model parameters
paras_dir = 'trained_parameters/vanilla_v2'
results_dir = 'results/MNIST/vanilla_v2'


latent_size = 2
model_2D = VAE(input_size=784, hidden_size=500, latent_size=latent_size).to(device)
checkpt  = torch.load(os.path.join(paras_dir, 'mnist_zdim' + str(latent_size) + '.pkl'))
model_2D.load_state_dict(checkpt['model_state_dict'])

latent_size = 5
model_5D = VAE(input_size=784, hidden_size=500, latent_size=latent_size).to(device)
checkpt  = torch.load(os.path.join(paras_dir, 'mnist_zdim' + str(latent_size) + '.pkl'))
model_5D.load_state_dict(checkpt['model_state_dict'])

latent_size = 10
model_10D = VAE(input_size=784, hidden_size=500, latent_size=latent_size).to(device)
checkpt  = torch.load(os.path.join(paras_dir, 'mnist_zdim' + str(latent_size) + '.pkl'))
model_10D.load_state_dict(checkpt['model_state_dict'])

latent_size = 20
model_20D = VAE(input_size=784, hidden_size=500, latent_size=latent_size).to(device)
checkpt  = torch.load(os.path.join(paras_dir, 'mnist_zdim' + str(latent_size) + '.pkl'))
model_20D.load_state_dict(checkpt['model_state_dict'])

with torch.no_grad():    
    
    all_reconst_loss1 = []
    all_reconst_loss2 = []
    all_reconst_loss3 = []
    all_reconst_loss4 = []

    # Reconstruction
    for batch_idx, (batch_x, _) in enumerate(test_loader):
        true_imgs = batch_x.view(-1, 1, 28, 28)
        save_image(true_imgs, os.path.join(results_dir, 'origin_imgs.png'), nrow=10)
        # break
        x = true_imgs.to(device).view(-1, 784)
        reconst_x = model_2D(x)[-1]
        reconst_imgs = reconst_x.view(-1, 1, 28, 28)
        reconst_loss1 = torch.sum((x - reconst_x).pow(2)) / len(x)
        save_image(reconst_imgs, os.path.join(results_dir, 'reconst_imgs-2D_batch' + str(batch_idx) + '.png'), nrow=10)
        reconst_x = model_5D(x)[-1]
        reconst_imgs = reconst_x.view(-1, 1, 28, 28)
        reconst_loss2 = torch.sum((x - reconst_x).pow(2)) / len(x)
        save_image(reconst_imgs, os.path.join(results_dir, 'reconst_imgs-5D_batch' + str(batch_idx) + '.png'), nrow=10)
        reconst_x = model_10D(x)[-1]
        reconst_imgs = reconst_x.view(-1, 1, 28, 28)
        reconst_loss3 = torch.sum((x - reconst_x).pow(2)) / len(x)
        save_image(reconst_imgs, os.path.join(results_dir, 'reconst_imgs-10D_batch' + str(batch_idx) + '.png'), nrow=10)
        reconst_x = model_20D(x)[-1]
        reconst_imgs = reconst_x.view(-1, 1, 28, 28)
        reconst_loss4 = torch.sum((x - reconst_x).pow(2)) / len(x)
        save_image(reconst_imgs, os.path.join(results_dir, 'reconst_imgs-20D_batch' + str(batch_idx) + '.png'), nrow=10)


        all_reconst_loss1.append(reconst_loss1.cpu().item())
        all_reconst_loss2.append(reconst_loss2.cpu().item())
        all_reconst_loss3.append(reconst_loss3.cpu().item())
        all_reconst_loss4.append(reconst_loss4.cpu().item())

    print(f"mean: {np.mean(all_reconst_loss1)}.  standard error: {np.std(all_reconst_loss1) / np.sqrt(batch_idx+1)}")
    print(f"mean: {np.mean(all_reconst_loss2)}.  standard error: {np.std(all_reconst_loss2) / np.sqrt(batch_idx+1)}")
    print(f"mean: {np.mean(all_reconst_loss3)}.  standard error: {np.std(all_reconst_loss3) / np.sqrt(batch_idx+1)}")
    print(f"mean: {np.mean(all_reconst_loss4)}.  standard error: {np.std(all_reconst_loss4) / np.sqrt(batch_idx+1)}")    

exit()


