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

from models import VAE, VAE_EP, VAE_EP_fullcov
from utils import make_gif, plot_elbocurve

from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def compute_elbo(x, reconst_x, mean, log_var):
        reconst_error = -torch.nn.functional.binary_cross_entropy(reconst_x, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        elbo = (reconst_error - kl_divergence) / len(x)
        return elbo

def main_vanilla(model, input_size, hidden_size, latent_size, train_loader, test_loader, results_dir, paras_dir):
    
    # Select the optimizer
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create folders for results and model checkpoints
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(paras_dir, exist_ok=True)
    
    for directory in [results_dir, paras_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Save initial samples
    counter = 0
    noise = torch.randn(25, latent_size).to(device)
    generated_imgs = model.decode(noise).view(-1, 1, 28, 28)
    save_image(generated_imgs, os.path.join(results_dir, f'samples_zdim{latent_size}-0.png'), nrow=5)

    # Training setup
    num_epochs = 200
    train_elbo = []
    test_elbo = []
    best_train_elbo = float('-inf')  # Initialize best ELBO as negative infinity

    for epoch in tqdm(range(1, num_epochs + 1), position=0):
        # Training loop
        for batch_idx, (batch_x, _) in enumerate(train_loader):
            batch_data = batch_x.to(device).view(-1, input_size)
            batch_mean, batch_logvar, reconst_batch = model(batch_data)
            aver_loss = -compute_elbo(batch_data, reconst_batch, batch_mean, batch_logvar)
            
            optimizer.zero_grad()
            aver_loss.backward()
            optimizer.step()
            
            if (batch_idx + 1) % 100 == 0:
                print('Epoch {}/{}, Batch {}/{}, Aver_Loss: {:.2f}'.format(
                    epoch, num_epochs, batch_idx + 1, math.ceil(len(train_dataset) / batch_size), aver_loss.item()))
            
                    
        # Evaluation phase
        with torch.no_grad():
            # Calculate training ELBO
            total_elbo = torch.tensor(0.0).to(device)
            for batch_idx, (batch_x, _) in enumerate(train_loader):
                batch_data = batch_x.to(device).view(-1, input_size)
                batch_mean, batch_logvar, reconst_batch = model(batch_data)
                total_elbo += compute_elbo(batch_data, reconst_batch, batch_mean, batch_logvar)
            current_train_elbo = (total_elbo / (batch_idx + 1)).cpu().item()
            train_elbo.append(current_train_elbo)
            
            # Save model if it achieves better training ELBO
            if current_train_elbo > best_train_elbo:
                best_train_elbo = current_train_elbo
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_elbo': best_train_elbo
                }, os.path.join(paras_dir, f'mnist_zdim{latent_size}.pkl'))
                # torch.save(model.state_dict(), os.path.join(paras_dir, f'mnist_zdim{latent_size}.pkl'))
                print(f'New best model saved! ELBO: {best_train_elbo:.2f}')
            
            # Calculate test ELBO
            total_elbo = torch.tensor(0.0).to(device)
            for batch_idx, (batch_x, _) in enumerate(test_loader):
                batch_data = batch_x.to(device).view(-1, input_size)
                batch_mean, batch_logvar, reconst_batch = model(batch_data)
                total_elbo += compute_elbo(batch_data, reconst_batch, batch_mean, batch_logvar)
            test_elbo.append((total_elbo / (batch_idx + 1)).cpu().item())
            
            # Save periodic samples
            if epoch in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
                counter += 1
                generated_imgs = model.decode(noise).view(-1, 1, 28, 28)
                save_image(generated_imgs, os.path.join(results_dir, f'samples_zdim{latent_size}-{counter}.png'), nrow=5)

    # Create visualization
    make_gif(results_dir, counter + 1, latent_size)
    plot_elbocurve(train_elbo, test_elbo, latent_size, results_dir)

    return train_elbo, test_elbo

def main(model, input_size, hidden_size, latent_size, train_loader, test_loader, results_dir, paras_dir):
    
    # Select the optimizer
    learning_rate = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create folders for results and model checkpoints
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(paras_dir, exist_ok=True)
    
    for directory in [results_dir, paras_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Save initial samples
    counter = 0
    noise = torch.randn(25, latent_size).to(device)
    generated_imgs = model.decode_EP(noise, 1+noise*0.1).view(-1, 1, 28, 28)
    save_image(generated_imgs, os.path.join(results_dir, f'samples_zdim{latent_size}-0.png'), nrow=5)

    # Training setup
    num_epochs = 200
    train_elbo = []
    test_elbo = []
    best_train_elbo = float('-inf')  # Initialize best ELBO as negative infinity

    for epoch in tqdm(range(1, num_epochs + 1), position=0):
        # Training loop
        for batch_idx, (batch_x, _) in enumerate(train_loader):
            batch_data = batch_x.to(device).view(-1, input_size)
            batch_mean, batch_logvar, reconst_batch = model(batch_data)
            aver_loss = -compute_elbo(batch_data, reconst_batch, batch_mean, batch_logvar)
            
            optimizer.zero_grad()
            aver_loss.backward()
            optimizer.step()
            
            if (batch_idx + 1) % 100 == 0:
                print('Epoch {}/{}, Batch {}/{}, Aver_Loss: {:.2f}'.format(
                    epoch, num_epochs, batch_idx + 1, math.ceil(len(train_dataset) / batch_size), aver_loss.item()))
                    
        # Evaluation phase
        with torch.no_grad():
            # Calculate training ELBO
            total_elbo = torch.tensor(0.0).to(device)
            for batch_idx, (batch_x, _) in enumerate(train_loader):
                batch_data = batch_x.to(device).view(-1, input_size)
                batch_mean, batch_logvar, reconst_batch = model(batch_data)
                total_elbo += compute_elbo(batch_data, reconst_batch, batch_mean, batch_logvar)
            current_train_elbo = (total_elbo / (batch_idx + 1)).cpu().item()
            train_elbo.append(current_train_elbo)
            
            # Save model if it achieves better training ELBO
            if current_train_elbo > best_train_elbo:
                best_train_elbo = current_train_elbo
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_elbo': best_train_elbo
                }, os.path.join(paras_dir, f'mnist_zdim{latent_size}.pkl'))
                # torch.save(model.state_dict(), os.path.join(paras_dir, f'mnist_zdim{latent_size}.pkl'))
                print(f'New best model saved! ELBO: {best_train_elbo:.2f}')
            
            # Calculate test ELBO
            total_elbo = torch.tensor(0.0).to(device)
            for batch_idx, (batch_x, _) in enumerate(test_loader):
                batch_data = batch_x.to(device).view(-1, input_size)
                batch_mean, batch_logvar, reconst_batch = model(batch_data)
                total_elbo += compute_elbo(batch_data, reconst_batch, batch_mean, batch_logvar)
            test_elbo.append((total_elbo / (batch_idx + 1)).cpu().item())
            
            # Save periodic samples
            if epoch in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
                counter += 1
                generated_imgs = model.decode_EP(noise, 1+noise*0.1).view(-1, 1, 28, 28)
                save_image(generated_imgs, os.path.join(results_dir, f'samples_zdim{latent_size}-{counter}.png'), nrow=5)

    # Create visualization
    make_gif(results_dir, counter + 1, latent_size)
    plot_elbocurve(train_elbo, test_elbo, latent_size, results_dir)

    return train_elbo, test_elbo

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Build the data input pipeline
    batch_size = 100 #100
    train_dataset = torchvision.datasets.MNIST(root='./data/MNIST', train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = torchvision.datasets.MNIST(root='./data/MNIST', train=False, transform=transforms.ToTensor(), download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=20, pin_memory=True)

    # Build the model
    input_size = 784
    hidden_size = 500
    for latent_size in [2]:
    
        results_dir = 'results/MNIST/vanilla_v2'
        paras_dir   = 'trained_parameters/vanilla_v2'
        model = VAE(input_size, hidden_size, latent_size).to(device)
        train_elbo_VAE, test_elbo_VAE = main_vanilla(model, input_size, hidden_size, latent_size, train_loader, test_loader, results_dir, paras_dir)

        results_dir = 'results/MNIST/EP_v2'
        paras_dir   = 'trained_parameters/EP_v2'
        model = VAE_EP(input_size, hidden_size, latent_size).to(device)
        train_elbo_VAE_EP, test_elbo_VAE_EP = main(model, input_size, hidden_size, latent_size, train_loader, test_loader, results_dir, paras_dir)

        

        train_elbo_VAE = np.array(train_elbo_VAE)
        test_elbo_VAE  = np.array(test_elbo_VAE)

        train_elbo_VAE_EP = np.array(train_elbo_VAE_EP)
        test_elbo_VAE_EP  = np.array(test_elbo_VAE_EP)



        # Save data to a CSV file for future plotting
        # Flatten data into a long format with labels
        data_list = []

        # Function to store each array's values with corresponding epoch and label
        def add_data_to_list(data, label):
            for epoch, value in enumerate(data):
                data_list.append([epoch, label, value])

        # # Add data to the list
        add_data_to_list(train_elbo_VAE, "train (vanilla)")
        add_data_to_list(test_elbo_VAE, "test (vanilla)")
        add_data_to_list(train_elbo_VAE_EP, "train (EP)")
        add_data_to_list(test_elbo_VAE_EP, "test (EP)")


        # Convert to DataFrame
        df = pd.DataFrame(data_list, columns=["Epoch", "Label", "Value"])

        # Save CSV
        csv_filename = os.path.join('./', 'elbo_data_{}D_vanilla.csv'.format(latent_size))
        df.to_csv(csv_filename, index=False)
        print(f"ELBO data saved to {csv_filename}")


        # Clear previous plots
        plt.clf()

        # Plot with updated colors
        colors = {"train (vanilla)": "#4A5E65", "test (vanilla)": "#B95A58", 
                  "train (PTPE)": "#4A5E65", "test (PTPE)": "#B95A58"}

        linestyles = {"train (vanilla)": "--", "test (vanilla)": "--", 
                      "train (PTPE)": "-", "test (PTPE)": "-"}

        # Replot from saved CSV for verification
        for label, group in df.groupby("Label"):
            plt.plot(group["Epoch"], group["Value"], color=colors[label], linestyle=linestyles[label], label=label)

        # Labels and title
        plt.legend(loc='best')
        plt.xlabel('Epoch')
        plt.ylabel('Lower Bound')
        plt.title(f'MNIST, $N_z$={latent_size}')

        # Save and show plot
        plot_filename = os.path.join('./', 'elbocurve-{}D.png'.format(latent_size))
        plt.savefig(plot_filename)
        # plt.show()

        print(f"ELBO-curve plot saved as {plot_filename}")





