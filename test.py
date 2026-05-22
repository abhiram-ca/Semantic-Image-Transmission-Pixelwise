import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from spade_models import SPADEGenerator256, Encoder

# Hyperparameters
LATENT_DIM = 16
Z_DIM = 256
IMG_SIZE = 256 # Cityscapes resolution
BATCH_SIZE = 4
SNR_LEVELS = [0, 5, 10, 20] # dB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_noise(x, snr_db):
    # Calculate signal power
    signal_power = torch.mean(x ** 2)
    
    # Calculate noise power based on SNR
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    
    # Generate noise
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    
    return x + noise

def evaluate():
    os.makedirs("results", exist_ok=True)

    # Load Models
    encoder = Encoder(output_nc=LATENT_DIM).to(device)
    generator = SPADEGenerator256(semantic_nc=LATENT_DIM, z_dim=Z_DIM).to(device)

    if os.path.exists("encoder.pth"):
        encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
        generator.load_state_dict(torch.load("generator.pth", map_location=device))
        print("Loaded models.")
    else:
        print("No trained models found. Please run train.py first.")
        return 

    encoder.eval()
    generator.eval()

    # Data
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        dataset = datasets.VOCSegmentation(root='./data', year='2012', image_set='val', download=True, transform=transform)
    except:
        print("VOC dataset not found. Falling back to CIFAR-10...")
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Get one batch
    data = next(iter(dataloader))
    if isinstance(data, list) and len(data) == 2:
        real_imgs = data[0].to(device)
    else:
        real_imgs = data.to(device)

    with torch.no_grad():
        # Encode
        semantic_map_clean = encoder(real_imgs)
        
        # Visualize Semantic Map (Average channels + Normalize to -1..1)
        sem_vis = torch.mean(semantic_map_clean, dim=1, keepdim=True)
        sem_vis = (sem_vis - sem_vis.min()) / (sem_vis.max() - sem_vis.min() + 1e-5)
        sem_vis = sem_vis * 2 - 1
        sem_vis = sem_vis.repeat(1, 3, 1, 1)
        
        # Upsample to match image size for visualization
        sem_vis = torch.nn.functional.interpolate(sem_vis, size=(IMG_SIZE, IMG_SIZE), mode='nearest')
        
        results = [real_imgs, sem_vis]
        
        for snr in SNR_LEVELS:
            print(f"Testing SNR: {snr} dB")
            
            # Add Channel Noise
            semantic_map_noisy = add_noise(semantic_map_clean, snr)
            
            # Decode
            z = torch.randn(real_imgs.size(0), Z_DIM, dtype=torch.float32, device=device)
            rec_imgs = generator(semantic_map_noisy, z)
            
            results.append(rec_imgs)
            
        # Concatenate results
        final_grid = torch.cat(results, dim=0)
        save_image(final_grid, "results/evaluation_grid.png", nrow=BATCH_SIZE, normalize=True)
        print("Saved evaluation grid to results/evaluation_grid.png")

if __name__ == "__main__":
    evaluate()
