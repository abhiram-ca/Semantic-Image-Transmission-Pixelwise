import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.utils import save_image
from spade_models import SPADEGenerator256, Encoder

# params must match train.py
LATENT_DIM = 16
Z_DIM = 256
IMG_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AWGNChannel(nn.Module):
    """Additive White Gaussian Noise Channel"""
    def __init__(self, snr_db=10.0):
        super().__init__()
        self.snr_db = snr_db

    def set_snr(self, snr_db):
        self.snr_db = snr_db

    def forward(self, signal):
        """
        signal: [B, C, H, W] or [B, C] (flattened semantic map)
        Adds AWGN based on SNR in dB
        """
        # Compute signal power
        signal_power = torch.mean(signal ** 2)
        
        # Convert SNR from dB to linear
        snr_linear = 10 ** (self.snr_db / 10.0)
        
        # Compute noise power
        noise_power = signal_power / snr_linear
        
        # Generate Gaussian noise
        noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        
        # Add noise to signal
        noisy_signal = signal + noise
        
        return noisy_signal, noise_power.item()

class SemanticTransmitter(nn.Module):
    """Transmitter: encodes image to semantic latent"""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image):
        """
        image: [B, 3, H, W] normalized to [-1, 1]
        Returns: semantic_map [B, LATENT_DIM, H, W]
        """
        semantic = self.encoder(image)
        return semantic

class SemanticReceiver(nn.Module):
    """Receiver: decodes semantic latent + noise to image"""
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, semantic, z=None):
        """
        semantic: [B, LATENT_DIM, H, W] (possibly noisy)
        z: [B, Z_DIM] random latent (if None, sampled randomly)
        Returns: reconstructed image [B, 3, H, W]
        """
        if z is None:
            z = torch.randn(semantic.size(0), Z_DIM, device=semantic.device)
        
        image = self.generator(semantic, z)
        return image

class SemanticCommNetwork:
    """End-to-end semantic communication system"""
    def __init__(self, encoder_path="encoder.pth", generator_path="generator.pth", device="cuda"):
        self.device = device
        
        # Load models
        self.encoder = Encoder(output_nc=LATENT_DIM).to(device)
        self.generator = SPADEGenerator256(semantic_nc=LATENT_DIM, z_dim=Z_DIM).to(device)
        
        if not os.path.exists(encoder_path) or not os.path.exists(generator_path):
            raise FileNotFoundError(f"Models not found: {encoder_path}, {generator_path}")
        
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        self.generator.load_state_dict(torch.load(generator_path, map_location=device))
        
        self.encoder.eval()
        self.generator.eval()
        
        # Build components
        self.transmitter = SemanticTransmitter(self.encoder)
        self.channel = AWGNChannel(snr_db=10.0)
        self.receiver = SemanticReceiver(self.generator)

    def set_snr(self, snr_db):
        """Set channel SNR (dB)"""
        self.channel.set_snr(snr_db)

    def transmit_receive(self, image, snr_db=10.0, add_noise=True, seed=None):
        """
        Full pipeline: image -> encode -> channel -> decode -> reconstructed image
        
        Args:
            image: [B, 3, H, W] tensor in [-1, 1]
            snr_db: SNR in dB
            add_noise: whether to add channel noise
            seed: random seed for reproducible z
        
        Returns:
            dict with:
                - semantic: encoded semantic map (before channel)
                - semantic_noisy: semantic after channel noise
                - reconstructed: decoded image
                - noise_power: noise power added by channel
        """
        self.set_snr(snr_db)
        
        if seed is not None:
            torch.manual_seed(seed)
        
        with torch.no_grad():
            # Transmitter
            semantic = self.transmitter(image)  # [B, C, H, W]
            
            # Channel
            if add_noise:
                semantic_noisy, noise_power = self.channel(semantic)
            else:
                semantic_noisy = semantic.clone()
                noise_power = 0.0
            
            # Receiver
            z = torch.randn(image.size(0), Z_DIM, device=self.device)
            reconstructed = self.receiver(semantic_noisy, z)  # [B, 3, H, W]
        
        return {
            "semantic": semantic,
            "semantic_noisy": semantic_noisy,
            "reconstructed": reconstructed,
            "noise_power": noise_power,
            "snr_db": snr_db
        }

def compute_metrics(img_real, img_fake, semantic_original=None, semantic_noisy=None):
    """Compute PSNR, SSIM, and Cosine Similarity of semantic maps"""
    real_np = (img_real.squeeze(0).cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    fake_np = (img_fake.squeeze(0).cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    
    mse = np.mean((real_np.astype(np.float32) - fake_np.astype(np.float32))**2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    ssim = None
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        ssim = ssim_fn(real_np, fake_np, channel_axis=2, data_range=255)
    except Exception as e:
        print(f"SSIM Error: {e}")
        ssim = None
    
    # Cosine similarity of semantic maps
    cos_sim = None
    if semantic_original is not None and semantic_noisy is not None:
        try:
            from torch.nn.functional import cosine_similarity
            # Flatten semantic maps
            sem_orig_flat = semantic_original.squeeze(0).reshape(semantic_original.size(1), -1)  # [C, H*W]
            sem_noisy_flat = semantic_noisy.squeeze(0).reshape(semantic_noisy.size(1), -1)  # [C, H*W]
            # Compute cosine similarity per channel
            cos_sims = cosine_similarity(sem_orig_flat, sem_noisy_flat, dim=1)  # [C]
            cos_sim = cos_sims.mean().item()  # average across channels
        except Exception as e:
            print(f"Cosine Similarity Error: {e}")
            cos_sim = None
    
    return {"psnr": psnr, "ssim": ssim, "cosine_similarity": cos_sim}

def preprocess_image(path):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0)  # [1, 3, H, W]

def test_semantic_network(img_path, snr_values=[20, 15, 10, 5, 0, -5], output_dir="semantic_comm_results"):
    """Test semantic network over different SNR levels"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize network
    net = SemanticCommNetwork(device=device)
    
    # Load image
    image = preprocess_image(img_path).to(device)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    
    results = []
    
    for snr in snr_values:
        print(f"\n=== Testing at SNR = {snr} dB ===")
        
        output = net.transmit_receive(image, snr_db=snr, add_noise=True, seed=42)
        
        reconstructed = output["reconstructed"]
        semantic = output["semantic"]
        semantic_noisy = output["semantic_noisy"]  # <- extract from output dict
        
        # Compute metrics
        metrics = compute_metrics(image, reconstructed, semantic_original=semantic, semantic_noisy=semantic_noisy)
        output.update(metrics)
        results.append(output)
        
        print(f"PSNR: {metrics['psnr']:.4f} dB")
        if metrics['ssim'] is not None:
            print(f"SSIM: {metrics['ssim']:.6f}")
        if metrics['cosine_similarity'] is not None:
            print(f"Cosine Similarity: {metrics['cosine_similarity']:.6f}")
        
        # Prepare images
        real_vis = (image * 0.5 + 0.5).clamp(0, 1)
        recon_vis = (reconstructed * 0.5 + 0.5).clamp(0, 1)
        
        # Save semantic map visualization
        sem_argmax = semantic.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        sem_img = Image.fromarray(sem_argmax, mode='P')
        VOC_PALETTE = [0,0,0, 128,0,0, 0,128,0, 128,128,0, 0,0,128, 128,0,128, 0,128,128, 128,128,128, 64,0,0, 192,0,0, 64,128,0, 192,128,0, 64,0,128, 192,0,128, 64,128,128, 192,128,128, 0,64,0, 128,64,0, 0,192,0] + [0]*(256*3 - 21*3)
        sem_img.putpalette(VOC_PALETTE)
        sem_tensor = torch.tensor(np.array(sem_img.convert("RGB")).transpose(2,0,1), dtype=torch.float32).unsqueeze(0)/255.0
        sem_tensor = sem_tensor.to(device)
        sem_tensor = torch.nn.functional.interpolate(sem_tensor, size=(IMG_SIZE, IMG_SIZE), mode='nearest')
        
        # Create comparison: original | transmitted (semantic) | reconstructed
        # Stack horizontally: [1, 3, H, W*3]
        real_img = real_vis.squeeze(0)  # [3, H, W]
        sem_img_t = sem_tensor.squeeze(0)  # [3, H, W]
        recon_img = recon_vis.squeeze(0)  # [3, H, W]
        
        # Concatenate along width dimension
        cmp_horizontal = torch.cat([real_img, sem_img_t, recon_img], dim=2)  # [3, H, W*3]
        
        cmp_pil = transforms.ToPILImage()(cmp_horizontal.cpu())
        
        # Add metrics text below
        draw = ImageDraw.Draw(cmp_pil)
        psnr_val = metrics['psnr']
        ssim_val = metrics['ssim'] if metrics['ssim'] is not None else "N/A"
        cos_sim_val = metrics['cosine_similarity'] if metrics['cosine_similarity'] is not None else "N/A"
        
        # Format cosine similarity
        if isinstance(cos_sim_val, float):
            cos_sim_str = f"{cos_sim_val:.4f}"
        else:
            cos_sim_str = str(cos_sim_val)
        
        text = f"SNR: {snr} dB | PSNR: {psnr_val:.4f} dB | SSIM: {str(ssim_val)} | CosSim: {cos_sim_str}"
        
        # Draw text at bottom
        text_y = cmp_pil.height - 30
        draw.rectangle([(0, text_y), (cmp_pil.width, cmp_pil.height)], fill=(255,255,255))
        draw.text((10, text_y + 5), text, fill=(0,0,0))
        
        cmp_pil.save(os.path.join(output_dir, f"{img_name}_cmp_snr_{snr}_dB.png"))
    
    # Save SNR-wise metrics summary
    summary_path = os.path.join(output_dir, f"{img_name}_metrics.txt")
    with open(summary_path, "w") as f:
        f.write(f"Image: {img_path}\n")
        f.write(f"{'SNR (dB)':<12} {'PSNR (dB)':<15} {'SSIM':<12} {'Cosine Sim':<12}\n")
        f.write("-" * 52 + "\n")
        for res in results:
            snr_val = res["snr_db"]
            psnr_val = res["psnr"]
            ssim_val = res["ssim"] if res["ssim"] is not None else "N/A"
            cos_sim_val = res["cosine_similarity"] if res["cosine_similarity"] is not None else "N/A"
            
            # Format with 4 decimals
            if isinstance(ssim_val, float):
                ssim_str = f"{ssim_val:.4f}"
            else:
                ssim_str = str(ssim_val)
            
            if isinstance(cos_sim_val, float):
                cos_sim_str = f"{cos_sim_val:.4f}"
            else:
                cos_sim_str = str(cos_sim_val)
            
            f.write(f"{snr_val:<12.1f} {psnr_val:<15.4f} {ssim_str:<12} {cos_sim_str:<12}\n")
    
    print(f"\nResults saved to {output_dir}/")
    return results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Image path")
    p.add_argument("--snr", type=float, nargs="+", default=[20, 15, 10, 5, 0, -5], help="SNR values (dB)")
    p.add_argument("--outdir", default="semantic_transmission_results", help="Output directory")
    args = p.parse_args()
    
    test_semantic_network(args.image, snr_values=args.snr, output_dir=args.outdir)