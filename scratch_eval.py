import torch
from PIL import Image
import torchvision.transforms as transforms
from semantic_comm_network import SemanticCommNetwork, compute_metrics
import pandas as pd

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SemanticCommNetwork(device=device)
    
    # Load image
    img_path = r"C:\Users\abhir\Desktop\Semantic-Image-Transmission-Pixelwise-main\Semantic-Image-Transmission-Pixelwise-main\test_inputs\image2.jpg"
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    results = []

    snrs = [10, 20, 50]
    
    # AWGN
    for snr in snrs:
        out = net.transmit_receive(img_tensor, snr_db=snr, channel_type="awgn")
        metrics = compute_metrics(img_tensor, out['reconstructed'], out['semantic'], out['semantic_noisy'])
        results.append({"Channel": "AWGN", "SNR (dB)": snr, "Params": "-", "PSNR": metrics['psnr'], "SSIM": metrics['ssim'], "CosSim": metrics['cosine_similarity']})
        
    # Rayleigh
    for snr in snrs:
        out = net.transmit_receive(img_tensor, snr_db=snr, channel_type="rayleigh")
        metrics = compute_metrics(img_tensor, out['reconstructed'], out['semantic'], out['semantic_noisy'])
        results.append({"Channel": "Rayleigh", "SNR (dB)": snr, "Params": "-", "PSNR": metrics['psnr'], "SSIM": metrics['ssim'], "CosSim": metrics['cosine_similarity']})
        
    # Rician
    k_factors = [1, 5, 10]
    for snr in snrs:
        for k in k_factors:
            out = net.transmit_receive(img_tensor, snr_db=snr, channel_type="rician", k_factor=k)
            metrics = compute_metrics(img_tensor, out['reconstructed'], out['semantic'], out['semantic_noisy'])
            results.append({"Channel": "Rician", "SNR (dB)": snr, "Params": f"K={k}", "PSNR": metrics['psnr'], "SSIM": metrics['ssim'], "CosSim": metrics['cosine_similarity']})
            
    # Nakagami
    m_factors = [1.0, 2.5, 5.0]
    for snr in snrs:
        for m in m_factors:
            out = net.transmit_receive(img_tensor, snr_db=snr, channel_type="nakagami", m_factor=m)
            metrics = compute_metrics(img_tensor, out['reconstructed'], out['semantic'], out['semantic_noisy'])
            results.append({"Channel": "Nakagami", "SNR (dB)": snr, "Params": f"m={m}", "PSNR": metrics['psnr'], "SSIM": metrics['ssim'], "CosSim": metrics['cosine_similarity']})

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
