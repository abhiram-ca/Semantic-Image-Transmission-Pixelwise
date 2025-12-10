import os
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw
import io
import base64
from semantic_comm_network import SemanticCommNetwork, compute_metrics
from torchvision import transforms

app = Flask(__name__)
CORS(app)

# Initialize network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    net = SemanticCommNetwork(device=device)
    print(f"✓ Model loaded on {device}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    net = None

os.makedirs("uploads", exist_ok=True)

def image_to_base64(pil_image):
    """Convert PIL image to base64"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def tensor_to_pil(tensor):
    """Convert tensor to PIL image"""
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    img_array = (tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(img_array)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(device), "model_loaded": net is not None})

@app.route("/api/process", methods=["POST"])
def process_image():
    """Process image with semantic transmission"""
    try:
        if not net:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.json
        image_b64 = data.get("image")
        snr = float(data.get("snr", 10.0))
        
        if not image_b64:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode image
        image_data = base64.b64decode(image_b64.split(",")[1])
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Preprocess
        tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image_tensor = tf(pil_image).unsqueeze(0).to(device)
        
        # Process
        with torch.no_grad():
            output = net.transmit_receive(image_tensor, snr_db=snr, add_noise=True, seed=42)
        
        reconstructed = output["reconstructed"]
        semantic = output["semantic"]
        semantic_noisy = output["semantic_noisy"]
        
        # Metrics
        metrics = compute_metrics(image_tensor, reconstructed, 
                                 semantic_original=semantic, 
                                 semantic_noisy=semantic_noisy)
        
        # Convert to PIL
        real_pil = tensor_to_pil(image_tensor.squeeze(0))
        recon_pil = tensor_to_pil(reconstructed.squeeze(0))
        
        # Semantic map
        sem_argmax = semantic.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        sem_pil = Image.fromarray(sem_argmax, mode='P')
        VOC_PALETTE = [0,0,0, 128,0,0, 0,128,0, 128,128,0, 0,0,128, 128,0,128, 0,128,128, 128,128,128, 64,0,0, 192,0,0, 64,128,0, 192,128,0, 64,0,128, 192,0,128, 64,128,128, 192,128,128, 0,64,0, 128,64,0, 0,192,0] + [0]*(256*3 - 21*3)
        sem_pil.putpalette(VOC_PALETTE)
        sem_pil = sem_pil.convert("RGB")
        
        # Create comparison image (original | semantic | reconstructed)
        width, height = 256, 256
        gap = 8
        total_w = width * 3 + gap * 2
        text_h = 50
        
        comp_img = Image.new("RGB", (total_w, height + text_h), (255, 255, 255))
        comp_img.paste(real_pil, (0, 0))
        comp_img.paste(sem_pil, (width + gap, 0))
        comp_img.paste(recon_pil, ((width + gap) * 2, 0))
        
        # Draw metrics
        draw = ImageDraw.Draw(comp_img)
        psnr_val = metrics["psnr"]
        ssim_val = metrics["ssim"] if metrics["ssim"] is not None else "N/A"
        cos_sim = metrics.get("cosine_similarity", 0.0)
        
        text = f"SNR: {snr:.1f} dB | PSNR: {psnr_val:.4f} dB | SSIM: {str(ssim_val)[:6]} | CosSim: {cos_sim:.4f}"
        draw.text((10, height + 10), text, fill=(0, 0, 0))
        
        return jsonify({
            "success": True,
            "original": image_to_base64(real_pil),
            "semantic": image_to_base64(sem_pil),
            "reconstructed": image_to_base64(recon_pil),
            "comparison": image_to_base64(comp_img),
            "metrics": {
                "psnr": float(psnr_val),
                "ssim": float(ssim_val) if isinstance(ssim_val, (int, float)) else None,
                "cosine_similarity": float(cos_sim),
                "snr": float(snr)
            }
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)