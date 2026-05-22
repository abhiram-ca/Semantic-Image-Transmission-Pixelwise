import os
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from spade_models import SPADEGenerator256, Encoder

# params must match train.py
LATENT_DIM = 16
Z_DIM = 256
IMG_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VOC palette for visualization (21 classes)
VOC_PALETTE = [
    0,0,0, 128,0,0, 0,128,0, 128,128,0, 0,0,128, 128,0,128,
    0,128,128, 128,128,128, 64,0,0, 192,0,0, 64,128,0, 192,128,0,
    64,0,128, 192,0,128, 64,128,128, 192,128,128, 0,64,0, 128,64,0,
    0,192,0
] + [0]*(256*3 - 21*3)

def load_models(device):
    encoder = Encoder(output_nc=LATENT_DIM).to(device)
    generator = SPADEGenerator256(semantic_nc=LATENT_DIM, z_dim=Z_DIM).to(device)
    if not os.path.exists("encoder.pth") or not os.path.exists("generator.pth"):
        raise SystemExit("encoder.pth or generator.pth not found in working dir.")
    encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
    generator.load_state_dict(torch.load("generator.pth", map_location=device))
    encoder.eval(); generator.eval()
    return encoder, generator

def preprocess_image(path):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0)  # 1,C,H,W

def save_segmentation_argmax(semantic_tensor, out_path):
    # semantic_tensor: [1, C, H, W]
    sem = semantic_tensor.detach().cpu()
    arg = sem.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)  # H,W
    im = Image.fromarray(arg, mode='P')
    im.putpalette(VOC_PALETTE)
    im.save(out_path)

def save_channel_heatmaps(semantic_tensor, out_dir, basename, max_ch=3):
    sem = semantic_tensor.detach().cpu().squeeze(0)  # C,H,W
    C = sem.shape[0]
    for ch in range(min(max_ch, C)):
        arr = sem[ch].numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        im = Image.fromarray((arr*255).astype(np.uint8))
        im = im.convert("L").resize((IMG_SIZE, IMG_SIZE))
        im.save(os.path.join(out_dir, f"{basename}_seg_ch{ch}.png"))

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    encoder, generator = load_models(device)
    for img_path in args.images:
        x = preprocess_image(img_path).to(device)
        with torch.no_grad():
            semantic = encoder(x)                 # [1, C, H, W]
            z = torch.randn(x.size(0), Z_DIM, device=device)
            fake = generator(semantic, z)
            out_name = os.path.splitext(os.path.basename(img_path))[0]

            # save fake and comparison
            fake_vis = fake * 0.5 + 0.5
            save_image(fake_vis.clamp(0,1), os.path.join(args.outdir, f"{out_name}_fake.png"))
            save_image(torch.cat([(x * 0.5 + 0.5).clamp(0,1), fake_vis.clamp(0,1)], dim=0),
                       os.path.join(args.outdir, f"{out_name}_cmp.png"))

            # save segmentation argmax visualization
            save_segmentation_argmax(semantic, os.path.join(args.outdir, f"{out_name}_seg_argmax.png"))

            # save a few channel heatmaps for inspection
            save_channel_heatmaps(semantic, args.outdir, out_name, max_ch=3)

            print("Saved outputs for:", img_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs="+", required=True, help="One or more image paths")
    p.add_argument("--outdir", default="test_outputs", help="Output folder")
    args = p.parse_args()
    main(args)