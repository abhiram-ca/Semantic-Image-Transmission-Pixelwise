import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from spade_models import SPADEGenerator256, Encoder
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# params must match train.py
EPOCHS = 50
LATENT_DIM = 16
Z_DIM = 256
IMG_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def save_semantic_png(semantic_tensor, out_path_argmax, out_path_raw=None):
    """
    Save two PNGs:
     - argmax (palette) visualization -> out_path_argmax
     - raw-channel RGB visualization (first 3 channels) -> out_path_raw (optional)
    semantic_tensor: torch.Tensor [1, C, H, W]
    """
    sem = semantic_tensor.detach().cpu().squeeze(0)  # C,H,W
    C, H, W = sem.shape

    # Argmax palette image (categorical view)
    arg = sem.argmax(dim=0).numpy().astype(np.uint8)  # H,W
    im = Image.fromarray(arg, mode='P')
    im.putpalette(VOC_PALETTE)
    im.save(out_path_argmax)

    if out_path_raw:
        # Raw visualization: map first 3 channels to RGB (normalize per-channel)
        if C >= 3:
            chs = []
            for i in range(3):
                a = sem[i].numpy()
                a = (a - a.min()) / (a.max() - a.min() + 1e-8)
                chs.append((a * 255).astype(np.uint8))
            rgb = np.stack(chs, axis=-1)  # H,W,3
            Image.fromarray(rgb).save(out_path_raw)
        else:
            # fewer than 3 channels: zero-pad channels
            rgb = np.zeros((H, W, 3), dtype=np.uint8)
            for i in range(C):
                a = sem[i].numpy()
                a = (a - a.min()) / (a.max() - a.min() + 1e-8)
                rgb[..., i] = (a * 255).astype(np.uint8)
            Image.fromarray(rgb).save(out_path_raw)

def save_segmentation_argmax(semantic_tensor, out_path):
    """
    semantic_tensor: torch.Tensor [1, C, H, W] (encoder output)
    Saves palette PNG showing argmax across channels.
    """
    sem = semantic_tensor.detach().cpu()
    arg = sem.argmax(dim=1).squeeze(0).numpy().astype('uint8')  # H,W
    im = Image.fromarray(arg, mode='P')
    im.putpalette(VOC_PALETTE)
    im.save(out_path)

def save_channel_heatmaps(semantic_tensor, out_dir, basename, max_ch=3):
    """
    Save up to max_ch channel heatmaps from semantic_tensor (torch.Tensor [1,C,H,W]).
    Files: <basename>_seg_ch0.png, ...
    """
    sem = semantic_tensor.detach().cpu().squeeze(0)  # C,H,W
    C, H, W = sem.shape
    for ch in range(min(max_ch, C)):
        arr = sem[ch].numpy()
        # normalize to [0,1]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        im = Image.fromarray((arr * 255).astype('uint8'), mode='L')
        im = im.resize((IMG_SIZE, IMG_SIZE))
        im.save(os.path.join(out_dir, f"{basename}_seg_ch{ch}.png"))

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    encoder, generator = load_models(device)
    for img_path in args.images:
        out_name = os.path.splitext(os.path.basename(img_path))[0]
        x = preprocess_image(img_path).to(device)
        with torch.no_grad():
            semantic = encoder(x)                 # [1, C, H, W]

            # save raw latent semantic map (npy/pt)
            np.save(os.path.join(args.outdir, f"{out_name}_semantic.npy"), semantic.detach().cpu().numpy())
            torch.save(semantic.detach().cpu(), os.path.join(args.outdir, f"{out_name}_semantic.pt"))

            # save semantic PNGs: argmax palette + raw-channel RGB
            save_semantic_png(
                semantic,
                os.path.join(args.outdir, f"{out_name}_semantic_argmax.png"),
                os.path.join(args.outdir, f"{out_name}_semantic_raw.png")
            )

            # prepare z and generate
            z = torch.randn(x.size(0), Z_DIM, device=device)
            np.save(os.path.join(args.outdir, f"{out_name}_z.npy"), z.detach().cpu().numpy())
            fake = generator(semantic, z)

            # existing fake + cmp saves
            fake_vis = fake * 0.5 + 0.5
            real_vis = (x * 0.5 + 0.5).clamp(0,1)
            save_image(fake_vis.clamp(0,1), os.path.join(args.outdir, f"{out_name}_fake.png"))
            save_image(torch.cat([real_vis, fake_vis.clamp(0,1)], dim=0),
                       os.path.join(args.outdir, f"{out_name}_cmp.png"))

            # save segmentation argmax visualization
            seg_arg_path = os.path.join(args.outdir, f"{out_name}_seg_argmax.png")
            save_segmentation_argmax(semantic, seg_arg_path)

            # save channel heatmaps
            save_channel_heatmaps(semantic, args.outdir, out_name, max_ch=3)

            # --- NEW: compute PSNR / SSIM and save final comparison image + metrics.txt ---
            # convert to uint8 PIL images
            real_np = (real_vis.squeeze(0).cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
            fake_np = (fake_vis.squeeze(0).cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

            # compute PSNR
            mse = np.mean((real_np.astype(np.float32) - fake_np.astype(np.float32))**2)
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))

            # try SSIM if skimage available
            try:
                from skimage.metrics import structural_similarity as ssim_fn
                ssim_val = ssim_fn(real_np, fake_np, multichannel=True, data_range=255)
            except Exception:
                ssim_val = None

            # write metrics to text file
            metrics_path = os.path.join(args.outdir, f"{out_name}_metrics.txt")
            with open(metrics_path, "w") as f:
                f.write(f"PSNR: {psnr:.4f}\n")
                if ssim_val is not None:
                    f.write(f"SSIM: {ssim_val:.6f}\n")
                else:
                    f.write("SSIM: not available (skimage missing)\n")

            # build final comparison image (real | seg-argmax | fake) and draw metrics below
            real_img = Image.fromarray(real_np)
            seg_img = Image.open(seg_arg_path).convert("RGB").resize(real_img.size, Image.NEAREST)
            fake_img = Image.fromarray(fake_np)

            # horizontal concat
            width, height = real_img.width, real_img.height
            gap = 8
            total_w = width * 3 + gap*2
            # leave space at bottom for text
            text_h = 40
            final_img = Image.new("RGB", (total_w, height + text_h), (255,255,255))
            final_img.paste(real_img, (0,0))
            final_img.paste(seg_img, (width + gap, 0))
            final_img.paste(fake_img, ((width + gap) * 2, 0))

            # draw metrics
            draw = ImageDraw.Draw(final_img)
            txt = f"PSNR: {psnr:.4f}"
            if ssim_val is not None:
                txt += f"   SSIM: {ssim_val:.6f}"
            draw.text((10, height + 5), txt, fill=(0,0,0))

            final_cmp_path = os.path.join(args.outdir, f"{out_name}_final_cmp.png")
            final_img.save(final_cmp_path)

            print(f"Saved: {final_cmp_path}  (metrics -> {metrics_path})")

# VOC_PALETTE must be defined (if not already, add it above)
VOC_PALETTE = [
    0,0,0, 128,0,0, 0,128,0, 128,128,0, 0,0,128, 128,0,128,
    0,128,128, 128,128,128, 64,0,0, 192,0,0, 64,128,0, 192,128,0,
    64,0,128, 192,0,128, 64,128,128, 192,128,128, 0,64,0, 128,64,0,
    0,192,0
] + [0]*(256*3 - 21*3)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs="+", required=True, help="One or more image paths")
    p.add_argument("--outdir", default="test_outputs", help="Output folder")
    p.add_argument("--samples", type=int, default=1, help="Number of z samples per image")
    args = p.parse_args()
    main(args)