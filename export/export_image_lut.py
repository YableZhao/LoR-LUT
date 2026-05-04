
import os
import argparse
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from core.core_lut import LoRIA3DLUT, cp_residual_to_lut, create_identity_lut
from export.export_cube import write_cube

def load_image(path):
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img).unsqueeze(0)  # [1,3,H,W]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out_cube", type=str, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    G = cfg["model"]["G"]; K = cfg["model"]["K"]; R = cfg["model"]["R"]

    model = LoRIA3DLUT(G=G, K=K, R=R).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # load image
    x = load_image(args.image).to(device)
    x_lr = TF.resize(x, [256,256], antialias=True)

    with torch.no_grad():
        # Bases branch (supports K=0)
        if K > 0:
            alpha = model.weight_pred(x_lr)           # [1,K]
            fused = model.fuse_bases(alpha)           # [1,G,G,G,3]
        else:
            alpha = torch.zeros((1,0), device=device, dtype=x_lr.dtype)
            if hasattr(model, "identity") and model.identity is not None:
                fused = model.identity.unsqueeze(0).expand(1, -1, -1, -1, -1).to(x_lr.dtype)
            else:
                fused = create_identity_lut(G).to(device=device, dtype=x_lr.dtype).unsqueeze(0)

        # Residual branch (supports R=0)
        if R > 0:
            u,v,w,c = model.resid_pred(x_lr)          # [1,R,G], [1,R,3]
            delta = cp_residual_to_lut(u,v,w,c)       # [1,G,G,G,3]
        else:
            delta = torch.zeros_like(fused)

        Lstar = fused + delta                         # [1,G,G,G,3]

    lut_np = Lstar.squeeze(0).clamp(0,1).detach().cpu().numpy()
    os.makedirs(os.path.dirname(args.out_cube), exist_ok=True)
    write_cube(lut_np, args.out_cube, title="LoR-LUT")
    print(f"Saved cube to: {args.out_cube}")
