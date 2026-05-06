
import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

try:
    import imageio.v3 as _iio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

def _list_files(d, exts):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(d, f"*{e}"))
    files = sorted(files)
    return files

class PairedFolderDataset(Dataset):
    """
    Expect structure:
    root/
      train/ or val/
        input/
        gt/
    Filenames must match one-to-one between input/ and gt/.
    """
    def __init__(self, root, split="train", in_dir="input", gt_dir="gt", exts=(".jpg",".jpeg",".png",".tif",".tiff"),
                 patch=512, augment=True):
        self.root = root
        self.split = split
        self.in_dir = os.path.join(root, split, in_dir)
        self.gt_dir = os.path.join(root, split, gt_dir)
        self.exts = exts
        self.patch = patch
        self.augment = augment

        ins = _list_files(self.in_dir, exts)
        gts = _list_files(self.gt_dir, exts)
        name2path_in = {os.path.basename(p): p for p in ins}
        name2path_gt = {os.path.basename(p): p for p in gts}
        names = sorted(list(set(name2path_in.keys()) & set(name2path_gt.keys())))
        if len(names) == 0:
            raise RuntimeError(f"No paired files found under {self.in_dir} and {self.gt_dir}.")
        self.pairs = [(name2path_in[n], name2path_gt[n]) for n in names]

    def __len__(self):
        return len(self.pairs)

    def _read_tensor(self, path):
        """Read image preserving bit depth, return float32 [3,H,W] tensor in [0,1].

        For 16-bit TIFFs (e.g., PPR10K source), uses imageio so the full ~16-bit
        precision is preserved. PIL.convert('RGB') would silently quantize to 8-bit
        (~118x precision loss per channel verified on PPR10K). Falls back to PIL
        for formats imageio can't handle.
        """
        ext = os.path.splitext(path)[1].lower()
        if _HAS_IMAGEIO and ext in (".tif", ".tiff"):
            arr = _iio.imread(path)
            # Normalize to [0,1] float32 according to dtype
            if arr.dtype == np.uint16:
                arr = arr.astype(np.float32) / 65535.0
            elif arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            elif np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
            else:
                arr = arr.astype(np.float32)
            # Channel handling: HWC -> CHW; grayscale -> 3-channel; drop alpha
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        # Non-TIFF (jpg/png) — PIL is fine, no precision issue
        img = Image.open(path).convert("RGB")
        return TF.to_tensor(img)

    def __getitem__(self, idx):
        pin, pgt = self.pairs[idx]
        tin = self._read_tensor(pin)
        tgt = self._read_tensor(pgt)

        # 强制让 tin 和 tgt 具有相同空间尺寸（取两者公共的中心区域）
        _, Hin, Win = tin.shape
        _, Hgt, Wgt = tgt.shape
        if Hin != Hgt or Win != Wgt:
            Hc = min(Hin, Hgt)
            Wc = min(Win, Wgt)
            # 分别按各自中心裁到公共尺寸
            top_in  = (Hin - Hc) // 2; left_in  = (Win - Wc) // 2
            top_gt  = (Hgt - Hc) // 2; left_gt  = (Wgt - Wc) // 2
            tin = tin[:, top_in:top_in+Hc, left_in:left_in+Wc]
            tgt = tgt[:, top_gt:top_gt+Hc, left_gt:left_gt+Wc]

        # random crop (train) or center crop (val) if patch > 0
        if self.patch and self.patch > 0:
            _, H, W = tin.shape
            p = self.patch
            if self.split == "train":
                if H >= p and W >= p:
                    top = torch.randint(0, H - p + 1, (1,)).item()
                    left = torch.randint(0, W - p + 1, (1,)).item()
                    tin = tin[:, top:top+p, left:left+p]
                    tgt = tgt[:, top:top+p, left:left+p]
            else:
                if H >= p and W >= p:
                    top = (H - p) // 2
                    left = (W - p) // 2
                    tin = tin[:, top:top+p, left:left+p]
                    tgt = tgt[:, top:top+p, left:left+p]

        # simple augmentation
        if self.split == "train" and self.augment:
            if torch.rand(1).item() < 0.5:
                tin = TF.hflip(tin); tgt = TF.hflip(tgt)
            if torch.rand(1).item() < 0.5:
                tin = TF.vflip(tin); tgt = TF.vflip(tgt)

        # predictor input (downsample to 256 for robustness & speed)
        h = 256
        img_lr = TF.resize(tin, [h, h], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

        return {
            "img_lr": img_lr,
            "img_in": tin,
            "img_gt": tgt,
            "name": os.path.basename(pin),
        }
