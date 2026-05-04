
import torch
import torch.nn.functional as F
from losses.delta_e import delta_e_2000_srgb

def psnr(x, y, eps=1e-8):
    mse = F.mse_loss(x, y, reduction='none').mean(dim=[1,2,3])
    psnr = -10.0 * torch.log10(mse + eps)
    return psnr

def deltaE2000_mean(x, y):
    de = delta_e_2000_srgb(x, y)  # [B,H,W]
    # 兼容 3D/4D
    if de.dim() == 4:
        return de.mean(dim=[1,2,3])
    else:  # [B,H,W]
        return de.mean(dim=[1,2])

# -----------------------
# Additional common metrics
# -----------------------

def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    window_2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,k,k]
    window = window_2d.repeat(channels, 1, 1, 1)  # [C,1,k,k]
    return window

def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5,
         data_range: float = 1.0) -> torch.Tensor:
    """
    Structural Similarity (SSIM) index.
    Returns per-image values with shape [N].
    """
    if x.shape != y.shape:
        raise ValueError("ssim expects x and y with the same shape")
    if x.dim() != 4:
        raise ValueError("ssim expects 4D tensors [N,C,H,W]")

    # Ensure float for numerical stability
    x = x.float()
    y = y.float()

    N, C, H, W = x.shape
    window = _gaussian_window(window_size, sigma, C, x.device, x.dtype)

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=C)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=C) - mu_xy

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + 1e-12)
    # average over C,H,W -> [N]
    return ssim_map.mean(dim=[1, 2, 3])
