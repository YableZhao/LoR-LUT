
# LoR-LUT (Starter)

A minimal, ready-to-train implementation that runs directly on Google Colab, featuring:
- Multi-basis LUT fusion + **low-rank (CP) residual** (LoR-LUT)
- Differentiable trilinear interpolation (default); tetrahedral interpolation interface reserved
- Losses: L1 + ΔE2000 (Lab) + TV(LUT) + L2(ΔL) (optional LPIPS)
- Paired-Folder data loading (any paired dataset, aligned by filename)
- Export the final `.cube` LUT for any input image (non-spatial-gating version)

> **Directory Structure**
```
LoR-LUT/
  core/
    core_lut.py
  data/
    paired_folder.py
  losses/
    delta_e.py
    lpips_wrapper.py
  utils/
    color.py
    metrics.py
    tv.py
  export/
    export_image_lut.py
    export_cube.py
  config/
    default.yaml
  train.py
  evaluate.py
  requirements.txt
  README.md
  notebooks/
    colab_quickstart.ipynb
```

---

## 0. Getting Started on Colab (Recommended)

1. Clone this repo and upload the entire folder to your Google Drive (e.g., `/MyDrive/LoR-LUT`).
2. Create a new Notebook in Colab, or directly use `notebooks/colab_quickstart.ipynb` (open it in Colab after uploading to Drive).
3. Run the following in Colab (if using our provided notebook, this will execute automatically):
```python
!nvidia-smi  # Check GPU
%cd /content
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/LoR-LUT
!pip install -r requirements.txt
```

---

## 1. Prepare Paired Data (Paired-Folder)

Organize your data as follows:
```
/dataset_root/
  train/
    input/   # Input images (e.g., camera raw / lower quality)
    gt/      # Corresponding target images (e.g., expert retouched / DSLR reference)
  val/
    input/
    gt/
```

**Requirements**: Filenames in `input/` and `gt/` must match one-to-one (e.g., `0001.jpg` exists in both directories). Supported formats: `.jpg/.png/.tif`.
Datasets such as the sRGB paired version of MIT-Adobe FiveK or DPED can be organized into this structure.

---

## 2. Start Training (Example)

```bash
# Train on /dataset_root with default parameters
python train.py   --data.root /content/drive/MyDrive/datasets/FiveK_paired   --work_dir runs/fivek_lor   --cfg config/default.yaml
```

**Optional parameter overrides**:
- `--train.patch 512`: Random crop size for training (default 512)
- `--train.batch 16`: Batch size
- `--model.G 33`, `--model.K 8`, `--model.R 8`
- `--loss.lpips 0.05`: LPIPS weight (requires `lpips` package)
- `--optim.lr 1e-3`, `--train.iters 120000`, etc.

Training logs are saved in `work_dir`. `best.ckpt` is selected based on validation PSNR/ΔE.

---

## 3. Evaluation and .cube Export

**Evaluation (computes PSNR / ΔE2000 / optional LPIPS)**
```bash
python evaluate.py   --data.root /content/drive/MyDrive/datasets/FiveK_paired/val   --ckpt runs/fivek_lor/best.ckpt   --out_dir runs/fivek_lor/val_vis
```

**Export the final LUT (.cube) for a given image**
```bash
python export/export_image_lut.py   --ckpt runs/fivek_lor/best.ckpt   --image /content/sample.jpg   --out_cube /content/sample_Lstar.cube
```

> The exported `.cube` file can be used directly in color grading tools / ISP pipelines. Interpolation assumes trilinear (tetrahedral can be switched on the deployment side).

---

## 4. Tips (Consistent with the Paper)

- The model predicts parameters from images downsampled to 256², then performs a single LUT lookup on the full image, enabling real-time inference at 4K resolution.
- It is recommended to **train in sRGB 0–1** channel space and use ΔE2000 as the perceptual color difference loss. If you have a proper RAW → linear sRGB pipeline, you can switch to linear space on this basis.
- Tetrahedral interpolation is more robust at small grid sizes but more complex to implement. This implementation defaults to trilinear for numerical stability and readability.

---

## 4.1 K=0 Full Residual Mode

**Purpose**: Remove the basis LUTs (K=0) and perform image-adaptive enhancement solely through low-rank residuals, while maintaining a stable starting point and loss compatibility.

- **Implementation (Residual around Identity)**
  - Final LUT: L = Identity + delta.
  - When K=0, the model does not predict alpha or fuse basis LUTs; instead, it uses a fixed identity LUT (buffer) as the bias.
  - The residual branch (R>0) operates as usual. When R=0, the residual branch is disabled (delta=0), leaving only basis LUT fusion. If both K=0 and R=0, L degenerates to the identity LUT.

- **Training and Losses**
  - L1 / ΔE2000 / LPIPS: Same as standard configuration, computed on the output image.
  - TV / Monotonicity: Computed on L_final (Identity + delta); the identity LUT is not updated — gradients flow only to delta.
  - DL2 (L2 on delta): Keep as is.
  - Alpha L2: Automatically skipped when K=0 (code handles empty alpha).

- **Logging / Visualization**
  - Alpha metrics (e.g., amax) in training logs show 0 when K=0.
  - In the viewer, Fused LUT displays as Identity; Residual and Final LUT visualizations remain functional.

- **Usage**
  - Set `model.K` to 0 in the config (`config/default.yaml`).
  - Other settings (e.g., `model.R`, loss weights) can be configured as usual.

- **Runtime Benchmark**
  - `bench_runtime.py` supports K=0: skips alpha prediction and basis fusion, reports weight prediction as 0ms, and uses the identity LUT as fused.

**Note**: If the identity LUT were made trainable, it would degenerate into "global basis LUT + residual", similar to K=1 with alpha≡1, which is no longer "fully residual". The current implementation uses a fixed identity LUT (buffer) to maintain the strict definition of full residual mode.

---

## 5. License and Acknowledgments

This Starter is intended for rapid academic/research reproduction. You are welcome to cite your implementation in publications.
