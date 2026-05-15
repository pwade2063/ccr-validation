# CCR Image Processing — Validation

A reproducible validation of the [Seattle Aquarium CCR](https://github.com/Seattle-Aquarium) Underwater Image Enhancer (UIE) pipeline. Runs their model on their test set, compares AI-processed outputs to hand-edited ground truth, and reports image quality metrics.

## What this is

The CCR program uses an AI model to automate the manual Adobe Lightroom editing step in their ROV survey image workflow. This notebook measures how close the AI output is to the hand-edited result using SSIM and PSNR on the 20-image reference test set.

## Results (flat06-249-best_model.pth, 20-image test set)

| | SSIM | PSNR |
|---|---|---|
| Raw baseline vs. ground truth | 0.22 avg | 13.9 dB avg |
| AI-processed vs. ground truth | 0.41 avg | 12.1 dB avg |
| Delta | **+0.19 (positive on all 20 images)** | **−1.8 dB (negative on 14/20 images)** |

The model consistently improves structural similarity but degrades pixel-level fidelity in most images, visible as blurriness and washed-out areas. See `output/comparison_grid.png` for the full visual comparison.

Full per-image results: [`output/metrics.csv`](output/metrics.csv)

## Reproducing

```bash
# 1. Python environment
python3.11 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "opencv-python==4.10.0.84" "numpy<2" Pillow rawpy scikit-image \
            matplotlib pandas tqdm jupyter huggingface_hub PyYAML ipykernel

# 2. UIE source code
git clone --depth 1 https://github.com/Seattle-Aquarium/underwater-auto-image-encoder.git

# 3. gpr_tools (GPR → DNG converter, macOS — requires cmake)
brew install cmake
git clone --depth 1 https://github.com/keenanjohnson/gpr_tools.git /tmp/gpr_src
cd /tmp/gpr_src && mkdir build && cd build && cmake .. && make
cp source/app/gpr_tools/gpr_tools /usr/local/bin/

# 4. Test images
mkdir -p test_images/input_GPR test_images/output_JPEG
# Download the 20 GPR and JPEG files from:
# https://github.com/Seattle-Aquarium/CCR_image_processing/tree/main/testing_image_sets
# (use curl/wget directly — git clone is slow due to binary file history)

# 5. Model weights
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Seattle-Aquarium/flat06-249-best_model.pth',
                'flat06-249-best_model.pth', local_dir='model')
"

# 6. Run
python -m ipykernel install --user --name ccr-validation --display-name "CCR Validation"
jupyter notebook validation.ipynb
```

Then **Kernel → Restart & Run All**.

## Files

| File | Description |
|---|---|
| `validation.ipynb` | The notebook — run this |
| `output/metrics.csv` | Per-image SSIM and PSNR results |
| `output/comparison_grid.png` | Visual: raw vs AI-processed vs hand-edited for all 20 images |
| `output/metrics_chart.png` | Bar chart of SSIM and PSNR per image |

## Related issues

- [CCR_development #29](https://github.com/Seattle-Aquarium/CCR_development/issues/29) — AI/ML to help process GoPro .GPR photos
- [CCR_image_processing #15](https://github.com/Seattle-Aquarium/CCR_image_processing/issues/15) — Blurry artifacts in U-Shape Transformer model
- [CCR_image_processing #14](https://github.com/Seattle-Aquarium/CCR_image_processing/issues/14) — Dataset partitioning (flat vs complex)
