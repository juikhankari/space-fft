# space-fft

SNR analysis for M51 stacked FITS images from the [ryspark/asymmetry](https://github.com/ryspark/asymmetry) dataset.

## What it does

1. Downloads M51 stacked FITS files (B, V, or R band) from the source repo.
2. Displays the image with a zscale stretch.
3. Lets you interactively select two boxes by clicking corners:
   - **Cyan box** — signal region (a spiral arm)
   - **Orange box** — sky background (same size recommended)
4. Computes SNR using:

```
SNR = (S · t) / sqrt(S·t + Sky·t + D·t + RN²)
```

where S = net signal/pixel, Sky = sky background/pixel, t = exposure time, D = dark current, RN = read noise.

## Setup

```bash
pip install -r requirements.txt
python snr_m51.py
```

## Notes

- First run downloads ~30 MB FITS files into `data/` (cached for subsequent runs).
- Default dark current = 0.002 e⁻/px/s and read noise = 10 e⁻; edit constants at the top of `snr_m51.py` to match your camera.
- If exposure time is not in the FITS header, you'll be prompted to enter it.
