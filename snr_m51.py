"""
SNR calculation for M51 FITS data.

Workflow:
  1. Download stacked FITS from ryspark/asymmetry (Git LFS via gh CLI).
  2. Subtract global background with sep (same approach as Lab 1.1).
  3. User draws a box over a spiral arm  → signal aperture.
  4. User draws a SMALL box over blank sky → sky sample.
     The sky sample is scaled up to the same pixel count as the signal box.
  5. Compute SNR = (S_net * t) / sqrt(S_net*t + Sky_scaled*t + D*t*N + RN²*N)
"""

import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import urllib.request
import sep
import os

# --- Instrument parameters (adjust to match actual camera) ---
DARK_CURRENT = 0.002  # e-/pixel/s
READ_NOISE   = 10.0   # e- RMS

# --- Data source ---
SOURCE_REPO = "ryspark/asymmetry"
SOURCE_PATH = "data/m51/2024_04_28/stacked"
BANDS = {
    "B": "m51_B_stacked.fits",
    "V": "m51_V_stacked.fits",
    "R": "m51_R_stacked.fits",
}
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _gh_download_url(filename: str) -> str:
    api_path = f"repos/{SOURCE_REPO}/contents/{SOURCE_PATH}/{filename}"
    result = subprocess.run(
        ["gh", "api", api_path], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)["download_url"]


def download_fits(band: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = BANDS[band]
    local_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(local_path):
        print(f"Fetching signed download URL for {filename} …")
        url = _gh_download_url(filename)
        print(f"Downloading {filename} (~30 MB) …")
        urllib.request.urlretrieve(url, local_path)
        print("  done.")
    return local_path


def load_image(band: str) -> tuple[np.ndarray, fits.Header]:
    path = download_fits(band)
    with fits.open(path) as hdul:
        data   = hdul[0].data.astype(float)
        header = hdul[0].header
    return data, header


# ---------------------------------------------------------------------------
# Background subtraction (Lab 1.1 technique: sep.Background)
# ---------------------------------------------------------------------------

def subtract_background(data: np.ndarray) -> tuple[np.ndarray, sep.Background]:
    """
    Estimate and subtract the 2-D background using sep, matching the
    approach in Lab 1.1 (sep.Background with a NaN mask).
    """
    d    = np.ascontiguousarray(data, dtype=np.float64)
    mask = np.isnan(d)
    bkg  = sep.Background(d, mask=mask, bw=64, bh=64)
    return d - bkg.back(), bkg


# ---------------------------------------------------------------------------
# Interactive box selection
# ---------------------------------------------------------------------------

def pick_box(ax, label: str, color: str) -> tuple[int, int, int, int]:
    """Click any two opposite corners to define a rectangle."""
    print(f"\nClick TWO CORNERS of the {label} box …")
    pts = plt.ginput(2, timeout=60)
    if len(pts) < 2:
        raise RuntimeError("Box selection cancelled.")
    (x0, y0), (x1, y1) = pts
    x0, x1 = sorted([int(round(x0)), int(round(x1))])
    y0, y1 = sorted([int(round(y0)), int(round(y1))])
    rect = patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2, edgecolor=color, facecolor="none", label=label,
    )
    ax.add_patch(rect)
    ax.legend(loc="upper right", fontsize=8)
    plt.draw()
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Pixel extraction (Lab 1.1 technique: sum pixels, track area separately)
# ---------------------------------------------------------------------------

def extract_box(data: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Return the flat array of valid (non-NaN) pixels inside the box."""
    region = data[y0:y1, x0:x1].ravel()
    return region[~np.isnan(region)]


# ---------------------------------------------------------------------------
# SNR calculation
# ---------------------------------------------------------------------------

def compute_snr(
    signal_pixels: np.ndarray,
    sky_pixels: np.ndarray,
    exposure_time: float,
    dark_current: float = DARK_CURRENT,
    read_noise:   float = READ_NOISE,
) -> dict:
    """
    Scale the small sky box to match the signal aperture pixel count, then
    compute aperture-level SNR.

    Lab 1.1 analogue:
      counts          = sum(signal_box)
      inner_area (N)  = len(signal_pixels)
      sky_per_pixel   = mean(sky_box)          ← tiny sample, scaled up
      sky_scaled      = sky_per_pixel * N      ← equivalent of background_counts_per_pixel * inner_area
      S_net           = counts - sky_scaled    ← background-subtracted signal

    SNR formula (aperture level):
      SNR = (S_net · t) / sqrt(S_net·t + Sky_scaled·t + D·t·N + RN²·N)
    """
    N             = len(signal_pixels)
    counts        = float(np.sum(signal_pixels))
    sky_per_pixel = float(np.mean(sky_pixels))   # mean of small sky sample
    sky_scaled    = sky_per_pixel * N             # blown up to signal aperture size
    S_net         = counts - sky_scaled           # net source counts

    t = exposure_time
    numerator   = S_net * t
    denominator = np.sqrt(
        S_net      * t +
        sky_scaled * t +
        dark_current * t * N +
        read_noise ** 2  * N
    )
    snr = numerator / denominator

    # Verification: SNR = flux / flux_err
    # flux     = S_net * t  (total net counts)
    # flux_err = denominator  (same quantity, different framing)
    flux      = S_net * t
    flux_err  = denominator
    snr_check = flux / flux_err

    return {
        "N_signal":     N,
        "N_sky_sample": len(sky_pixels),
        "counts":       counts,
        "sky_per_pixel":sky_per_pixel,
        "sky_scaled":   sky_scaled,
        "S_net":        S_net,
        "flux":         flux,
        "flux_err":     flux_err,
        "SNR":          snr,
        "SNR_verified": snr_check,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    band = input("Choose band [B/V/R] (default V): ").strip().upper() or "V"
    if band not in BANDS:
        raise ValueError(f"Unknown band '{band}'. Choose B, V, or R.")

    data, header = load_image(band)

    exptime = header.get("EXPTIME", header.get("EXPOSURE", None))
    if exptime is None:
        exptime = float(input("Exposure time not found in header. Enter seconds: "))
    else:
        print(f"Exposure time: {exptime} s")

    print("Subtracting background with sep …")
    sub, bkg = subtract_background(data)
    print(f"  Global RMS: {bkg.globalrms:.4f} ADU")

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(sub)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(sub, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(
        f"M51  |  Band: {band}  |  background-subtracted\n"
        "1. Draw SPIRAL ARM box (cyan)  →  2. Draw small SKY box (orange)"
    )
    plt.tight_layout()
    plt.ion()
    plt.show()

    print("\n=== Box selection ===")
    print("Click any two opposite corners per box.")

    sx0, sy0, sx1, sy1 = pick_box(ax, "Signal — spiral arm", "cyan")
    bx0, by0, bx1, by1 = pick_box(ax, "Sky background (small sample)", "orange")

    sig_pix = extract_box(sub, sx0, sy0, sx1, sy1)
    sky_pix = extract_box(sub, bx0, by0, bx1, by1)

    res = compute_snr(sig_pix, sky_pix, exptime)

    print("\n=== Results ===")
    print(f"  Band                   : {band}")
    print(f"  Exposure time          : {exptime:.1f} s")
    print(f"  Signal box             : ({sx0},{sy0}) → ({sx1},{sy1})  — {res['N_signal']} px")
    print(f"  Sky box (raw sample)   : ({bx0},{by0}) → ({bx1},{by1})  — {res['N_sky_sample']} px")
    print(f"  Sky mean / pixel       : {res['sky_per_pixel']:.4f} ADU")
    print(f"  Sky scaled to aperture : {res['sky_scaled']:.1f} ADU  "
          f"({res['N_sky_sample']} → {res['N_signal']} px)")
    print(f"  Total counts (signal)  : {res['counts']:.1f} ADU")
    print(f"  Net signal             : {res['S_net']:.1f} ADU")
    print(f"  SNR                    : {res['SNR']:.2f}")
    print(f"\n=== Verification (flux / flux_err) ===")
    print(f"  flux                   : {res['flux']:.2f} ADU·s")
    print(f"  flux_err               : {res['flux_err']:.2f} ADU·s")
    print(f"  SNR (verified)         : {res['SNR_verified']:.2f}")
    print(f"  Match                  : {np.isclose(res['SNR'], res['SNR_verified'])}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
