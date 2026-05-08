"""
SNR calculation for M51 FITS data.

Downloads stacked FITS files from ryspark/asymmetry GitHub repo,
lets user select signal (spiral arm) and background (sky) boxes,
then computes SNR = (S * t) / sqrt(S*t + Sky*t + D*t + RN^2).
"""

import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import urllib.request
import os

# --- Instrument parameters (typical values for the telescope used) ---
DARK_CURRENT = 0.002   # e-/pixel/s — adjust for actual camera
READ_NOISE   = 10.0    # e- RMS — adjust for actual camera

# --- Data source ---
SOURCE_REPO = "ryspark/asymmetry"
SOURCE_PATH = "data/m51/2024_04_28/stacked"
BANDS = {"B": "m51_B_stacked.fits",
         "V": "m51_V_stacked.fits",
         "R": "m51_R_stacked.fits"}
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _gh_download_url(filename: str) -> str:
    """Use the gh CLI to get a fresh signed LFS download URL."""
    api_path = f"repos/{SOURCE_REPO}/contents/{SOURCE_PATH}/{filename}"
    result = subprocess.run(
        ["gh", "api", api_path],
        capture_output=True, text=True, check=True
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


def pick_box(ax, label: str, color: str) -> tuple[int, int, int, int]:
    """Ask the user to click two corners of a rectangular box."""
    print(f"\nClick the TOP-LEFT corner of the {label} box …")
    pts = plt.ginput(2, timeout=60)
    if len(pts) < 2:
        raise RuntimeError("Box selection cancelled.")
    (x0, y0), (x1, y1) = pts
    x0, x1 = sorted([int(x0), int(x1)])
    y0, y1 = sorted([int(y0), int(y1)])
    rect = patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2, edgecolor=color, facecolor="none", label=label
    )
    ax.add_patch(rect)
    ax.legend(loc="upper right")
    plt.draw()
    return x0, y0, x1, y1


def box_stats(data: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[float, float]:
    """Return (mean, std) of pixels inside the box."""
    region = data[y0:y1, x0:x1]
    return float(np.mean(region)), float(np.std(region))


def compute_snr(
    signal_mean: float,
    sky_mean: float,
    exposure_time: float,
    n_pixels: int,
    dark_current: float = DARK_CURRENT,
    read_noise: float = READ_NOISE,
) -> float:
    """
    SNR = (S * t) / sqrt(S*t + Sky*t + D*t + RN^2)

    Parameters
    ----------
    signal_mean   : mean counts/pixel in signal box (ADU/s assumed if exposure applied)
    sky_mean      : mean counts/pixel in sky box
    exposure_time : integration time in seconds (from FITS header or fallback)
    n_pixels      : number of pixels in the signal aperture
    dark_current  : dark current in e-/pixel/s
    read_noise    : read noise in e- RMS
    """
    S   = signal_mean - sky_mean   # net source signal per pixel (ADU/pixel)
    Sky = sky_mean                 # sky background per pixel

    # Per-pixel SNR, then scale to aperture
    numerator   = S * exposure_time
    denominator = np.sqrt(
        S   * exposure_time +
        Sky * exposure_time +
        dark_current * exposure_time +
        read_noise ** 2
    )
    snr_per_pixel = numerator / denominator
    # Total SNR for aperture (assuming uncorrelated pixels)
    snr_total = snr_per_pixel * np.sqrt(n_pixels)
    return snr_total, snr_per_pixel


def main():
    band = input("Choose band [B/V/R] (default V): ").strip().upper() or "V"
    if band not in BANDS:
        raise ValueError(f"Unknown band '{band}'. Choose from B, V, R.")

    data, header = load_image(band)

    # Try to read exposure time from header
    exptime = header.get("EXPTIME", header.get("EXPOSURE", None))
    if exptime is None:
        exptime = float(input("Exposure time not in header. Enter exposure time in seconds: "))
    else:
        print(f"Exposure time from header: {exptime} s")

    # Display image with zscale stretch
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(
        f"M51  |  Band: {band}  |  {data.shape[1]}×{data.shape[0]} px\n"
        "Select SPIRAL ARM box first, then SKY box"
    )
    plt.tight_layout()
    plt.ion()
    plt.show()

    print("\n=== Box selection ===")
    print("Click TWO corners (any two opposite corners) per box.")

    sx0, sy0, sx1, sy1 = pick_box(ax, "Signal (spiral arm)", "cyan")
    bx0, by0, bx1, by1 = pick_box(ax, "Sky background",      "orange")

    sig_mean, sig_std   = box_stats(data, sx0, sy0, sx1, sy1)
    sky_mean, sky_std   = box_stats(data, bx0, by0, bx1, by1)
    n_pix               = (sx1 - sx0) * (sy1 - sy0)

    snr_total, snr_px = compute_snr(sig_mean, sky_mean, exptime, n_pix)

    print("\n=== Results ===")
    print(f"  Band              : {band}")
    print(f"  Exposure time     : {exptime:.1f} s")
    print(f"  Signal box        : ({sx0},{sy0}) → ({sx1},{sy1})  ({n_pix} px)")
    print(f"  Sky box           : ({bx0},{by0}) → ({bx1},{by1})")
    print(f"  Signal mean       : {sig_mean:.4f}  ± {sig_std:.4f} ADU")
    print(f"  Sky mean          : {sky_mean:.4f}  ± {sky_std:.4f} ADU")
    print(f"  Net signal/pixel  : {sig_mean - sky_mean:.4f} ADU")
    print(f"  SNR (per pixel)   : {snr_px:.2f}")
    print(f"  SNR (aperture)    : {snr_total:.2f}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
