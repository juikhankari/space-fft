"""
Compare raw flux counts in B vs R for the same selected region of M51.

Downloads B and R FITS files from ryspark/asymmetry, displays them
side-by-side, lets you draw one box on the V-band reference image,
then reports total counts and the B/R ratio for that region.
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

SOURCE_REPO = "ryspark/asymmetry"
SOURCE_PATH = "data/m51/2024_04_28/stacked"
BANDS = {
    "B": "m51_B_stacked.fits",
    "V": "m51_V_stacked.fits",
    "R": "m51_R_stacked.fits",
}
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _gh_download_url(filename):
    result = subprocess.run(
        ["gh", "api", f"repos/{SOURCE_REPO}/contents/{SOURCE_PATH}/{filename}"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["download_url"]


def load_band(band):
    os.makedirs(DATA_DIR, exist_ok=True)
    filename  = BANDS[band]
    local     = os.path.join(DATA_DIR, filename)
    if not os.path.exists(local):
        print(f"Fetching {filename} …")
        urllib.request.urlretrieve(_gh_download_url(filename), local)
        print("  done.")
    with fits.open(local) as h:
        return h[0].data.astype(float), h[0].header


def pick_box(ax, color="cyan"):
    print("Click TWO CORNERS of the box …")
    pts = plt.ginput(2, timeout=60)
    if len(pts) < 2:
        raise RuntimeError("Cancelled.")
    (x0, y0), (x1, y1) = pts
    x0, x1 = sorted([int(round(x0)), int(round(x1))])
    y0, y1 = sorted([int(round(y0)), int(round(y1))])
    rect = patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2, edgecolor=color, facecolor="none"
    )
    ax.add_patch(rect)
    plt.draw()
    return x0, y0, x1, y1


def box_sum(data, x0, y0, x1, y1):
    region = data[y0:y1, x0:x1]
    valid  = region[~np.isnan(region)]
    return float(np.sum(valid)), float(np.mean(valid)), len(valid)


def main():
    print("Loading B, V, R bands …")
    data_b, hdr_b = load_band("B")
    data_v, _     = load_band("V")
    data_r, hdr_r = load_band("R")

    interval = ZScaleInterval()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (band, img) in zip(axes, [("B", data_b), ("V", data_v), ("R", data_r)]):
        vmin, vmax = interval.get_limits(img)
        ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"M51  [{band}]")
        ax.axis("off")

    fig.suptitle("Draw a box on any panel — same pixel coords applied to all bands", fontsize=11)
    plt.tight_layout()
    plt.ion()
    plt.show()

    # Let user pick box on whichever panel they click
    print("\nDraw your box on any of the three images.")
    x0, y0, x1, y1 = pick_box(axes[1])   # ginput works across the whole figure

    # Draw the same box on B and R panels too
    for ax in [axes[0], axes[2]]:
        ax.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor="cyan", facecolor="none"
        ))
    plt.draw()

    # Extract counts from the same region in B and R
    t_b = float(hdr_b.get("EXPTIME", hdr_b.get("EXPOSURE", 1)))
    t_r = float(hdr_r.get("EXPTIME", hdr_r.get("EXPOSURE", 1)))

    sum_b, mean_b, n_b = box_sum(data_b, x0, y0, x1, y1)
    sum_r, mean_r, n_r = box_sum(data_r, x0, y0, x1, y1)

    # Normalise by exposure time so we're comparing count rates
    rate_b = sum_b / t_b
    rate_r = sum_r / t_r

    print(f"\n=== Box: ({x0},{y0}) → ({x1},{y1})  |  {n_b} pixels ===")
    print(f"\n  {'':30s}  {'B':>14}  {'R':>14}")
    print(f"  {'Exposure time (s)':30s}  {t_b:>14.1f}  {t_r:>14.1f}")
    print(f"  {'Total raw counts (ADU)':30s}  {sum_b:>14.1f}  {sum_r:>14.1f}")
    print(f"  {'Mean counts/pixel (ADU)':30s}  {mean_b:>14.4f}  {mean_r:>14.4f}")
    print(f"  {'Count rate (ADU/s)':30s}  {rate_b:>14.2f}  {rate_r:>14.2f}")
    print(f"\n  Raw counts  R / B  = {sum_r / sum_b:.4f}")
    print(f"  Count rate  R / B  = {rate_r / rate_b:.4f}")
    print(f"  (R brighter = ratio > 1, B brighter = ratio < 1)")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
