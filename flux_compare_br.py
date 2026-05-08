"""
Compare raw flux counts and SNR across B, V, R for the same region of M51.

Downloads all three FITS files from ryspark/asymmetry, shows them side by
side, asks for two boxes (signal + small sky sample), then reports raw counts
and SNR = (S_net * t) / sqrt(S_net*t + Sky_scaled*t + D*t*N + RN²*N)
for every filter on the same bounding box.
"""

import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sep
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

DARK_CURRENT = 0.002   # e-/pixel/s
READ_NOISE   = 10.0    # e- RMS


def _gh_download_url(filename):
    result = subprocess.run(
        ["gh", "api", f"repos/{SOURCE_REPO}/contents/{SOURCE_PATH}/{filename}"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["download_url"]


def load_band(band):
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = BANDS[band]
    local    = os.path.join(DATA_DIR, filename)
    if not os.path.exists(local):
        print(f"Fetching {filename} …")
        urllib.request.urlretrieve(_gh_download_url(filename), local)
        print("  done.")
    with fits.open(local) as h:
        return h[0].data.astype(float), h[0].header


def subtract_background(data):
    d    = np.ascontiguousarray(data, np.float64)
    mask = np.isnan(d)
    bkg  = sep.Background(d, mask=mask, bw=64, bh=64)
    return d - bkg.back()


def pick_box(fig, label, color):
    print(f"Click TWO CORNERS of the {label} box …")
    pts = plt.ginput(2, timeout=60)
    if len(pts) < 2:
        raise RuntimeError("Cancelled.")
    (x0, y0), (x1, y1) = pts
    x0, x1 = sorted([int(round(x0)), int(round(x1))])
    y0, y1 = sorted([int(round(y0)), int(round(y1))])
    for ax in fig.axes:
        ax.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor=color, facecolor="none", label=label,
        ))
    fig.axes[1].legend(loc="upper right", fontsize=7)
    plt.draw()
    return x0, y0, x1, y1


def extract_pixels(data, x0, y0, x1, y1):
    region = data[y0:y1, x0:x1].ravel()
    return region[~np.isnan(region)]


def compute_snr(sig_pix, sky_pix, exptime):
    N             = len(sig_pix)
    counts        = float(np.sum(sig_pix))
    sky_per_pixel = float(np.mean(sky_pix))
    sky_scaled    = sky_per_pixel * N
    S_net         = counts - sky_scaled
    t             = exptime
    flux          = S_net * t
    flux_err      = np.sqrt(
        S_net      * t +
        sky_scaled * t +
        DARK_CURRENT * t * N +
        READ_NOISE**2 * N
    )
    return {
        "counts":        counts,
        "sky_per_pixel": sky_per_pixel,
        "sky_scaled":    sky_scaled,
        "S_net":         S_net,
        "flux":          flux,
        "flux_err":      flux_err,
        "SNR":           flux / flux_err,
    }


def main():
    print("Loading B, V, R …")
    images  = {}
    headers = {}
    subs    = {}
    for band in ["B", "V", "R"]:
        img, hdr       = load_band(band)
        images[band]   = img
        headers[band]  = hdr
        subs[band]     = subtract_background(img)

    interval = ZScaleInterval()
    colors   = {"B": "Blues_r", "V": "gray", "R": "Reds_r"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, band in zip(axes, ["B", "V", "R"]):
        vmin, vmax = interval.get_limits(subs[band])
        ax.imshow(subs[band], origin="lower", cmap=colors[band], vmin=vmin, vmax=vmax)
        ax.set_title(f"M51  [{band}]")
        ax.axis("off")

    fig.suptitle(
        "1. Draw SIGNAL box (cyan)  →  2. Draw small SKY box (orange)",
        fontsize=11
    )
    plt.tight_layout()
    plt.ion()
    plt.show()

    sx0, sy0, sx1, sy1 = pick_box(fig, "Signal", "cyan")
    bx0, by0, bx1, by1 = pick_box(fig, "Sky",    "orange")

    exptime = {b: float(headers[b].get("EXPTIME", headers[b].get("EXPOSURE", 1)))
               for b in ["B", "V", "R"]}

    results = {}
    for band in ["B", "V", "R"]:
        sig_pix = extract_pixels(subs[band], sx0, sy0, sx1, sy1)
        sky_pix = extract_pixels(subs[band], bx0, by0, bx1, by1)
        results[band] = compute_snr(sig_pix, sky_pix, exptime[band])

    N = results["B"]["SNR"]   # just for reference

    print(f"\n=== Signal box: ({sx0},{sy0}) → ({sx1},{sy1})  |  {len(extract_pixels(subs['V'], sx0, sy0, sx1, sy1))} px ===")
    print(f"=== Sky box:    ({bx0},{by0}) → ({bx1},{by1}) ===\n")

    w = 28
    header_row = f"  {'':>{w}}  {'B':>14}  {'V':>14}  {'R':>14}"
    print(header_row)
    print("  " + "-" * (w + 48))

    rows = [
        ("Exposure time (s)",       lambda r, b: exptime[b],          ".1f"),
        ("Total counts (ADU)",      lambda r, b: r["counts"],         ".1f"),
        ("Sky/pixel (ADU)",         lambda r, b: r["sky_per_pixel"],  ".4f"),
        ("Sky scaled to aperture",  lambda r, b: r["sky_scaled"],     ".1f"),
        ("Net signal S_net (ADU)",  lambda r, b: r["S_net"],          ".1f"),
        ("flux  (S_net * t)",       lambda r, b: r["flux"],           ".1f"),
        ("flux_err",                lambda r, b: r["flux_err"],       ".2f"),
        ("SNR  (flux / flux_err)",  lambda r, b: r["SNR"],            ".2f"),
    ]

    for label, fn, fmt in rows:
        vals = [fn(results[b], b) for b in ["B", "V", "R"]]
        row  = f"  {label:>{w}}  " + "  ".join(f"{v:{fmt}:>14}" for v in vals)
        print(row)

    print("\n  Ratios (relative to V):")
    for band in ["B", "R"]:
        snr_ratio   = results[band]["SNR"]   / results["V"]["SNR"]
        count_ratio = results[band]["counts"] / results["V"]["counts"]
        print(f"    {band}/V  counts={count_ratio:.4f}   SNR={snr_ratio:.4f}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
