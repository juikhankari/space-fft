"""
SNR comparison: m3 (Lab 1.2 approach) vs M51 spiral arm.

Replicates the exact flux / flux_err calculation from SUBMITTED_Lab_1_2.ipynb
for m3 on the 0.7m telescope, then compares the resulting per-star SNRs to
the M51 aperture SNR of 15348.91.

Usage:
    python snr_m3.py --b path/to/m3_B.fits --v path/to/m3_V.fits \
                     --bias path/to/bias_dir --dark path/to/dark_dir

    All four paths can be Google Drive downloads or local copies.
    The bias/dark dirs should contain .fit files for the 2026-04-16 night.
"""

import argparse
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
import sep
from astropy.io import fits
from tqdm import tqdm

# --- Parameters from the notebook (m3, 0.7m) ---
R1        = 10      # aperture radius (px)
R2        = 12      # background annulus outer radius (px)
N_IMAGES  = 30      # number of coadded images
SIGMA_DET = 100     # sep detection threshold (sigma)
BW = BH   = 4       # background box size for sep
MAX_AREA  = 32      # max source area (px^2) to keep

M51_SNR = 15348.91  # reference value from snr_m51.py


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def load_fits_stack(pattern: str) -> list[np.ndarray]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return [fits.getdata(f).astype(float) for f in files]


def avg_bias_frame(bias_dir: str) -> tuple[np.ndarray, float]:
    frames = load_fits_stack(f"{bias_dir}/bias*.fit")
    avg    = np.mean(frames, axis=0)
    # gain from header of first bias file (same as notebook: 1/EGAIN)
    with fits.open(sorted(glob.glob(f"{bias_dir}/bias*.fit"))[0]) as h:
        gain = 1.0 / h[0].header["EGAIN"]
    print(f"  Gain (1/EGAIN): {gain:.4f} e-/count")
    sigma_readout = float(np.std(avg * gain))
    print(f"  Readout noise:  {sigma_readout:.4f} e-")
    return avg, sigma_readout, gain


def avg_dark_frames(dark_dir: str, avg_bias: np.ndarray) -> dict:
    avg_darks = {}
    for fn in sorted(glob.glob(f"{dark_dir}/dark*.fit")):
        with fits.open(fn) as h:
            t = float(h[0].header["EXPTIME"])
            avg_darks.setdefault(t, []).append(h[0].data.astype(float))
    return {t: np.mean(stack, axis=0) - avg_bias for t, stack in avg_darks.items()}


# ---------------------------------------------------------------------------
# Aperture photometry + SNR (notebook recipe)
# ---------------------------------------------------------------------------

def compute_snr_sources(
    science_data: dict[str, np.ndarray],
    headers: dict,
    sigma_readout: float,
) -> dict:
    """
    Replicate Lab 1.2 cell 23 exactly.
    Returns source_data dict with flux, flux_err, and snr per filter.
    """
    # Source extraction on V band (notebook uses V)
    src_img = science_data["V"]
    mask    = np.isnan(src_img)
    bkg     = sep.Background(
        np.ascontiguousarray(src_img, np.float64),
        mask=mask, bw=BW, bh=BH
    )
    sub = src_img - bkg.back()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sources = sep.extract(
            np.ascontiguousarray(sub, np.float64),
            SIGMA_DET, err=bkg.globalrms, mask=mask, deblend_cont=1e-4
        )
    sources = sources[sources["a"] * sources["b"] * np.pi < MAX_AREA]
    print(f"  Detected {len(sources)} sources")

    source_data = {}
    for filt in ["B", "V"]:
        img     = np.ascontiguousarray(science_data[filt], np.float64)
        nan_msk = np.isnan(img)
        ones    = np.ones_like(img)
        exptime = float(headers[filt]["EXPTIME"])

        counts       = sep.sum_circle(img,  sources["x"], sources["y"], R1, mask=nan_msk)[0]
        inner_area   = sep.sum_circle(ones, sources["x"], sources["y"], R1, mask=nan_msk)[0]
        bkg_counts   = sep.sum_circann(img,  sources["x"], sources["y"], R1, R2, mask=nan_msk)[0]
        outer_area   = sep.sum_circann(ones, sources["x"], sources["y"], R1, R2, mask=nan_msk)[0]
        bkg_per_pix  = bkg_counts / outer_area

        # per-pixel background variance from the annulus (notebook approach)
        ny, nx = img.shape
        Y, X   = np.ogrid[:ny, :nx]
        var_bkg = []
        for i in tqdm(range(len(sources)), desc=f"Annulus var [{filt}]", leave=False):
            dR = np.sqrt((X - sources["x"][i])**2 + (Y - sources["y"][i])**2)
            ann_pix = img[(dR >= R1) & (dR < R2) & ~nan_msk]
            var_bkg.append(float(np.var(ann_pix)) if ann_pix.size else 0.0)
        var_bkg = np.array(var_bkg)

        flux     = (counts - inner_area * bkg_per_pix) / exptime
        flux_err = np.sqrt(
            (counts + var_bkg * inner_area + N_IMAGES * inner_area * sigma_readout**2)
            / exptime**2
        )
        snr = flux / flux_err

        source_data[filt] = {
            "flux":     flux,
            "flux_err": flux_err,
            "snr":      snr,
            "exptime":  exptime,
        }

    return source_data, sources


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b",    required=True, help="Path to m3 B-band stacked FITS")
    ap.add_argument("--v",    required=True, help="Path to m3 V-band stacked FITS")
    ap.add_argument("--bias", required=True, help="Dir containing bias*.fit files")
    ap.add_argument("--dark", required=True, help="Dir containing dark*.fit files")
    args = ap.parse_args()

    print("Loading science images …")
    science = {}
    headers = {}
    for filt, path in [("B", args.b), ("V", args.v)]:
        with fits.open(path) as h:
            science[filt] = h[0].data.astype(h[0].data.dtype.newbyteorder("="))
            headers[filt] = h[0].header

    print("Loading calibration …")
    avg_bias, sigma_readout, gain = avg_bias_frame(args.bias)

    print("Extracting sources and computing SNR …")
    source_data, sources = compute_snr_sources(science, headers, sigma_readout)

    # --- Results ---
    for filt in ["B", "V"]:
        snr = source_data[filt]["snr"]
        valid = snr[np.isfinite(snr) & (snr > 0)]
        print(f"\n=== m3 [{filt}]  (exptime = {source_data[filt]['exptime']:.1f} s) ===")
        print(f"  Sources with valid SNR : {len(valid)}")
        print(f"  Median SNR             : {np.median(valid):.2f}")
        print(f"  Mean SNR               : {np.mean(valid):.2f}")
        print(f"  Max SNR                : {np.max(valid):.2f}")
        print(f"  Min SNR                : {np.min(valid):.2f}")

    print(f"\n=== Comparison to M51 spiral arm ===")
    print(f"  M51 aperture SNR       : {M51_SNR:.2f}")
    for filt in ["B", "V"]:
        snr    = source_data[filt]["snr"]
        valid  = snr[np.isfinite(snr) & (snr > 0)]
        ratio  = np.median(valid) / M51_SNR
        print(f"  m3 [{filt}] median / M51  : {ratio:.4f}  "
              f"({'~same order' if 0.1 < ratio < 10 else 'different by >10x'})")

    # --- Plot SNR histogram ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for filt, color in [("B", "steelblue"), ("V", "goldenrod")]:
        snr   = source_data[filt]["snr"]
        valid = snr[np.isfinite(snr) & (snr > 0)]
        ax.hist(valid, bins=30, histtype="step", color=color, label=f"m3 [{filt}]")
    ax.axvline(M51_SNR, color="red", ls="--", label=f"M51 ({M51_SNR:.0f})")
    ax.set_xlabel("SNR  (flux / flux_err)")
    ax.set_ylabel("Number of sources")
    ax.set_title("m3 per-star SNR vs M51 spiral arm aperture SNR")
    ax.legend()
    plt.tight_layout()
    plt.savefig("snr_comparison.png", dpi=150)
    print("\nSaved plot to snr_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
