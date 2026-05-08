"""
SNR comparison: m3 (Lab 1.2 approach) vs M51 spiral arm.

Calibrates raw M3 frames from:
  data_04_08/2026_04_08/   — science (B 20s x20, V 10s x20)
  data_04_16/2026_04_16/   — bias, darks, flats

Then replicates the exact flux/flux_err SNR from SUBMITTED_Lab_1_2.ipynb
and compares to M51 aperture SNR of 15348.91.
"""

import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
import sep
from astropy.io import fits
from tqdm import tqdm

# --- Paths (edit if you moved the unzipped folders) ---
DIR_08 = "/Users/juikhankari/Downloads/data_04_08/2026_04_08"
DIR_16 = "/Users/juikhankari/Downloads/data_04_16/2026_04_16"

# --- Aperture params from notebook (m3, 0.7m) ---
R1       = 10    # aperture radius px
R2       = 12    # background annulus outer radius px
SIGMA    = 100   # sep detection threshold
BW = BH  = 4    # sep background box size
MAX_AREA = 32   # max source ellipse area px²

M51_SNR = 15348.91


# ---------------------------------------------------------------------------
# Step 1 — Calibration frames
# ---------------------------------------------------------------------------

def load_stack(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    return np.array([fits.getdata(f).astype(float) for f in files]), files


def make_avg_bias():
    frames, _ = load_stack(f"{DIR_16}/bias*.fit")
    avg = np.mean(frames, axis=0)
    # gain = 1/EGAIN (same calculation as notebook)
    with fits.open(sorted(glob.glob(f"{DIR_16}/bias*.fit"))[0]) as h:
        gain = 1.0 / h[0].header["EGAIN"]
    sigma_readout = float(np.std(avg * gain))   # e-
    print(f"  Bias frames    : {len(frames)}")
    print(f"  Gain           : {gain:.4f} e-/ADU")
    print(f"  Readout noise  : {sigma_readout:.4f} e-")
    return avg, sigma_readout, gain


def make_avg_darks(avg_bias):
    """Returns dict: exptime -> mean dark (bias-subtracted)."""
    dark_files = sorted(glob.glob(f"{DIR_16}/dark*.fit"))
    by_time = {}
    for fn in dark_files:
        with fits.open(fn) as h:
            t = float(h[0].header["EXPTIME"])
            by_time.setdefault(t, []).append(h[0].data.astype(float))
    avg_darks = {t: np.mean(np.array(stack), axis=0) - avg_bias
                 for t, stack in by_time.items()}
    print(f"  Dark exptimes  : {sorted(avg_darks.keys())}")
    return avg_darks


def get_dark_for(exptime, avg_darks):
    """Return (avg_dark, t_dark) for the closest available dark exptime."""
    t_dark = min(avg_darks.keys(), key=lambda t: abs(t - exptime))
    return avg_darks[t_dark], t_dark


def make_mean_flat(filt, avg_bias, avg_darks):
    """Bias+dark correct and normalise flats for one filter."""
    flat_files = sorted(glob.glob(f"{DIR_16}/flat_{filt}_*.fit"))
    if not flat_files:
        raise FileNotFoundError(f"No flats for filter {filt}")
    corrected = []
    for fn in flat_files:
        with fits.open(fn) as h:
            t   = float(h[0].header["EXPTIME"])
            raw = h[0].data.astype(float)
        dark, _ = get_dark_for(t, avg_darks)
        cf = raw - avg_bias - dark
        corrected.append(cf / np.nanmean(cf))   # normalise to mean 1
    mean_flat = np.mean(corrected, axis=0)
    print(f"  Flat [{filt}]       : {len(flat_files)} frames")
    return mean_flat


# ---------------------------------------------------------------------------
# Step 2 — Calibrate + stack science images
# ---------------------------------------------------------------------------

def calibrate_and_stack(filt, avg_bias, avg_darks, mean_flat):
    sci_files = sorted(glob.glob(f"{DIR_08}/M3_{filt}_*.fit"))
    if not sci_files:
        raise FileNotFoundError(f"No science frames for {filt}")
    with fits.open(sci_files[0]) as h:
        exptime = float(h[0].header["EXPTIME"])
        header  = h[0].header

    dark, t_dark = get_dark_for(exptime, avg_darks)
    time_ratio   = exptime / t_dark

    stack = []
    for fn in sci_files:
        raw = fits.getdata(fn).astype(float)
        cal = (raw - avg_bias - dark * time_ratio) / mean_flat
        stack.append(cal)

    stacked = np.mean(stack, axis=0)
    n_images = len(stack)
    print(f"  Science [{filt}]   : {n_images} frames  exptime={exptime}s  dark={t_dark}s")
    return stacked, header, exptime, n_images


# ---------------------------------------------------------------------------
# Step 3 — Source extraction + flux/flux_err SNR (Lab 1.2 recipe)
# ---------------------------------------------------------------------------

def extract_and_snr(science, headers, exptimes, n_images_dict, sigma_readout):
    # Source detection on V (same as notebook)
    v_img = np.ascontiguousarray(science["V"], np.float64)
    mask  = np.isnan(v_img)
    bkg   = sep.Background(v_img, mask=mask, bw=BW, bh=BH)
    sub   = v_img - bkg.back()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sources = sep.extract(
            np.ascontiguousarray(sub, np.float64),
            SIGMA, err=bkg.globalrms, mask=mask, deblend_cont=1e-4
        )
    sources = sources[sources["a"] * sources["b"] * np.pi < MAX_AREA]
    print(f"  Detected {len(sources)} sources")

    source_data = {}
    for filt in ["B", "V"]:
        img     = np.ascontiguousarray(science[filt], np.float64)
        nan_msk = np.isnan(img)
        ones    = np.ones_like(img)
        t       = exptimes[filt]
        n_im    = n_images_dict[filt]

        counts      = sep.sum_circle(img,  sources["x"], sources["y"], R1, mask=nan_msk)[0]
        inner_area  = sep.sum_circle(ones, sources["x"], sources["y"], R1, mask=nan_msk)[0]
        bkg_counts  = sep.sum_circann(img,  sources["x"], sources["y"], R1, R2, mask=nan_msk)[0]
        outer_area  = sep.sum_circann(ones, sources["x"], sources["y"], R1, R2, mask=nan_msk)[0]
        bkg_per_pix = bkg_counts / outer_area

        # per-pixel annulus variance (Lab 1.2 approach)
        ny, nx = img.shape
        Y, X   = np.ogrid[:ny, :nx]
        var_bkg = []
        for i in tqdm(range(len(sources)), desc=f"Annulus var [{filt}]", leave=False):
            dR      = np.sqrt((X - sources["x"][i])**2 + (Y - sources["y"][i])**2)
            ann_pix = img[(dR >= R1) & (dR < R2) & ~nan_msk]
            var_bkg.append(float(np.var(ann_pix)) if ann_pix.size else 0.0)
        var_bkg = np.array(var_bkg)

        flux     = (counts - inner_area * bkg_per_pix) / t
        flux_err = np.sqrt(
            (counts + var_bkg * inner_area + n_im * inner_area * sigma_readout**2)
            / t**2
        )
        snr = flux / flux_err

        source_data[filt] = {"flux": flux, "flux_err": flux_err, "snr": snr, "exptime": t}

    return source_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Step 1: Calibration frames ===")
    avg_bias, sigma_readout, gain = make_avg_bias()
    avg_darks = make_avg_darks(avg_bias)

    mean_flat = {}
    for filt in ["B", "V"]:
        mean_flat[filt] = make_mean_flat(filt, avg_bias, avg_darks)

    print("\n=== Step 2: Calibrate + stack science images ===")
    science  = {}
    headers  = {}
    exptimes = {}
    n_images = {}
    for filt in ["B", "V"]:
        stacked, hdr, t, n = calibrate_and_stack(
            filt, avg_bias, avg_darks, mean_flat[filt]
        )
        science[filt]  = stacked
        headers[filt]  = hdr
        exptimes[filt] = t
        n_images[filt] = n

    print("\n=== Step 3: Source extraction + SNR ===")
    source_data = extract_and_snr(
        science, headers, exptimes, n_images, sigma_readout
    )

    print("\n=== Results ===")
    for filt in ["B", "V"]:
        snr   = source_data[filt]["snr"]
        valid = snr[np.isfinite(snr) & (snr > 0)]
        print(f"\n  m3 [{filt}]  (exptime={exptimes[filt]:.0f}s, n_images={n_images[filt]})")
        print(f"    Valid sources : {len(valid)}")
        print(f"    Median SNR    : {np.median(valid):.2f}")
        print(f"    Mean SNR      : {np.mean(valid):.2f}")
        print(f"    Max SNR       : {np.max(valid):.2f}")
        print(f"    Min SNR       : {np.min(valid):.2f}")

    print(f"\n=== Comparison to M51 spiral arm SNR = {M51_SNR:.2f} ===")
    for filt in ["B", "V"]:
        snr   = source_data[filt]["snr"]
        valid = snr[np.isfinite(snr) & (snr > 0)]
        ratio = np.median(valid) / M51_SNR
        print(f"  m3 [{filt}] median / M51 = {ratio:.4f}")

    # Histogram
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
    print("\nSaved plot → snr_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
