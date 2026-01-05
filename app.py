# Exoplanet Finder v3.4 - 19 FITS, NASA-style stitching, safe BLS

import os
import io
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings("ignore")

try:
    from astropy.io import fits
    from astropy.timeseries import BoxLeastSquares
    HAS_ASTROPY = True
    print("Astropy loaded")
except ImportError:
    HAS_ASTROPY = False
    print("Astropy NOT available")


def memory_cleanup():
    gc.collect()


def read_fits_safe(path, max_points=60000):
    """Read TIME and FLUX-like column from Kepler/TESS FITS safely, with downsample."""
    try:
        with fits.open(path, memmap=False) as hdul:
            for h in hdul[1:3]:
                if hasattr(h, 'columns') and h.data is not None:
                    cols = h.columns.names
                    if cols is None:
                        continue
                    flux_col = next((c for c in cols if any(x in c.upper() for x in ['PDCSAP_FLUX', 'SAP_FLUX', 'FLUX'])), None)
                    time_col = next((c for c in cols if 'TIME' in c.upper()), None)
                    if flux_col and time_col:
                        time = np.array(h.data[time_col], dtype=float)
                        flux = np.array(h.data[flux_col], dtype=float)
                        mask = np.isfinite(time) & np.isfinite(flux)
                        time, flux = time[mask], flux[mask]
                        if len(time) > max_points:
                            step = max(1, len(time)//max_points)
                            time, flux = time[::step], flux[::step]
                        if len(time) > 20:
                            return time, flux, flux_col
        return None, None, None
    except Exception:
        return None, None, None


def smart_detrend(time, flux, flux_type):
    """Light detrend for PDCSAP, stronger for raw/SAP."""
    n = len(flux)
    if n < 30:
        med = np.nanmedian(flux)
        std = np.nanstd(flux) or 1.0
        return (flux - med) / std, "normalize"

    if flux_type and 'PDCSAP' in flux_type.upper():
        med = np.nanmedian(flux)
        std = np.nanstd(flux) or 1.0
        return (flux - med) / std, "pdcsap"

    try:
        win = min(201, max(11, (n//25) | 1))
        trend = savgol_filter(flux, win, 2)
        rel = flux / trend - 1.0
        return rel, "savgol"
    except Exception:
        k = max(5, n//25)
        trend = median_filter(flux, k)
        rel = flux / trend - 1.0
        return rel, "median"


def nasa_ml_score(sde, depth, n_transits, rms, baseline_days):
    """Heuristic confidence 0-1 inspired by NASA vetting metrics."""
    score = 0.0
    score += min(35.0, sde * 3.5)
    score += min(25.0, depth * 5000.0)
    score += min(20.0, n_transits * 2.0)
    score += min(10.0, np.log10(baseline_days + 1.0) * 4.0)
    score += min(10.0, 1.0/(rms + 1e-3))
    return min(1.0, score / 100.0)


def get_status(conf):
    if conf > 85:
        return "🟢 CONFIRMED-LIKE"
    if conf > 65:
        return "🟡 STRONG CANDIDATE"
    return "🔵 WEAK SIGNAL"


def ultimate_exoplanet_analyzer(files, sde_thresh=6.0, min_period=0.2, max_period=500.0):
    print(f"Analysis: {len(files)} files")
    if not files or not HAS_ASTROPY:
        return "❌ Need FITS files and astropy", None

    segments = []
    quarter_labels = []

    for i, fobj in enumerate(files):
        fname = os.path.basename(fobj.name)
        t, flux, ftype = read_fits_safe(fobj.name)
        if t is None or len(t) < 30:
            print(f"Skip {fname}")
            continue
        rel, method = smart_detrend(t, flux, ftype)
        segments.append((t, rel))
        qlabel = fname.split('-')[1][:4] if '-' in fname else f"Q{i+1}"
        quarter_labels.append(f"{qlabel}: {len(t)} pts ({method})")
        print(f"OK {fname}: {len(t)} pts")

    if len(segments) < 2:
        return "❌ Need at least 2 valid quarters", None

    # NASA-like quarter stitching
    seg_sorted = sorted(segments, key=lambda x: np.nanmedian(x[0]))
    all_t, all_f = [], []
    prev_end = None
    for t_seg, f_seg in seg_sorted:
        if prev_end is None:
            all_t.extend(t_seg)
            all_f.extend(f_seg)
            prev_end = t_seg[-1]
        else:
            overlap_start = max(t_seg[0], prev_end - 5.0)
            mask_ov = t_seg >= overlap_start
            if np.any(mask_ov) and len(all_t) > 100:
                prev_ov = np.array(all_f[-100:])
                new_ov = np.array(f_seg[mask_ov][:100])
                if np.sum(np.isfinite(prev_ov)) > 10 and np.sum(np.isfinite(new_ov)) > 10:
                    scale = np.nanmedian(prev_ov) / np.nanmedian(new_ov)
                    f_seg = f_seg * scale
            all_t.extend(t_seg)
            all_f.extend(f_seg)
            prev_end = t_seg[-1]

    time_all = np.array(all_t, dtype=float)
    flux_all = np.array(all_f, dtype=float)
    order = np.argsort(time_all)
    time_all, flux_all = time_all[order], flux_all[order]

    baseline = time_all[-1] - time_all[0]
    if len(time_all) < 200 or baseline <= 0:
        return "❌ Too few points after stitching", None

    # Normalize globally
    med = np.nanmedian(flux_all)
    flux_rel = flux_all/med - 1.0 if med != 0 else flux_all

    # Safe BLS grid
    min_p = max(min_period, baseline/1000.0)
    max_p = min(max_period, baseline/2.0)
    if max_p <= min_p:
        return f"❌ Period range invalid: min={min_p:.3f}, max={max_p:.3f}", None

    if baseline > 80.0:
        periods = np.logspace(np.log10(min_p), np.log10(max_p), 15000)
    else:
        periods = np.linspace(min_p, max_p, 12000)

    max_duration = min(min_p/3.0, 0.25)
    min_duration = max(0.0005, max_duration/20.0)
    if max_duration <= min_duration:
        max_duration = min_duration * 2.0
    durations = np.linspace(min_duration, max_duration, 8)

    print(f"BLS grid: {len(periods)} periods × {len(durations)} durations")

    bls = BoxLeastSquares(time_all, flux_rel)
    power_list = []
    for d in durations:
        try:
            if d >= periods[0]:
                continue
            res = bls.power(periods, d)
            power_list.append(res.power)
        except Exception as e:
            print(f"Duration {d:.5f} skipped: {e}")
            continue

    if not power_list:
        return "❌ BLS grid failed. Try larger Min Period.", None

    power_max = np.max(power_list, axis=0)
    peak_idx = int(np.argmax(power_max))
    best_period = float(periods[peak_idx])

    # SDE
    exclude_w = max(20, int(len(periods)*0.01))
    lo = max(0, peak_idx - exclude_w)
    hi = min(len(periods), peak_idx + exclude_w)
    noise = np.concatenate([power_max[:lo], power_max[hi:]]) if hi > lo else power_max
    if len(noise) > 20:
        sde = float((power_max[peak_idx] - np.median(noise)) / (np.std(noise) or 1.0))
    else:
        sde = 0.0

    phase = ((time_all - time_all[0])/best_period + 0.5) % 1.0
    depth = float(-np.nanmin(flux_rel))
    rms = float(np.nanstd(flux_rel))
    n_trans = float(baseline / best_period)

    ml_p = float(nasa_ml_score(sde, depth, n_trans, rms, baseline))
    confidence = float(min(100.0, sde*7.8 + ml_p*28.0))
    status = get_status(confidence)

    quarters_str = " | ".join(quarter_labels[:6])
    if len(quarter_labels) > 6:
        quarters_str += " ..."

    result = (
        f"{status}
"
        f"Confidence: {confidence:.0f}%\n\n"
        f"SDE: {sde:.3f} (thr {sde_thresh})\n"
        f"Best period: {best_period:.6f} d\n"
        f"Depth: {depth:.5f} ({depth*100:.3f}%)\n"
        f"Transits (est): {n_trans:.1f}\n"
        f"RMS: {rms:.6f}\n\n"
        f"Stitched quarters: {len(segments)}\n"
        f"{quarters_str}\n"
        f"Points: {len(time_all):,} | Baseline: {baseline:.1f} d"
    )

    # Plots: LC + BLS + phase-folded in one image
    fig = plt.figure(figsize=(14, 11), facecolor='black')
    ax1 = plt.subplot(3,1,1)
    ax1.plot(time_all, flux_rel, 'lightblue', lw=0.6)
    ax1.set_ylabel('Flux', color='white')
    ax1.set_title('Detrended Light Curve (stitched)', color='cyan')
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(3,1,2)
    ax2.semilogx(periods, power_max, 'gold', lw=1.2)
    ax2.axvline(best_period, color='lime', ls='--', lw=2, label=f'{best_period:.4f} d')
    thr_line = np.median(power_max) + sde_thresh*np.std(power_max)
    ax2.axhline(thr_line, color='orange', ls=':', lw=1.5, label='SDE threshold')
    ax2.set_ylabel('BLS Power', color='white')
    ax2.set_title('BLS Periodogram', color='cyan')
    ax2.legend(facecolor='black', framealpha=0.8)
    ax2.grid(alpha=0.3)

    phase_center = (phase - 0.5) * best_period
    order_p = np.argsort(phase_center)
    ax3 = plt.subplot(3,1,3)
    ax3.plot(phase_center[order_p], flux_rel[order_p], 'lightcoral', lw=0, marker='.', ms=2, alpha=0.7)
    ax3.set_xlabel('Time from transit center (days)', color='white')
    ax3.set_ylabel('Flux', color='white')
    ax3.set_title('Phase-folded light curve', color='cyan')
    ax3.grid(alpha=0.3)

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor('#0a0a1a')
        ax.tick_params(colors='white')

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=200, facecolor='black', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')

    memory_cleanup()
    return result, img


# ---- Gradio UI ----

css = """
body { background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 60%, #2a2a5a 100%); color: #e8e8ff; }
.gr-button { background: linear-gradient(45deg, #00d4ff, #0099cc) !important; border-radius: 18px !important; font-weight: bold !important; }
.gr-file, .gr-slider { background: rgba(15,20,40,0.95) !important; border-radius: 12px !important; }
"""

with gr.Blocks(css=css, title="Exoplanet Finder v3.4") as demo:
    gr.Markdown("# 🔭 Exoplanet Finder v3.4 — NASA-style BLS")
    with gr.Row():
        file_input = gr.File(file_count="multiple", file_types=[".fits"], label="FITS files (one candidate, many quarters)")
        with gr.Column():
            sde_slider = gr.Slider(4, 12, value=6.0, step=0.5, label="SDE threshold")
            min_p_slider = gr.Slider(0.1, 10, value=0.2, step=0.1, label="Min period (days)")
            max_p_slider = gr.Slider(20, 1000, value=365.0, step=5.0, label="Max period (days)")
    analyze_btn = gr.Button("🔬 Run BLS analysis", variant="primary")
    out_text = gr.Textbox(label="Results", lines=14)
    out_img = gr.Image(label="Plots")

    analyze_btn.click(
        ultimate_exoplanet_analyzer,
        inputs=[file_input, sde_slider, min_p_slider, max_p_slider],
        outputs=[out_text, out_img]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
