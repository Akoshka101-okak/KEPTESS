# ORIGINAL Exoplanet Finder UI + AI comparison (3 graphs + detailed results)
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from PIL import Image
import gradio as gr
import joblib

# AI Model
try:
    AI_MODEL = joblib.load('planet_longquiet.pkl')
    print('AI loaded')
except:
    AI_MODEL = None

try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except:
    HAS_SAVGOL = False

def choose_flux_column(colnames):
    names_up = [c.upper() for c in colnames]
    for prefer in ('PDCSAP_FLUX', 'SAP_FLUX', 'FLUX'):
        if prefer in names_up:
            return colnames[names_up.index(prefer)]
    return None

def read_time_flux_from_hdu(hdu):
    cols = hdu.columns.names
    flux_col = choose_flux_column(cols)
    if flux_col is None or 'TIME' not in [c.upper() for c in cols]:
        return None, None
    time_name = next((c for c in cols if c.upper() == 'TIME'), None)
    flux_name = next((c for c in cols if c.upper() == flux_col.upper()), None)
    time = np.array(hdu.data[time_name], dtype=float)
    flux = np.array(hdu.data[flux_name], dtype=float)
    return time, flux

def read_fits_file_auto(path):
    try:
        with fits.open(path, memmap=False) as hdul:
            for h in hdul:
                if hasattr(h, 'data') and h.data is not None:
                    t, f = read_time_flux_from_hdu(h)
                    if t is not None and f is not None:
                        return t, f
            try:
                t, f = read_time_flux_from_hdu(hdul[1])
                return t, f
            except:
                return None, None
    except:
        return None, None

def clean_and_normalize_segment(time, flux):
    mask = np.isfinite(time) & np.isfinite(flux)
    time = np.array(time[mask], dtype=float)
    flux = np.array(flux[mask], dtype=float)
    if len(time) == 0:
        return None, None
    order = np.argsort(time)
    time = time[order]
    flux = flux[order]
    med = np.nanmedian(flux)
    if med == 0 or not np.isfinite(med):
        med = 1.0
    return time, flux / med

def stitch_segments(segments):
    segs = [(np.nanmedian(t), t, f) for t, f in segments if t is not None and f is not None and len(t) > 0]
    if len(segs) == 0:
        return None, None
    segs.sort(key=lambda x: x[0])
    aligned = []
    base_time, base_flux = segs[0][1], segs[0][2]
    aligned.append((base_time, base_flux))
    for _, t, f in segs[1:]:
        overlap_mask_in_new = (t >= aligned[-1][0][0]) & (t <= aligned[-1][0][-1])
        overlap_mask_in_old = (aligned[-1][0] >= t[0]) & (aligned[-1][0] <= t[-1])
        if np.any(overlap_mask_in_new) and np.any(overlap_mask_in_old):
            new_med = np.nanmedian(f[overlap_mask_in_new])
            old_med = np.nanmedian(aligned[-1][1][overlap_mask_in_old])
            if np.isfinite(new_med) and np.isfinite(old_med) and old_med != 0:
                f *= old_med / new_med
        aligned.append((t, f))
    time_all = np.concatenate([t for t, f in aligned])
    flux_all = np.concatenate([f for t, f in aligned])
    order = np.argsort(time_all)
    time_all = time_all[order]
    flux_all = flux_all[order]
    med_total = np.nanmedian(flux_all)
    if med_total == 0 or not np.isfinite(med_total):
        med_total = 1.0
    flux_all /= med_total
    return time_all, flux_all

def detrend_flux(time, flux):
    n = len(flux)
    if n < 10:
        return flux - np.nanmedian(flux), np.ones_like(flux)
    if HAS_SAVGOL:
        win = min(201, max(7, (n // 50) | 1))
        try:
            trend = savgol_filter(flux, win, 2, mode='interp')
        except:
            k = max(3, n // 50)
            from scipy.ndimage import median_filter
            trend = median_filter(flux, k, mode='nearest')
    else:
        k = max(3, n // 50)
        pad = k//2
        fpad = np.pad(flux, pad_width=pad, mode='edge')
        trend = np.array([np.median(fpad[i:i+k]) for i in range(n)])
    mask = np.isfinite(trend) & (np.abs(trend) > 0)
    if not np.all(mask):
        fallback = np.nanmedian(trend[mask]) if np.any(mask) else 1.0
        trend[~mask] = fallback
    return flux / trend - 1.0, trend

def compute_sde(power, peak_index, exclude_width=50):
    p = np.array(power, dtype=float)
    n = len(p)
    mask = np.ones(n, dtype=bool)
    lo = max(0, peak_index - exclude_width)
    hi = min(n, peak_index + exclude_width)
    mask[lo:hi] = False
    noise = p[mask]
    median = np.median(noise) if len(noise) >= 10 else np.median(p)
    std = np.std(noise) if len(noise) >= 10 else np.std(p)
    return (p[peak_index] - median) / std if std > 0 else 0.0

def analyze_exoplanet(file_objs, sde_threshold=7.5):
    if not file_objs:
        return 'Upload FITS files', None, None

    segments, failed = [], []
    for f in file_objs:
        t, flux = read_fits_file_auto(f.name)
        if t is None or flux is None or len(t) == 0:
            failed.append(os.path.basename(f.name))
            continue
        t_clean, f_clean = clean_and_normalize_segment(t, flux)
        if t_clean is None:
            failed.append(os.path.basename(f.name))
            continue
        segments.append((t_clean, f_clean))

    if not segments:
        return f'No valid data. Failed: {', '.join(failed)}', None, None

    time_all, flux_all = stitch_segments(segments)
    if time_all is None or len(time_all) < 50:
        return 'Too few points', None, None

    mask = np.isfinite(time_all) & np.isfinite(flux_all)
    time_all = time_all[mask]
    flux_all = flux_all[mask]

    flux_rel, trend = detrend_flux(time_all, flux_all)

    total_span = time_all[-1] - time_all[0]
    max_period = min(500.0, total_span / 2.0)
    n_periods = min(40000, max(3000, int(total_span * 50)))
    periods = np.linspace(0.3, max_period, n_periods)
    durations = np.linspace(0.005, 0.2, 12)

    bls = BoxLeastSquares(time_all, flux_rel)
    power_matrix = np.zeros((len(durations), len(periods)))
    for i, d in enumerate(durations):
        power_matrix[i] = bls.power(periods, d).power

    power_per_period = np.max(power_matrix, axis=0)
    idx_peak = np.argmax(power_per_period)
    best_period = periods[idx_peak]
    best_duration = durations[np.argmax(power_matrix[:, idx_peak])]
    sde_val = compute_sde(power_per_period, idx_peak)
    detected = sde_val >= sde_threshold

    # AI FEATURES (6 exactly)
    log_period = np.log10(best_period)
    depth = np.nanmax(np.abs(flux_rel))
    duration = best_duration
    sde = sde_val
    planet_radius = np.sqrt(depth) * 1.3
    multi = 0

    ai_features = np.array([[log_period, depth, duration, sde, planet_radius, multi]])
    planet_proba = 0
    if AI_MODEL is not None:
        try:
            planet_proba = AI_MODEL.predict_proba(ai_features)[0][1] * 100
        except:
            pass

    # ORIGINAL 3 PLOTS
    fig = plt.figure(figsize=(12, 10), facecolor='black')

    # 1. Time domain
    ax1 = plt.subplot(3,1,1)
    ax1.plot(time_all, flux_rel, '.', ms=0.5, color='cyan')
    ax1.set_ylabel('Flux rel')
    ax1.grid(alpha=0.3)
    ax1.set_title('Detrended lightcurve')

    # 2. Periodogram
    ax2 = plt.subplot(3,1,2)
    ax2.plot(periods, power_per_period, 'm-', lw=0.8)
    noise_mask = np.ones_like(power_per_period, dtype=bool)
    w = max(1, int(len(periods)*0.002))
    lo, hi = max(0, idx_peak-w), min(len(periods), idx_peak+w)
    noise_mask[lo:hi] = False
    noise_med = np.median(power_per_period[noise_mask])
    noise_std = np.std(power_per_period[noise_mask])
    det_level = noise_med + sde_threshold * noise_std if noise_std > 0 else noise_med
    ax2.axvline(best_period, color='red', ls='--', lw=2, label=f'P={best_period:.3f}d')
    ax2.axhline(det_level, color='orange', ls=':', lw=2, label=f'SDE thresh={sde_threshold}')
    ax2.set_ylabel('BLS Power')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Phase folded
    ax3 = plt.subplot(3,1,3)
    phase = ((time_all - time_all[0]) / best_period) % 1
    phase = (phase + 0.5) % 1
    order = np.argsort(phase)
    phase_days = (phase[order] - 0.5) * best_period
    flux_sorted = flux_rel[order]
    ax3.plot(phase_days, flux_sorted, '.', ms=0.5, alpha=0.6, color='cyan')
    nbins = 50
    bins = np.linspace(-0.5*best_period, 0.5*best_period, nbins+1)
    inds = np.digitize(phase_days, bins) - 1
    binned = [np.nanmedian(flux_sorted[inds==i]) if np.any(inds==i) else np.nan for i in range(nbins)]
    ax3.plot(0.5*(bins[:-1]+bins[1:]), binned, 'r-', lw=3)
    ax3.set_xlim(-0.2*best_period, 0.2*best_period)
    ax3.set_xlabel('Days from transit center')
    ax3.set_ylabel('Flux rel')
    ax3.grid(alpha=0.3)
    ax3.set_title(f'Phase folded (P={best_period:.3f}d)')

    plt.tight_layout()
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', dpi=150, facecolor='black')
    plt.close()
    img1 = Image.open(buf1).convert('RGB')

    # DETAILED RESULTS TABLE
    status = 'DETECTED' if detected else 'NOT CONFIRMED'
    result = f'''BLS ALGORITHM:
Status: {status}
Period: {best_period:.6f} days
Duration: {best_duration:.4f} fraction
SDE: {sde_val:.3f} (threshold {sde_threshold})
Depth: {depth:.6f}
Files: {len(segments)}/{len(file_objs)} OK
Span: {total_span:.1f} days

AI MODEL:
Planet probability: {planet_proba:.1f}%

COMPARISON:
- BLS SDE >7.5 = strong candidate
- AI >50% = likely planet
- Both low = probable false positive'''

    return result, img1

css = """
body { background: #0b0c10; color: #c5c6c7; font-family: 'Segoe UI', sans-serif; }
h1 { color: #66fcf1; text-align: center; }
.gr-button { background: #1f2833; color: #66fcf1; border-radius: 8px; }
.gr-button:hover { background: #45a29e; }
.gr-textbox { background: rgba(31,40,51,0.95); border-radius: 8px; }
.gr-image { background: rgba(31,40,51,0.8); border-radius: 8px; }
"""

with gr.Blocks(css=css, title='Exoplanet Finder - BLS vs AI') as app:
    gr.Markdown('# 🚀 Exoplanet Finder - BLS Algorithm vs AI Model')
    gr.Markdown('Upload Kepler/TESS FITS files. Compare **BLS SDE** (classical) vs **AI probability** side-by-side.')

    with gr.Row():
        file_input = gr.File(file_count='multiple', file_types=['.fits'], label='FITS files (multiple OK)')

    sde_slider = gr.Slider(5.0, 12.0, value=7.5, step=0.1, label='SDE threshold')

    analyze_btn = gr.Button('🔍 Analyze + AI Prediction', variant='primary', size='lg')

    output_text = gr.Textbox(label='Detailed Results (BLS vs AI)', lines=12, max_lines=15)
    output_graphs = gr.Image(label='3 Plots: Time | Periodogram | Phase-folded', type='pil')

    analyze_btn.click(
        analyze_exoplanet, 
        inputs=[file_input, sde_slider], 
        outputs=[output_text, output_graphs]
    )

if __name__ == '__main__':
    app.launch(server_name='0.0.0.0', server_port=7860, share=True)