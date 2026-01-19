# Exoplanet Finder app.py FIXED - AI_MODEL 6 features + NameError fix
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from PIL import Image
import gradio as gr
import joblib

# Load AI model (handles missing file)
try:
    AI_MODEL = joblib.load('planet_longquiet.pkl')
    print('AI_MODEL loaded OK')
except:
    AI_MODEL = None
    print('No AI model - using BLS only')

# Savgol filter
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
    flux_norm = flux / med
    return time, flux_norm

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
                scale = old_med / new_med
                f = f * scale
        aligned.append((t, f))
    time_all = np.concatenate([t for t, f in aligned])
    flux_all = np.concatenate([f for t, f in aligned])
    order = np.argsort(time_all)
    time_all = time_all[order]
    flux_all = flux_all[order]
    med_total = np.nanmedian(flux_all)
    if med_total == 0 or not np.isfinite(med_total):
        med_total = 1.0
    flux_all = flux_all / med_total
    return time_all, flux_all

def detrend_flux(time, flux):
    n = len(flux)
    if n < 10:
        trend = np.ones_like(flux)
    else:
        if HAS_SAVGOL:
            win = min(201, max(7, (n // 50) | 1))
            try:
                trend = savgol_filter(flux, window_length=win, polyorder=2, mode='interp')
            except:
                k = max(3, n // 50)
                from scipy.ndimage import median_filter
                trend = median_filter(flux, size=k, mode='nearest')
        else:
            k = max(3, n // 50)
            pad = k//2
            fpad = np.pad(flux, pad_width=pad, mode='edge')
            trend = np.array([np.median(fpad[i:i+k]) for i in range(len(flux))])
    mask = np.isfinite(trend) & (np.abs(trend) > 0)
    if not np.all(mask):
        fallback = np.nanmedian(trend[mask]) if np.any(mask) else 1.0
        trend[~mask] = fallback
    flux_rel = flux / trend - 1.0
    return flux_rel, trend

def compute_sde(power, peak_index, exclude_width=50):
    p = np.array(power, dtype=float)
    n = len(p)
    mask = np.ones(n, dtype=bool)
    lo = max(0, peak_index - exclude_width)
    hi = min(n, peak_index + exclude_width)
    mask[lo:hi] = False
    noise = p[mask]
    if len(noise) < 10:
        median = np.median(p)
        std = np.std(p)
    else:
        median = np.median(noise)
        std = np.std(noise)
    if std == 0:
        return 0.0
    return (p[peak_index] - median) / std

def analyze_exoplanet(file_objs, sde_threshold=7.5, min_period=0.3, max_period_user=None):
    if not file_objs or len(file_objs) == 0:
        return 'Upload FITS files.', None

    segments = []
    failed = []
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

    if len(segments) == 0:
        return f'No valid data. Failed: {', '.join(failed)}', None

    time_all, flux_all = stitch_segments(segments)
    if time_all is None or len(time_all) < 10:
        return 'Too few points after stitching.', None

    mask = np.isfinite(time_all) & np.isfinite(flux_all)
    time_all = time_all[mask]
    flux_all = flux_all[mask]
    if len(time_all) < 10:
        return 'Too few points after cleaning.', None

    flux_rel, trend = detrend_flux(time_all, flux_all)

    total_span = time_all[-1] - time_all[0]
    if total_span <= 0:
        return 'Invalid time stamps.', None

    if max_period_user is None:
        max_period = max(min(500.0, total_span / 2.0), 1.0)
    else:
        max_period = min(max_period_user, total_span/2.0)

    n_periods = min(40000, max(3000, int(total_span * 50)))
    periods = np.linspace(min_period, max_period, n_periods)
    durations = np.linspace(0.005, 0.2, 12)

    bls = BoxLeastSquares(time_all, flux_rel)
    power_matrix = np.zeros((len(durations), len(periods)))
    for i, d in enumerate(durations):
        res = bls.power(periods, d)
        power_matrix[i, :] = res.power

    power_per_period = np.max(power_matrix, axis=0)
    idx_peak = np.argmax(power_per_period)
    best_period = periods[idx_peak]
    idx_best_dur = np.argmax(power_matrix[:, idx_peak])
    best_duration = durations[idx_best_dur]
    sde = compute_sde(power_per_period, idx_peak)
    detected = sde >= sde_threshold

    # AI MODEL FEATURES (FIXED 6 features)
    log_period = np.log10(best_period)
    depth = np.nanmax(np.abs(flux_rel))
    duration = best_duration
    sde_val = sde
    planet_radius = np.sqrt(depth) * 1.3  # R_earth scale
    multi = 0  # single system

    ai_features = np.array([[log_period, depth, duration, sde_val, planet_radius, multi]])

    planet_proba = 0
    if AI_MODEL is not None and hasattr(AI_MODEL, 'predict_proba'):
        try:
            planet_proba = AI_MODEL.predict_proba(ai_features)[0][1] * 100
        except Exception as e:
            planet_proba = 0
            print(f'AI error: {e}')

    # Plots (3 panels)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # 1. Time series
    ax1.plot(time_all, flux_rel, '.', markersize=0.5)
    ax1.set_ylabel('Flux rel')
    ax1.grid(alpha=0.3)

    # 2. Periodogram
    noise_mask = np.ones_like(power_per_period, dtype=bool)
    w = max(1, int(len(periods)*0.002))
    lo = max(0, idx_peak - w)
    hi = min(len(periods), idx_peak + w)
    noise_mask[lo:hi] = False
    noise_median = np.median(power_per_period[noise_mask])
    noise_std = np.std(power_per_period[noise_mask])
    detection_level = noise_median + sde_threshold * noise_std if noise_std > 0 else noise_median
    ax2.plot(periods, power_per_period, linewidth=0.8)
    ax2.axvline(best_period, color='red', linestyle='--', label=f'P={best_period:.2f}d')
    ax2.axhline(detection_level, color='orange', linestyle=':', label=f'SDE={sde:.1f}')
    ax2.set_xlabel('Period (days)')
    ax2.set_ylabel('BLS Power')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Phase-folded
    phase = ((time_all - time_all[0]) / best_period) % 1.0
    phase = (phase + 0.5) % 1.0
    order = np.argsort(phase)
    phase_sorted = phase[order]
    flux_sorted = flux_rel[order]
    phase_days = (phase_sorted - 0.5) * best_period
    ax3.plot(phase_days, flux_sorted, '.', markersize=0.5, alpha=0.6)
    nbins = 50
    bins = np.linspace(-0.5*best_period, 0.5*best_period, nbins+1)
    inds = np.digitize(phase_days, bins) - 1
    binned = [np.nanmedian(flux_sorted[inds == i]) if np.any(inds==i) else np.nan for i in range(nbins)]
    ax3.plot(0.5*(bins[:-1]+bins[1:]), binned, 'r-', linewidth=2)
    ax3.set_xlim(-0.2*best_period, 0.2*best_period)
    ax3.set_xlabel('Phase (days from transit center)')
    ax3.set_ylabel('Flux rel')
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor='black', edgecolor='none')
    plt.close()
    buf.seek(0)
    img = Image.open(buf).convert('RGB')

    # Result text
    status = 'DETECTED' if detected else 'NOT CONFIRMED'
    result = f'''{status} | SDE: {sde:.2f} | P: {best_period:.3f}d
AI Planet Probability: {planet_proba:.1f}%
Files OK: {len(segments)}/{len(file_objs)}'''

    return result, img

css = """
body { background-color: #0b0c10; color: #c5c6c7; }
.gr-button { background-color: #1f2833; color: #66fcf1; }
.gr-button:hover { background-color: #45a29e; }
.gr-textbox, .gr-image { background-color: rgba(31,40,51,0.95); border-radius: 8px; }
"""

with gr.Blocks(css=css) as app:
    gr.Markdown('# рџљЂ Exoplanet Finder - Kepler/TESS FITS + AI')
    with gr.Row():
        file_input = gr.File(file_count='multiple', file_types=['.fits'], label='Upload FITS files')
    analyze_btn = gr.Button('Analyze', variant='primary')
    output_text = gr.Textbox(label='Results', lines=5)
    output_img = gr.Image(label='Lightcurve + Periodogram + Phase-fold')

    analyze_btn.click(analyze_exoplanet, inputs=[file_input], outputs=[output_text, output_img])

if __name__ == '__main__':
    app.launch(server_name='0.0.0.0', server_port=7860)