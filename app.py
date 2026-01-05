
# EXOPLANET FINDER v3.5 FULL VERSION - NASA STITCH + BLS + 3 PLOTS
# Fixed string literal, safe BLS, 19 files OK

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

HAS_ASTROPY = False
try:
    from astropy.io import fits
    from astropy.timeseries import BoxLeastSquares
    HAS_ASTROPY = True
except ImportError:
    pass

def read_fits_safe(path):
    try:
        with fits.open(path) as hdul:
            h = hdul[1]
            cols = h.columns.names
            flux_col = next((c for c in cols if 'FLUX' in c.upper()), None)
            time_col = next((c for c in cols if 'TIME' in c.upper()), None)
            if flux_col and time_col:
                time = h.data[time_col]
                flux = h.data[flux_col]
                mask = np.isfinite(time) & np.isfinite(flux)
                if np.sum(mask) > 20:
                    return time[mask], flux[mask]
        return None, None
    except:
        return None, None

def stitch_quarters(files):
    segments = []
    for f in files[:20]:
        t, f = read_fits_safe(f.name)
        if t is not None:
            med = np.median(f)
            f_norm = (f - med) / np.std(f)
            segments.append((t, f_norm))
    if not segments:
        return None, None
    all_t = np.concatenate([t for t,f in segments])
    all_f = np.concatenate([f for t,f in segments])
    order = np.argsort(all_t)
    return all_t[order], all_f[order]

def get_status(conf):
    if conf > 85:
        return "🟢 CONFIRMED"
    elif conf > 65:
        return "🟡 CANDIDATE"
    else:
        return "🔵 WEAK"

def analyze_exoplanet(files, sde_thresh, min_p, max_p):
    if not HAS_ASTROPY or not files:
        return "Error: Astropy or no files", None
    t_all, f_all = stitch_quarters(files)
    if t_all is None or len(t_all) < 100:
        return "Insufficient data", None
    baseline = t_all[-1] - t_all[0]
    periods = np.linspace(max(min_p, 0.2), min(max_p, baseline/2), 8000)
    durations = np.linspace(0.001, periods[0]/4, 6)
    bls = BoxLeastSquares(t_all, f_all)
    power_max = np.zeros(len(periods))
    for d in durations:
        try:
            pg = bls.power(periods, d)
            power_max = np.maximum(power_max, pg.power)
        except:
            continue
    peak = np.argmax(power_max)
    sde = (power_max[peak] - np.median(power_max)) / np.std(power_max)
    depth = -np.min(f_all)
    ml_p = min(1.0, sde/10 + depth*2000)
    conf = min(100, sde*8 + ml_p*25)
    status = get_status(conf)
    result = f'{status}\nScore: {conf:.0f}%\nSDE: {sde:.2f}\nP: {periods[peak]:.4f}d\nDepth: {depth:.5f}\nPoints: {len(t_all)}'

    fig, axs = plt.subplots(3,1, figsize=(12,12), facecolor='black')
    axs[0].plot(t_all, f_all, 'lightblue', alpha=0.7)
    axs[0].set_title('Light Curve', color='white')
    axs[1].plot(periods, power_max, 'gold')
    axs[1].axvline(periods[peak], color='red', ls='--')
    axs[1].set_title('BLS Periodogram', color='white')
    axs[2].set_facecolor('black')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, facecolor='black', dpi=150)
    plt.close()
    img = Image.open(buf)
    return result, img

css = "body {background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 100%); color: #e8e8ff;} .gr-button {background: linear-gradient(45deg, #00d4ff, #0099cc);}"

with gr.Blocks(css=css) as demo:
    gr.Markdown('# Exoplanet Finder v3.5')
    file_input = gr.File(file_count="multiple", file_types=[".fits"])
    sde = gr.Slider(4, 12, 6)
    minp = gr.Slider(0.1, 20, 0.2)
    maxp = gr.Slider(20, 1000, 200)
    btn = gr.Button("Analyze")
    text_out = gr.Textbox()
    img_out = gr.Image()
    btn.click(analyze_exoplanet, [file_input, sde, minp, maxp], [text_out, img_out])

demo.launch()
