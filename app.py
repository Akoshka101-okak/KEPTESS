# EXOPLANET FINDER v3.7 - SINGLE FILES OK + NASA SORT
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
import io
import warnings
warnings.filterwarnings("ignore")

HAS_ASTROPY = False
try:
    from astropy.io import fits
    from astropy.timeseries import BoxLeastSquares
    HAS_ASTROPY = True
except:
    pass

def extract_quarter(fname):
    parts = fname.split('-')
    if len(parts) > 1 and len(parts[1]) >= 8:
        try:
            year = int(parts[1][:4])
            doy = int(parts[1][4:7])
            quarter = ((doy - 1) // 90) + 1
            return quarter * 10000 + year * 100 + doy
        except:
            pass
    return 0

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
                if np.sum(mask) > 20:  # ✅ Минимум 20 точек
                    return time[mask], flux[mask]
    except:
        pass
    return None, None

def stitch_nasa_order(files):
    segments = []
    for f in files:
        fname = os.path.basename(f.name)
        t, f = read_fits_safe(f.name)
        if t is not None:
            med = np.median(f)
            f_norm = (f - med) / np.std(f)
            sort_key = extract_quarter(fname)
            segments.append((t, f_norm, fname, sort_key))
    
    if not segments:
        return None, None
    
    # ✅ Сортировка по кварталу (даже для 1 файла)
    segments.sort(key=lambda x: x[3])
    
    all_t, all_f = [], []
    prev_end = None
    
    for t_seg, f_seg, fname, _ in segments:
        if prev_end is None:
            all_t.extend(t_seg)
            all_f.extend(f_seg)
            prev_end = t_seg[-1]
        else:
            # Overlap для множественных
            ov_start = max(t_seg[0], prev_end - 4.0)
            mask = t_seg >= ov_start
            if np.any(mask):
                prev_ov = np.array(all_f[-80:])
                new_ov = f_seg[mask][:80]
                if len(new_ov) > 10:
                    scale = np.nanmedian(prev_ov) / np.nanmedian(new_ov)
                    f_seg = f_seg * np.clip(scale, 0.8, 1.2)
            all_t.extend(t_seg)
            all_f.extend(f_seg)
            prev_end = t_seg[-1]
    
    order = np.argsort(all_t)
    return np.array(all_t)[order], np.array(all_f)[order]

def get_status(conf, n_points):
    if conf > 85: return "🟢 CONFIRMED"
    elif conf > 65: return "🟡 CANDIDATE"
    elif n_points > 2000: return "🔵 STRONG"
    return "⚪ WEAK LC"

def analyze_exoplanet(files, sde_thresh, min_p, max_p):
    if not HAS_ASTROPY or not files:
        return "Error: Astropy missing", None
    
    t_all, f_all = stitch_nasa_order(files)
    if t_all is None:
        return "No valid FITS", None
    
    n_points = len(t_all)
    
    # ✅ 20+ точек = OK для любого количества файлов!
    if n_points < 20:
        return f"Too few points: {n_points}", None
    
    baseline = t_all[-1] - t_all[0]
    minp = max(min_p, 0.15)
    maxp = min(max_p, baseline/2)
    
    if maxp < minp:
        return "Period range invalid", None
    
    # ✅ Адаптивное количество периодов
    if n_points < 1000:
        n_periods = 5000
        periods = np.linspace(minp, maxp, n_periods)
    else:
        n_periods = 14000
        if baseline > 80:
            periods = np.logspace(np.log10(minp), np.log10(maxp), n_periods)
        else:
            periods = np.linspace(minp, maxp, n_periods)
    
    max_dur = minp / 3.0
    durations = np.linspace(0.0008, max_dur, 8)
    
    bls = BoxLeastSquares(t_all, f_all)
    power_max = np.zeros(len(periods))
    
    for d in durations:
        try:
            if d < periods[0]:
                pg = bls.power(periods, d)
                power_max = np.maximum(power_max, pg.power)
        except:
            continue
    
    peak = np.argmax(power_max)
    sde = (power_max[peak] - np.median(power_max)) / np.std(power_max)
    depth = -np.min(f_all)
    ml_p = min(1.0, sde/10 + depth*2000)
    conf = min(100, sde*8 + ml_p*25 + (n_points-20)/200)
    status = get_status(conf, n_points)
    
    result = f'{status}\nScore: {conf:.0f}%\nSDE: {sde:.2f}\nPeriod: {periods[peak]:.4f}d\nDepth: {depth:.5f}\nPoints: {n_points}\nFiles: {len(files)}'
    
    fig, axs = plt.subplots(3,1, figsize=(12,12), facecolor='black')
    axs[0].plot(t_all, f_all, 'lightblue', alpha=0.7, linewidth=0.8)
    axs[0].set_title(f'Light Curve ({n_points} pts)', color='white')
    axs[1].semilogx(periods, power_max, 'gold')
    axs[1].axvline(periods[peak], color='lime', ls='--')
    axs[1].set_title('BLS Periodogram', color='white')
    phase = ((t_all - t_all[0])/periods[peak] + 0.5) % 1
    phase_days = (phase - 0.5) * periods[peak]
    axs[2].plot(phase_days, f_all, 'lightcoral', alpha=0.7)
    axs[2].set_title('Phase-folded', color='white')
    for ax in axs:
        ax.set_facecolor('#0a0a1a')
        ax.tick_params(colors='white')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, facecolor='black', dpi=150)
    plt.close()
    img = Image.open(buf)
    return result, img

css = "body {background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 100%); color: #e8e8ff;} .gr-button {background: linear-gradient(45deg, #00d4ff, #0099cc);}"

with gr.Blocks(css=css) as demo:
    gr.Markdown('# 🚀 Exoplanet Finder v3.7 - 1..19 FILES OK')
    gr.Markdown('**Single FITS ✅ Multi FITS ✅ NASA quarters sorting**')
    file_input = gr.File(file_count="multiple", file_types=[".fits"], label="NASA FITS (1-19 files)")
    sde = gr.Slider(4, 12, 6, label="SDE Threshold")
    minp = gr.Slider(0.1, 20, 0.2, label="Min Period (days)")
    maxp = gr.Slider(20, 1000, 200, label="Max Period (days)")
    btn = gr.Button("🔍 ANALYZE LC", variant="primary", size="lg")
    text_out = gr.Textbox(label="Results")
    img_out = gr.Image(label="Plots")
    btn.click(analyze_exoplanet, [file_input, sde, minp, maxp], [text_out, img_out])

demo.launch()