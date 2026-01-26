import os
import io
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from PIL import Image
import gradio as gr
import joblib

# ии модель
try:
    AI_MODEL = joblib.load('planet_longquiet.pkl')
except:
    AI_MODEL = None

try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except:
    HAS_SAVGOL = False

plt.style.use('dark_background')

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
            return read_time_flux_from_hdu(hdul[1]) if len(hdul) > 1 else (None, None)
    except:
        return None, None

def clean_and_normalize_segment(time, flux):
    mask = np.isfinite(time) & np.isfinite(flux)
    time = np.array(time[mask], dtype=float)
    flux = np.array(flux[mask], dtype=float)
    if len(time) == 0:
        return None, None
    order = np.argsort(time)
    time, flux = time[order], flux[order]
    med = np.nanmedian(flux)
    return time, flux / (med if np.isfinite(med) and med != 0 else 1.0)

def stitch_segments(segments):
    segs = [(np.nanmedian(t), t, f) for t, f in segments if len(t) > 0]
    if not segs:
        return None, None
    segs.sort(key=lambda x: x[0])
    aligned = [(segs[0][1], segs[0][2])]
    for _, t, f in segs[1:]:
        if len(aligned):
            prev_t, prev_f = aligned[-1]
            overlap_new = (t >= prev_t[0]) & (t <= prev_t[-1])
            overlap_old = (prev_t >= t[0]) & (prev_t <= t[-1])
            if overlap_new.any() and overlap_old.any():
                scale = np.nanmedian(prev_f[overlap_old]) / np.nanmedian(f[overlap_new])
                f *= scale if np.isfinite(scale) else 1.0
        aligned.append((t, f))
    time_all = np.concatenate([t for t, f in aligned])
    flux_all = np.concatenate([f for t, f in aligned])
    order = np.argsort(time_all)
    time_all, flux_all = time_all[order], flux_all[order]
    med = np.nanmedian(flux_all)
    flux_all /= med if np.isfinite(med) and med != 0 else 1.0
    return time_all, flux_all

def detrend_flux(time, flux):
    n = len(flux)
    if n < 20:
        trend = np.ones_like(flux)
    elif HAS_SAVGOL:
        win = min(201, max(11, n//30 | 1))
        try:
            trend = savgol_filter(flux, win, 2, mode='interp')
        except:
            trend = np.ones_like(flux)
    else:
        k = max(5, n//50)
        from scipy.ndimage import median_filter
        trend = median_filter(flux, k, mode='nearest')
    trend = np.nan_to_num(trend, nan=1.0)
    return flux / trend - 1.0, trend

def compute_sde(power, peak_idx, exclude_width=50):
    p = np.array(power, dtype=float)
    n = len(p)
    mask = np.ones(n, dtype=bool)
    lo, hi = max(0, peak_idx-exclude_width), min(n, peak_idx+exclude_width)
    mask[lo:hi] = False
    noise = p[mask]
    med, std = np.median(noise), np.std(noise)
    return (p[peak_idx] - med) / std if std > 0 else 0.0

def analyze_exoplanet(files, sde_thresh=7.5):
    if not files:
        return 'Upload FITS files', None

    segments, failed = [], []
    for f in files:
        t, flux = read_fits_file_auto(f.name)
        if t is None or len(t) < 10:
            failed.append(os.path.basename(f.name))
            continue
        t_clean, f_clean = clean_and_normalize_segment(t, flux)
        if t_clean is not None:
            segments.append((t_clean, f_clean))

    if not segments:
        return f'No valid data. Failed: {', '.join(failed)}', None

    time_all, flux_all = stitch_segments(segments)
    if time_all is None or len(time_all) < 50:
        return 'Too few points after processing', None

    mask = np.isfinite(time_all) & np.isfinite(flux_all)
    time_all, flux_all = time_all[mask], flux_all[mask]

    flux_rel, _ = detrend_flux(time_all, flux_all)

    span = time_all[-1] - time_all[0]
    max_p = min(500, span/2)
    n_p = min(40000, max(5000, int(span*30)))
    periods = np.linspace(0.3, max_p, n_p)
    durations = np.linspace(0.005, 0.2, 15)

    bls = BoxLeastSquares(time_all, flux_rel)
    power_max = np.max([bls.power(periods, d).power for d in durations], axis=0)
    peak_idx = np.argmax(power_max)
    best_p = periods[peak_idx]
    sde = compute_sde(power_max, peak_idx)
    detected = sde >= sde_thresh

    # AI
    log_p = np.log10(best_p)
    depth = np.nanmax(np.abs(flux_rel))
    dur = durations[np.argmax([bls.power(periods, d).power[peak_idx] for d in durations])]
    ai_sde = sde
    r_planet = np.sqrt(depth) * 1.3
    multi = 0
    ai_feats = np.array([[log_p, depth, dur, ai_sde, r_planet, multi]])
    ai_prob = AI_MODEL.predict_proba(ai_feats)[0,1]*100 if AI_MODEL is not None else 0

    # PLOTS
    fig = plt.figure(figsize=(14, 12), facecolor='black')
    gs = fig.add_gridspec(3, 1, hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    from scipy.interpolate import interp1d
    if len(time_all) > 100:
        f_int = interp1d(time_all, flux_rel, kind='cubic', bounds_error=False, fill_value='extrapolate')
        t_smooth = np.linspace(time_all[0], time_all[-1], 1000)
        ax1.plot(t_smooth, f_int(t_smooth), 'cyan', lw=1.2)
    ax1.plot(time_all, flux_rel, 'cyan', lw=0.4, alpha=0.4)
    ax1.set_ylabel('Flux rel')
    ax1.grid(alpha=0.2)
    ax1.set_title('Lightcurve', color='white')

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(periods, power_max, 'magenta', lw=1.5)
    w = max(1, len(periods)//1000)
    lo, hi = max(0, peak_idx-w), min(len(periods), peak_idx+w)
    noise_mask = np.ones_like(power_max, dtype=bool)
    noise_mask[lo:hi] = False
    n_med = np.median(power_max[noise_mask])
    n_std = np.std(power_max[noise_mask])
    det_lev = n_med + sde_thresh * n_std
    ax2.axvline(best_p, 'r--', lw=3, label=f'P={best_p:.3f}d')
    ax2.axhline(det_lev, 'lime:', lw=2, label=f'SDE={sde_thresh}')
    ax2.set_ylabel('Power')
    ax2.legend()
    ax2.grid(alpha=0.2)
    ax2.set_title('Periodogram', color='white')

    ax3 = fig.add_subplot(gs[2])
    phase = ((time_all - time_all[0]) / best_p) % 1
    phase = (phase + 0.5) % 1
    order = np.argsort(phase)
    phase_days = (phase[order] - 0.5) * best_p
    flux_sort = flux_rel[order]
    ax3.plot(phase_days, flux_sort, 'cyan', lw=0.8, alpha=0.6)
    nbins = 80
    bins = np.linspace(-0.5*best_p, 0.5*best_p, nbins+1)
    inds = np.digitize(phase_days, bins) - 1
    binned = [np.nanmedian(flux_sort[inds==i]) if np.any(inds==i) else np.nan for i in range(nbins)]
    bin_c = 0.5*(bins[:-1]+bins[1:])
    ax3.plot(bin_c, binned, 'red', lw=4)
    ax3.set_xlim(-0.25*best_p, 0.25*best_p)
    ax3.set_xlabel('Phase days')
    ax3.set_ylabel('Flux rel')
    ax3.grid(alpha=0.2)
    ax3.set_title(f'Phase Folded P={best_p:.3f}d', color='white')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
    img = Image.open(buf).convert('RGB')

    status = 'DETECTED' if detected else 'NOT CONFIRMED'
    result = f'''BLS:
Status: {status}
P: {best_p:.6f}d | Dur: {dur:.4f} | SDE: {sde:.3f}
Depth: {depth:.1e} | Span: {span:.1f}d
Files: {len(segments)}/{len(files)}

AI: {ai_prob:.1f}% planet

| Metric | BLS | AI |
|--------|-----|----|
| Signal | SDE {sde:.2f} | {ai_prob:.0f}% |
| Verdict | {'STRONG' if sde>10 else 'WEAK' if sde>7.5 else 'NO'} | {'PLANET' if ai_prob>70 else 'CANDIDATE' if ai_prob>30 else 'FP'} |

KOI-score: {min(100, (sde/15 + ai_prob/2)/2):.0f}%'''

    return result, img

css = '''
body { background: linear-gradient(135deg, #0b0c10 0%, #1a1a2e 50%, #16213e 100%); color: #c5c6c7; font-family: 'Segoe UI', sans-serif; }
h1 { color: #66fcf1; text-align: center; }
.gr-button { background: linear-gradient(45deg, #1f2833, #45a29e); color: #66fcf1; border-radius: 12px; font-weight: bold; }
.gr-button:hover { background: linear-gradient(45deg, #45a29e, #66fcf1); color: #0b0c10; }
.gr-textbox { background: rgba(31,40,51,0.9); border-radius: 12px; border: 1px solid #66fcf1; color: #c5c6c7; }
.gr-image { background: rgba(31,40,51,0.8); border-radius: 12px; border: 2px solid #66fcf1; }
.gr-dropdown { background: rgba(31,40,51,0.9); border-radius: 12px; border: 1px solid #66fcf1; color: #c5c6c7; }
'''

# DROPDOWN SPRAVKA
terms = {
    "SDE": "Signal Detection Efficiency. Сила BLS сигнала. >7.5=сильный, >15=отличный",
    "BLS": "Box Least Squares. Поиск транзитов методом наименьших квадратов",
    "PDCSAP_FLUX": "Pipeline flux. Очищенный (РЕКОМЕНДУЕТСЯ)",
    "SAP_FLUX": "Simple Aperture. Сырой flux",
    "LLC": "Long Cadence. 30мин (~9k точек/квартал)",
    "SLC": "Short Cadence. 1мин (~250k точек/квартал)",
    "log_period": "log10(Period) для ML нормализации",
    "depth": "Глубина транзита ΔF/F",
    "AI Probability": "Нейросеть вероятность планеты >70%=CONFIRMED",
    "KOI": "Kepler Object of Interest. NASA каталог (9564)",
    "phase folded": "Фазовая кривая. Сложение транзитов"
}

def show_term(term):
    return f"**{term}**: {terms.get(term, 'Термин не найден')}"

with gr.Blocks(css=css) as app:
    gr.Markdown('# 🌌 Exoplanet Finder - Dark Lines + AI')

    with gr.Row():
        files = gr.File(file_count='multiple', file_types=['.fits'], label='FITS files (multi-OK)')
        sde_slider = gr.Slider(5, 15, 7.5, step=0.1, label='SDE threshold')

    analyze_btn = gr.Button('🔍 Analyze + AI Predict', variant='primary', size='lg')

    with gr.Row():
        result_text = gr.Textbox(label='BLS vs AI Comparison', lines=15)
        result_img = gr.Image(label='Dark Smooth Plots', type='pil')

    analyze_btn.click(analyze_exoplanet, [files, sde_slider], [result_text, result_img])

    # DROPDOWN SPRAVKA ACCORDION
    with gr.Accordion("📚 СПРАВКА (Выберите термин)", open=False):
        with gr.Row():
            term_dropdown = gr.Dropdown(
                choices=list(terms.keys()), 
                label="Термины", 
                value="SDE",
                elem_id="term_select"
            )
            search_box = gr.Textbox(
                label="🔍 Поиск", 
                placeholder="SDE, BLS, depth...",
                lines=1
            )
        help_output = gr.Markdown(label="Описание", value="**SDE**: Signal Detection Efficiency...")

        # DROPDOWN + SEARCH
        term_dropdown.change(show_term, term_dropdown, help_output)
        search_box.submit(lambda q: show_term(next((k for k in terms if q.lower() in k.lower()), 'SDE')), search_box, help_output)

if __name__ == '__main__':
    app.launch(server_name='0.0.0.0', server_port=7860)