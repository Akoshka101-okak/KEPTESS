import gradio as gr
import numpy as np
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
import matplotlib.pyplot as plt
import io
from PIL import Image

# Try savgol
try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False

# ---------- Helpers ----------
def running_median(x, window):
    if window <= 1:
        return x
    pad = window // 2
    x_padded = np.pad(x, pad, mode='edge')
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(x_padded[i:i+window])
    return out

def detrend_flux(time, flux):
    n = len(flux)
    if n < 20:
        trend = np.ones_like(flux)
    else:
        if _HAS_SAVGOL:
            win = min(601, max(5, (n // 10) | 1))
            if win >= n:
                win = (n - 1) if (n - 1) % 2 == 1 else (n - 2)
                if win < 3:
                    win = 3
            try:
                trend = savgol_filter(flux, window_length=win, polyorder=2, mode='interp')
            except Exception:
                trend = running_median(flux, max(3, n//10))
        else:
            trend = running_median(flux, max(3, n//10))
    finite = np.isfinite(trend)
    if not np.any(finite):
        trend = np.ones_like(trend)
    else:
        med = np.nanmedian(trend[finite])
        trend = np.where(np.isfinite(trend) & (np.abs(trend) > 0), trend, med)
    flux_rel = (flux / trend) - 1.0
    return flux_rel, trend

def estimate_transit_properties(time, flux_rel, period, duration):
    dur_days = duration * period
    phases = ((time / period) % 1.0 + 0.5) % 1.0
    phase_center = 0.5
    in_transit = np.abs((phases - phase_center) * period) <= (dur_days/2.0)
    n_in = np.sum(in_transit)
    depth = -np.nanmedian(flux_rel[in_transit]) if n_in>0 else 0.0
    return depth, int(n_in)

def check_harmonics(time, flux_rel, candidate_period, durations, bls, periods_power_map):
    candidates = [candidate_period, candidate_period/2.0, candidate_period/3.0, candidate_period/4.0, candidate_period*2.0]
    best = candidate_period
    best_score = -np.inf
    for p in candidates:
        if not (periods_power_map['min_p'] <= p <= periods_power_map['max_p']):
            continue
        pow_interp = np.interp(p, periods_power_map['periods'], periods_power_map['powers'])
        try:
            res_local = bls.power(np.array([p]), durations)
            power_local = np.max(res_local.power)
            dur_idx = np.argmax(res_local.power)
            dur_local = durations[dur_idx % len(durations)]
        except Exception:
            power_local = pow_interp
            dur_local = durations[len(durations)//2]
        depth, n_in = estimate_transit_properties(time, flux_rel, p, dur_local)
        score = power_local + (depth * 1e3) + np.log1p(n_in)
        if score > best_score:
            best_score = score
            best = p
    return best

def detect_mission_from_header(hdr):
    # Read several common header keys
    keys = {}
    for k in ("MISSION","TELESCOP","INSTRUME","OBJECT"):
        v = hdr.get(k)
        if v is not None:
            keys[k] = str(v).upper()
    # Heuristics
    if any("KEPLER" in v for v in keys.values()):
        return "KEPLER"
    if any("TESS" in v for v in keys.values()):
        return "TESS"
    return "UNKNOWN"

# ---------- Main analysis ----------
def analyze_fits(fits_file, sde_slider=6.0, auto_sde=True, max_period_grid_points=20000):
    if fits_file is None:
        return "❌ Файл не выбран", None

    # Read FITS and header (safely)
    try:
        with fits.open(fits_file.name, memmap=False) as hdul:
            header0 = hdul[0].header if len(hdul) > 0 else {}
            mission = detect_mission_from_header(header0)
            data = None
            for h in hdul:
                if hasattr(h, 'data') and h.data is not None:
                    cols = getattr(h.data, 'columns', None)
                    if cols is not None:
                        names = [n.upper() for n in cols.names]
                        if 'TIME' in names and any(x in names for x in ('PDCSAP_FLUX','SAP_FLUX','FLUX')):
                            data = h.data
                            break
            if data is None:
                data = hdul[1].data
            colnames = [n.upper() for n in data.columns.names]
            time_col = None
            flux_col = None
            for n in colnames:
                if n == 'TIME':
                    time_col = n
                if n in ('PDCSAP_FLUX','SAP_FLUX','FLUX'):
                    flux_col = n
            if time_col is None or flux_col is None:
                return "❌ В FITS нет TIME или подходящего столбца яркости (PDCSAP_FLUX / SAP_FLUX / FLUX)", None
            time = np.array(data[time_col], dtype=float)
            flux = np.array(data[flux_col], dtype=float)
    except Exception as e:
        return f"❌ Ошибка чтения FITS: {e}", None

    # Determine SDE threshold by mission (if auto)
    if auto_sde:
        if mission == "KEPLER":
            sde_threshold = 7.5  # среднее между 7-8
        elif mission == "TESS":
            sde_threshold = 9.5  # среднее между 9-10
        else:
            sde_threshold = float(sde_slider)  # fallback to slider
    else:
        sde_threshold = float(sde_slider)

    # Clean
    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]
    if len(time) < 20:
        return "❌ Недостаточно данных для анализа (<20 точек)", None

    if not np.all(np.diff(time) >= 0):
        order = np.argsort(time)
        time = time[order]
        flux = flux[order]

    flux_norm = flux / np.nanmedian(flux)
    flux_rel, trend = detrend_flux(time, flux_norm)

    # Period grid
    baseline = time[-1] - time[0]
    if baseline <= 0:
        return "❌ Неправильные метки времени в FITS", None

    min_period = 0.3
    max_period = max(1.0, min(500.0, baseline * 0.5))
    # adaptive n_periods but bounded
    approx_points = int(min(max_period/min_period * 50, max_period * 200))
    n_periods = min(max(2000, approx_points), max_period_grid_points)
    periods = np.linspace(min_period, max_period, n_periods)

    durations = np.concatenate((
        np.linspace(0.005, 0.02, 4),
        np.linspace(0.02, 0.08, 6),
        np.linspace(0.08, 0.25, 4)
    ))

    # BLS (memory conscious)
    bls = BoxLeastSquares(time, flux_rel)
    power_matrix = np.zeros((len(durations), len(periods)), dtype=float)
    for i, d in enumerate(durations):
        try:
            res = bls.power(periods, d)
            power_matrix[i, :] = res.power
        except Exception:
            chunk = 1000
            pow_chunk = np.empty(len(periods))
            for start in range(0, len(periods), chunk):
                stop = min(len(periods), start+chunk)
                resc = bls.power(periods[start:stop], d)
                pow_chunk[start:stop] = resc.power
            power_matrix[i, :] = pow_chunk

    power_per_period = np.max(power_matrix, axis=0)
    idx_peak = np.argmax(power_per_period)
    best_period = periods[idx_peak]
    best_power = power_per_period[idx_peak]
    idx_dur = np.argmax(power_matrix[:, idx_peak])
    best_duration = durations[idx_dur]

    median_power = np.median(power_per_period)
    std_power = np.std(power_per_period)
    sde = (best_power - median_power)/std_power if std_power>0 else 0.0

    # refine with local BLS and transit estimate
    try:
        res_best = bls.power(np.array([best_period]), durations)
        best_local_idx = np.argmax(res_best.power)
        best_local_dur = durations[best_local_idx % len(durations)]
    except Exception:
        best_local_dur = best_duration
    depth, n_in_transit = estimate_transit_properties(time, flux_rel, best_period, best_local_dur)

    # harmonic check
    periods_power_map = {'periods': periods, 'powers': power_per_period, 'min_p': min_period, 'max_p': max_period}
    refined_period = check_harmonics(time, flux_rel, best_period, durations, bls, periods_power_map)
    if refined_period != best_period:
        best_period = refined_period
        try:
            res_ref = bls.power(np.array([best_period]), durations)
            best_local_idx = np.argmax(res_ref.power)
            best_local_dur = durations[best_local_idx % len(durations)]
            best_power = np.max(res_ref.power)
        except Exception:
            best_local_dur = best_duration
        depth, n_in_transit = estimate_transit_properties(time, flux_rel, best_period, best_local_dur)
        sde = (best_power - median_power)/std_power if std_power>0 else sde

    # Build plots
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), facecolor="#0b0c10")
    ax0, ax1, ax2 = axes

    ax0.plot(time, flux_rel, linewidth=0.6, color='cyan')
    ax0.set_title("Детрендированная кривая блеска", color='white')
    ax0.set_xlabel("Время (дни)", color='white')
    ax0.set_ylabel("ΔFlux (отн.)", color='white')
    ax0.tick_params(colors='white')
    ax0.grid(True, color='gray', alpha=0.4, linestyle='--')

    ax1.plot(periods, power_per_period, linewidth=0.8, color='lime')
    detection_level = median_power + sde_threshold * std_power if 'sde_threshold' in locals() else median_power + sde_threshold * std_power
    ax1.axhline(detection_level, color='red', linestyle='--', linewidth=0.9,
                label=f"SDE threshold ({sde_threshold:.1f})")
    ax1.axvline(best_period, color='white', linestyle=':', linewidth=0.8,
                label=f"Selected period {best_period:.4f} d")
    ax1.set_title("Периодограмма BLS (максимум по длительностям)", color='white')
    ax1.set_xlabel("Период (дни)", color='white')
    ax1.set_ylabel("Power", color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor="#0b0c10", edgecolor='gray', labelcolor='white')
    ax1.grid(True, color='gray', alpha=0.4, linestyle='--')

    phase = ((time / best_period) % 1.0 + 0.5) % 1.0
    order = np.argsort(phase)
    phase_sorted = phase[order]
    flux_sorted = flux_rel[order]
    phase_days = (phase_sorted - 0.5) * best_period

    ax2.plot(phase_days, flux_sorted, '.', ms=2, color='cyan', alpha=0.6)
    nbins = 100
    bins = np.linspace(-0.5 * best_period, 0.5 * best_period, nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    digitized = np.digitize(phase_days, bins) - 1
    bin_med = np.array([np.median(flux_sorted[digitized == i]) if np.any(digitized == i) else np.nan for i in range(nbins)])
    ax2.plot(bin_centers, bin_med, color='red', lw=1.5, label='Binned median')

    ax2.set_xlim(-0.2 * best_period, 0.2 * best_period)
    ax2.set_title(f"Фазовая кривая (P = {best_period:.6f} д)", color='white')
    ax2.set_xlabel("Время от транзита (дни)", color='white')
    ax2.set_ylabel("ΔFlux (отн.)", color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor="#0b0c10", edgecolor='gray', labelcolor='white')
    ax2.grid(True, color='gray', alpha=0.4, linestyle='--')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0b0c10', dpi=160)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # Interpretation
    is_candidate = (sde >= sde_threshold) and (depth > 1e-4) and (n_in_transit >= 3)
    header_info = f"Detected mission: {mission}. Auto SDE: {auto_sde}. Used SDE threshold: {sde_threshold:.2f}"

    if is_candidate:
        result_text = (f"{header_info}\n\n🌍 Кандидат обнаружен!\n"
                       f"Период: {best_period:.6f} д\n"
                       f"SDE: {sde:.3f} (порог {sde_threshold:.2f})\n"
                       f"Оценочная глубина: {depth:.3e}\n"
                       f"Точек в транзите: {n_in_transit}\n"
                       f"Длительность (фр. периода): {best_local_dur:.4f}")
    else:
        result_text = (f"{header_info}\n\n❌ Кандидат не подтверждён автоматически.\n"
                       f"Selected period: {best_period:.6f} д\n"
                       f"SDE: {sde:.3f} (порог {sde_threshold:.2f})\n"
                       f"Depth est.: {depth:.3e}, points in transit: {n_in_transit}\n"
                       "Если уверен в сигнале — попробуй уменьшить порог SDE или проверить длиннее базу наблюдений.")

    return result_text, img

# --------- Gradio UI ----------
with gr.Blocks(css="""
body {
    background-image: url('https://images.unsplash.com/photo-1581325785936-3e14a9ef9f83?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
    background-size: cover;
    background-position: center;
    color: #c5c6c7;
    font-family: 'Segoe UI', sans-serif;
}
.gr-button {
    background-color: #1f2833;
    color: #66fcf1;
    border-radius: 8px;
    border: none;
    padding: 12px 20px;
    font-size: 16px;
    transition: 0.2s;
}
.gr-button:hover {
    background-color: #45a29e;
    color: #0b0c10;
}
.gr-textbox, .gr-image {
    background-color: rgba(31, 40, 51, 0.85);
    border-radius: 8px;
    padding: 10px;
}
""") as app:

    gr.Markdown("<h1 style='color:#66fcf1; text-align:center'>🚀 AI Exoplanet Detector — PRO</h1>")
    gr.Markdown("<p style='color:#c5c6c7; text-align:center'>Загрузите FITS (Kepler/TESS). Алгоритм автоматически определит миссию и предложит порог SDE (можно переопределить).</p>")

    with gr.Row():
        file_input = gr.File(label="Выберите FITS-файл", file_types=['.fits'])
        sde_slider = gr.Slider(minimum=3, maximum=12, value=6, step=0.5, label="SDE threshold (manual override)")
        auto_checkbox = gr.Checkbox(value=True, label="Auto SDE by mission (Kepler/TESS)")

    result_text = gr.Textbox(label="Результат анализа", interactive=False)
    result_image = gr.Image(label="Графики анализа", type="pil")
    analyze_btn = gr.Button("🔎 Анализировать")

    analyze_btn.click(analyze_fits, inputs=[file_input, sde_slider, auto_checkbox], outputs=[result_text, result_image])

app.launch()
