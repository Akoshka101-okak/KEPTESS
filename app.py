import gradio as gr
import numpy as np
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
import matplotlib.pyplot as plt
import io
from PIL import Image

# Попытка подключить Savitzky-Golay, если доступен
try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False

# --------- Вспомогательные функции ----------
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
            win = min(201, max(5, (n // 5) | 1))
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
    trend = np.where(np.isfinite(trend) & (np.abs(trend) > 0), trend, np.nanmedian(trend[np.isfinite(trend)]))
    flux_rel = (flux / trend) - 1.0
    return flux_rel, trend

# --------- Основная функция анализа ----------
def analyze_fits(fits_file):
    if fits_file is None:
        return "❌ Файл не выбран", None

    try:
        with fits.open(fits_file.name, memmap=False) as hdul:
            data = None
            for h in hdul:
                if hasattr(h, 'data') and h.data is not None:
                    cols = getattr(h.data, 'columns', None)
                    if cols is not None:
                        names = [n.upper() for n in cols.names]
                        if 'TIME' in names and ('PDCSAP_FLUX' in names or 'SAP_FLUX' in names or 'FLUX' in names):
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
                if n in ('PDCSAP_FLUX', 'SAP_FLUX', 'FLUX'):
                    flux_col = n
            if time_col is None or flux_col is None:
                return "❌ В FITS нет TIME или подходящего столбца яркости", None

            time = data[time_col]
            flux = data[flux_col]
    except Exception as e:
        return f"❌ Ошибка чтения FITS: {e}", None

    # Очистка
    mask = np.isfinite(time) & np.isfinite(flux)
    time = np.array(time[mask], dtype=float)
    flux = np.array(flux[mask], dtype=float)
    if len(time) < 20:
        return "❌ Недостаточно данных для анализа (<20 точек)", None
    if not np.all(np.diff(time) >= 0):
        order = np.argsort(time)
        time = time[order]
        flux = flux[order]

    flux_norm = flux / np.nanmedian(flux)
    flux_rel, trend = detrend_flux(time, flux_norm)

    # Настройка периодов
    total_time = time[-1] - time[0]
    min_p = 0.3
    max_p = max(min(500, total_time/2.0), 1.0)

    n_periods = min(20000, int(total_time*100))  # адаптивная сетка
    periods = np.linspace(min_p, max_p, n_periods)
    durations = np.linspace(0.01, 0.2, 10)

    bls = BoxLeastSquares(time, flux_rel)
    power_matrix = np.zeros((len(durations), len(periods)), dtype=float)
    for i, dur in enumerate(durations):
        res = bls.power(periods, dur)
        power_matrix[i, :] = res.power

    power_per_period = np.max(power_matrix, axis=0)
    idx_peak = np.argmax(power_per_period)
    best_period = periods[idx_peak]
    best_power = power_per_period[idx_peak]
    idx_dur = np.argmax(power_matrix[:, idx_peak])
    best_duration = durations[idx_dur]

    median_power = np.median(power_per_period)
    std_power = np.std(power_per_period)
    sde = (best_power - median_power)/std_power if std_power>0 else 0.0

    SDE_THRESHOLD = 6.0

    # ----- Построение графиков -----
    fig, axes = plt.subplots(3, 1, figsize=(9,10))
    ax0, ax1, ax2 = axes

    # Детрендированная кривая
    ax0.plot(time, flux_rel, linewidth=0.6, color='cyan')
    ax0.set_title("Детрендированная кривая блеска", color='white')
    ax0.set_xlabel("Время (дни)", color='white')
    ax0.set_ylabel("ΔFlux (отн.)", color='white')
    ax0.tick_params(colors='white')
    ax0.grid(True, color='gray', alpha=0.4, linestyle='--')

    # Периодограмма
    ax1.plot(periods, power_per_period, linewidth=0.8, color='lime')
    detection_level = median_power + SDE_THRESHOLD*std_power
    ax1.axhline(detection_level, color='red', linestyle='--', linewidth=0.9, label=f"SDE threshold ({SDE_THRESHOLD:.1f})")
    ax1.axvline(best_period, color='white', linestyle=':', linewidth=0.8, label=f"Best period {best_period:.3f} d")
    ax1.set_title("Периодограмма BLS", color='white')
    ax1.set_xlabel("Период (дни)", color='white')
    ax1.set_ylabel("Power", color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor="#0b0c10", edgecolor='gray', labelcolor='white')
    ax1.grid(True, color='gray', alpha=0.4, linestyle='--')

    # Фазовая кривая
    phase = ((time / best_period) % 1.0 + 0.5) % 1.0
    order = np.argsort(phase)
    phase_sorted = phase[order]
    flux_sorted = flux_rel[order]
    phase_days = (phase_sorted - 0.5)*best_period

    ax2.plot(phase_days, flux_sorted, '.', ms=2, color='cyan', alpha=0.6)
    nbins = 100
    bins = np.linspace(-0.5*best_period, 0.5*best_period, nbins+1)
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    digitized = np.digitize(phase_days, bins)-1
    bin_med = np.array([np.median(flux_sorted[digitized==i]) if np.any(digitized==i) else np.nan for i in range(nbins)])
    ax2.plot(bin_centers, bin_med, color='red', lw=1.5, label='Binned median')

    ax2.set_xlim(-0.2*best_period, 0.2*best_period)
    ax2.set_title(f"Фазовая кривая (P={best_period:.4f} d)", color='white')
    ax2.set_xlabel("Время от транзита (дни)", color='white')
    ax2.set_ylabel("ΔFlux (отн.)", color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor="#0b0c10", edgecolor='gray', labelcolor='white')
    ax2.grid(True, color='gray', alpha=0.4, linestyle='--')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0b0c10', dpi=150)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # ----- Интерпретация -----
    if sde >= SDE_THRESHOLD:
        result_text = (f"🌍 Кандидат обнаружен!\n"
                       f"Период: {best_period:.6f} д\n"
                       f"SDE: {sde:.3f} (порог {SDE_THRESHOLD})\n"
                       f"Длительность (фр. периода): {best_duration:.4f}")
    else:
        result_text = (f"❌ Кандидат не подтверждён.\n"
                       f"Период: {best_period:.6f} д\n"
                       f"SDE: {sde:.3f} (порог {SDE_THRESHOLD})\n"
                       f"Длительность (фр. периода): {best_duration:.4f}\n"
                       "Примечание: сигнал может быть слабым. Проверьте графики или увеличьте набор данных.")

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
    gr.Markdown("<p style='color:#c5c6c7; text-align:center'>Загрузите FITS (Kepler/TESS). Алгоритм автоматически проведёт детренд, BLS-анализ и покажет фазовую кривую.</p>")

    with gr.Row():
        file_input = gr.File(label="Выберите FITS-файл", file_types=['.fits'])
        result_text = gr.Textbox(label="Результат анализа", interactive=False)

    result_image = gr.Image(label="Графики анализа", type="pil")
    analyze_btn = gr.Button("🔎 Анализировать")

    analyze_btn.click(analyze_fits, inputs=file_input, outputs=[result_text, result_image])

app.launch()
