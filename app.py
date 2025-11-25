import gradio as gr
import numpy as np
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
import matplotlib.pyplot as plt
import io
from PIL import Image

# Попытка использовать Savitzky–Golay
try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False


# -------------------------------
# ЧТЕНИЕ ОДНОГО FITS-ФАЙЛА
# -------------------------------
def read_fits_file(file_obj):
    hdul = fits.open(file_obj.name)
    
    # Kepler / TESS light curve tables
    if "TIME" in hdul[1].columns.names:
        time = hdul[1].data["TIME"]
    else:
        raise ValueError("Не найден столбец TIME")

    # flux column detection
    flux_col = None
    for col in ["PDCSAP_FLUX", "SAP_FLUX", "FLUX"]:
        if col in hdul[1].columns.names:
            flux_col = col
            break

    if flux_col is None:
        raise ValueError("Не найден столбец flux (например, PDCSAP_FLUX)")

    flux = hdul[1].data[flux_col]

    # Clean
    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]

    hdul.close()
    return time, flux


# -------------------------------
# ОБЪЕДИНЕНИЕ НЕСКОЛЬКИХ ФАЙЛОВ
# -------------------------------
def merge_fits_files(file_list):
    all_time = []
    all_flux = []

    for f in file_list:
        t, fl = read_fits_file(f)
        all_time.append(t)
        all_flux.append(fl)

    time = np.concatenate(all_time)
    flux = np.concatenate(all_flux)

    # Сортировка по времени
    order = np.argsort(time)
    time = time[order]
    flux = flux[order]

    # нормировка
    flux = flux / np.nanmedian(flux)

    return time, flux


# -------------------------------
# ПОИСК ТРАНЗИТОВ
# -------------------------------
def analyze(time, flux):
    # detrend
    if _HAS_SAVGOL:
        try:
            trend = savgol_filter(flux, 101, 2)
            flux = flux / trend
        except Exception:
            pass

    # periods grid
    periods = np.linspace(0.5, 30, 3000)
    durations = np.linspace(0.05, 0.25, 5)

    bls = BoxLeastSquares(time, flux)
    res = bls.autopower(durations)

    sde = (res.power - np.median(res.power)) / np.std(res.power)
    best = np.argmax(sde)

    best_period = res.period[best]
    best_sde = sde[best]
    depth = res.depth[best]

    # график
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(res.period, sde, lw=0.8)
    ax.axvline(best_period, color="red", linestyle="--")
    ax.set_xlabel("Период (дни)")
    ax.set_ylabel("SDE")
    ax.set_title("Box Least Squares – SDE periodogram")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130)
    buf.seek(0)
    img = Image.open(buf)

    # решение
    threshold = 7.5
    status = "⚠️ Не подтверждено"
    if best_sde >= threshold:
        status = "✅ Найден уверенный сигнал"

    result_text = (
        f"ПЕРИОД: {best_period:.6f} дней\n"
        f"SDE: {best_sde:.3f}\n"
        f"Глубина: {depth:.3e}\n"
        f"Статус: {status}"
    )

    return img, result_text



# -------------------------------
# GRADIO UI (всё на одном экране)
# -------------------------------
def process(files):
    if not files:
        return None, "Загрузите хотя бы 1 FITS файл"

    try:
        time, flux = merge_fits_files(files)
        img, result_text = analyze(time, flux)
        return img, result_text
    except Exception as e:
        return None, f"Ошибка: {str(e)}"


with gr.Blocks() as app:
    gr.Markdown("# 🪐 Exoplanet Finder — объединение FITS + анализ\nЗагрузи несколько FITS-файлов Kepler/TESS.")

    file_input = gr.File(label="Загрузите несколько FITS файлов", file_count="multiple")

    run_btn = gr.Button("Анализировать")

    output_img = gr.Image(label="SDE график")
    output_text = gr.Textbox(label="Результаты", lines=6)

    run_btn.click(process, inputs=file_input, outputs=[output_img, output_text])

app.launch()
