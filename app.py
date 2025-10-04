import gradio as gr
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def analyze_fits(fits_file):
    # Открываем FITS-файл
    with fits.open(fits_file.name) as hdul:
        data = hdul[1].data
        time = data['TIME']
        flux = data['PDCSAP_FLUX']

    # Убираем NaN (пустые значения)
    mask = ~np.isnan(time) & ~np.isnan(flux)
    time = time[mask]
    flux = flux[mask]

    # Нормализуем яркость
    flux = flux / np.median(flux)

    # Создаем график
    plt.figure(figsize=(8, 4))
    plt.plot(time, flux, color="blue", lw=0.5)
    plt.xlabel("Время (дни)")
    plt.ylabel("Нормализованный поток")
    plt.title("Кривая блеска (Light Curve)")
    plt.grid(True)

    # Проверяем наличие возможного транзита
    flux_min = np.min(flux)
    flux_std = np.std(flux)
    if flux_min < (1 - 3 * flux_std):
        result = "⚡ Обнаружен возможный транзит — кандидат в экзопланеты!"
    else:
        result = "🌙 Экзопланет не обнаружено."

    return plt, result


app = gr.Interface(
    fn=analyze_fits,
    inputs=gr.File(label="Загрузите FITS-файл"),
    outputs=[gr.Plot(label="Кривая блеска"), gr.Textbox(label="Результат анализа")],
    title="TESS / Kepler Planet Finder",
    description="Этот инструмент анализирует световые кривые и помогает находить кандидатов в экзопланеты.",
    allow_flagging="never",
    live=False
)

app.launch()


