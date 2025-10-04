import gradio as gr
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import io
import PIL.Image

def analyze_fits(fits_file):
    # Открываем FITS-файл
    with fits.open(fits_file.name) as hdul:
        data = hdul[1].data
        time = data['TIME']

        # Проверяем, какая колонка есть
        if 'PDCSAP_FLUX' in data.columns.names:
            flux = data['PDCSAP_FLUX']
        else:
            flux = data['SAP_FLUX']

    # Убираем NaN
    mask = ~np.isnan(time) & ~np.isnan(flux)
    time = time[mask]
    flux = flux[mask]

    # Нормируем поток
    flux = flux / np.median(flux)

    # --- Простейший анализ: поиск транзита ---
    dips = flux[flux < 0.99]  # точки, где яркость упала >1%
    if len(dips) > 5 and np.std(flux) < 0.05:
        result = "🌍 Кандидат в планеты (наблюдаются периодические падения яркости)"
    else:
        result = "✖️ Шум или не планета (яркость падает случайно)"

    # Строим график
    plt.figure(figsize=(8, 4))
    plt.plot(time, flux, color="blue", lw=0.5)
    plt.xlabel("Время (дни)")
    plt.ylabel("Нормированная яркость")
    plt.title("Кривая блеска (Light Curve)")
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    img = PIL.Image.open(buf)

    # Возвращаем картинку и текст
    return np.array(img), result


app = gr.Interface(
    fn=analyze_fits,
    inputs=gr.File(label="Загрузите FITS-файл"),
    outputs=[
        gr.Image(label="Кривая блеска"),
        gr.Textbox(label="Результат анализа")
    ],
    title="TESS / Kepler Planet Finder",
    description="Инструмент анализирует световые кривые и определяет, есть ли признаки транзита экзопланеты."
)

app.launch()

