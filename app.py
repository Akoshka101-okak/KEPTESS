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

        # Берём правильную колонку флюкса
        if 'PDCSAP_FLUX' in data.columns.names:
            flux = data['PDCSAP_FLUX']
        else:
            flux = data['SAP_FLUX']

    # Убираем NaN
    mask = ~np.isnan(time) & ~np.isnan(flux)
    time = time[mask]
    flux = flux[mask]

    # Строим график
    plt.figure(figsize=(8, 4))
    plt.plot(time, flux, color="blue", lw=0.5)
    plt.xlabel("Время (дни)")
    plt.ylabel("Яркость звезды")
    plt.title("Кривая блеска (Light Curve)")
    plt.grid(True)

    # Сохраняем картинку в память
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    # Превращаем в картинку
    img = PIL.Image.open(buf)
    return np.array(img)

app = gr.Interface(
    fn=analyze_fits,
    inputs=gr.File(label="Загрузите FITS-файл"),
    outputs=gr.Image(label="Кривая блеска"),
    title="TESS / Kepler Planet Finder",
    description="Этот инструмент анализирует световые кривые и помогает находить кандидатов в экзопланеты."
)

app.launch()
