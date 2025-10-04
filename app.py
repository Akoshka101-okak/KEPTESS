import gradio as gr
import numpy as np
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
import matplotlib.pyplot as plt
import io
from PIL import Image

# Функция анализа FITS
def analyze_fits(fits_file):
    with fits.open(fits_file.name) as hdul:
        data = hdul[1].data
        time = data['TIME']
        flux = data['PDCSAP_FLUX']

    mask = ~np.isnan(time) & ~np.isnan(flux)
    time = time[mask]
    flux = flux[mask]

    flux = flux / np.median(flux)

    bls = BoxLeastSquares(time, flux)
    periods = np.linspace(0.5, 20, 10000)
    results = bls.power(periods, 0.05)

    best_period = results.period[np.argmax(results.power)]
    power = np.max(results.power)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(time, flux, color="cyan", lw=0.5)
    ax[0].set_title("Кривая блеска (Light Curve)", color='white')
    ax[0].set_xlabel("Время (дни)", color='white')
    ax[0].set_ylabel("Яркость (отн.)", color='white')
    ax[0].tick_params(colors='white')
    ax[0].grid(True, color='gray', linestyle='--', alpha=0.5)

    ax[1].plot(results.period, results.power, color="lime")
    ax[1].set_title("Периодограмма BLS", color='white')
    ax[1].set_xlabel("Период (дни)", color='white')
    ax[1].set_ylabel("Мощность сигнала", color='white')
    ax[1].tick_params(colors='white')
    ax[1].grid(True, color='gray', linestyle='--', alpha=0.5)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', facecolor='#0b0c10')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)

    if power > 10:
        result_text = f"🌍 Обнаружен кандидат в экзопланеты (Период: {best_period:.2f} дней)"
    else:
        result_text = "❌ Экзопланета не обнаружена"

    return result_text, img

# Интерфейс с космическим фоном
with gr.Blocks(css="""
    body {
        background-image: url('https://images.unsplash.com/photo-1581325785936-3e14a9ef9



   
