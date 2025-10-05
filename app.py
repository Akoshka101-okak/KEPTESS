import gradio as gr
import numpy as np
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
import matplotlib.pyplot as plt
import io
from PIL import Image

# ------------------ Функция анализа ------------------
def analyze_fits(fits_file):
    if fits_file is None:
        return "❌ Файл не выбран", None

    # Открываем FITS
    try:
        with fits.open(fits_file.name) as hdul:
            data = hdul[1].data
            time = data['TIME']
            # flux может быть PDCSAP_FLUX или SAP_FLUX
            if 'PDCSAP_FLUX' in data.columns.names:
                flux = data['PDCSAP_FLUX']
            elif 'SAP_FLUX' in data.columns.names:
                flux = data['SAP_FLUX']
            else:
                return "❌ Нет подходящего столбца с данными яркости", None
    except Exception as e:
        return f"❌ Ошибка чтения FITS: {e}", None

    # Убираем NaN
    mask = ~np.isnan(time) & ~np.isnan(flux)
    time = time[mask]
    flux = flux[mask]

    if len(time) < 10:
        return "❌ Недостаточно данных для анализа", None

    # Нормируем и центрируем
    flux = flux / np.median(flux)
    flux = flux - np.median(flux)

    # BLS анализ
    bls = BoxLeastSquares(time, flux)
    periods = np.linspace(0.5, 50, 20000)  # расширяем диапазон периодов до 50 дней
    results = bls.power(periods, 0.05)

    best_period = results.period[np.argmax(results.power)]
    power = np.max(results.power)

    # Рисуем графики
    fig, ax = plt.subplots(2, 1, figsize=(8,6))
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

    # ------------------ Порог мощности ------------------
    if power > 1:  
        result_text = f"🌍 Обнаружен кандидат в экзопланеты (Период: {best_period:.2f} дней, Power: {power:.2f})"
    else:
        result_text = f"❌ Экзопланета не обнаружена (Power: {power:.2f})"

    return result_text, img

# ------------------ Gradio интерфейс ------------------
with gr.Blocks(css="""
body {
    background-image: url('https://images.unsplash.com/photo-1581325785936-3e14a9ef9f83?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
    background-size: cover;
    background-position: center;
    color: #c5c6c7;
    font-family: Arial, sans-serif;
}
.gr-button {
    background-color: #1f2833;
    color: #66fcf1;
    border-radius: 8px;
    border: none;
    padding: 12px 20px;
    font-size: 16px;
}
.gr-button:hover {
    background-color: #45a29e;
    color: #0b0c10;
}
.gr-textbox, .gr-image {
    background-color: rgba(31, 40, 51, 0.8);
    border-radius: 8px;
    padding: 10px;
}
""") as app:

    gr.Markdown("<h1 style='color:#66fcf1; text-align:center'>🚀 AI Exoplanet Detector</h1>")
    gr.Markdown("<p style='color:#c5c6c7; text-align:center'>🔭 Загружайте световые кривые Kepler/TESS и ИИ найдёт признаки транзита планеты.</p>")

    with gr.Row():
        file_input = gr.File(label="Выберите FITS-файл", file_types=['.fits'])
        result_text = gr.Textbox(label="Результат", interactive=False)

    result_image = gr.Image(label="График анализа")
    analyze_btn = gr.Button("Анализировать")

    analyze_btn.click(analyze_fits, inputs=file_input, outputs=[result_text, result_image])

# ------------------ Запуск приложения ------------------
app.launch()
