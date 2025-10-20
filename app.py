import gradio as gr
import numpy as np
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
import matplotlib.pyplot as plt
import io
from PIL import Image

# ==================== ФУНКЦИЯ АНАЛИЗА ====================

def analyze_fits(fits_file):
    if fits_file is None:
        return "❌ Файл не выбран", None

    # ---------- Чтение FITS ----------
    try:
        with fits.open(fits_file.name) as hdul:
            data = hdul[1].data
            time = data['TIME']

            # Проверяем доступный столбец потока
            if 'PDCSAP_FLUX' in data.columns.names:
                flux = data['PDCSAP_FLUX']
            elif 'SAP_FLUX' in data.columns.names:
                flux = data['SAP_FLUX']
            else:
                return "❌ Нет подходящего столбца яркости (PDCSAP_FLUX или SAP_FLUX)", None

    except Exception as e:
        return f"❌ Ошибка при чтении FITS: {e}", None

    # ---------- Очистка данных ----------
    mask = ~np.isnan(time) & ~np.isnan(flux)
    time = time[mask]
    flux = flux[mask]

    if len(time) < 10:
        return "❌ Недостаточно данных для анализа", None

    # ---------- Нормализация ----------
    flux = flux / np.median(flux)
    flux = flux - np.median(flux)

    # ---------- Определяем миссию ----------
    file_name = fits_file.name.lower()
    if "tess" in file_name:
        periods = np.linspace(0.3, 20, 20000)
    else:
        periods = np.linspace(0.5, 50, 20000)

    # ---------- Поиск транзита методом Box Least Squares ----------
    bls = BoxLeastSquares(time, flux)
    results = bls.power(periods, 0.02)  # ширина транзита 2%

    best_period = results.period[np.argmax(results.power)]
    power = np.max(results.power)

    # ---------- Динамический порог ----------
    mean_power = np.mean(results.power)
    std_power = np.std(results.power)
    threshold = mean_power + 3 * std_power  # порог = среднее + 3σ

    # ---------- Визуализация ----------
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    # Кривая блеска
    ax[0].plot(time, flux, color="cyan", lw=0.5)
    ax[0].set_title("Кривая блеска (Light Curve)", color='white')
    ax[0].set_xlabel("Время (дни)", color='white')
    ax[0].set_ylabel("Яркость (отн.)", color='white')
    ax[0].tick_params(colors='white')
    ax[0].grid(True, color='gray', linestyle='--', alpha=0.5)

    # Периодограмма
    ax[1].plot(results.period, results.power, color="lime")
    ax[1].axhline(threshold, color="red", ls="--", lw=0.8, label="Порог обнаружения")
    ax[1].set_title("Периодограмма BLS", color='white')
    ax[1].set_xlabel("Период (дни)", color='white')
    ax[1].set_ylabel("Мощность сигнала", color='white')
    ax[1].tick_params(colors='white')
    ax[1].legend(facecolor="#0b0c10", edgecolor="gray", labelcolor='white')
    ax[1].grid(True, color='gray', linestyle='--', alpha=0.5)

    # ---------- Сохраняем изображение ----------
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', facecolor='#0b0c10')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)

    # ---------- Интерпретация результата ----------
    if power > threshold:
        result_text = f"🌍 Обнаружен кандидат в экзопланеты!\nПериод: {best_period:.2f} дней\nPower: {power:.3f}\nПорог: {threshold:.3f}"
    else:
        result_text = f"❌ Экзопланета не обнаружена.\nPower: {power:.3f}\nПорог: {threshold:.3f}"

    return result_text, img


# ==================== GRADIO ИНТЕРФЕЙС ====================

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

    gr.Markdown("<h1 style='color:#66fcf1; text-align:center'>🚀 AI Exoplanet Detector</h1>")
    gr.Markdown("<p style='color:#c5c6c7; text-align:center'>🔭 Загрузите световую кривую Kepler/TESS — модель обнаружит сигналы возможных экзопланет с помощью анализа транзитов.</p>")

    with gr.Row():
        file_input = gr.File(label="Выберите FITS-файл", file_types=['.fits'])
        result_text = gr.Textbox(label="Результат анализа", interactive=False)

    result_image = gr.Image(label="График анализа", type="pil")
    analyze_btn = gr.Button("🔎 Анализировать")

    analyze_btn.click(analyze_fits, inputs=file_input, outputs=[result_text, result_image])

# ==================== ЗАПУСК ПРИЛОЖЕНИЯ ====================
app.launch()


