import gradio as gr
import numpy as np
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
import matplotlib.pyplot as plt
import io

def analyze_fits(fits_file):
    # Открываем FITS-файл
    with fits.open(fits_file.name) as hdul:
        data = hdul[1].data
        time = data['TIME']
        flux = data['PDCSAP_FLUX']

    # Убираем пустые значения
    mask = ~np.isnan(time) & ~np.isnan(flux)
    time = time[mask]
    flux = flux[mask]

    # Нормализуем поток
    flux = flux / np.median(flux)

    # Применяем Box Least Squares (поиск транзитов)
    bls = BoxLeastSquares(time, flux)
    periods = np.linspace(0.5, 20, 10000)  # от 0.5 до 20 дней
    results = bls.power(periods, 0.05)

    best_period = results.period[np.argmax(results.power)]
    power = np.max(results.power)

    # Создаем график
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(time, flux, color="blue", lw=0.5)
    ax[0].set_title("Кривая блеска (Light Curve)")
    ax[0].set_xlabel("Время (дни)")
    ax[0].set_ylabel("Яркость (отн.)")
    ax[0].grid(True)

    ax[1].plot(results.period, results.power, color="green")
    ax[1].set_title("Периодограмма BLS")
    ax[1].set_xlabel("Период (дни)")
    ax[1].set_ylabel("Мощность сигнала")
    ax[1].grid(True)

    # Сохраняем график
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Простое решение для вывода текста
    if power > 10:  # если сигнал сильный
        result_text = f"🌍 Обнаружен кандидат в экзопланеты (Период: {best_period:.2f} дней)"
    else:
        result_text = "❌ Экзопланета не обнаружена"

    return result_text, buf

app = gr.Interface(
    fn=analyze_fits,
    inputs=gr.File(label="Загрузите FITS-файл (Kepler/TESS)"),
    outputs=[gr.Textbox(label="Результат"), gr.Image(label="График анализа")],
    title="AI Exoplanet Detector (Kepler/TESS)",
    description="ИИ анализирует световую кривую с помощью метода Box Least Squares и определяет, есть ли признаки транзита планеты."
)

app.launch()

