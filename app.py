import numpy as np
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
import matplotlib.pyplot as plt
import io
from PIL import Image

def analyze_fits(fits_file):
    if fits_file is None:
        return "❌ Файл не выбран", None

    # Открываем FITS
    with fits.open(fits_file.name) as hdul:
        data = hdul[1].data
        time = data['TIME']
        # Выбираем правильный столбец flux
        if 'PDCSAP_FLUX' in data.columns.names:
            flux = data['PDCSAP_FLUX']
        elif 'SAP_FLUX' in data.columns.names:
            flux = data['SAP_FLUX']
        else:
            return "❌ Нет подходящего столбца с данными яркости", None

    # Убираем NaN
    mask = ~np.isnan(time) & ~np.isnan(flux)
    time = time[mask]
    flux = flux[mask]

    if len(time) < 10:
        return "❌ Недостаточно данных для анализа", None

    # Нормируем
    flux = flux / np.median(flux)

    # BLS анализ
    bls = BoxLeastSquares(time, flux)
    periods = np.linspace(0.5, 30, 20000)  # расширяем диапазон периодов
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

    # Снижаем порог, чтобы реальные экзопланеты детектировались
    if power > 2.5:  
        result_text = f"🌍 Обнаружен кандидат в экзопланеты (Период: {best_period:.2f} дней, Power: {power:.2f})"
    else:
        result_text = f"❌ Экзопланета не обнаружена (Power: {power:.2f})"

    return result_text, img

   
