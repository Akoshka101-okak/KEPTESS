import os
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ✅ FIX: No display
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings("ignore")

try:
    from astropy.io import fits
    from astropy.timeseries import BoxLeastSquares
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("Astropy not available - basic mode")

# Safe ML mock
def ml_score(sde, depth):
    return min(1.0, (sde/12 + depth*2000)/2)

def safe_analyze(files, sde_thresh=6.0, min_p=0.2, max_p=500):
    """Safe version - no crashes."""
    if not files or not HAS_ASTROPY:
        return "❌ Astropy/FITS error. Upload valid Kepler FITS.", None
    
    try:
        segments = []
        for f in files[:3]:  # Limit memory
            try:
                with fits.open(f.name) as hdul:
                    hdu = hdul[1]
                    if not hasattr(hdu, 'data') or hdu.data is None:
                        continue
                    
                    cols = hdu.columns.names
                    flux_col = next((c for c in cols if 'FLUX' in c.upper()), None)
                    if not flux_col or 'TIME' not in [c.upper() for c in cols]:
                        continue
                    
                    time_name = next(c for c in cols if c.upper() == 'TIME')
                    time = hdu.data[time_name]
                    flux = hdu.data[flux_col]
                    
                    # Clean
                    mask = np.isfinite(time) & np.isfinite(flux)
                    time, flux = time[mask], flux[mask]
                    
                    if len(time) < 20:
                        continue
                    
                    # Normalize
                    med = np.nanmedian(flux)
                    flux_norm = (flux - med) / np.nanstd(flux)
                    segments.append((time, flux_norm))
                    
            except Exception as e:
                print(f"FITS error {f.name}: {e}")
                continue
        
        if len(segments) == 0:
            return "❌ No valid flux/time data found", None
        
        # Simple stitch
        time_all = np.concatenate([t for t,f in segments])
        flux_all = np.concatenate([f for t,f in segments])
        order = np.argsort(time_all)
        time_all, flux_all = time_all[order], flux_all[order]
        
        # BLS safe
        bls = BoxLeastSquares(time_all, flux_all)
        total_span = time_all[-1] - time_all[0]
        periods = np.linspace(max(0.2, total_span/1000), min(30, total_span/3), 5000)
        power = bls.power(periods, 0.1).power
        
        peak_idx = np.argmax(power)
        best_period = periods[peak_idx]
        sde = (power[peak_idx] - np.median(power)) / np.std(power)
        
        depth = -np.min(flux_all)
        ml_p = ml_score(sde, depth)
        confidence = min(100, sde*8 + ml_p*30)
        
        status = "🟢 DETECTED" if confidence > 70 else "🟡 Candidate"
        
        result = f"""
{status} | Score: {confidence:.0f}%
SDE: {sde:.2f} | ML: {ml_p:.1%}
Period: {best_period:.4f}d | Depth: {depth:.4f}
Points: {len(time_all)} | Span: {total_span:.1f}d
"""
        
        # Safe plot
        fig, ax = plt.subplots(figsize=(10,6), facecolor='black')
        ax.plot(time_all, flux_all, 'w.', ms=0.5, alpha=0.6)
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        plt.title("Light Curve + BLS Peak", color='white')
        
        buf = io.BytesIO()
        plt.savefig(buf, dpi=150, facecolor='black')
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        
        return result, img
        
    except Exception as e:
        return f"❌ Runtime: {str(e)[:100]}", None

# UI - MINIMAL NO CRASH
css = "body{background:#0a0a1a;color:#ddd;}.gr-button{background:#00aaff;}"

with gr.Blocks(css=css) as demo:
    gr.Markdown("# 🔭 Exoplanet Finder - Safe Version")
    
    file_input = gr.File(file_count="multiple", file_types=[".fits"])
    sde_slider = gr.Slider(4, 12, 6.0)
    
    btn = gr.Button("Analyze")
    output_text = gr.Textbox()
    output_img = gr.Image()
    
    btn.click(safe_analyze, [file_input, sde_slider], [output_text, output_img])

if __name__ == "__main__":
    demo.launch()