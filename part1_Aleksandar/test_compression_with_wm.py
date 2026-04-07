#3. JPG компресия със воден знак

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Зареждане
host = cv2.cvtColor(cv2.resize(cv2.imread('baboon.png'), (512, 512)), cv2.COLOR_BGR2GRAY)
wm   = cv2.cvtColor(cv2.resize(cv2.imread('bird.png'),   (512, 512)), cv2.COLOR_BGR2GRAY)

# Подготовка на watermark
wm_bin = (wm > 128).astype(np.uint8)

# LSB вграждане
watermarked = (host - host % 2 + wm_bin).astype(np.uint8)

# JPEG атака + извличане + метрики
qf_values = [10, 30, 50, 70, 90,100]
ber_list, psnr_list, ssim_list, nc_list = [], [], [], []

print(f"{'QF':<6} {'BER (%)':<12} {'PSNR (dB)':<12} {'SSIM':<10} {'NC'}")
print("-" * 55)

for qf in qf_values:
    # JPEG компресия
    _, enc       = cv2.imencode('.jpg', watermarked, [cv2.IMWRITE_JPEG_QUALITY, qf])
    attacked     = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

    # Извличане на watermark след атака
    extracted = attacked % 2

    # BER
    ber = np.mean(extracted != wm_bin) * 100

    # PSNR (атакувано vs оригинален хост)
    mse  = np.mean((host.astype(float) - attacked.astype(float))**2)
    psnr = 10 * np.log10(255**2 / mse)

    # SSIM
    ssim_val = ssim(host, attacked)

    # NC
    wm_f  = wm_bin.flatten().astype(float)
    ext_f = extracted.flatten().astype(float)
    nc    = np.dot(wm_f, ext_f) / (np.linalg.norm(wm_f) * np.linalg.norm(ext_f) + 1e-10)

    ber_list.append(ber)
    psnr_list.append(psnr)
    ssim_list.append(ssim_val)
    nc_list.append(nc)

    print(f"{qf:<6} {ber:<12.4f} {psnr:<12.2f} {ssim_val:<10.4f} {nc:.6f}")

# Графики
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('LSB Watermark robustness след JPEG атака', fontsize=13)

metrics = [
    (ber_list,  'BER (%)',    'Bit Error Rate',         '#E24B4A'),
    (psnr_list, 'PSNR (dB)', 'Peak Signal-to-Noise',   '#378ADD'),
    (ssim_list, 'SSIM',      'Structural Similarity',  '#1D9E75'),
    (nc_list,   'NC',        'Normalized Correlation', '#BA7517'),
]

for ax, (vals, ylabel, title, color) in zip(axes, metrics):
    ax.plot(qf_values, vals, marker='o', color=color, linewidth=2, markersize=7)
    for x, y in zip(qf_values, vals):
        ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('JPEG Quality Factor')
    ax.set_ylabel(ylabel)
    ax.set_xticks(qf_values)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ber_vs_qf.png', dpi=150, bbox_inches='tight')

