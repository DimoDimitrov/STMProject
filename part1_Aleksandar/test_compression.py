#2. Тестване при JPEG компресиране при различни QF

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Зареждане
img = cv2.imread('baboon.png')
original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

qf_values = [10, 30, 50, 70, 90, 100]

print(f"{'QF':<6} {'PSNR (dB)':<12} {'SSIM':<10} {'MSE':<12} {'Size (KB)'}")
print("-" * 50)

for qf in qf_values:
    # Компресия и декомпресия в паметта
    _, enc = cv2.imencode('.jpg', original, [cv2.IMWRITE_JPEG_QUALITY, qf])
    compressed = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

    # Метрики
    mse  = np.mean((original.astype(float) - compressed.astype(float))**2)
    psnr = 10 * np.log10(255**2 / mse)
    ssim_val = ssim(original, compressed)
    size_kb = len(enc) / 1024

    print(f"{qf:<6} {psnr:<12.2f} {ssim_val:<10.4f} {mse:<12.2f} {size_kb:.1f}")