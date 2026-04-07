#Имплементация на LSB Watermark

import matplotlib
matplotlib.use('TkAgg')  # <- това оправя проблема

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ... останалия код

# Зареждане и подготовка
host = cv2.resize(cv2.imread('baboon.png'), (1000, 1000))
host = cv2.cvtColor(host, cv2.COLOR_BGR2GRAY)

wm = cv2.resize(cv2.imread('bird.png'), (1000, 1000))
wm = cv2.cvtColor(wm, cv2.COLOR_BGR2GRAY)
wm_bin = (wm > 128).astype(np.uint8)  # вместо im2bw

# Вграждане (LSB)
watermarked = (host - host % 2 + wm_bin).astype(np.uint8)

# Извличане
extracted = (watermarked % 2 * 255).astype(np.uint8)

extracted_bits = watermarked % 2

# Метрики

wm_flat = wm_bin.flatten().astype(float)
ext_flat = extracted_bits.flatten().astype(float)

wm_flat -= np.mean(wm_flat)
ext_flat -= np.mean(ext_flat)

mse  = np.mean((host.astype(float) - watermarked.astype(float))**2)
psnr = 10 * np.log10(255**2 / mse)
ber  = np.mean(extracted_bits != wm_bin)
nc = np.dot(wm_flat, ext_flat) / \
     (np.linalg.norm(wm_flat) * np.linalg.norm(ext_flat) + 1e-10)

print(f'PSNR : {psnr:.2f} dB')
print(f'BER  : {ber*100:.4f}%')
print(f'NC   : {nc:.6f}')
print(f'MSE   : {mse:.6f}')

# Визуализация
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0,0].imshow(host,        cmap='gray'); axes[0,0].set_title('Original image')
axes[0,1].imshow(watermarked, cmap='gray'); axes[0,1].set_title('Watermarked image')
axes[1,0].imshow(np.abs(host.astype(int) - watermarked.astype(int)) * 20,
                 cmap='gray');             axes[1,0].set_title('Differnece (x20)')
axes[1,1].imshow(extracted,   cmap='gray'); axes[1,1].set_title('Extracted watermark')
for ax in axes.flat: ax.axis('off')
plt.tight_layout()
plt.savefig('result.png')  # запазва като файл вместо да показва прозорец