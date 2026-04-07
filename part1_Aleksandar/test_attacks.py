# 4. Другите атаки

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Зареждане и подготовка ───────────────────────────────
host = cv2.cvtColor(cv2.resize(cv2.imread('baboon.png'), (512,512)), cv2.COLOR_BGR2GRAY)
wm   = cv2.cvtColor(cv2.resize(cv2.imread('bird.png'),   (512,512)), cv2.COLOR_BGR2GRAY)

wm_bin      = (wm > 128).astype(np.uint8)
watermarked = (host - host % 2 + wm_bin).astype(np.uint8)

H, W = watermarked.shape

# ── Помощна функция за метрики ───────────────────────────
def compute(attacked):
    ext  = attacked % 2
    ber  = np.mean(ext != wm_bin) * 100
    mse  = np.mean((host.astype(float) - attacked.astype(float))**2)
    psnr = 10 * np.log10(255**2 / mse)
    sv   = ssim_fn(host, attacked)
    wf   = wm_bin.flatten().astype(float)
    ef   = ext.flatten().astype(float)
    nc   = np.dot(wf, ef) / (np.linalg.norm(wf) * np.linalg.norm(ef) + 1e-10)
    return ber, psnr, sv, nc

# ── Атаки ────────────────────────────────────────────────
attacks = {}

# Gaussian Noise
for sigma in [5, 10]:
    noise = np.random.normal(0, sigma, watermarked.shape)
    att   = np.clip(watermarked.astype(float) + noise, 0, 255).astype(np.uint8)
    attacks[f'Gaussian Noise σ={sigma}'] = compute(att)

# Gaussian Blur
for k in [3, 5]:
    att = cv2.GaussianBlur(watermarked, (k, k), 0)
    attacks[f'Gaussian Blur {k}×{k}'] = compute(att)

# Resize
for scale in [0.5, 0.25]:
    small = cv2.resize(watermarked, (int(W*scale), int(H*scale)))
    att   = cv2.resize(small, (W, H))
    attacks[f'Resize {int(scale*100)}%→100%'] = compute(att)

# Crop
for frac in [0.75, 0.5]:
    ch, cw = int(H*frac), int(W*frac)
    y0, x0 = (H-ch)//2, (W-cw)//2
    cropped = watermarked[y0:y0+ch, x0:x0+cw]
    att     = cv2.resize(cropped, (W, H))
    attacks[f'Crop {int((1-frac)*100)}%'] = compute(att)

# ── Принт на резултатите ─────────────────────────────────
print(f"{'Атака':<30} {'BER (%)':<12} {'PSNR (dB)':<12} {'SSIM':<10} {'NC'}")
print("-" * 70)
for name, (ber, psnr, sv, nc) in attacks.items():
    print(f"{name:<30} {ber:<12.4f} {psnr:<12.2f} {sv:<10.4f} {nc:.4f}")

# ── Таблица като PNG ─────────────────────────────────────
BG     = "#F8FAFC"
DARK   = "#1E293B"
GRAY   = "#64748B"
BORDER = "#CBD5E1"
BLUE   = "#2563EB"
TEAL   = "#0D9488"
RED    = "#DC2626"
AMBER  = "#D97706"

names = list(attacks.keys())
bers  = [attacks[n][0] for n in names]
psnrs = [attacks[n][1] for n in names]
ssims = [attacks[n][2] for n in names]
ncs   = [attacks[n][3] for n in names]

def status(ber):
    if ber < 10:   return "✓ Добро",   "#16A34A"
    if ber < 30:   return "⚠ Слабо",   AMBER
    if ber < 45:   return "✗ Лошо",    RED
    return              "✗ Неприемливо", RED

fig, ax = plt.subplots(figsize=(15, 4.8), facecolor=BG)
ax.set_facecolor(BG)
ax.axis('off')

cols       = ["Атака", "BER (%)", "PSNR (dB)", "SSIM", "NC", "Статус"]
col_widths = [0.26, 0.12, 0.13, 0.11, 0.11, 0.17]
x_starts   = [0.01]
for w in col_widths[:-1]:
    x_starts.append(x_starts[-1] + w)

row_h = 0.11
hdr_y = 0.92

ax.add_patch(plt.Rectangle((0.01, hdr_y-0.015), 0.97, row_h,
    transform=ax.transAxes, facecolor=DARK, zorder=2))
for col, x in zip(cols, x_starts):
    ax.text(x+0.01, hdr_y+0.038, col, transform=ax.transAxes,
            color="#FFFFFF", fontsize=10.5, fontweight='bold', va='center')

for i, (name, ber, psnr, sv, nc) in enumerate(zip(names, bers, psnrs, ssims, ncs)):
    y  = hdr_y - (i+1)*row_h - 0.01
    bg = "#FFFFFF" if i % 2 == 0 else "#F1F5F9"
    ax.add_patch(plt.Rectangle((0.01, y-0.015), 0.97, row_h,
        transform=ax.transAxes, facecolor=bg, zorder=1))

    st_label, st_color = status(ber)
    row_vals  = [name, f"{ber:.2f}%", f"{psnr:.2f}", f"{sv:.4f}", f"{nc:.4f}", st_label]
    row_clrs  = [DARK, RED if ber > 45 else AMBER if ber > 20 else "#16A34A",
                 BLUE, TEAL, DARK, st_color]
    row_bolds = ['normal','bold','bold','bold','normal','bold']

    for val, x, color, fw in zip(row_vals, x_starts, row_clrs, row_bolds):
        ax.text(x+0.01, y+0.038, val, transform=ax.transAxes,
                color=color, fontsize=10, fontweight=fw, va='center')

ax.add_patch(plt.Rectangle((0.01, hdr_y-len(names)*row_h-0.025), 0.97,
    (len(names)+1)*row_h+0.01, transform=ax.transAxes,
    facecolor='none', edgecolor=BORDER, linewidth=1.5, zorder=5))

ax.set_title("LSB Watermark — Robustness при различни атаки",
             color=DARK, fontsize=14, fontweight='bold', pad=14)

plt.tight_layout()
plt.savefig('lsb_attack_table.png', dpi=180, bbox_inches='tight', facecolor=BG)
print("\nТаблицата е записана като 'lsb_attack_table.png'")