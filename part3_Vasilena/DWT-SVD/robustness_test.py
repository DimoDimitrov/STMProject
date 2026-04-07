# robustness_dwt_svd.py

import numpy as np
import pywt
import os
from io import BytesIO
from PIL import Image, ImageFilter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import csv
import matplotlib.pyplot as plt

ALPHA = 0.1  # сила на водния знак


# ── core DWT–SVD алгоритъм ────────────────────────────────────────────────────
def dwt2(arr, wavelet='haar', level=1):
    coeffs = pywt.wavedec2(arr, wavelet=wavelet, level=level)
    return list(coeffs)


def idwt2(coeffs, wavelet='haar'):
    return pywt.waverec2(coeffs, wavelet=wavelet)


def embed(host_arr, wm_arr, alpha=ALPHA):
    """
    DWT–SVD вграждане:
      host -> DWT -> LL_h
      LL_h = U_h S_h V_h^T
      wm_resized -> SVD -> U_w S_w V_w^T
      S_h' = S_h + alpha * S_w
      LL_h' = U_h diag(S_h') V_h^T
      обратно DWT -> watermarked image

    Връща:
      watermarked_image, LL_original (за извличане)
    """

    # 1) DWT на host
    coeffs = dwt2(host_arr, wavelet='haar', level=1)
    LL_h = coeffs[0].astype(np.float64)

    # 2) SVD върху LL_h
    U_h, S_h, Vt_h = np.linalg.svd(LL_h, full_matrices=False)

    # 3) SVD върху watermark-а (оразмерен до LL)
    wm_img = Image.fromarray(wm_arr.astype(np.uint8))
    wm_resized_img = wm_img.resize((LL_h.shape[1], LL_h.shape[0]), Image.BILINEAR)
    W_resized = np.array(wm_resized_img, dtype=np.float64)

    U_w, S_w, Vt_w = np.linalg.svd(W_resized, full_matrices=False)

    # 4) комбиниране на сингулярните стойности
    k = min(len(S_h), len(S_w))
    S_h_new = S_h.copy()
    S_h_new[:k] = S_h[:k] + alpha * S_w[:k]

    # 5) нова LL' подлента
    LL_h_new = U_h @ np.diag(S_h_new) @ Vt_h

    # 6) обратно DWT
    coeffs[0] = LL_h_new
    watermarked = idwt2(coeffs, wavelet='haar')
    watermarked = np.clip(watermarked, 0, 255)

    return watermarked, LL_h  # връщаме и оригиналната LL за извличане


def extract(attacked_arr, LL_orig, wm_orig, alpha=ALPHA, wavelet='haar'):
    """
    DWT–SVD извличане (non-blind):

      attacked -> DWT -> LL_wm
      LL_orig = U_o S_o V_o^T
      LL_wm   = U_w S_wm V_w^T
      S_w_est ≈ (S_wm - S_o) / alpha

      watermark_orig (resized to LL) -> SVD -> U_wm, S_wm0, V_wm^T
      W_est_LL = U_wm diag(S_w_est) V_wm^T
      resize обратно до размера на watermark-а
    """

    # DWT на атакуваното изображение
    coeffs_wm = dwt2(attacked_arr, wavelet=wavelet, level=1)
    LL_wm = coeffs_wm[0].astype(np.float64)

    # SVD на оригиналната и водоотмаркираната LL
    _, S_o, _ = np.linalg.svd(LL_orig.astype(np.float64), full_matrices=False)
    _, S_wm, _ = np.linalg.svd(LL_wm, full_matrices=False)

    k = min(len(S_o), len(S_wm))
    S_w_est = (S_wm[:k] - S_o[:k]) / alpha

    # SVD върху watermark-а (оразмерен до LL размера)
    wm_img = Image.fromarray(wm_orig.astype(np.uint8))
    wm_resized_img = wm_img.resize((LL_orig.shape[1], LL_orig.shape[0]), Image.BILINEAR)
    W_resized = np.array(wm_resized_img, dtype=np.float64)
    U_wm, S_w0, Vt_wm = np.linalg.svd(W_resized, full_matrices=False)

    # изграждаме S_w_full с оценените сингулярни стойности
    S_w_full = np.zeros_like(S_w0)
    S_w_full[:k] = S_w_est

    W_est_LL = U_wm @ np.diag(S_w_full) @ Vt_wm

    # обратно до размера на watermark-а
    wm_h, wm_w = wm_orig.shape
    W_est_img = Image.fromarray(W_est_LL)
    W_est_resized = W_est_img.resize((wm_w, wm_h), Image.BILINEAR)
    W_est_resized = np.array(W_est_resized, dtype=np.float64)
    W_est_resized = np.clip(W_est_resized, 0, 255).astype(np.uint8)

    return W_est_resized


# ── метрики ───────────────────────────────────────────────────────────────────
def compute_nc(orig, extr):
    o = orig.astype(np.float64).ravel()
    e = extr.astype(np.float64).ravel()
    return float(np.dot(o, e) / (np.linalg.norm(o) * np.linalg.norm(e) + 1e-10))


def compute_ber(orig, extr, threshold=128):
    o_bits = (orig.astype(np.float64).ravel() >= threshold).astype(int)
    e_bits = (np.clip(extr, 0, 255).astype(np.float64).ravel() >= threshold).astype(int)
    return float(np.sum(o_bits != e_bits) / len(o_bits))


def all_metrics(host, attacked, wm_orig, wm_extracted):
    img_psnr = psnr(np.uint8(np.clip(host, 0, 255)),
                    np.uint8(np.clip(attacked, 0, 255)), data_range=255)
    img_ssim = ssim(np.uint8(np.clip(host, 0, 255)),
                    np.uint8(np.clip(attacked, 0, 255)), data_range=255)
    wm_nc  = compute_nc(wm_orig, wm_extracted)
    wm_ber = compute_ber(wm_orig, wm_extracted)
    return img_psnr, img_ssim, wm_nc, wm_ber


# ── атаки (същите като при DWT–DCT) ───────────────────────────────────────────
def attack_jpeg(arr, quality):
    img = Image.fromarray(np.uint8(arr))
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert('L'), dtype=np.float64)


def attack_jpeg2000(arr, quality_layers):
    img = Image.fromarray(np.uint8(arr))
    buf = BytesIO()
    img.save(buf, format='JPEG2000',
             quality_mode='rates', quality_layers=[quality_layers])
    buf.seek(0)
    return np.array(Image.open(buf).convert('L'), dtype=np.float64)


def attack_webp(arr, quality):
    img = Image.fromarray(np.uint8(arr))
    buf = BytesIO()
    img.save(buf, format='WEBP', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert('L'), dtype=np.float64)


def attack_png(arr):
    img = Image.fromarray(np.uint8(arr))
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return np.array(Image.open(buf).convert('L'), dtype=np.float64)


def attack_gaussian_noise(arr, sigma):
    return np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255)


def attack_salt_pepper(arr, amount):
    out = arr.copy()
    n = int(amount * out.size)
    idx = np.random.choice(out.size, n, replace=False)
    out.ravel()[idx[:n//2]] = 255
    out.ravel()[idx[n//2:]] = 0
    return out


def attack_blur(arr, radius):
    img = Image.fromarray(np.uint8(arr))
    return np.array(img.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float64)


def attack_median(arr):
    img = Image.fromarray(np.uint8(arr))
    return np.array(img.filter(ImageFilter.MedianFilter(size=3)), dtype=np.float64)


def attack_rotate(arr, angle):
    size = arr.shape[0]
    img = Image.fromarray(np.uint8(arr))
    return np.array(
        img.rotate(angle, expand=False).resize((size, size), Image.LANCZOS),
        dtype=np.float64)


def attack_crop(arr, pct):
    size = arr.shape[0]
    m = int(size * pct / 2)
    cropped = arr[m:size-m, m:size-m]
    return np.array(
        Image.fromarray(np.uint8(cropped)).resize((size, size), Image.LANCZOS),
        dtype=np.float64)


def attack_scale(arr, factor):
    size = arr.shape[0]
    new = int(size * factor)
    img = Image.fromarray(np.uint8(arr))
    return np.array(
        img.resize((new, new), Image.LANCZOS).resize((size, size), Image.LANCZOS),
        dtype=np.float64)


def attack_brightness(arr, factor):
    return np.clip(arr * factor, 0, 255)


# ── графики (идентични на DWT–DCT версията) ───────────────────────────────────
def plot_robustness_metrics(rows, output_dir):
    labels = [f"{r['Категория']} \n{r['Атака']}" for r in rows]
    psnr_vals = [r['PSNR (dB)'] for r in rows]
    ssim_vals = [r['SSIM'] for r in rows]
    nc_vals   = [r['NC'] for r in rows]
    ber_vals  = [r['BER'] for r in rows]

    x = np.arange(len(labels))
    width = 0.6

    # PSNR
    plt.figure(figsize=(16, 6))
    plt.bar(x, psnr_vals, width, color='steelblue')
    plt.xticks(x, labels, rotation=90)
    plt.ylabel('PSNR [dB]')
    plt.title('Устойчивост (DWT-SVD): PSNR по атаки')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'svd_robustness_psnr.png'))
    plt.close()

    # SSIM
    plt.figure(figsize=(16, 6))
    plt.bar(x, ssim_vals, width, color='seagreen')
    plt.xticks(x, labels, rotation=90)
    plt.ylabel('SSIM')
    plt.title('Устойчивост (DWT-SVD): SSIM по атаки')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'svd_robustness_ssim.png'))
    plt.close()

    # NC
    plt.figure(figsize=(16, 6))
    plt.bar(x, nc_vals, width, color='darkorange')
    plt.xticks(x, labels, rotation=90)
    plt.ylabel('NC')
    plt.ylim(0, 1.05)
    plt.title('Устойчивост (DWT-SVD): NC на watermark-а по атаки')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'svd_robustness_nc.png'))
    plt.close()

    # BER
    plt.figure(figsize=(16, 6))
    plt.bar(x, ber_vals, width, color='crimson')
    plt.xticks(x, labels, rotation=90)
    plt.ylabel('BER')
    plt.ylim(0, 1.05)
    plt.title('Устойчивост (DWT-SVD): BER на watermark-а по атаки')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'svd_robustness_ber.png'))
    plt.close()


def plot_compression_metrics(comp_rows, output_dir):
    formats = sorted(set(r['Формат'] for r in comp_rows))

    # PSNR
    plt.figure(figsize=(10, 6))
    for fmt in formats:
        xs, ys = [], []
        for r in comp_rows:
            if r['Формат'] != fmt:
                continue
            param = r['Параметър']
            if '=' in param:
                val = float(param.split('=')[1])
            else:
                val = 0.0
            xs.append(val)
            ys.append(r['PSNR'])
        plt.plot(xs, ys, marker='o', label=fmt)

    plt.xlabel('Параметър на компресията (качество / rate)')
    plt.ylabel('PSNR [dB]')
    plt.title('Компресия (DWT-SVD): PSNR при различни формати')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'svd_compression_psnr.png'))
    plt.close()

    # NC
    plt.figure(figsize=(10, 6))
    for fmt in formats:
        xs, ys = [], []
        for r in comp_rows:
            if r['Формат'] != fmt:
                continue
            param = r['Параметър']
            if '=' in param:
                val = float(param.split('=')[1])
            else:
                val = 0.0
            xs.append(val)
            ys.append(r['NC'])
        plt.plot(xs, ys, marker='o', label=fmt)

    plt.xlabel('Параметър на компресията (качество / rate)')
    plt.ylabel('NC')
    plt.ylim(0, 1.05)
    plt.title('Компресия (DWT-SVD): NC на watermark-а при различни формати')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'svd_compression_nc.png'))
    plt.close()


# ── ПАРАМЕТРИЧНО ИЗСЛЕДВАНЕ на компресия ─────────────────────────────────────
def compression_study(host, watermarked, wm_orig, LL_orig, output_dir):
    print("\n=== ПАРАМЕТРИЧНО ИЗСЛЕДВАНЕ (DWT-SVD): Компресионни формати ===")
    rows = []

    # JPEG
    for q in range(10, 101, 10):
        attacked = attack_jpeg(watermarked, q)
        extr = extract(attacked, LL_orig, wm_orig, alpha=ALPHA)
        p, s, nc, ber = all_metrics(host, attacked, wm_orig, extr)
        rows.append({'Формат': 'JPEG', 'Параметър': f'Q={q}',
                     'PSNR': round(p, 2), 'SSIM': round(s, 4),
                     'NC': round(nc, 4), 'BER': round(ber, 4)})

    # JPEG2000
    for rate in [1, 2, 5, 10, 20, 40]:
        try:
            attacked = attack_jpeg2000(watermarked, rate)
            extr = extract(attacked, LL_orig, wm_orig, alpha=ALPHA)
            p, s, nc, ber = all_metrics(host, attacked, wm_orig, extr)
            rows.append({'Формат': 'JPEG2000', 'Параметър': f'rate={rate}',
                         'PSNR': round(p, 2), 'SSIM': round(s, 4),
                         'NC': round(nc, 4), 'BER': round(ber, 4)})
        except Exception as e:
            print(f"JPEG2000 rate={rate} пропуснато: {e}")

    # WebP
    for q in range(10, 101, 10):
        attacked = attack_webp(watermarked, q)
        extr = extract(attacked, LL_orig, wm_orig, alpha=ALPHA)
        p, s, nc, ber = all_metrics(host, attacked, wm_orig, extr)
        rows.append({'Формат': 'WebP', 'Параметър': f'Q={q}',
                     'PSNR': round(p, 2), 'SSIM': round(s, 4),
                     'NC': round(nc, 4), 'BER': round(ber, 4)})

    # PNG baseline
    attacked = attack_png(watermarked)
    extr = extract(attacked, LL_orig, wm_orig, alpha=ALPHA)
    p, s, nc, ber = all_metrics(host, attacked, wm_orig, extr)
    rows.append({'Формат': 'PNG', 'Параметър': 'lossless',
                 'PSNR': round(p, 2), 'SSIM': round(s, 4),
                 'NC': round(nc, 4), 'BER': round(ber, 4)})

    csv_path = f"{output_dir}/svd_compression_study.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['Формат', 'Параметър', 'PSNR', 'SSIM', 'NC', 'BER'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Записано: {csv_path}")

    plot_compression_metrics(rows, output_dir)
    return rows


# ── пълно тестване на атаки ───────────────────────────────────────────────────
def run_robustness_tests(host_path, wm_path, output_dir='./result_tests'):
    os.makedirs(output_dir, exist_ok=True)

    host = np.array(Image.open(host_path).resize((2048, 2048),
                    Image.LANCZOS).convert('L'), dtype=np.float64)
    wm   = np.array(Image.open(wm_path).resize((128, 128),
                    Image.LANCZOS).convert('L'), dtype=np.float64)

    watermarked, LL_orig = embed(host, wm, alpha=ALPHA)
    wm_orig = np.uint8(wm)

    attacks = [
        ("Без атака",       "Без атака",           lambda x: x),
        ("Компресия",       "JPEG Q=90",           lambda x: attack_jpeg(x, 90)),
        ("Компресия",       "JPEG Q=70",           lambda x: attack_jpeg(x, 70)),
        ("Компресия",       "JPEG Q=50",           lambda x: attack_jpeg(x, 50)),
        ("Компресия",       "JPEG Q=30",           lambda x: attack_jpeg(x, 30)),
        ("Компресия",       "WebP Q=80",           lambda x: attack_webp(x, 80)),
        ("Компресия",       "PNG lossless",        lambda x: attack_png(x)),
        ("Шум",             "Gaussian σ=5",        lambda x: attack_gaussian_noise(x, 5)),
        ("Шум",             "Gaussian σ=15",       lambda x: attack_gaussian_noise(x, 15)),
        ("Шум",             "Salt&Pepper 1%",      lambda x: attack_salt_pepper(x, 0.01)),
        ("Шум",             "Salt&Pepper 5%",      lambda x: attack_salt_pepper(x, 0.05)),
        ("Филтриране",      "Gaussian Blur r=1",   lambda x: attack_blur(x, 1)),
        ("Филтриране",      "Gaussian Blur r=3",   lambda x: attack_blur(x, 3)),
        ("Филтриране",      "Median Filter 3×3",   lambda x: attack_median(x)),
        ("Геометрични",     "Ротация 5°",          lambda x: attack_rotate(x, 5)),
        ("Геометрични",     "Ротация 15°",         lambda x: attack_rotate(x, 15)),
        ("Геометрични",     "Ротация 45°",         lambda x: attack_rotate(x, 45)),
        ("Геометрични",     "Изрязване 10%",       lambda x: attack_crop(x, 0.10)),
        ("Геометрични",     "Изрязване 25%",       lambda x: attack_crop(x, 0.25)),
        ("Геометрични",     "Мащабиране 0.5×",     lambda x: attack_scale(x, 0.5)),
        ("Геометрични",     "Мащабиране 2×",       lambda x: attack_scale(x, 2.0)),
        ("Яркост",          "Яркост ×0.8",         lambda x: attack_brightness(x, 0.8)),
        ("Яркост",          "Яркост ×1.2",         lambda x: attack_brightness(x, 1.2)),
    ]

    print("=== ПЪЛНО ТЕСТВАНЕ НА АТАКИ (DWT-SVD) ===")
    rows = []
    for category, name, fn in attacks:
        attacked = fn(watermarked)
        extr = extract(attacked, LL_orig, wm_orig, alpha=ALPHA)
        p, s, nc, ber = all_metrics(host, attacked, wm_orig, extr)
        rows.append({'Категория': category, 'Атака': name,
                     'PSNR (dB)': round(p, 2), 'SSIM': round(s, 4),
                     'NC': round(nc, 4), 'BER': round(ber, 4)})

        safe = name.replace(' ', '_').replace('=', '').replace('°', 'deg').replace('×', 'x')
        Image.fromarray(extr).save(f"{output_dir}/svd_wm_{safe}.jpg")

    csv_path = f"{output_dir}/svd_robustness_results.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['Категория', 'Атака', 'PSNR (dB)', 'SSIM', 'NC', 'BER'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nЗаписано: {csv_path}")

    plot_robustness_metrics(rows, output_dir)

    # параметрично изследване на компресия
    compression_study(host, watermarked, wm_orig, LL_orig, output_dir)


if __name__ == '__main__':
    run_robustness_tests(
        host_path='./pictures/lena.png',
        wm_path='./pictures/wm.png',
        output_dir='./result_tests'
    )