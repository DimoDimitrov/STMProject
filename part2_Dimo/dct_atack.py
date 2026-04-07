# robustness_test.py

import numpy as np
import pywt
import os
from io import BytesIO
from PIL import Image, ImageFilter
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import csv
import matplotlib.pyplot as plt


# ── core алгоритъм ────────────────────────────────────────────────────────────
def apply_dct(arr):
    size = arr.shape[0]
    out = np.empty_like(arr)
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            blk = arr[i:i+8, j:j+8]
            out[i:i+8, j:j+8] = dct(dct(blk.T, norm='ortho').T, norm='ortho')
    return out


def inverse_dct(arr):
    size = arr.shape[0]
    out = np.empty_like(arr)
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            out[i:i+8, j:j+8] = idct(
                idct(arr[i:i+8, j:j+8].T, norm='ortho').T, norm='ortho')
    return out


def embed(host_arr, wm_arr):
    coeffs = list(pywt.wavedec2(host_arr, 'haar', level=1))
    dct_ll = apply_dct(coeffs[0])
    flat = wm_arr.ravel()
    ind = 0
    for x in range(0, dct_ll.shape[0], 8):
        for y in range(0, dct_ll.shape[1], 8):
            if ind < len(flat):
                dct_ll[x:x+8, y:y+8][5][5] = flat[ind]
                ind += 1
    coeffs[0] = inverse_dct(dct_ll)
    return np.clip(pywt.waverec2(coeffs, 'haar'), 0, 255)


def extract(wm_img_arr, wm_size=128):
    coeffs = pywt.wavedec2(wm_img_arr, 'haar', level=1)
    dct_ll = apply_dct(coeffs[0])
    vals = []
    for x in range(0, dct_ll.shape[0], 8):
        for y in range(0, dct_ll.shape[1], 8):
            vals.append(dct_ll[x:x+8, y:y+8][5][5])
    return np.array(vals[:wm_size**2]).reshape(wm_size, wm_size)


# ── метрики ───────────────────────────────────────────────────────────────────
def compute_nc(orig, extr):
    """Normalized Correlation [0–1]: сходство на извлечения watermark"""
    o = orig.astype(np.float64).ravel()
    e = extr.astype(np.float64).ravel()
    return float(np.dot(o, e) / (np.linalg.norm(o) * np.linalg.norm(e) + 1e-10))


def compute_ber(orig, extr, threshold=128):
    """Bit Error Rate [0–1]: дял грешно извлечени битове (по threshold бинаризация)"""
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


# ── атаки ─────────────────────────────────────────────────────────────────────
def attack_jpeg(arr, quality):
    img = Image.fromarray(np.uint8(arr))
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert('L'), dtype=np.float64)


def attack_jpeg2000(arr, quality_layers):
    """JPEG 2000 — wavelet-базирана компресия (по-добра от JPEG при ниско качество)"""
    img = Image.fromarray(np.uint8(arr))
    buf = BytesIO()
    img.save(buf, format='JPEG2000', quality_mode='rates', quality_layers=[quality_layers])
    buf.seek(0)
    return np.array(Image.open(buf).convert('L'), dtype=np.float64)


def attack_webp(arr, quality):
    img = Image.fromarray(np.uint8(arr))
    buf = BytesIO()
    img.save(buf, format='WEBP', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert('L'), dtype=np.float64)


def attack_png(arr):
    """PNG е lossless — не губи информация, служи като baseline"""
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


# ── ГРАФИКИ ───────────────────────────────────────────────────────────────────
def plot_robustness_metrics(rows, output_dir):
    """
    Чертае bar графики за PSNR, SSIM, NC и BER за всички атаки.
    """
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
    plt.title('Устойчивост: PSNR по атаки')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_psnr.png'))
    plt.close()

    # SSIM
    plt.figure(figsize=(16, 6))
    plt.bar(x, ssim_vals, width, color='seagreen')
    plt.xticks(x, labels, rotation=90)
    plt.ylabel('SSIM')
    plt.title('Устойчивост: SSIM по атаки')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_ssim.png'))
    plt.close()

    # NC
    plt.figure(figsize=(16, 6))
    plt.bar(x, nc_vals, width, color='darkorange')
    plt.xticks(x, labels, rotation=90)
    plt.ylabel('NC')
    plt.ylim(0, 1.05)
    plt.title('Устойчивост: NC на watermark-а по атаки')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_nc.png'))
    plt.close()

    # BER
    plt.figure(figsize=(16, 6))
    plt.bar(x, ber_vals, width, color='crimson')
    plt.xticks(x, labels, rotation=90)
    plt.ylabel('BER')
    plt.ylim(0, 1.05)
    plt.title('Устойчивост: BER на watermark-а по атаки')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_ber.png'))
    plt.close()


def plot_compression_metrics(comp_rows, output_dir):
    """
    Линейни графики за компресионните формати:
    PSNR и NC спрямо параметъра (качество/rate).
    """
    formats = sorted(set(r['Формат'] for r in comp_rows))

    # PSNR
    plt.figure(figsize=(10, 6))
    for fmt in formats:
        xs = []
        ys = []
        for r in comp_rows:
            if r['Формат'] != fmt:
                continue
            # в Параметър има "Q=90" или "rate=5" или "lossless"
            param = r['Параметър']
            if '=' in param:
                val = float(param.split('=')[1])
            else:
                # напр. PNG lossless – просто 0
                val = 0.0
            xs.append(val)
            ys.append(r['PSNR'])
        plt.plot(xs, ys, marker='o', label=fmt)

    plt.xlabel('Параметър на компресията (качество / rate)')
    plt.ylabel('PSNR [dB]')
    plt.title('Компресия: PSNR при различни формати')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compression_psnr.png'))
    plt.close()

    # NC
    plt.figure(figsize=(10, 6))
    for fmt in formats:
        xs = []
        ys = []
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
    plt.title('Компресия: NC на watermark-а при различни формати')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compression_nc.png'))
    plt.close()


# ── ПАРАМЕТРИЧНО ИЗСЛЕДВАНЕ на компресия ─────────────────────────────────────
def compression_study(host, watermarked, wm_orig, output_dir):
    """
    Сравнява JPEG, JPEG2000 и WebP при различни нива на компресия.
    Записва резултатите в compression_study.csv и връща списък от речници.
    """
    print("\n=== ПАРАМЕТРИЧНО ИЗСЛЕДВАНЕ: Компресионни формати ===")
    rows = []

    # JPEG: quality 10..100 стъпка 10
    for q in range(10, 101, 10):
        attacked = attack_jpeg(watermarked, q)
        extr = extract(attacked)
        p, s, nc, ber = all_metrics(host, attacked, wm_orig, extr)
        rows.append({'Формат': 'JPEG', 'Параметър': f'Q={q}',
                     'PSNR': round(p, 2), 'SSIM': round(s, 4),
                     'NC': round(nc, 4), 'BER': round(ber, 4)})
        print(f"JPEG  Q={q:3d} | PSNR={p:.2f} | SSIM={s:.4f} | NC={nc:.4f} | BER={ber:.4f}")

    # JPEG2000: rate 1..40
    for rate in [1, 2, 5, 10, 20, 40]:
        try:
            attacked = attack_jpeg2000(watermarked, rate)
            extr = extract(attacked)
            p, s, nc, ber = all_metrics(host, attacked, wm_orig, extr)
            rows.append({'Формат': 'JPEG2000', 'Параметър': f'rate={rate}',
                         'PSNR': round(p, 2), 'SSIM': round(s, 4),
                         'NC': round(nc, 4), 'BER': round(ber, 4)})
            print(f"JP2K  rate={rate:2d} | PSNR={p:.2f} | SSIM={s:.4f} | NC={nc:.4f} | BER={ber:.4f}")
        except Exception as e:
            print(f"JPEG2000 rate={rate} пропуснато: {e}")

    # WebP: quality 10..100 стъпка 10
    for q in range(10, 101, 10):
        attacked = attack_webp(watermarked, q)
        extr = extract(attacked)
        p, s, nc, ber = all_metrics(host, attacked, wm_orig, extr)
        rows.append({'Формат': 'WebP', 'Параметър': f'Q={q}',
                     'PSNR': round(p, 2), 'SSIM': round(s, 4),
                     'NC': round(nc, 4), 'BER': round(ber, 4)})
        print(f"WebP  Q={q:3d} | PSNR={p:.2f} | SSIM={s:.4f} | NC={nc:.4f} | BER={ber:.4f}")

    # PNG (lossless baseline)
    attacked = attack_png(watermarked)
    extr = extract(attacked)
    p, s, nc, ber = all_metrics(host, attacked, wm_orig, extr)
    rows.append({'Формат': 'PNG', 'Параметър': 'lossless',
                 'PSNR': round(p, 2), 'SSIM': round(s, 4),
                 'NC': round(nc, 4), 'BER': round(ber, 4)})
    print(f"PNG   lossless | PSNR={p:.2f} | SSIM={s:.4f} | NC={nc:.4f} | BER={ber:.4f}")

    csv_path = f"{output_dir}/compression_study.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Формат', 'Параметър', 'PSNR', 'SSIM', 'NC', 'BER'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Записано: {csv_path}")

    # >>> НОВО: графики за компресия
    plot_compression_metrics(rows, output_dir)

    return rows


# ── пълно тестване на атаки ───────────────────────────────────────────────────
def run_robustness_tests(host_path, wm_path, output_dir='./result_tests'):
    os.makedirs(output_dir, exist_ok=True)

    host = np.array(Image.open(host_path).resize((2048, 2048),
                    Image.LANCZOS).convert('L'), dtype=np.float64)
    wm   = np.array(Image.open(wm_path).resize((128, 128),
                    Image.LANCZOS).convert('L'), dtype=np.float64)

    watermarked = embed(host, wm)
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

    print("=== ПЪЛНО ТЕСТВАНЕ НА АТАКИ ===")
    rows = []
    for category, name, fn in attacks:
        attacked = fn(watermarked)
        extr = extract(attacked)
        extr_u8 = np.uint8(np.clip(extr, 0, 255))
        p, s, nc, ber = all_metrics(host, attacked, wm_orig, extr_u8)
        rows.append({'Категория': category, 'Атака': name,
                     'PSNR (dB)': round(p, 2), 'SSIM': round(s, 4),
                     'NC': round(nc, 4), 'BER': round(ber, 4)})
        print(f"[{category:15s}] {name:22s} | PSNR={p:.2f} | SSIM={s:.4f} | NC={nc:.4f} | BER={ber:.4f}")

        safe = name.replace(' ', '_').replace('=', '').replace('°', 'deg').replace('×', 'x')
        Image.fromarray(extr_u8).save(f"{output_dir}/wm_{safe}.jpg")

    csv_path = f"{output_dir}/robustness_results.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Категория', 'Атака', 'PSNR (dB)', 'SSIM', 'NC', 'BER'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nЗаписано: {csv_path}")

    # >>> НОВО: графики за всички атаки
    plot_robustness_metrics(rows, output_dir)

    # Параметрично изследване на компресия
    compression_study(host, watermarked, wm_orig, output_dir)


if __name__ == '__main__':
    run_robustness_tests(
        host_path='STMProject\part2_Dimo\images\image.png',
        wm_path='STMProject\part2_Dimo\images\watermark.png',
        output_dir='./result_tests'
    )