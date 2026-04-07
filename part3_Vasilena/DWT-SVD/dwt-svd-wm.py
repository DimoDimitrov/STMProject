import os
import numpy as np
import pywt
from PIL import Image, ImageDraw, ImageFont

# ---------------------- Paths / config ----------------------

current_path = os.path.dirname(os.path.abspath(__file__))

# image names in ./pictures/
image = 'lena.png'
watermark = 'wm.png'

# DWT / SVD parameters
DWT_MODEL = 'haar'
DWT_LEVEL = 1
ALPHA = 0.1          # embedding strength
WATERMARK_SIZE = 128 # watermark size used in slides

# ensure folders exist
os.makedirs(os.path.join(current_path, 'dataset'), exist_ok=True)
os.makedirs(os.path.join(current_path, 'result'), exist_ok=True)

# ---------------------- Helpers ----------------------

def convert_image(image_name, size):
    """
    Load ./pictures/<image_name>, resize to (size,size), convert to grayscale,
    save to ./dataset, and return as float64 array.
    """
    src_path = os.path.join(current_path, 'pictures', image_name)
    dst_path = os.path.join(current_path, 'dataset', image_name)

    img = Image.open(src_path).resize((size, size), Image.BILINEAR)
    img = img.convert('L')
    img.save(dst_path)

    image_array = np.array(img.getdata(), dtype=np.float64).reshape((size, size))
    return image_array


def process_coefficients(imArray, model, level):
    coeffs = pywt.wavedec2(data=imArray, wavelet=model, level=level)
    coeffs_H = list(coeffs)
    return coeffs_H


def print_image_from_array(image_array, name, label=None):
    image_array_copy = image_array.clip(0, 255)
    image_array_copy = image_array_copy.astype("uint8")
    img = Image.fromarray(image_array_copy)

    if label:
        draw = ImageDraw.Draw(img)
        font_size = max(10, img.width // 40)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        padding = 10

        if img.mode == 'L':
            bg_color = 0
            text_color = 255
        elif img.mode == 'RGBA':
            bg_color = (0, 0, 0, 180)
            text_color = (255, 255, 255, 255)
        else:  # RGB
            bg_color = (0, 0, 0)
            text_color = (255, 255, 255)

        bbox = draw.textbbox((padding, padding), label, font=font)
        draw.rectangle(
            [bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5],
            fill=bg_color
        )
        draw.text((padding, padding), label, fill=text_color, font=font)

    out_path = os.path.join(current_path, 'result', name)
    img.save(out_path)

# ---------------------- DWT–SVD embed/extract ----------------------

def embed_watermark_svd(LL, watermark_array, alpha=ALPHA):
    """
    DWT-SVD embedding (глобално, без блокове):

      LL_host = U_h * S_h * V_h^T
      W_resized -> SVD: U_w * S_w * V_w^T
      S_h' = S_h + alpha * S_w (формула от статиите)

      LL' = U_h * diag(S_h') * V_h^T
    """
    # SVD върху LL на host
    U_h, S_h, Vt_h = np.linalg.svd(LL, full_matrices=False)

    # SVD върху watermark (оразмерен до LL)
    wm_img = Image.fromarray(watermark_array.astype(np.uint8))
    wm_resized_img = wm_img.resize((LL.shape[1], LL.shape[0]), Image.BILINEAR)
    W_resized = np.array(wm_resized_img, dtype=np.float64)

    U_w, S_w, Vt_w = np.linalg.svd(W_resized, full_matrices=False)

    # комбиниране на сингулярните стойности
    k = min(len(S_h), len(S_w))
    S_h_new = S_h.copy()
    S_h_new[:k] = S_h[:k] + alpha * S_w[:k]

    # ново LL'
    LL_new = U_h @ np.diag(S_h_new) @ Vt_h

    # за извличане ще ни трябва U_w, Vt_w (може да ги пресметнем пак от оригиналния watermark)
    return LL_new


def recover_watermark_svd(original_LL, watermarked_LL, original_watermark, alpha=ALPHA):
    """
    Non-blind извличане:

      original_LL  -> S_h
      watermarked_LL -> S_hw
      S_w_est ≈ (S_hw - S_h) / alpha

      watermark_resized ≈ U_w * diag(S_w_est) * V_w^T
      (U_w, V_w от SVD на watermark-а).
    """
    # SVD на оригиналния и водоотмаркирания LL
    _, S_h, _ = np.linalg.svd(original_LL, full_matrices=False)
    _, S_hw, _ = np.linalg.svd(watermarked_LL, full_matrices=False)

    # оценка на сингулярните стойности на watermark-а
    k = min(len(S_h), len(S_hw))
    S_w_est = (S_hw[:k] - S_h[:k]) / alpha

    # SVD на оригиналния watermark (оразмерен до LL)
    wm_img = Image.fromarray(original_watermark.astype(np.uint8))
    wm_resized_img = wm_img.resize((original_LL.shape[1], original_LL.shape[0]),
                                   Image.BILINEAR)
    W_resized = np.array(wm_resized_img, dtype=np.float64)
    U_w, S_w, Vt_w = np.linalg.svd(W_resized, full_matrices=False)

    # подменяме собствените сингулярни стойности със S_w_est
    S_w_full = np.zeros_like(S_w)
    S_w_full[:k] = S_w_est
    W_est_LL = U_w @ np.diag(S_w_full) @ Vt_w

    # връщаме към оригиналния размер на watermark-а
    wm_h, wm_w = original_watermark.shape
    W_est_img = Image.fromarray(W_est_LL)
    W_est_resized = W_est_img.resize((wm_w, wm_h), Image.BILINEAR)
    W_est_resized = np.array(W_est_resized, dtype=np.float64)

    W_est_resized = np.clip(W_est_resized, 0, 255).astype(np.uint8)
    return W_est_resized

# ---------------------- Main pipeline (steps like DWT–DCT) ----------------------
def visualize_svd_components(LL, prefix="LL"):
    """
    Записва:
      - U матрицата като изображение
      - S (като диагонална матрица) като изображение
      - Vt матрицата като изображение
      - 3 реконструкции с първите k сингулярни стойности (k=5,20,100)
    """
    U, S, Vt = np.linalg.svd(LL, full_matrices=False)

    # Помощна функция за нормализиране до [0,255]
    def norm_to_uint8(arr):
        arr = arr.astype(np.float64)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr, dtype=np.uint8)
        arr_norm = (arr - mn) / (mx - mn)
        return (arr_norm * 255).clip(0, 255).astype(np.uint8)

    # U
    U_img = norm_to_uint8(U)
    print_image_from_array(
        U_img,
        f'svd_{prefix}_U.jpg',
        label=f'SVD на {prefix}: матрица U'
    )

    # S като диагонална матрица
    S_mat = np.zeros_like(LL)
    k = min(len(S), LL.shape[0], LL.shape[1])
    for i in range(k):
        S_mat[i, i] = S[i]
    S_img = norm_to_uint8(S_mat)
    print_image_from_array(
        S_img,
        f'svd_{prefix}_S_diag.jpg',
        label=f'SVD на {prefix}: диагонална матрица S'
    )

    # Vt
    Vt_img = norm_to_uint8(Vt)
    print_image_from_array(
        Vt_img,
        f'svd_{prefix}_Vt.jpg',
        label=f'SVD на {prefix}: матрица V^T'
    )

    # Частични реконструкции с първите k сингулярни стойности
    for k_approx in [5, 20, 100]:
        k_use = min(k_approx, len(S))
        S_k = np.zeros_like(S)
        S_k[:k_use] = S[:k_use]
        LL_k = U @ np.diag(S_k) @ Vt
        LL_k_uint8 = LL_k.clip(0, 255).astype(np.uint8)

        print_image_from_array(
            LL_k_uint8,
            f'svd_{prefix}_recon_{k_use}.jpg',
            label=f'SVD на {prefix}: реконструкция с първите {k_use} сингулярни стойности'
        )

def w2d(dummy=None):
    model = DWT_MODEL
    level = DWT_LEVEL

    # 1) Оригинална картинка (input image)
    image_array = convert_image(image, 2048)
    print_image_from_array(
        image_array,
        'step1_original.jpg',
        label='Стъпка 1: Оригинална картинка'
    )

    # 2) Watermark
    watermark_array = convert_image(watermark, WATERMARK_SIZE)
    print_image_from_array(
        watermark_array,
        'step2_watermark.jpg',
        label='Стъпка 2: Watermark'
    )

    # 3) DWT на входното изображение -> LL, LH, HL, HH
    coeffs_image = process_coefficients(image_array, model, level=level)
    LL_original = coeffs_image[0]
    print_image_from_array(
        LL_original,
        'step3_dwt_LL_band.jpg',
        label='Стъпка 3: DWT коефициенти (LL подлента)'
    )

    # 3a) Визуализация на SVD компонентите на LL
    visualize_svd_components(LL_original, prefix="LL")

    # 4) SVD върху LL и SVD върху watermark-а + вграждане в сингулярните стойности
    LL_embedded = embed_watermark_svd(LL_original, watermark_array, alpha=ALPHA)
    print_image_from_array(
        LL_embedded,
        'step4_LL_with_watermark.jpg',
        label='Стъпка 4: LL с вграден watermark (DWT-SVD)'
    )

    # 5) Замяна на LL с LL' и обратен DWT -> изображение с watermark
    coeffs_image[0] = LL_embedded
    image_array_H = pywt.waverec2(coeffs_image, model)
    print_image_from_array(
        image_array_H,
        'step5_image_with_watermark.jpg',
        label='Стъпка 5: Реконструирана картинка с watermark'
    )

    # 6) DWT на водоотмаркираното изображение (за извличане)
    coeffs_watermarked = process_coefficients(image_array_H, model, level=level)
    LL_watermarked = coeffs_watermarked[0]

    # 7) Извличане на watermark-а чрез разлика на сингулярните стойности
    recovered_wm = recover_watermark_svd(
        LL_original,
        LL_watermarked,
        watermark_array,
        alpha=ALPHA
    )
    print_image_from_array(
        recovered_wm,
        'step6_recovered_watermark.jpg',
        label='Стъпка 6: Възстановен watermark (DWT-SVD)'
    )

if __name__ == '__main__':
    w2d("test")