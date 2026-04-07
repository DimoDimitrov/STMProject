import numpy as np
import pywt
import os
from PIL import Image
from scipy.fftpack import dct
from scipy.fftpack import idct
from PIL import Image, ImageDraw, ImageFont

current_path = str(os.path.dirname(os.path.abspath(__file__)))

image = 'lena.png'
watermark = 'wm.png'


def convert_image(image_name, size):
    img = Image.open('./pictures/' + image_name).resize((size, size), 1)
    img = img.convert('L')
    img.save('./dataset/' + image_name)

    image_array = np.array(img.getdata(), dtype=np.float64).reshape((size, size))
    print(image_array[0][0])
    print(image_array[10][10])

    return image_array


def process_coefficients(imArray, model, level):
    coeffs = pywt.wavedec2(data=imArray, wavelet=model, level=level)
    coeffs_H = list(coeffs)
    return coeffs_H


def embed_mod2(coeff_image, coeff_watermark, offset=0):
    for i in range(coeff_watermark.__len__()):
        for j in range(coeff_watermark[i].__len__()):
            coeff_image[i*2+offset][j*2+offset] = coeff_watermark[i][j]
    return coeff_image


def embed_mod4(coeff_image, coeff_watermark):
    for i in range(coeff_watermark.__len__()):
        for j in range(coeff_watermark[i].__len__()):
            coeff_image[i*4][j*4] = coeff_watermark[i][j]
    return coeff_image


def embed_watermark(watermark_array, orig_image):
    watermark_flat = watermark_array.ravel()
    ind = 0

    for x in range(0, orig_image.__len__(), 8):
        for y in range(0, orig_image.__len__(), 8):
            if ind < watermark_flat.__len__():
                subdct = orig_image[x:x+8, y:y+8]
                subdct[5][5] = watermark_flat[ind]
                orig_image[x:x+8, y:y+8] = subdct
                ind += 1

    return orig_image


def apply_dct(image_array):
    size = image_array[0].__len__()
    all_subdct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subpixels = image_array[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_subdct[i:i+8, j:j+8] = subdct
    return all_subdct


def inverse_dct(all_subdct):
    size = all_subdct[0].__len__()
    all_subidct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subidct = idct(idct(all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
            all_subidct[i:i+8, j:j+8] = subidct
    return all_subidct


def get_watermark(dct_watermarked_coeff, watermark_size):
    subwatermarks = []

    for x in range(0, dct_watermarked_coeff.__len__(), 8):
        for y in range(0, dct_watermarked_coeff.__len__(), 8):
            coeff_slice = dct_watermarked_coeff[x:x+8, y:y+8]
            subwatermarks.append(coeff_slice[5][5])

    watermark = np.array(subwatermarks).reshape(watermark_size, watermark_size)
    return watermark


def recover_watermark(image_array, model='haar', level=1):
    coeffs_watermarked_image = process_coefficients(image_array, model, level=level)
    dct_watermarked_coeff = apply_dct(coeffs_watermarked_image[0])

    watermark_array = get_watermark(dct_watermarked_coeff, 128)
    watermark_array = np.uint8(watermark_array)

    img = Image.fromarray(watermark_array)
    img.save('./result/recovered_watermark.jpg')


def print_image_from_array(image_array, name, label=None):
    image_array_copy = image_array.clip(0, 255)
    image_array_copy = image_array_copy.astype("uint8")
    img = Image.fromarray(image_array_copy)

    if label:
        draw = ImageDraw.Draw(img)
        font_size = max(10, img.width // 40)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", font_size)
        except:
            font = ImageFont.load_default()

        padding = 10

        # Адаптиране на цветовете спрямо режима на картинката
        if img.mode == 'L':
            bg_color = 0        # черно (grayscale int)
            text_color = 255    # бяло (grayscale int)
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

    img.save('./result/' + name)


def w2d(img):
    model = 'haar'
    level = 1

    image_array = convert_image(image, 2048)
    print_image_from_array(image_array, 'step1_original.jpg',
        label='Стъпка 1: Оригинална картинка')

    watermark_array = convert_image(watermark, 128)
    print_image_from_array(watermark_array, 'step2_watermark.jpg',
        label='Стъпка 2: Watermark')

    coeffs_image = process_coefficients(image_array, model, level=level)
    print_image_from_array(coeffs_image[0], 'step3_dwt_LL_band.jpg',
        label='Стъпка 3: DWT коефициенти (LL подлента)')

    dct_array = apply_dct(coeffs_image[0])
    print_image_from_array(dct_array, 'step4_dct_of_LL.jpg',
        label='Стъпка 4: DCT върху LL подлентата')

    dct_array = embed_watermark(watermark_array, dct_array)
    print_image_from_array(dct_array, 'step5_dct_with_watermark.jpg',
        label='Стъпка 5: DCT коефициенти с вграден watermark')

    coeffs_image[0] = inverse_dct(dct_array)
    print_image_from_array(coeffs_image[0], 'step6_idct_result.jpg',
        label='Стъпка 6: Обратен DCT (IDCT) след вграждането')

    image_array_H = pywt.waverec2(coeffs_image, model)
    print_image_from_array(image_array_H, 'image_with_watermark.jpg',
        label='Стъпка 7: Реконструирана картинка с watermark')

    recover_watermark(image_array=image_array_H, model=model, level=level)

w2d("test")
