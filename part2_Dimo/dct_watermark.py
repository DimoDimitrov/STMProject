import cv2
import numpy as np
from pathlib import Path

# --- Paths (relative to this script) ---
BASE = Path(__file__).resolve().parent
IMAGE_PATH = BASE / "images" / "image.png"
WATERMARK_PATH = BASE / "images" / "watermark.png"
OUT_DIR = BASE / "result"
OUT_DIR.mkdir(exist_ok=True)

# --- Зареждане на изображения ---
image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
watermark = cv2.imread(str(WATERMARK_PATH), cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")
if watermark is None:
    raise FileNotFoundError(f"Could not load watermark: {WATERMARK_PATH}")

# Resize watermark до размера на изображението
watermark = cv2.resize(watermark, (image.shape[1], image.shape[0]))

# --- DCT на изображението ---
image_f = np.float32(image)
dct_image = cv2.dct(image_f)

# --- Нормализиране на watermark ---
watermark = np.float32(watermark) / 255.0

# --- Вграждане ---
alpha = 20  # сила на водния знак
dct_watermarked = dct_image + alpha * watermark

# --- Обратно преобразване (IDCT) ---
watermarked_image = cv2.idct(dct_watermarked)
watermarked_image = np.uint8(np.clip(watermarked_image, 0, 255))

cv2.imwrite(str(OUT_DIR / "watermarked.png"), watermarked_image)

# Визуална разлика (водният знак е в DCT домейна — в пространството често е почти невидим)
diff = cv2.absdiff(watermarked_image, image)
max_d = float(np.max(diff))
mean_d = float(np.mean(diff))
print(f"Spatial change vs original: mean_abs={mean_d:.4f}, max_abs={max_d:.1f} (0 = identical)")
cv2.imwrite(str(OUT_DIR / "difference.png"), diff)
# Усилена разлика за екран — иначе човешкото око не вижда нищо при малък alpha
gain = 25.0
amplified = np.clip(diff.astype(np.float32) * gain, 0, 255).astype(np.uint8)
cv2.imwrite(str(OUT_DIR / "difference_amplified.png"), amplified)

# --- Симулиране на атака (JPEG компресия) ---
cv2.imwrite(
    str(OUT_DIR / "compressed.jpg"),
    watermarked_image,
    [int(cv2.IMWRITE_JPEG_QUALITY), 50],
)
attacked = cv2.imread(str(OUT_DIR / "compressed.jpg"), cv2.IMREAD_GRAYSCALE)

# --- Извличане на watermark ---
attacked_f = np.float32(attacked)
dct_attacked = cv2.dct(attacked_f)

extracted = (dct_attacked - dct_image) / alpha
extracted = np.uint8(np.clip(extracted * 255, 0, 255))

cv2.imwrite(str(OUT_DIR / "extracted.png"), extracted)
print(
    "Done. Outputs:",
    OUT_DIR / "watermarked.png",
    OUT_DIR / "difference.png",
    OUT_DIR / "difference_amplified.png",
    OUT_DIR / "compressed.jpg",
    OUT_DIR / "extracted.png",
)
