import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# 2D DCT
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# 2D IDCT
def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def load_grayscale_image(path: Path) -> np.ndarray:
    """Load an image as float32 grayscale (0..255)."""
    img = plt.imread(path)

    # If image has multiple channels, convert to grayscale.
    if img.ndim == 3:
        img = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]

    # Many readers return floats in [0, 1]; rescale to [0, 255].
    if img.max() <= 1.0:
        img = img * 255.0

    return np.float32(img)


def jpeg_quant_matrix() -> np.ndarray:
    """Standard JPEG luminance quantization matrix."""
    return np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ],
        dtype=np.float64,
    )


def scaled_quant_matrix(quality: int) -> np.ndarray:
    """Scale quantization matrix from JPEG-like quality in [1..100]."""
    quality = max(1, min(100, quality))
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    q = np.floor((jpeg_quant_matrix() * scale + 50) / 100)
    q[q < 1] = 1
    return q


def pad_to_block_size(img: np.ndarray, block_size: int = 8) -> tuple[np.ndarray, int, int]:
    h, w = img.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded, h, w


def compress_decompress_image(img: np.ndarray, quality: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Apply block DCT + quantization + inverse DCT to whole image.

    Returns:
        reconstructed: reconstructed image (float32, 0..255)
        kept_ratio: fraction of non-zero quantized coefficients
        dct_coeffs: per-pixel-view of DCT coeff magnitudes (cropped to original image size)
        quantized_coeffs: per-pixel-view of quantized DCT coeffs (cropped)
    """
    block_size = 8
    padded, h, w = pad_to_block_size(img, block_size=block_size)
    centered = padded.astype(np.float64) - 128.0
    recon = np.zeros_like(centered)
    q = scaled_quant_matrix(quality)
    nonzero = 0
    total = 0

    # Store full coefficient fields so we can visualize them.
    dct_coeffs = np.zeros_like(centered)
    quantized_coeffs = np.zeros_like(centered)

    for row in range(0, centered.shape[0], block_size):
        for col in range(0, centered.shape[1], block_size):
            block = centered[row : row + block_size, col : col + block_size]
            coeff = dct2(block)
            quantized = np.round(coeff / q)
            nonzero += int(np.count_nonzero(quantized))
            total += quantized.size

            dct_coeffs[row : row + block_size, col : col + block_size] = coeff
            quantized_coeffs[row : row + block_size, col : col + block_size] = quantized

            dequantized = quantized * q
            recon[row : row + block_size, col : col + block_size] = idct2(dequantized)

    reconstructed = np.clip(recon + 128.0, 0, 255)
    return (
        reconstructed[:h, :w].astype(np.float32),
        nonzero / total,
        dct_coeffs[:h, :w],
        quantized_coeffs[:h, :w],
    )


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = float(np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255.0 ** 2) / mse)


parser = argparse.ArgumentParser(description="DCT-compress the full grayscale image.")
parser.add_argument(
    "--image",
    type=str,
    default="image2.png",
    help="Path to input image (default: image.png in this folder).",
)
parser.add_argument("--quality", type=int, default=50, help="Compression quality 1..100 (higher = better).")
args = parser.parse_args()

image_path = Path(args.image)
if not image_path.is_absolute():
    # First respect the current working directory; if missing, fallback to script dir.
    cwd_candidate = Path.cwd() / image_path
    script_candidate = Path(__file__).resolve().parent / image_path
    image_path = cwd_candidate if cwd_candidate.exists() else script_candidate

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

img = load_grayscale_image(image_path)

if img.shape[0] < 8 or img.shape[1] < 8:
    raise ValueError("Image must be at least 8x8 pixels.")

reconstructed, kept_ratio, dct_coeffs, quantized_coeffs = compress_decompress_image(img, quality=args.quality)
score = psnr(img, reconstructed)
print(f"Quality={args.quality}, PSNR={score:.2f} dB, kept_coefficients={kept_ratio*100:.2f}%")

# Визуализация
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title(f"Reconstructed (Q={args.quality})")
plt.imshow(reconstructed, cmap='gray', vmin=0, vmax=255)
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("DCT")
plt.imshow(np.log1p(np.abs(dct_coeffs)), cmap='gray')
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Quantized DCT")
plt.imshow(np.log1p(np.abs(quantized_coeffs)), cmap='gray')
plt.axis("off")

plt.show()


# cd "d:\Codes\UNI\STM\project"
# .\.venv\Scripts\Activate.ps1
# python "STMProject\part2_Dimo\dct_image.py" --image "STMProject\part2_Dimo\images\image.png" --quality 50