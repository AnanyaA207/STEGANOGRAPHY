import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_psnr_ssim(orig_path, stego_path):
    # Read images
    A = cv2.imread(orig_path)
    B = cv2.imread(stego_path)

    if A is None or B is None:
        print("❌ One of the images couldn't be read. Check the file paths.")
        return None, None

    # Convert both to same size just in case
    if A.shape != B.shape:
        B = cv2.resize(B, (A.shape[1], A.shape[0]))

    # Compute PSNR
    psnr_val = cv2.PSNR(A, B)

    # Convert to grayscale for SSIM
    A_gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    B_gray = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)

    # Compute SSIM with higher precision
    ssim_val = ssim(A_gray, B_gray, data_range=A_gray.max() - A_gray.min())

    # Round off but keep good precision
    psnr_val = round(psnr_val, 4)
    ssim_val = round(ssim_val, 6)

    print(f"✅ PSNR: {psnr_val} dB | SSIM: {ssim_val}")
    return psnr_val, ssim_val
