import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def evaluate(real: cv2.Mat, sr_result: cv2.Mat):
    psnr = peak_signal_noise_ratio(real, sr_result)
    print("Peak SNR: %f"%(psnr))
    ssim = structural_similarity(real, sr_result)
    print("Structural similarity: %f"%(ssim))
