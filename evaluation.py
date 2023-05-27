import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def evaluate(real: cv2.Mat, sr_result: cv2.Mat):
    '''
    一个简单的评估函数，评估生成的图像质量。
    '''
    if len(real.shape)==2: # Real image is processed as grayscale
        sr_result = cv2.cvtColor(sr_result, cv2.COLOR_BGR2GRAY)
    psnr = peak_signal_noise_ratio(real, sr_result)
    print("Peak SNR: %f (More is better)"%(psnr))
    ssim = structural_similarity(real, sr_result)
    print("Structural similarity: %f (More is better)"%(ssim))
