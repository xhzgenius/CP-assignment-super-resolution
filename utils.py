'''
这是我写的一些功能函数。
---
——XHZ
'''
import cv2
import numpy as np
import torch
import torch.nn.functional as F

DEBUG = True
DEBUG = False
def debug(*args):
    if DEBUG:
        print(*args)

def solve_equation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''用SVD解方程组AX = B。
    
    Returns:
    ---
    X: 最优解向量。
    '''
    # x_dim = A.shape[1]
    # U, Sigma, V = np.linalg.svd(A)
    # X_vec = np.matmul(np.linalg.inv(V.T), np.matmul(U.T, B)[:x_dim]/Sigma)
    X_vec = np.matmul(np.linalg.pinv(A), B)
    # 他奶奶滴，我怎么不知道pinv这么一个好东西居然是用SVD得出来的，居然可以用来干这个
    return X_vec

def my_conv_transpose_2d(image: cv2.Mat, H_kernel: cv2.Mat) -> cv2.Mat:
    '''
    计算转置卷积（反卷积）的一个功能函数。用于逆模糊步骤。作者：XHZ。
    '''
    if len(image.shape)==2:
        image = torch.as_tensor(image, dtype=torch.float32, device="cuda").unsqueeze(0).unsqueeze(0)
                                                                        # (w, h) -> (batch, c, w, h)
        id3 = torch.as_tensor([[1]], dtype=torch.float32, device="cuda")
    else: # RGB or BGR or something
        image = torch.as_tensor(image, dtype=torch.float32, device="cuda").unsqueeze(0).permute([0, 3, 1, 2]) 
                                                                        # (w, h, c) -> (batch, c, w, h)
        id3 = torch.as_tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device="cuda")
    kernel = torch.as_tensor(H_kernel, dtype=torch.float32, device="cuda") # (w, h)
    kernel = torch.einsum("ij,kl->ijkl", id3, kernel) # (w, h) -> (batch, c, w, h)
    # debug("tmp.shape:", image.shape, "kernel.shape:", kernel.shape)
    image = F.conv_transpose2d(image, kernel, padding=len(H_kernel)//2, stride=1) # 转置卷积（逆模糊）
    image = image.squeeze(dim=0).permute([1, 2, 0]).cpu().numpy() # (batch, c, w, h) -> (w, h, c)
    # debug("tmp.shape after conv_transpose2d:", image.shape)
    return image
