import os
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

OUTPUT = True
DEBUG = True
# DEBUG = False
def debug(*args):
    if DEBUG:
        print(*args)

# 使用高分辨率图像合成低分辨率图像序列
def lr_generation(ratio: float, 
                  image_file_path: str = "./ori.jpg", 
                  serie_len: int = 10,
                  rotate_angle_range: float = 10, 
                  shift_pixel_range: float = 5, 
                  blur_kernel_size: int = 15, 
                  blur_kernel_sigma: float = 5, 
                  noise_sigma: float = 10, 
                  use_grayscale: bool = False
                  ):
    """
    使用高分辨率图像合成低分辨率图像序列。

    Params: 
    ---
    ratio: 超分系数
    
    其余参数顾名思义，懒得写了
    
    Returns: 
    ---
    X_gt, Y_seq, H_kernel
    
    X_gt: ground truth
    
    Y_seq: downsampled images
    
    H_kernel: blur kernel
    """
    if OUTPUT:
        os.makedirs("./outputs/", exist_ok=True)

    # 读取高分辨率图像ori.jpg为合成数据素材
    X_ori = cv2.imread(image_file_path)
    if use_grayscale:
        X_ori = cv2.cvtColor(X_ori, cv2.COLOR_RGB2GRAY)
        if OUTPUT:
            cv2.imwrite("./outputs/X_ori_grayscale.jpg", X_ori)

    # 添加帧间运动，生成高分辨率图像序列，以第一张为真值(ground truth)
    X_seq: List[cv2.Mat] = [X_ori]
    for i in range(1, serie_len):
        alpha = np.random.uniform(-rotate_angle_range, rotate_angle_range)
        delta_x = np.random.uniform(-shift_pixel_range, shift_pixel_range)
        delta_y = np.random.uniform(-shift_pixel_range, shift_pixel_range)
        # 旋转变换矩阵（ahpha为正代表逆时针）
        rotation_matrix = cv2.getRotationMatrix2D(center=(0, 0), angle=alpha, scale=1)
        rotation_matrix = cv2.getRotationMatrix2D(center=(0, 0), angle=i*3-15, scale=1)
        # 平移变换矩阵
        shift_matrix = np.float32([[0, 0, delta_x], [0, 0, delta_y]])
        shift_matrix = np.float32([[0, 0, i*5-25], [0, 0, i*5-25]]) # TODO: This is only for testing. 
        # 对图像应用变换
        X_new: cv2.Mat = cv2.warpAffine(src=X_ori, M=rotation_matrix+shift_matrix, 
                                        dsize=(X_ori.shape[1], X_ori.shape[0]), borderValue=(0, 0, 0))
                                        # dsize里面长和宽是相反的...
        X_seq.append(X_new)
        debug("Ground truth: rotation is", alpha, "and shift is", delta_x, delta_y)
    X_gt = X_seq[0]
    X_gt = X_gt.astype(np.float32)

    # 使用上一步的高分辨率图像序列通过添加图像模糊（统一模糊核）、降采样、图像噪声等图像退化过程生成低分辨率图像序列，
    # 以第一张低分辨率图像为参考图像。
    H_kernel_1d: cv2.Mat = cv2.getGaussianKernel(ksize=blur_kernel_size, sigma=blur_kernel_sigma) # 这是一维高斯！
    H_kernel_2d = H_kernel_1d * H_kernel_1d.T
    Y_seq: List[cv2.Mat] = []
    for i, X in enumerate(X_seq):
        X = cv2.filter2D(src=X, ddepth=-1, kernel=H_kernel_2d) # 模糊
        X = cv2.resize(src=X, dsize=(X.shape[1]//ratio, X.shape[0]//ratio)).astype(np.float32) # 降采样。dsize里面长和宽是相反的...
        noise = np.random.normal(0, noise_sigma, X.shape) # 高斯噪声
        X += noise
        Y_seq.append(X)
        if OUTPUT:
            cv2.imwrite("./outputs/Y_%d.jpg"%i, X)

    # 返回高分辨率图像真值，低分辨率图像序列，以及模糊核
    return X_gt, Y_seq, H_kernel_2d


# 以第一张低分辨率图像为参考图像，计算序列中其他图像相对参考图像的运动
def mv_est(Y_seq: List[cv2.Mat], radius: int = 500, max_corners: int = 10):
    """
    使用光流算法找不同图像之间的参考点，然后用SVD解线性方程组得到坐标变换矩阵。

    Params: 
    ---
    Y_seq = [Y1,...,YN]: 低分辨率图像序列
    
    Returns: 
    ---
    mv_seq = [mv1,...,mvN]: 序列中其他图像相对参考图像的坐标变换（2x3矩阵）
    """
    mv_seq = []
    if len(Y_seq[0].shape)==3: # RGB or BGR or something
        points = cv2.goodFeaturesToTrack(cv2.cvtColor(Y_seq[0].astype(np.uint8), cv2.COLOR_BGR2GRAY), 
                                         maxCorners=max_corners, qualityLevel=0.3, minDistance=10)
    else: # Grayscale
        points = cv2.goodFeaturesToTrack(Y_seq[0].astype(np.uint8), maxCorners=max_corners, qualityLevel=0.3, minDistance=10)
    points: cv2.Mat = points.astype(np.float32)
    points = points.squeeze() # (max_corners, 1, 2) -> (max_corners, 2)
    for i, Y_i in enumerate(Y_seq):
        points_new, st, err = cv2.calcOpticalFlowPyrLK(Y_i.astype(np.uint8), Y_seq[0].astype(np.uint8), points, None)
        points_new: cv2.Mat = points_new.astype(np.float32)
        points_new = points_new.squeeze() # (max_corners, 1, 2) -> (max_corners, 2)
        debug("Points:", points)
        debug("Points_new:", points_new)
        
        # X1*x + X2*y + X3 = x' => AX = B
        # Y1*x + Y2*y + Y3 = y'
        # We have x, y, x', y'. We want X and Y. 
        A_x = np.c_[points, np.ones(max_corners)]
        B_x = points_new[:, 0]
        debug("A_x:", A_x)
        debug("B_x:", B_x)
        U_x, Sigma_x, V_x = np.linalg.svd(A_x)
        Sigma_x = np.diag(Sigma_x)
        debug("U_x, Sigma_x, V_x's shapes:", U_x.shape, Sigma_x.shape, V_x.shape)
        X_vec = V_x*np.c_[np.linalg.inv(Sigma_x), np.zeros((3, max_corners-3))]*U_x.T*B_x
        debug("X_vec:", X_vec)
        
        A_y = points
        B_y = points_new[:, 1]
        U_y, Sigma_y, V_y = np.linalg.svd(A_y)
        Sigma_y = np.diag(Sigma_y)
        Y_vec = np.linalg.inv(Sigma_y*V_y.T)*U_y.T*B_y
        debug("Y_vec:", Y_vec)
        
        
    return mv_seq


# 实现一轮梯度下降迭代
def grad_desent(X: cv2.Mat, Y_seq: List[cv2.Mat], mv_seq, ratio, beta, H_kernel, loss: str = "L2"):
    """
    X : 上一轮重建的高分辨率图像
    Y_seq = [Y1,...,YN] : 低分辨率图像序列
    mv_seq = [mv1,...,mvN] : 估计的运动向量，需要注意的是输入的运动向量是低分辨率图像之间的
    ratio: 超分系数
    beta: 学习率
    loss: 损失函数，可选：["L2", "L1", "L1_BTV"]
    """
    grad_functions: dict[function] = {
        "L2": grad_L2, 
        "L1": grad_L1, 
        "L1_BTV": grad_BTV
    }
    grad: cv2.Mat = grad_functions[loss](X, Y_seq, mv_seq, ratio, H_kernel)
    X -= beta*grad


# 基于L2范数的梯度
def grad_L2(X: cv2.Mat, Y_seq: List[cv2.Mat], mv_seq, ratio, H_kernel) -> cv2.Mat:
    """
    Params: 
    ---
    X : 上一轮重建的高分辨率图像
    Y_seq = [Y1,...,YN] : 低分辨率图像序列
    mv_seq = [mv1,...,mvN] : 估计的运动向量，需要注意的是输入的运动向量是低分辨率图像之间的
    ratio: 超分系数
    H_kernel: 模糊核
    
    公式: 
    ---
    \Sigma F^T_k H^T D^T (D H F_k X - Y_k)
    
    H: 模糊矩阵
    D: 降采样矩阵
    F_k: 几何变换矩阵
    """
    grads = []
    for i in range(len(Y_seq)):
        F_i = np.concatenate([mv_seq[i], np.float32([[0, 0, 1]])]) # (x, y)的坐标变换矩阵
        X_new: cv2.Mat = cv2.warpAffine(src=X, M=F_i[:2], dsize=(X.shape[1]//ratio, X.shape[0]//ratio), borderValue=(0, 0, 0)) # 几何变换
        X_new = cv2.filter2D(src=X_new, ddepth=-1, kernel=H_kernel) # 模糊
        X_new = cv2.resize(src=X_new, dsize=(X.shape[1]//ratio, X.shape[0]//ratio)) # 降采样。dsize里面长和宽是相反的...
        
        debug("X_new:", X_new)
        debug("Y_seq[i]:", Y_seq[i])
        estimated_error = X_new-Y_seq[i]
        debug("estimated error's shape:", estimated_error.shape)
        debug("estimated error:",estimated_error)
        print("estimated error's L2 norm:", 
              np.sqrt(np.sum(np.square(estimated_error)) / 
                      (np.shape(estimated_error)[0] * np.shape(estimated_error)[1]))
              )
        
        tmp = cv2.resize(src=estimated_error, dsize=(X.shape[1], X.shape[0])) # 上采样
        tmp = my_conv_transpose_2d(tmp, H_kernel) # 逆模糊（转置卷积）
        F_i_inverse = np.linalg.inv(F_i)
        debug("Inverse (x, y) transform matrix:", F_i_inverse)
        grad: cv2.Mat = cv2.warpAffine(src=tmp, M=F_i_inverse[:2], dsize=(X.shape[1], X.shape[0]), borderValue=(0, 0, 0)) # 逆几何变换
        debug("grad's shape:", grad.shape)
        grads.append(grad)
    return np.sum(grads, axis=0)

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
    debug("tmp.shape:", image.shape, "kernel.shape:", kernel.shape)
    image = F.conv_transpose2d(image, kernel, padding=len(H_kernel)//2, stride=1) # 转置卷积（逆模糊）
    image = image.squeeze(dim=0).permute([1, 2, 0]).cpu().numpy() # (batch, c, w, h) -> (w, h, c)
    debug("tmp.shape after conv_transpose2d:", image.shape)
    return image

# 基于L1范数的梯度
def grad_L1(X: cv2.Mat, Y_seq: List[cv2.Mat], mv_seq, ratio, H_kernel) -> cv2.Mat:
    """
    Params: 
    ---
    X : 上一轮重建的高分辨率图像
    Y_seq = [Y1,...,YN] : 低分辨率图像序列
    mv_seq = [mv1,...,mvN] : 估计的运动向量，需要注意的是输入的运动向量是低分辨率图像之间的
    ratio: 超分系数
    H_kernel: 模糊核
    
    公式: 
    ---
    \Sigma F^T_k H^T D^T sign(D H F_k X - Y_k)
    
    H: 模糊矩阵
    D: 降采样矩阵
    F_k: 几何变换矩阵
    """
    grads = []
    for i in range(len(Y_seq)):
        F_i = np.concatenate([mv_seq[i], np.float32([[0, 0, 1]])]) # (x, y)的坐标变换矩阵
        X_new: cv2.Mat = cv2.warpAffine(src=X, M=F_i[:2], dsize=(X.shape[1]//ratio, X.shape[0]//ratio), borderValue=(0, 0, 0)) # 几何变换
        X_new = cv2.filter2D(src=X_new, ddepth=-1, kernel=H_kernel) # 模糊
        X_new = cv2.resize(src=X_new, dsize=(X.shape[1]//ratio, X.shape[0]//ratio)) # 降采样。dsize里面长和宽是相反的...
        
        debug("X_new:", X_new)
        debug("Y_seq[i]:", Y_seq[i])
        estimated_error = X_new-Y_seq[i]
        debug("estimated error's shape:", estimated_error.shape)
        debug("estimated error:",estimated_error)
        print("estimated error's L1 norm:", 
              np.sum(np.absolute(estimated_error)) / 
              (np.shape(estimated_error)[0] * np.shape(estimated_error)[1]))
        signed_estimated_error = np.sign(estimated_error)
        
        tmp = cv2.resize(src=signed_estimated_error, dsize=(X.shape[1], X.shape[0])) # 上采样
        tmp = my_conv_transpose_2d(tmp, H_kernel) # 逆模糊（转置卷积）
        F_i_inverse = np.linalg.inv(F_i)
        debug("Inverse (x, y) transform matrix:", F_i_inverse)
        grad: cv2.Mat = cv2.warpAffine(src=tmp, M=F_i_inverse[:2], dsize=(X.shape[1], X.shape[0]), borderValue=(0, 0, 0)) # 逆几何变换
        debug("grad's shape:", grad.shape)
        grads.append(grad)
    return np.sum(grads, axis=0)


# 基于L1范数，使用双边全变分正则项的梯度
def grad_BTV(X, Y_seq, mv_seq, ratio, lamda, P, alpha):
    """
    X : 上一轮重建的高分辨率图像
    Y_seq = [Y1,...,YN] : 低分辨率图像序列
    mv_seq = [mv1,...,mvN] : 估计的运动向量，需要注意的是输入的运动向量是低分辨率图像之间的
    ratio: 超分系数
    lamda: 正则项系数
    P, alpha: BTV相关的参数
    """

    pass


# 完成迭代优化的主函数
def main():
    X_ori, Y_seq, H_kernel= lr_generation(10, "./kq.jpg", rotate_angle_range=0, shift_pixel_range=0, noise_sigma=5, use_grayscale=False)
    mv_seq = mv_est(Y_seq)
    # X_start = cv2.resize(Y_seq[0], dsize=(X_ori.shape[1], X_ori.shape[0]))
    X_start = np.zeros(X_ori.shape, dtype=np.float32)
    for epoch in range(1, 51):
        # grad_desent(X_start, Y_seq, mv_seq, ratio=10, beta=0.5, H_kernel=H_kernel, loss="L1")
        grad_desent(X_start, Y_seq, mv_seq, ratio=10, beta=0.005, H_kernel=H_kernel, loss="L2")
    # for epoch in range(1, 21):
        # grad_desent(X_start, Y_seq, mv_seq, ratio=10, beta=0.0003, H_kernel=H_kernel, loss="L2")
        print("epoch %d completed. "%epoch)
        cv2.imwrite("./outputs/epoch_%d.jpg"%epoch, X_start)


if __name__ == '__main__':
    main()
