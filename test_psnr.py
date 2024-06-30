import cv2
import numpy as np
import math
import os

def psnr_test(true_file,test_file_path,true_file_path):
    for index,file in enumerate(os.listdir(test_file_path)):
        test_path = os.path.join(test_file_path,file)
        true_path = os.path.join(true_file_path,os.listdir(true_file_path)[true_file[index]])
        img1 = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # 计算均方误差 MSE
        mse_r = np.mean((img1[:, :, 0] - img2[:, :, 0]) ** 2)
        mse_g = np.mean((img1[:, :, 1] - img2[:, :, 1]) ** 2)
        mse_b = np.mean((img1[:, :, 2] - img2[:, :, 2]) ** 2)

        mse = (mse_r + mse_g + mse_b) / 3
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 10 * math.log10((max_pixel ** 2) / mse)

        print(f"{index} PSNR value is {psnr} dB")

if __name__ == '__main__':
    psnr_test([0,8,16,24,32,40],'my_logs/myvideo/testset_190000','data/nerf_llff_data/myvideo/images_8')