import cv2
import numpy as np

def bilateral_filter(img, d=7, sigma_color=75, sigma_space=75):
    """
    Bilateral filter

    [Args]:
        img: Input image (H, W, C), uint8 type
        d: Diameter of pixel neighborhood (kernel size, odd number)
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
    
    [Returns]:
        Filtered image as uint8 numpy array
    """
    H, W, C = img.shape
    pad = d // 2  

    padded_img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    output = np.zeros_like(img, dtype=np.float32)
    
    # 공간 가우시안 커널 미리 계산
    y, x = np.ogrid[-pad:pad+1, -pad:pad+1]
    spatial_kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma_space ** 2))
    
    # 채널별 처리
    for c in range(C):
        input_channel = padded_img[:,:,c].astype(np.float32)
        output_channel = output[:,:,c]
        
        for i in range(H):
            for j in range(W):
                window = input_channel[i:i+d, j:j+d]
                center = input_channel[i+pad, j+pad]
                
                # 색상 차이 기반 가우시안 
                color_diff = window - center
                color_kernel = np.exp(-(color_diff ** 2) / (2.0 * sigma_color ** 2))
                
                # 가중치 결합
                weights = spatial_kernel * color_kernel
                weights_sum = np.sum(weights)

                if weights_sum > 1e-6:  # 0으로 나누기 방지
                    output_channel[i, j] = np.sum(weights * window) / weights_sum
                else:
                    output_channel[i, j] = center

    return np.clip(output, 0, 255).astype(np.uint8)


def median_filter(img, kernel_size):
    """
    Median filter
    
    [Args]:
        img: Input image (H, W, C), uint8 type
        kernel_size: size of filter, odd number

    
    [Returns]:
        Filtered image as uint8 numpy array
    """
    H, W, C = img.shape
    pad = kernel_size // 2
    padded_img = np.pad(img, ((pad, pad), (pad , pad), (0, 0)), mode='edge')
    output= np.zeros_like(img, dtype=np.float32)
    for c in range(C):
        for y in range(H):
            for x in range(W):
                window = padded_img[y:y + kernel_size, x:x + kernel_size, c]
                output[y,x,c] = np.median(window)
    return output
