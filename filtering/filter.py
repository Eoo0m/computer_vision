import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_laplacian_filter(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]], dtype=np.float32)   #laplacian filter

    kernel = (1/8) * kernel    # Normalization for the kernel.

    H, W = image.shape
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)   #zero padding to maintain the image size
    output = np.zeros_like(image)

    for x in range(H):
        for y in range(W):
            region = padded_image[x:x+3, y:y+3]
            filtered_value = np.sum(region * kernel)
            output[x, y] = filtered_value    #replace pixel by calculated value
    return output.astype(np.uint8)
    

'''
[args]: 
image: color image(H,W,C), uint8 type
kernel_size: size of kernel. asymmetric padding if even number.
sigma: blurring hyperparameter.
A larger sigma results in a smoother, more diffused blur.

'''
#1d kernel
def apply_gaussian_filter_separable(image, kernel_size, sigma):
    H, W, C = image.shape
    
    # 1D Gaussian 커널
    scope = np.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
    kernel = np.exp(-(scope ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    
    # 패딩
    pad1 = kernel_size // 2
    pad2 = kernel_size - pad1 - 1
    padded_image = np.pad(image, ((pad1, pad2), (pad1, pad2), (0, 0)), mode='constant')
    
    output = np.zeros_like(image, dtype=np.float32)
    
    for c in range(C):
        input_channel = padded_image[:, :, c].astype(np.float32)
        
        # 수평 방향
        intermediate = np.zeros((H + pad1 + pad2, W + pad1 + pad2), dtype=np.float32)
        for x in range(kernel_size):
            intermediate[pad1:H + pad1, pad1:W + pad1] += input_channel[pad1:H + pad1, x:W + x] * kernel[x]
        
        # 수직 방향
        for y in range(kernel_size):
            output[:, :, c] += intermediate[y:H + y, pad1:W + pad1] * kernel[y]
    
    return output.astype(np.uint8)
    
#2d kernel efficient
def apply_gaussian_filter_2D(image, kernel_size, sigma):
    H, W, C = image.shape
    scope = np.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
    xx, yy = np.meshgrid(scope,scope)
    kernel = (1/(2 * np.pi * sigma ** 2)) * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    kernel /= np.sum(kernel)    # Normalization for the kernel.

    #asymmetric padding for even number.
    pad1 = kernel_size // 2
    pad2 = kernel_size - pad1 - 1
    padded_image = np.pad(
        image,
        pad_width=((pad1, pad2), (pad1, pad2), (0, 0)),
        mode='constant',
        constant_values=0
    ) #zero padding to maintain the image size
    output = np.zeros_like(image, dtype=np.float64)  

    for c in range(C):
        for x in range(kernel_size):
            for y in range(kernel_size):
                region = padded_image[y:y+H, x:x+W,c]
                output[:, :, c] += region * kernel[y][x]

    return output.astype(np.uint8)
    
#2d kernel not vectorized
def apply_gaussian_filter_original(image, kernel_size, sigma):
    H, W, C = image.shape
    scope = np.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
    xx, yy = np.meshgrid(scope,scope)
    kernel = (1/(2 * np.pi * sigma ** 2)) * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    kernel /= np.sum(kernel)    # Normalization for the kernel.

    #asymmetric padding for even number.
    pad1 = kernel_size // 2
    pad2 = kernel_size - pad1 - 1
    padded_image = np.pad(
        image,
        pad_width=((pad1, pad2), (pad1, pad2), (0, 0)),
        mode='constant',
        constant_values=0
    ) #zero padding to maintain the image size
    output = np.zeros_like(image)

    for y in range(H):
        for x in range(W):
            for c in range(C):
                region = padded_image[y:y+kernel_size, x:x+kernel_size,c]
                filtered_value = np.sum(region * kernel)
                output[y, x, c] = filtered_value

    return output.astype(np.uint8)
    
def apply_laplacian_of_gaussian_filter(image, kernel_size, sigma):
    H, W = image.shape
    scope = np.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
    x, y = np.meshgrid(scope, scope)

    # LoG 수식
    kernel = (1 - ((x**2 + y**2) / (2 * sigma**2))) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(np.abs(kernel))     # Normalization for the kernel.
    
    pad1  = kernel_size // 2
    pad2  = kernel_size - pad1 - 1
    padded_image = np.pad(
        image,
        pad_width=((pad1, pad2), (pad1, pad2)),
        mode='constant',
        constant_values=0
    ) #zero padding to maintain the image size
    output = np.zeros_like(image)
    #convolution
    for x in range(H):
        for y in range(W):
            region = padded_image[x:x+kernel_size, y:y+kernel_size]
            filtered_value = np.sum(region * kernel)
            output[x, y] = filtered_value    #replace pixel by calculated value



    return output.astype(np.uint8)
