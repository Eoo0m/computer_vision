import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
[[  0 128 255]         
 [ 64 192 64]   
 [255 128  0]]
->
[[192  64  64]
 [128   0 255]
 [128 255   0]]
 Shift the DC component to the center
 '''
def fftshift(img):
    H, W = img.shape
    H_half = H // 2
    W_half = W // 2
    
    #concatanate Q4,Q3 and Q1,Q2 and concatanate top and buttom
    top = np.concatenate((img[H_half:, W_half:],img[H_half:, 0:W_half]), axis=1)       
    bottom = np.concatenate((img[0:H_half, W_half:],img[0:H_half, 0:W_half] ), axis=1)
    img = np.concatenate((top, bottom), axis=0)


    return img
'''
[[192  64  64]
 [128   0 255]
 [128 255   0]]
 ->
[[  0 128 255]     
 [ 64 192 64] 
 [255 128  0]]
 Shift the DC component back to its original position
 '''
 
def ifftshift(img):
    H, W = img.shape
    #for even size
    H_half = (H + 1) // 2
    W_half = (W + 1) // 2
    
    #concatanate Q4,Q3 and Q1,Q2 and concatanate top and buttom. same
    top = np.concatenate((img[H_half:, W_half:],img[H_half:, 0:W_half]), axis=1)
    bottom = np.concatenate((img[0:H_half, W_half:],img[0:H_half, 0:W_half] ), axis=1)
    img = np.concatenate((top, bottom), axis=0)

    return img


def get_magnitude(img):
    """
    This function should get the frequency magnitude of the input image.
    Make sure that the spectrum image is shifted to the center using the implemented fftshift function.
    """
    img = np.abs(np.fft.fft2(img))
    img = fftshift(img)
    return img


def get_phase(img): 
    """
    This function should get the frequency phase of the input image.
    Make sure that the spectrum image is shifted to the center using the implemented fftshift function.
    """
    img = np.angle(np.fft.fft2(img))
    img = fftshift(img)

    return img


def swap_phase(img1, img2):
    """
    This fucntion should swap the phase of two images.
    Use get_magnitude() and get_phase() functions implemented above..
    To implement swap_phase(), use np.exp, np.real, np.fft.ifft2.
    """
    magnitude1, phase1 = get_magnitude(img1), get_phase(img1)
    magnitude1, phase1  = ifftshift(magnitude1), ifftshift(phase1)
    magnitude2, phase2 = get_magnitude(img2), get_phase(img2)
    magnitude2, phase2  = ifftshift(magnitude2), ifftshift(phase2)
    
    #represent img re^jΘ
    new_img1 = magnitude1 * np.exp(1j * phase2)  # img1 magnitude + img2 phase
    new_img2 = magnitude2 * np.exp(1j * phase1)  # img2 magnitude + img1 phase

    new_img1 = np.real(np.fft.ifft2(new_img1))
    new_img2 = np.real(np.fft.ifft2(new_img2))

    return new_img1, new_img2


def low_pass_filter(img, r=30):
    '''
    This function should return an image that goes through low-pass filter.
    '''
    magnitude, phase = get_magnitude(img), get_phase(img)
    H,W = img.shape
    mask = create_circular_mask(H, W,radius=r)
    magnitude *= mask
    
    #inverse fourier transform
    magnitude, phase  = ifftshift(magnitude), ifftshift(phase) 
    img = magnitude * np.exp(1j * phase) 
    img = np.abs(np.fft.ifft2(img))

    return np.clip(img, 0, 255).astype(np.uint8)

def high_pass_filter(img, r=30):
    '''
    This function should return an image that goes through high-pass filter.
    '''
    magnitude, phase = get_magnitude(img), get_phase(img)
    H,W = img.shape
    mask = create_circular_mask(H, W,radius=r)
    magnitude *= ~mask

    #inverse fourier transform
    magnitude, phase  = ifftshift(magnitude), ifftshift(phase) 
    img = magnitude * np.exp(1j * phase) 
    img = np.abs(np.fft.ifft2(img))
    
    return np.clip(img, 0, 255).astype(np.uint8)

def band_pass_filter(img, r_low=50, r_high=100):
    '''
    This function should return an image that goes through band-pass filter.
    '''
    magnitude, phase = get_magnitude(img), get_phase(img)
    H,W = img.shape
    mask1 = create_circular_mask(H, W,radius=r_low)
    mask2 = create_circular_mask(H, W,radius=r_high)
    magnitude *= (mask2.astype(int) - mask1.astype(int))

    #inverse fourier transform
    magnitude, phase  = ifftshift(magnitude), ifftshift(phase) 
    img = magnitude * np.exp(1j * phase) 
    img = np.abs(np.fft.ifft2(img))

    return np.clip(img, 0, 255).astype(np.uint8)

def sharpening(img, r=20, alpha=1):
    '''
    Use adequate technique(s) to sharpen the image.
    Hint: Use fourier transform
    [Args]
        - img : input image
        - r : radius
        - alpha : strength of edge
    '''
    
    magnitude, phase = get_magnitude(img), get_phase(img)
    H,W = img.shape
    mask = create_circular_mask(H, W,radius=r)
    high_magnitude = magnitude * ~mask
    magnitude += alpha * high_magnitude

    #inverse fourier transform
    magnitude, phase  = ifftshift(magnitude), ifftshift(phase) 
    img = magnitude * np.exp(1j * phase) 
    img = np.abs(np.fft.ifft2(img))

    return np.clip(img, 0, 255).astype(np.uint8)

#####################################################################################

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

