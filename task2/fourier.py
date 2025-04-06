import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def fftshift(img):
    '''
    This function should shift the spectrum image to the center.
    You should not use any kind of built in shift function. Please implement your own.
    '''

    return img


def ifftshift(img):
    '''
    This function should do the reverse of what fftshift function does.
    You should not use any kind of built in shift function. Please implement your own.
    '''

    return img


def get_magnitude(img):
    """
    This function should get the frequency magnitude of the input image.
    Make sure that the spectrum image is shifted to the center using the implemented fftshift function.
    """

    return img


def get_phase(img):
    """
    This function should get the frequency phase of the input image.
    Make sure that the spectrum image is shifted to the center using the implemented fftshift function.
    """

    return img


def swap_phase(img1, img2):
    """
    This fucntion should swap the phase of two images.
    Use get_magnitude() and get_phase() functions implemented above..
    To implement swap_phase(), use np.exp, np.real, np.fft.ifft2.
    """

    return new_img1, new_img2


def low_pass_filter(img, r=30):
    '''
    This function should return an image that goes through low-pass filter.
    '''

    return img

def high_pass_filter(img, r=20):
    '''
    This function should return an image that goes through high-pass filter.
    '''

    return img

def band_pass_filter(img, r_low=50, r_high=100):
    '''
    This function should return an image that goes through band-pass filter.
    '''

    return img

def sharpening(img, r=20, alpha=1):
    '''
    Use adequate technique(s) to sharpen the image.
    Hint: Use fourier transform
    [Args]
        - img : input image
        - r : radius
        - alpha : strength of edge
    '''

    return img
