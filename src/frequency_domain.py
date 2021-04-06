import math

import numpy as np
from numpy.fft import fft2
import cv2

from color_to_gray_operations import luminosity_method as to_gray
from histogram_processing import stretch_histogram


def to_frequency_domain(img, make_gray=True):
    if len(img.shape) == 3 and make_gray:
        img = to_gray(img)
    return fft2(img)


def construct_lowpass_filter(dims, img_shape):
    if isinstance(dims, int):
        dims = (dims, dims)

    kernel = np.zeros(img_shape)
    kernel[0 : dims[0], 0 : dims[1]] = np.ones((dims[0], dims[1]))
    centered = np.fft.fftshift(kernel)
    return centered
    

def main(img_file='../input_data/tunnels/tunnel_1.png'):
    img = cv2.imread(img_file)
    
    # this method will automatically convert `img` to grayscale. set param `make_gray` = `False` if you don't want this
    dft_img = to_frequency_domain(img)
    dft_shift = np.fft.fftshift(dft_img)
    magnitude = 20 * np.log(np.abs(dft_shift.real) + 1)
    phase = dft_shift.imag

    kernel = construct_lowpass_filter(130, dft_img.shape)
    filtered = dft_shift * kernel
    filtered = np.fft.ifft2(filtered)
    filtered = np.abs(filtered.real)
    print(filtered)
    
    
    cv2.imwrite('../output_data/frequency_domain/lowpass_filter.png', stretch_histogram(kernel))
    cv2.imwrite('../output_data/frequency_domain/magnitude.png', magnitude)
    cv2.imwrite('../output_data/frequency_domain/phase.png', phase)
    cv2.imwrite('../output_data/frequency_domain/filtered.png', filtered)
    

main()