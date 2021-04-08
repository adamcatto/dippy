import math
import os

import numpy as np
from numpy.fft import fft2
import cv2

from color_to_gray_operations import luminosity_method as to_gray
from histogram_processing import stretch_histogram


def to_frequency_domain(img, make_gray=True):
    if len(img.shape) == 3 and make_gray:
        img = to_gray(img)
    return fft2(img)


def construct_highpass_filter(dims, img_shape, start_x, start_y):
    if isinstance(dims, int):
        dims = (dims, dims)

    kernel = np.ones(img_shape)
    kernel[start_x : dims[0] + start_x, start_y : dims[1] + start_y] = np.zeros((dims[0], dims[1]))
    #centered = np.fft.fftshift(kernel)
    centered = kernel
    return centered


def construct_lowpass_filter(dims, img_shape, start_x, start_y):
    if isinstance(dims, int):
        dims = (dims, dims)

    kernel = np.zeros(img_shape)
    kernel[start_x : dims[0] + start_x, start_y : dims[1] + start_y] = np.ones((dims[0], dims[1]))
    #centered = np.fft.fftshift(kernel)
    centered = kernel
    return centered
    

def main(img_file='../input_data/tunnels/tunnel_1.png', out_dir='../output_data/frequency_domain/original_tunnel/'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    img = cv2.imread(img_file)
    
    # this method will automatically convert `img` to grayscale. set param `make_gray` = `False` if you don't want this
    dft_img = to_frequency_domain(img)
    dft_shift = np.fft.fftshift(dft_img)
    magnitude = 20 * np.log(np.abs(dft_shift.real) + 1)
    phase = dft_shift.imag

    dims = int(img.shape[0] / 4), int(img.shape[1] / 4)
    if isinstance(dims, int):
        dims = (dims, dims)

    start_x = int((dft_img.shape[0] - dims[0]) / 2)
    start_y = int((dft_img.shape[1] - dims[1]) / 2)

    # highpass filtering

    highpass_kernel = construct_highpass_filter(dims, dft_img.shape, start_x, start_y)
    highpass_fft_filtered = dft_shift * highpass_kernel

    highpass_filtered = np.fft.ifft2(highpass_fft_filtered)
    # get magnitude
    highpass_filtered = np.abs(highpass_filtered.real)
    print(highpass_filtered)

    # lowpass filtering

    lowpass_kernel = construct_lowpass_filter(dims, dft_img.shape, start_x, start_y)
    lowpass_fft_filtered = dft_shift * lowpass_kernel

    lowpass_filtered = np.fft.ifft2(lowpass_fft_filtered)
    # get magnitude
    lowpass_filtered = np.abs(lowpass_filtered.real)
    print(lowpass_filtered)
    
    cv2.imwrite(os.path.join(out_dir, 'highpass_fft_filtered.png'), np.abs(highpass_fft_filtered.real))
    cv2.imwrite(os.path.join(out_dir, 'lowpass_fft_filtered.png'), np.abs(lowpass_fft_filtered.real))
    cv2.imwrite(os.path.join(out_dir, 'lowpass_filter.png'), stretch_histogram(lowpass_kernel))
    cv2.imwrite(os.path.join(out_dir, 'highpass_filter.png'), stretch_histogram(highpass_kernel))
    cv2.imwrite(os.path.join(out_dir, 'magnitude.png'), magnitude)
    cv2.imwrite(os.path.join(out_dir, 'phase.png'), phase)
    cv2.imwrite(os.path.join(out_dir, 'highpass_filtered.png'), stretch_histogram(highpass_filtered))
    cv2.imwrite(os.path.join(out_dir, 'lowpass_filtered.png'), stretch_histogram(lowpass_filtered))


def run(img_type_list=[]):
    if not isinstance(img_type_list, list):
        img_type_list = [img_type_list]

    img_types = {
        'L0_smooth': ('../input_data/L0-smooth/grayscale_tunnel_1.png', '../output_data/frequency_domain/L0_smooth/'),
        'original': ('../input_data/tunnels/tunnel_1.png', '../output_data/frequency_domain/original_tunnel/'),
        'overlay_smooth_rolling': ('../output_data/smooth_segment_blend/overlay_smooth_rolling.png', '../output_data/frequency_domain/overlay_smooth_rolling/'),
        'overlay_smooth_original': ('../output_data/smooth_segment_blend/overlay_smooth_original.png', '../output_data/frequency_domain/overlay_smooth_original/')
    }

    if not img_type_list:
        print('computing all')
        for img_type in img_types.keys():
            # recall that * unpacks the tuple returned by img_types[img_type], to be used as separate args to `main()`
            main(*img_types[img_type])
    else:
        for img_type in img_type_list:
            # recall that * unpacks the tuple returned by img_types[img_type], to be used as separate args to `main()`
            main(*img_types[img_type])


if __name__ == '__main__':
    run()