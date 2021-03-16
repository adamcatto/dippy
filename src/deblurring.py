import numpy as np
import cv2
import skimage
from skimage.filters import gaussian
from scipy.fftpack import dct
from fcmeans import FCM

import histogram_processing
from color_to_gray_operations import luminosity_method


"""
based on:
Javaran, T. A., Hassanpour, H., & Abolghasemi, V. (2017). Automatic estimation and segmentation of partial blur in 
natural images. The Visual Computer, 33(2), 151-161.
"""


def compute_lp_norm(img_block, p=1):
    flattened_array = np.ravel(img_block)
    p_transformed_values = np.power(flattened_array, p)
    return np.sum(p_transformed_values)


def compute_block_blurriness(img_block):
    assert img_block.shape[0] == img_block.shape[1]
    assert img_block.shape[0] % 2 == 1

    blurred_block = gaussian(img_block)
    dct_original = dct(dct(img_block.T, norm='ortho').T, norm='ortho')
    dct_blurred = dct(dct(blurred_block.T, norm='ortho').T, norm='ortho')
    blur_metric_original = compute_lp_norm(dct_original)
    blur_metric_blurred = compute_lp_norm(dct_blurred)
    
    return blur_metric_blurred / (blur_metric_original + 0.01)


def compute_img_blurriness(img):
    print(img.shape)
    padded_img = skimage.util.pad(array=img, pad_width=22)
    result_img = np.zeros(img.shape)

    for row_index, row in enumerate(img):
        print(str(row_index) + ' / ' + str(img.shape[0]))
        for col_index, col in enumerate(row):
            window_9 = padded_img[row_index - 4 + 22 : row_index + 4 + 22 + 1, col_index - 4 + 22 : col_index + 4 + 22 + 1]
            window_19 = padded_img[row_index - 9 + 22 : row_index + 9 + 22 + 1, col_index - 9 + 22 : col_index + 9 + 22 + 1]
            window_45 = padded_img[row_index + 22 - 22: row_index + 22 + 22 + 1, col_index - 22 + 22 : col_index + 22 + 22 + 1]

            blur_9 = compute_block_blurriness(window_9)
            blur_19 = compute_block_blurriness(window_19)
            blur_45 = compute_block_blurriness(window_45)

            average_blur = (blur_9 + blur_19 + blur_45) / 3
            if average_blur > 1.4:
                average_blur = 1.4
            if average_blur < 0:
                average_blur = 0.01
            result_img[row_index, col_index] = average_blur
            
            
            

    return result_img


def segment_blur_map(blur_map):
    fcm = FCM(n_clusters=2)
    fcm.fit(blur_map)
    seg = fcm.predict(blur_map)
    return seg





def main():
    #img = luminosity_method(cv2.imread('../input_data/grayscale_tunnels/grayscale_tunnel_1.png'))
    #img = luminosity_method(cv2.imread('../input_data/blur_car.jpg'))
    img = luminosity_method(cv2.imread('../input_data/javaran-book.png'))
    out_file = '../output_data/deblur/local/javaran-book.png'
    blur_map = compute_img_blurriness(img)
    stretched_blur_map = histogram_processing.stretch_histogram(blur_map)
    cv2.imwrite(out_file, stretched_blur_map)
    #segmented = segment_blur_map(stretched_blur_map).reshape(img.shape)
    
    print(stretched_blur_map.shape)
    #cv2.imwrite('../output_data/deblur/local/javaran-book-segmented.png', segmented)


main()