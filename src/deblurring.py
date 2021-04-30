import os

import numpy as np
from matplotlib import pyplot as plt
import cv2
import skimage
from skimage import io
from skimage.filters import gaussian
from scipy.fftpack import dct
from fcmeans import FCM

import histogram_processing
from color_to_gray_operations import luminosity_method
from base import alpha_blend, make_bool


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
    blur_map = blur_map[..., np.newaxis]
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


def construct_heatmaps(dir='../output_data/deblur/local/'):
    files = os.listdir(dir)

    for f in files:
        print(f)
        if f[-4] != '.':
            continue
        f_ = os.path.join(dir, f)
        image = io.imread(f_)
        m = np.mean(np.ravel(image))
        for i, row in enumerate(image):
            for j, col in enumerate(row):
                if i < 47 or j < 47: 
                    image[i][j] = m

        cv2.imwrite('../output_data/deblur/local/' + f, image)
        heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        
        cv2.imwrite(dir + 'heatmap/' + f, heatmap)


def test_blend_smooth_with_segment(alpha=0.75):
    smooth_file = '../input_data/L0-smooth/tunnel_1.png'
    segment_file = '../output_data/edge-segmented/tunnel_1.png'
    smooth_img = cv2.imread(smooth_file)
    segment_img = cv2.imread(segment_file)
    blended = alpha_blend(smooth_img, segment_img, alpha)
    cv2.imwrite('../output_data/smooth_segment_blend/tunnel_1.png', blended)


def test_overlay_original_on_smoothed():
    #smooth_file = '../input_data/L0-smooth/tunnel_1.png'
    #segment_file = '../output_data/edge-segmented/tunnel_1_rolling.png'
    #original_file = '../output_data/edge-segmented/rolling_guided.png'

    smooth_file = '../output_data/smooth_segment_blend/overlay_smooth_rolling.png'
    segment_file = '/Users/adamcatto/SRC/dippy/output_data/frequency_domain/overlay_smooth_rolling/highpass_filtered.png'
    original_file = '../output_data/edge-segmented/rolling_guided.png'


    smooth_img = cv2.imread(smooth_file)
    segment_img = make_bool(luminosity_method(cv2.imread(segment_file)), threshold=20)
    original_img = cv2.imread(original_file)

    for i, row in enumerate(segment_img):
        for j, col in enumerate(row):
            if segment_img[i, j]:
                smooth_img[i, j] = original_img[i, j]

    return smooth_img


"""
bm = cv2.imread('../output_data/deblur/local/gray_tunnel_1.png')
s = segment_blur_map(bm)
print(s)
"""
#overlay = test_overlay_original_on_smoothed()
#cv2.imwrite('../output_data/smooth_segment_blend/overlay_smooth_rolling.png', overlay)
#cv2.imwrite('../output_data/smooth_segment_blend/overlay_frequency_domain_highpass.png', overlay)

#test_blend_smooth_with_segment()

#construct_heatmaps()

#main()