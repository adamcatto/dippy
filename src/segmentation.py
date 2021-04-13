from __future__ import annotations
import os
from typing import Tuple, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt

from color_to_gray_operations import luminosity_method as lm
from histogram_processing import stretch_histogram, equalize_histogram, compute_image_pdf


# todo: create configs.py and name separate configs for separate files/experiments
config = {}
config['out_dir'] = '../output_data/segmentation/'
config['image_paths'] = {}
config['image_paths']['original'] = '../input_data/grayscale_tunnels/grayscale_tunnel_1.png'
config['image_paths']['L0_smooth'] = '../input_data/L0-smooth/grayscale_tunnel_1.png'
config['image_paths']['overlay'] = '../output_data/smooth_segment_blend/overlay_smooth_rolling.png'


def compute_threshold_between_class_variance(pdf, threshold, global_mean):
    # compute probabilities of being in classes 1 and 2
    p1 = sum(pdf[0: threshold + 1])
    p2 = 1 - p1
    if p1 == 0 or p2 == 0:
        return 0
    # compute mean values of classes 1 and 2
    mean_1 = sum([x * pdf[x] for x in range(1, threshold + 1)]) / p1
    mean_2 = sum([x * pdf[x] for x in range(threshold + 1, len(pdf))]) / p2
    # compute between-class variance
    between_class_variance = (p1 * ((mean_1 - global_mean) ** 2)) + (p2 * ((mean_2 - global_mean) ** 2))
    return between_class_variance


def get_otsu_threshold(img, img_type, return_only_threshold=True, write_pdf_thresholded=True):
    """
    param :: return_only_threshold
    if True  ==> returns only integer threshold value
    if False ==> returns Tuple[threshold, between_class_variance]
    i.e. return both (1) threshold maximizing Otsu objective, (2) value of between-class variance at that threshold

    param :: write_pdf_threshold
    if True  ==> write plot of image's PDF with a vertical dashed line at position where otsu objective is maximized
    if False ==> continue
    """
    if len(img.shape) == 3:
        img = lm(img)
    
    pdf, bins = compute_image_pdf(img)
    global_mean = sum([x * pdf[x] for x in range(1, len(pdf))])
    global_variance = sum([((pdf[x] - global_mean) ** 2) * x for x in range(1, len(pdf))])
    
    threshold_variances = {}
    
    for threshold in range(1, len(pdf)):
        between_class_variance = compute_threshold_between_class_variance(pdf, threshold, global_mean)
        threshold_variances[threshold] = between_class_variance

    max_item = (0, 0)

    for threshold, between_class_variance in threshold_variances.items():
        if between_class_variance > max_item[1]:
            max_item = (threshold, between_class_variance)

    if write_pdf_thresholded:
        fig = plt.figure()
        plt.plot(bins[0:len(pdf)], pdf)
        # draw dashed vertical line at position where otsu objective is maximized
        plt.axvline(x=max_item[0], color='red', linestyle='--')
        fig.savefig(os.path.join(config['out_dir'], img_type + '_histogram_thresholded.png'))

    if return_only_threshold:
        return max_item[0]
    
    # else
    return max_item


def compute_separability_score(global_variance=None, between_class_variance=None, pdf=None, threshold=None, 
         img=None, img_path=None) -> float:
    if global_variance and between_class_variance:
        return between_class_variance / global_variance
    
    elif pdf and threshold:
        global_mean = sum([x * pdf[x] for x in range(1, len(pdf))])
        global_variance = sum([ ((pdf[x] - global_mean) ** 2) * x for x in range(1, len(pdf))])
        between_class_variance = compute_threshold_between_class_variance(pdf, threshold, global_mean)
        return compute_separability_score(global_variance=global_variance, between_class_variance=between_class_variance)

    elif img and threshold:
        pdf, bins = compute_image_pdf(img)
        return compute_separability_score(pdf=pdf, threshold=threshold)

    elif img_path and threshold:
        img = lm(cv2.imread(img_path))
        return compute_separability_score(img=img, threshold=threshold)


def get_binary_segmentation(img, threshold, numeric=True, stretch=True):
    segmented = img >= threshold

    if numeric:
        segmented = segmented.astype(int)
    
    if stretch:
        segmented = stretch_histogram(segmented)

    print(segmented)
    return segmented


def global_threshold(img, threshold_method):
    """
    threshold_method in ['otsu', 'niblack', 'sauvola]
    """
    pass


def local_adaptive_threshold(img: np.array, window_dims: Union[int, Tuple[int, int]]):
    pass


def write_otsu_segmented_image(img, out_dir, img_type):
    if len(img.shape) == 3:
        img = lm(img)
    optimal_otsu_threshold = get_otsu_threshold(img, img_type, return_only_threshold=True)
    otsu_segmented = get_binary_segmentation(img, optimal_otsu_threshold)
    #print(optimal_otsu_threshold)
    cv2.imwrite(os.path.join(out_dir, img_type + '_otsu.png'), otsu_segmented)


def run_experiments(config):
    out_dir = config['out_dir']
    for img_type, path in config['image_paths'].items():
        img = lm(cv2.imread(path))
        write_otsu_segmented_image(img, out_dir, img_type)


run_experiments(config)

# todo: make experiments function that loops over all images and applies all segmentation methods to each of them
