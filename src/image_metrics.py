from typing import Mapping, Tuple, Union

import numpy as np
import cv2
from skimage.util import view_as_windows


def luminance_similarity(img_1, img_2, regularization_term=0.01, dynamic_range=255):
    mean_1, mean_2 = np.mean(img_1), np.mean(img_2)
    stabilizer = (regularization_term * dynamic_range) ** 2
    similarity = (2 * mean_1 * mean_2 + stabilizer) / ((mean_1 ** 2) + (mean_2 ** 2) + stabilizer)
    return similarity


def contrast_similarity(img_1, img_2, regularization_term=0.03, dynamic_range=255):
    variance_1, variance_2 = np.var(img_1), np.var(img_2)
    stabilizer = (regularization_term * dynamic_range) ** 2
    similarity = (2 * variance_1 * variance_2 + stabilizer) / ((variance_1 ** 2) + (variance_2 ** 2) + stabilizer)
    return similarity


def structure_similarity(img_1, img_2, regularization_term=0.015, dynamic_range=255):
    variance_1, variance_2 = np.var(img_1), np.var(img_2)
    stabilizer = (regularization_term * dynamic_range) ** 2
    covariance = np.cov(img_1, img_2)
    similarity = (covariance + stabilizer) / (variance_1 * variance_2 + stabilizer)
    return similarity


# todo: add option to parametrize regularization_term and dynamic_range (or infer dynamic_range)
def ssim(img_1, img_2, luminance_weight, contrast_weight, structure_weight):
    luminance = luminance_similarity(img_1, img_2) ** luminance_weight
    contrast = contrast_similarity(img_1, img_2) ** contrast_weight
    structure = structure_similarity(img_1, img_2) ** structure_weight
    return luminance * contrast * structure


def _compute_tile_eme_score(tile, regularization_term=0.01):
    contrast_difference = np.max(tile) / (np.min(tile) + regularization_term)
    eme = 20 * np.log(contrast_difference)
    return eme


def compute_eme_score(img: np.ndarray, block_size: Union[Tuple[int, int], int], 
                    regularization_term: float = 0.01) -> float:
    if isinstance(block_size, int):
        block_size = (block_size, block_size)
    
    max_eme = 0
    # create block_size[0] x block_size[1] tiles
    #M = img.shape[0] // block_size[0]
    #N = img.shape[1] // block_size[1]
    # note: if instead want tiles to have block_size, simply let M, N = block_size
    M, N = block_size
    tiled_img = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]
    
    eme_scores = [_compute_tile_eme_score(tile, regularization_term) for tile in tiled_img]
    eme_score = np.mean(eme_scores)
    print(len(eme_scores))
    return eme_score / (block_size[0] * block_size[1])


def edge_detector_quality_measure(edge_map: np.ndarray, window_size: Union[int, Tuple[int, int]], 
        prior_distribution: Mapping[int, int], stride_len: int) -> float:
    """
    Prior-based metric – specify a prior distribution of ratio of edge pixels to total pixels over windows.
    Note to self: test this with both windows (with varying stride length) and blocks.

    This can be done using blocks by setting `stride_len` = `window_size` (or `window_size[0]`)
    """
    num_window_pixels = window_size[0] * window_size[1]
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    hist = {x: 0 for x in range(num_window_pixels + 1)}
    windows = view_as_windows(edge_map, window_size, stride_len)
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            w = windows[i][j]
            s = np.sum(np.ravel(w))
            hist[s] += 1

    prior_hist = np.array([x[1] for x in dict(sorted(prior_distribution)).items()])
    posterior_hist = np.array([x[1] for x in dict(sorted(hist)).items()])
    diff_hist = posterior_hist - prior_hist
    abs_diff_hist = np.abs(diff_hist)
    inverse_diff_hist = (abs_diff_hist + 1) ** -1
    score = np.sum(inverse_diff_hist) / num_window_pixels
    return score


def test_image_metrics(path='../input_data/tunnel_1.png'):
    img = cv2.imread(path)
    print(compute_eme_score(img, 4))


#test_image_metrics(path='../input_data/test_for_eme.png')