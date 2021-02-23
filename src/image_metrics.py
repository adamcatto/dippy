from typing import Union

import numpy as np
import cv2
#import image_slicer


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


def compute_eme_score(img: np.array, block_size, regularization_term: float = 0.01) -> float:
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


def test_image_metrics(path='../input_data/tunnel_1.png'):
    img = cv2.imread(path)
    print(compute_eme_score(img, 4))


#test_image_metrics(path='../input_data/test_for_eme.png')