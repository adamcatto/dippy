import numpy as np



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


