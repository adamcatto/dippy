import os
import math
from typing import Mapping

import numpy as np
from numpy.lib import stride_tricks
import cv2
from scipy.stats import median_absolute_deviation

import histogram_processing


NOISE_PATH = '../input_data/noisy_segments/'


def estimate_noise_parameters(img_segment, return_histogram=True):
    hist, bins = histogram_processing.compute_image_histogram(img_segment)
    mean_value = np.mean(hist)
    standard_deviation = np.sqrt(np.var(hist))
    if return_histogram:
        return mean_value, standard_deviation, hist, bins
    return mean_value, standard_deviation


def generate_gaussian_pdf(mean: float, standard_deviation: float) -> Mapping:
    pdf = {}

    for x in range(0, 256):
        term_1 = 1 / (math.sqrt(2 * math.pi) * standard_deviation)
        term_2 = (math.exp(x - mean) ** 2) / (2 * (variance ** 2))
        pdf[x] = term_1 * term_2
    
    return pdf


def generate_rayleigh_pdf(mean, standard_deviation):
    pdf = {}
    b = (4 * (standard_deviation ** 2)) / (4 - math.pi)
    a = mean - math.sqrt((math.pi * b) / 4)

    for x in range(0, 256):
        if x < a:
            pdf[x] = 0
        else:
            term_1 = 2 / b
            term_2 = x - a
            term_3 = math.exp( (- (x - a) ** 2) / b )
            pdf[x] = term_1 * term_2 * term_3
    
    return pdf


def generate_erlang_pdf(mean, standard_deviation):
    pdf = {}
    a = mean / (standard_deviation ** 2)
    b = a * mean        # (mean ** 2 ) * (standard_deviation ** 2)

    for x in range(0, 256):
        exponential = math.exp(-a * x)
        coef_numerator = (a ** b) * (z ** (b - 1))
        coef_denominator = math.factorial(b - 1)
        coef = coef_numerator / coef_denominator
        pdf[x] = coef * exponential

    return pdf


def generate_exponential_pdf(mean, standard_deviation):
    pdf = {}
    a = 1 / mean

    for x in range(0, 256):
        exponential = a * math.exp(-1 * a * x)
        pdf[x] = exponential

    return pdf


def generate_uniform_pdf(mean, standard_deviation):
    pdf = {}
    b = (math.sqrt(3) * standard_deviation) / mean
    a = (2 * mean) - b

    for x in range(0, 256):
        if (z >= a and z <= b):
            pdf[x] = 1 / (b - a)
        else:
            pdf[x] = 0

    return pdf


def estimate_noise_model(img_segment):
    mean, standard_deviation, hist, bins = estimate_noise_parameters(img_segment, return_histogram=True)
    pdf = histogram_processing.compute_image_pdf(img_segment)

    gaussian_pdf = generate_gaussian_pdf(mean, standard_deviation)
    rayleigh_pdf = generate_rayleigh_pdf(mean, standard_deviation)
    erlang_pdf = generate_erlang_pdf(mean, standard_deviation)
    exponential_pdf = generate_exponential_pdf(mean, standard_deviation)
    uniform_pdf = generate_uniform_pdf(mean, standard_deviation)

    gaussian_distance = median_absolute_deviation(np.array(pdf.values()), np.array(gaussian_pdf.values()))
    rayleigh_distance = median_absolute_deviation(np.array(pdf.values()), np.array(rayleigh_pdf.values()))
    erlang_distance = median_absolute_deviation(np.array(pdf.values()), np.array(erlang_pdf.values()))
    exponential_distance = median_absolute_deviation(np.array(pdf.values()), np.array(exponential_pdf.values()))
    uniform_distance = median_absolute_deviation(np.array(pdf.values()), np.array(uniform_pdf.values()))

    distances = {
        'gaussian': gaussian_distance,
        'rayleigh': rayleigh_distance,
        'erlang': erlang_distance,
        'exponential': exponential_distance,
        'uniform': uniform_distance
    }

    min_distance = tuple(d[0], d[1] for d in distances.items() if d[1] = min(distances.values()))

    return min_distance


def compute_arithmetic_mean_img(img_segment):
    height, width = img_segment.shape
    flattened_array = np.ravel(img_segment)
    average = np.sum(flattened_array) / (height * width)
    return average


def compute_geometric_mean_img(img_segment):
    height, width = img_segment.shape
    flattened_array = np.ravel(img_segment)
    geometric_mean = np.prod(flattened_array) ** (1 / (height * width))
    return geometric_mean


def compute_harmonic_mean_img(img_segment):
    height, width = img_segment.shape
    flattened_array = np.ravel(img_segment)
    inverted_array_values = 1 / flattened_array
    harmonic_mean = height * width / np.sum(inverted_array_values)
    return harmonic_mean


def compute_contraharmonic_mean_img(img_segment, order_param):
    flattened_array = np.ravel(img_segment)
    numerator_values = np.power(flattened_array, order_param + 1)
    denominator_values = np.power(flattened_array, order_param)
    contraharmonic_mean = np.sum(numerator_values) / np.sum(denominator_values)
    return contraharmonic_mean


# this is slow, don't actually use it – just use vectorized implementation
def apply_mean_filter(img, window_size, filter_type='arithmetic'):
    window_height, window_width = window_size
    image_height, image_width = img.shape
    restored_img = np.zeros(img.shape)
    for row_index in range(window_height // 2, (image_height - window_height) // 2):
        for column_index in range(window_width // 2, (image_width - window_width) // 2):
            window = img[row_index - window_height: row_index + window_height][column_index - window_width: column_index + window_width]
            if filter_type == 'arithmetic':
                restored_img[row_index][column_index] = compute_arithmetic_mean_img(window)
            if filter_type == 'geometric':
                restored_img[row_index][column_index] = compute_geometric_mean_img(window)
            if filter_type == 'harmonic':
                restored_img[row_index][column_index] = compute_harmonic_mean_img(window)
            if filter_type == 'contraharmonic':
                restored_img[row_index][column_index] = compute_contraharmonic_mean_img(window)

    return restored_img


def test():
    img = cv2.imread('../nyc_tunnel/1_nyc_tunnel.png')
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    cv2.imwrite('../output_data/denoised/1_nyc_tunnel.png', dst)


#test()