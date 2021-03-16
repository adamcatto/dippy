import os
import math
from typing import Mapping

import numpy as np
import cv2
from numpy.lib import stride_tricks
from scipy.stats import wasserstein_distance
from skimage.util import random_noise

import histogram_processing
import color_to_gray_operations


NOISE_PATH = '../input_data/noisy_segments/'


def estimate_noise_parameters(img_segment, return_histogram=True):
    hist, bins = histogram_processing.compute_image_histogram(img_segment)
    pdf, bins = histogram_processing.compute_image_pdf(img_segment)
    cdf, bins = histogram_processing.compute_image_cdf(img_segment)
    #print(hist)
    mean_value = sum([p[0] * p[1] for p in zip(pdf, bins)])
    variance = sum([(p[1] - mean_value) ** 2 * p[0] for p in list(zip(pdf, bins))])
    standard_deviation = np.sqrt(variance)
    if return_histogram:
        return mean_value, standard_deviation, hist, bins
    return mean_value, standard_deviation


def generate_gaussian_pdf(mean: float, standard_deviation: float) -> Mapping:
    pdf = {}

    for x in range(0, 256):
        term_1 = 1 / (math.sqrt(2 * math.pi) * standard_deviation)
        term_2 = (math.exp(-(x - mean) ** 2)) / (2 * (standard_deviation ** 2))
        pdf[x] = term_1 * term_2
    return pdf


def generate_rayleigh_pdf(mean, standard_deviation):
    pdf = {}
    b = (4 * (standard_deviation ** 2)) / (4 - math.pi)
    #print(standard_deviation)
    a = mean - math.sqrt((math.pi * b) / 4)
    #print(a)

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
        coef_numerator = (a ** b) * (x ** (b - 1))
        coef_denominator = math.factorial(round(b - 1))
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
        if (x >= a and x <= b):
            pdf[x] = 1 / (b - a)
        else:
            pdf[x] = 0

    return pdf


def estimate_noise_model(img_segment):
    mean, standard_deviation, hist, bins = estimate_noise_parameters(img_segment, return_histogram=True)
    pdf, bins = histogram_processing.compute_image_pdf(img_segment)
    #standard_deviation = np.sqrt(np.var(pdf.values()))

    gaussian_pdf = generate_gaussian_pdf(mean, standard_deviation)
    rayleigh_pdf = generate_rayleigh_pdf(mean, standard_deviation)
    erlang_pdf = generate_erlang_pdf(mean, standard_deviation)
    exponential_pdf = generate_exponential_pdf(mean, standard_deviation)
    uniform_pdf = generate_uniform_pdf(mean, standard_deviation)

    gaussian_distance = wasserstein_distance(pdf, np.array(list(gaussian_pdf.values())))
    rayleigh_distance = wasserstein_distance(pdf, np.array(list(rayleigh_pdf.values())))
    erlang_distance = wasserstein_distance(pdf, np.array(list(erlang_pdf.values())))
    exponential_distance = wasserstein_distance(pdf, np.array(list(exponential_pdf.values())))
    uniform_distance = wasserstein_distance(pdf, np.array(list(uniform_pdf.values())))

    distances = {
        'gaussian': gaussian_distance,
        'rayleigh': rayleigh_distance,
        'erlang': erlang_distance,
        'exponential': exponential_distance,
        'uniform': uniform_distance
    }

    min_distance = tuple((d[0], d[1]) for d in distances.items() if d[1] == min(distances.values()))

    return distances, min_distance


def compute_arithmetic_mean_img(img_segment):
    height, width = img_segment.shape
    flattened_array = np.ravel(img_segment)
    average = np.sum(flattened_array) / (height * width)
    return average


def compute_geometric_mean_img(img_segment):
    height, width = img_segment.shape
    flattened_array = np.ravel(img_segment)
    geometric_mean = np.power(np.prod(flattened_array), (1 / (height * width)))
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


def compute_adaptive_filter(window, global_noise_variance):
    flattened_array = np.ravel(window)
    window_mean = np.mean(flattened_array)
    window_variance = np.var(flattened_array)
    central_pixel_value = window[window.shape[0] // 2][window.shape[1] // 2]
    new_pixel_value = central_pixel_value - ((global_noise_variance / (window_variance + 0.01)) * (central_pixel_value - window_mean))
    return new_pixel_value


# this is slow, don't actually use it – just use vectorized implementation
def apply_mean_filter(img, window_size, filter_type='arithmetic', global_noise_variance=None, order_param=None):
    window_height, window_width = window_size
    image_height, image_width = img.shape
    restored_img = np.zeros(img.shape)
    
    for row_index in range(window_height // 2, image_height - window_height // 2):
        for column_index in range(window_width // 2, image_width - (window_width // 2)):
            #print(row_index - window_height, row_index + window_height)
            window = img[row_index - window_height//2: row_index + window_height//2, column_index - window_width//2: column_index + window_width//2]
            #print(row_index - window_height//2, row_index + window_height//2)
            if filter_type == 'arithmetic':
                restored_img[row_index, column_index] = compute_arithmetic_mean_img(window)
                #print(compute_arithmetic_mean_img(window))
            if filter_type == 'geometric':
                restored_img[row_index, column_index] = compute_geometric_mean_img(window)
            if filter_type == 'harmonic':
                restored_img[row_index, column_index] = compute_harmonic_mean_img(window)
            if filter_type == 'contraharmonic':
                restored_img[row_index, column_index] = compute_contraharmonic_mean_img(window, order_param=order_param)
            if filter_type == 'local_adaptive':
                restored_img[row_index, column_index] = compute_adaptive_filter(window, global_noise_variance)
            
            #print(restored_img[row_index, column_index])
    
    return restored_img


def add_rayleigh_noise(img):
    rayleigh_noise_img = np.random.rayleigh(scale=25.0, size=img.shape)
    return img + rayleigh_noise_img


def add_gaussian_noise(img):
    gaussian_noise_img = np.random.normal(scale=25.0, size=img.shape)
    return img + gaussian_noise_img


def add_salt_and_pepper_noise(img):
    sp_noise_img = random_noise(img, mode='s&p')
    return img + sp_noise_img


def add_gamma_noise(img):
    gamma_noise_img = np.random.gamma(shape=2.0, scale=25.0, size=img.shape)
    return img + gamma_noise_img


def add_exponential_noise(img):
    exponential_noise_img = np.random.exponential(scale=25.0, size=img.shape)
    return img + exponential_noise_img


def test():
    #img = cv2.imread('../nyc_tunnel/1_nyc_tunnel.png')
    img = cv2.imread('../input_data/tunnels/tunnel_1.png')
    img = color_to_gray_operations.luminosity_method(img)
    noise_segment = color_to_gray_operations.luminosity_method(cv2.imread('../input_data/noisy_segments/tunnel_1_noise_segment.png'))

    noise_mean, noise_sd, noise_hist, noise_bins = estimate_noise_parameters(noise_segment)


    ######## APPLY FILTERS ########
    for j in range(7, 67, 10):
        #for filter_type in [ 'local_adaptive', 'arithmetic', 'geometric', 'harmonic', 'contraharmonic']:
        for filter_type in ['local_adaptive']:
            print('applying ' + filter_type + ' filter with window height/width = ' + str(j))
            if filter_type == 'local_adaptive':
                new_img = apply_mean_filter(img, window_size=(j, j), filter_type='local_adaptive', global_noise_variance=noise_sd)
                cv2.imwrite('../output_data/denoised/_tunnel_1' + filter_type + '_' + str(j) + '.png', new_img)
            elif filter_type == 'contraharmonic':
                for op in range(-2, 2):
                    print('with order_param = ' + str(op))
                    new_img = apply_mean_filter(img, window_size=(j, j), filter_type=filter_type, global_noise_variance=None, order_param=op)
                    cv2.imwrite('../output_data/denoised/_tunnel_1' + filter_type + '_' + str(j) + '_op=' + str(op) + '.png', new_img)
            else:
                new_img = apply_mean_filter(img, window_size=(j, j), filter_type=filter_type, global_noise_variance=None, order_param=None)
                cv2.imwrite('../output_data/denoised/tunnel_1_' + filter_type + '_' + str(j) + '.png', new_img)
            print('done')
            

    #dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

test()


def add_noise_test():
    img = cv2.imread('../input_data/visolo_tunnel.jpg')
    img = color_to_gray_operations.luminosity_method(img)
    cv2.imwrite('../input_data/visolo_tunnel_gray.jpg', img)

    noisy_img_rayleigh = add_rayleigh_noise(img)
    cv2.imwrite('../output_data/add_noise/rayleigh_visolo_tunnel_gray.jpg', noisy_img_rayleigh)

    noisy_img_gamma = add_gamma_noise(img)
    cv2.imwrite('../output_data/add_noise/gamma_visolo_tunnel_gray.jpg', noisy_img_gamma)

    noisy_img_gaussian = add_gaussian_noise(img)
    cv2.imwrite('../output_data/add_noise/gaussian_visolo_tunnel_gray.jpg', noisy_img_gaussian)

    noisy_img_exponential = add_exponential_noise(img)
    cv2.imwrite('../output_data/add_noise/exponential_visolo_tunnel_gray.jpg', noisy_img_exponential)


def estimate_noise_model_test():
    rayleigh = cv2.imread('../input_data/noisy_segments/visolo_segment_rayleigh.png')
    rayleigh = color_to_gray_operations.luminosity_method(rayleigh)
    cv2.imwrite('../input_data/noisy_segments/visolo_segment_rayleigh.png', rayleigh)

    gaussian = cv2.imread('../input_data/noisy_segments/visolo_segment_gaussian.png')
    gaussian = color_to_gray_operations.luminosity_method(gaussian)
    cv2.imwrite('../input_data/noisy_segments/visolo_segment_gaussian.png', gaussian)

    gamma = cv2.imread('../input_data/noisy_segments/visolo_segment_gamma.png')
    gamma = color_to_gray_operations.luminosity_method(gamma)
    cv2.imwrite('../input_data/noisy_segments/visolo_segment_gamma.png', gamma)

    exponential = cv2.imread('../input_data/noisy_segments/visolo_segment_exponential.png')
    exponential = color_to_gray_operations.luminosity_method(exponential)
    cv2.imwrite('../input_data/noisy_segments/visolo_segment_exponential.png', exponential)


    #### estimate noise models
    print('rayleigh:')
    rayleigh_estim, closest = estimate_noise_model(rayleigh)
    print('closest: ' + str(closest))
    print(rayleigh_estim)

    print('\n\ngaussian:')
    gaussian_estim, closest = estimate_noise_model(gaussian)
    print('closest: ' + str(closest))
    print(gaussian_estim)

    print('\n\ngamma:')
    gamma_estim, closest = estimate_noise_model(gamma)
    print('closest: ' + str(closest))
    print(gamma_estim)

    print('\n\nexponential:')
    exponential_estim, closest = estimate_noise_model(exponential)
    print('closest: ' + str(closest))
    print(exponential_estim)


def stretch_histograms(dir='../output_data/denoised/'):
    for f in os.listdir(dir):
        file_name = os.path.join(dir, f)
        print(file_name)
        img = cv2.imread(file_name)
        img = color_to_gray_operations.luminosity_method(img)
        
        stretched = histogram_processing.stretch_histogram(img)
        cv2.imwrite('../output_data/denoised_stretched/' + f, stretched)


#### ORDER STATISTICS ####
def median_filter(window):
    flattened_array = np.ravel(window)
    return np.median(flattened_array)


def max_filter(window):
    flattened_array = np.ravel(window)
    return np.max(flattened_array)


def min_filter(window):
    flattened_array = np.ravel(window)
    return np.min(flattened_array)


def midpoint_filter(window):
    min_ = min_filter(window)
    max_ = max_filter(window)
    return (max_ + min_) / 2


def alpha_trimmed_mean_filter(window, d):
    height, width = window.shape
    sorted_flattened_array = np.array(list(sorted(np.ravel(window))))
    num_to_remove = round(d * len(sorted_flattened_array))
    relevant_values = sorted_flattened_array[num_to_remove: len(sorted_flattened_array) - num_to_remove]
    return (1 / ((height * width) - (num_to_remove * 2))) * np.sum(relevant_values)


def adaptive_median_filter(window, max_window_size):
    pass