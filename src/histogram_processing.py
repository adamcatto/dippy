import numpy as np
import cv2

from skimage import morphology
from skimage.filters import rank

import base
from color_to_gray_operations import luminosity_method, average_method
from image_metrics import compute_eme_score


def compute_image_histogram(img):
    if len(img.shape) == 2:
        hist, bins = np.histogram(img, bins=[x for x in range(0, 257)])
        return hist, bins

    b, g, r = cv2.split(img)

    # now each of b, g, r are 2-dimensional arrays; can call `compute_image_histogram()` on each of them
    blue_hist, blue_bins = compute_image_histogram(b)
    green_hist, green_bins = compute_image_histogram(g)
    red_hist, red_bins = compute_image_histogram(r)

    return ((blue_hist, blue_bins), (green_hist, green_bins), (red_hist, red_bins)) # blue_bins = green_bins = red_bins


def compute_image_pdf(img):
    # in any case, image dimensions (without channels) will be in positions 0 and 1 of `img.shape`
    total_pixels = img.shape[0] * img.shape[1]

    # case when image is grayscale
    if len(img.shape) == 2:
        hist, bins = compute_image_histogram(img)
        pdf = hist / total_pixels
        return pdf, bins

    # otherwise we assume `img` has 3 channels
    b, g, r = cv2.split(img)
    b, g, r = tuple(compute_image_pdf(x) for x in [b, g, r])

    return b, g, r


def compute_cdf_from_pdf(pdf):
    cumulative = 0
    cdf = np.zeros(len(pdf))

    for i, value in enumerate(pdf):
        cdf[i] = cumulative + value
        cumulative += value

    return cdf


def compute_image_cdf(img):
    if len(img.shape) == 2:
        pdf, bins = compute_image_pdf(img)
        cdf = compute_cdf_from_pdf(pdf)
        return cdf, bins

    # otherwise we assume 3 channels; get pdf for each of channels
    bgr = cv2.split(img)
    b, g, r = tuple(compute_image_cdf(x) for x in bgr)
    return b, g, r


def stretch_histogram(img):
    max_value, min_value = np.max(img), np.min(img)
    stretched_hist = 255 * (img - min_value) / (0.01 + max_value - min_value)
    return stretched_hist


def get_hist_eq_mapping_from_cdf(cdf, bins):
    transformed_cdf = np.array([255 * x for x in cdf])
    mapping = {x: y for x, y in list(zip(bins, transformed_cdf))}
    return mapping


def get_hist_eq_mapping_from_image(img):
    if len(img.shape) == 2:
        cdf, bins = compute_image_cdf(img) 
        mapping = get_hist_eq_mapping_from_cdf(cdf, bins)
        return mapping


def get_naive_hist_eq_mapping_from_cdf(img, cdf):
    print(cdf)
    if len(img.shape) == 2:
        max_value, min_value = np.max(img), np.min(img) + 1
        mapping = {i: np.log(min_value) + (np.log(max_value) - np.log(min_value)) * cdf[i] for i in range(0, 256)}
        return mapping

    assert isinstance(cdf, tuple) and len(cdf) == 3
    bgr = cv2.split(img)
    mappings = tuple(get_naive_hist_eq_mapping_from_cdf(x, c) for x, c in list(zip(bgr, cdf)))
    return mappings


def get_image_from_mapping(img, mapping):
    if len(img.shape) == 2:
        equalized_img = np.vectorize(mapping.__getitem__)(img)
        return equalized_img

    assert isinstance(mapping, tuple) and len(mapping) == 3

    bgr = cv2.split(img)
    channels = tuple(get_image_from_mapping(img, m) for img, m in list(zip(bgr, mapping)))
    img = np.dstack(channels)
    return img


def equalize_histogram(img):
    if len(img.shape) == 2:
        mapping = get_hist_eq_mapping_from_image(img)
        equalized_img = np.vectorize(mapping.__getitem__)(img)
        #print(equalized_img)
        return equalized_img

    # otherwise equalize by channel
    bgr = cv2.split(img)
    bgr_equalized_histograms =list(equalize_histogram(x) for x in bgr)
    img = np.dstack(bgr_equalized_histograms)
    return img


def bi_histogram_equalization(img):
    if len(img.shape) == 2:
        threshold = int(np.round(np.mean(img)))
        #print(threshold)
        #print(img)

        lower_img = np.where(img < threshold, img, np.nan).astype(int) 
        higher_img = np.where(img >= threshold, img, np.nan).astype(int)
        #print(lower_img)
        #print(higher_img)

        # todo: do we really need flattening? can't we just do this with NaN handling on lower/higher img?
        flattened = img.flatten()
        
        lower_mask = flattened < threshold
        higher_mask = flattened >= threshold

        lower_values = flattened[lower_mask]
        higher_values = flattened[higher_mask]

        #lower_hist, bins = np.histogram(lower_values, bins=[x for x in range(0, threshold + 1)])
        #higher_hist, bins = np.histogram(higher_values, bins=[x for x in range(threshold, 257)])
        lower_hist, lower_bins = np.histogram(lower_img, bins=[x for x in range(0, threshold + 1)])
        higher_hist, higher_bins = np.histogram(higher_img, bins=[x for x in range(threshold, 257)])

        total_pixels_lower, total_pixels_higher = len(lower_values), len(higher_values)
        

        lower_pdf = lower_hist / total_pixels_lower
        higher_pdf = higher_hist / total_pixels_higher
        

        lower_cdf = compute_cdf_from_pdf(lower_pdf)
        higher_cdf = compute_cdf_from_pdf(higher_pdf)

        lower_mapping = get_hist_eq_mapping_from_cdf(lower_cdf, lower_bins)
        lower_mapping[np.nan] = 0
        lower_mapping[-9223372036854775808] = 0
        higher_mapping = get_hist_eq_mapping_from_cdf(higher_cdf, higher_bins)
        higher_mapping[np.nan] = 0
        higher_mapping[-9223372036854775808] = 0
        
        #print(higher_mapping)

        lower_equalized_img = np.vectorize(lower_mapping.__getitem__)(lower_img)
        higher_equalized_img = np.vectorize(higher_mapping.__getitem__)(higher_img)
        #higher_equalized_img = higher_img
        #print(lower_equalized_img)
        #print(lower_equalized_img)
        
        equalized_img = lower_equalized_img + higher_equalized_img
        print(np.max(equalized_img))
        return equalized_img

    # case for color images
    # todo: just write a wrapper to handle color images, e.g. `return colorize(bi_histogram_equalization, img)`
    bgr = cv2.split(img)
    equalized_channels = tuple(bi_histogram_equalization(x) for x in bgr)
    equalized_img = np.dstack(equalized_channels)
    return equalized_img


def binned_histogram_equalization(img, num_bins):
    pass


def adaptive_histogram_equalization(img):
    pass


def local_histogram_equalization(img, disk_size):
    if len(img.shape) == 2:
        kernel = morphology.rectangle(disk_size, disk_size)
        equalized_img = rank.equalize(img, kernel)
        return equalized_img

    bgr = cv2.split(img)
    equalized_channels = tuple(local_histogram_equalization(x, disk_size) for x in bgr)
    equalized_img = np.dstack(equalized_channels)
    return equalized_img


def average_histogram_equalization(img):
    gray_img = luminosity_method(img)
    gray_hist = compute_image_histogram(gray_img)
    gray_img_mapping = get_hist_eq_mapping_from_image(gray_img)
    #gray_img_equalized = np.vectorize(gray_img_mapping.__getitem__)(gray_img)
    b, g, r = cv2.split(img)

    b_eq = np.vectorize(gray_img_mapping.__getitem__)(b)
    g_eq = np.vectorize(gray_img_mapping.__getitem__)(g)
    r_eq = np.vectorize(gray_img_mapping.__getitem__)(r)

    reconstructed = np.dstack([b_eq, g_eq, r_eq])
    return reconstructed
    #bgr_hists = tuple(compute_image_histogram(i) for i in bgr)
    #average_hist = sum([x[0] for x in bgr_hists]) / 3


def alternative_average_hist_eq(img):
    bgr = cv2.split(img)
    bins = [x for x in range(0, 257)]
    total_pixels = img.shape[0] * img.shape[1]
    hists = [compute_image_histogram(x)[0] for x in bgr]
    average_hist = (hists[0] + hists[1] + hists[2]) / 3
    pdf = average_hist / total_pixels
    cdf = compute_cdf_from_pdf(pdf)
    hist_eq_mapping = get_hist_eq_mapping_from_cdf(cdf, bins)

    b_eq = np.vectorize(hist_eq_mapping.__getitem__)(bgr[0])
    g_eq = np.vectorize(hist_eq_mapping.__getitem__)(bgr[1])
    r_eq = np.vectorize(hist_eq_mapping.__getitem__)(bgr[2])

    reconstructed = np.dstack([b_eq, g_eq, r_eq])
    return reconstructed


    





def stretch_piecewise_linear(img, threshold):
    pass


def stretch_piecewise_non_linear(img, threshold):
    pass


####################
###### tests #######
####################


def experiments(out_path='../output_data/histogram_processing/'):
    img = cv2.imread('../output_data/smooth_output/tunnel_1.png')
    #img = cv2.imread('/Users/adamcatto/src/L0-Smoothing/src/output_tunnels/tunnel_1.png')
    original_eme = compute_eme_score(luminosity_method(img), 10)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(img)
    if len(img.shape) == 3:
        cdfs = compute_image_cdf(img)
        cdfs = tuple(x[0] for x in cdfs)
    else:
        cdfs = compute_image_cdf(img)[0]

    

    naive_eq_mapping = get_naive_hist_eq_mapping_from_cdf(img, cdfs)
    naive_equalized_img = stretch_histogram(get_image_from_mapping(img, naive_eq_mapping))
    naive_equalized_img_eme = compute_eme_score(luminosity_method(naive_equalized_img), 10)
    #print(naive_equalized_img)
    equalized_global = equalize_histogram(img)
    equalized_global_eme = compute_eme_score(luminosity_method(equalized_global), 10)
    #equalized_global = 255 - base.normalize_by_channel(img - equalized_global)
    bi_equalized = bi_histogram_equalization(img)
    bi_equalized_eme = compute_eme_score(luminosity_method(bi_equalized), 10)
    #bi_equalized = stretch_histogram(img - bi_equalized)
    equalized_local = local_histogram_equalization(img, 200)
    equalized_local_eme = compute_eme_score(luminosity_method(equalized_local), 10)

    equalized_average = average_histogram_equalization(img)
    equalized_average_eme = compute_eme_score(luminosity_method(equalized_average), 10)

    alternative_equalized_average = alternative_average_hist_eq(img)
    alternative_equalized_average_eme = compute_eme_score(luminosity_method(alternative_equalized_average), 10)
    
    print('original eme: ' + str(original_eme))
    print('naive eme: ' + str(naive_equalized_img_eme))
    print('global eme: ' + str(equalized_global_eme))
    print('bihistogram equalized eme: ' + str(bi_equalized_eme))
    print('local eme: ' + str(equalized_local_eme))
    print('average method eme: ' + str(equalized_average_eme))
    print('alternative average eme: ' + str(alternative_equalized_average_eme))


    cv2.imwrite(out_path + 'equalized_global.png', equalized_global)
    cv2.imwrite(out_path + 'naive_equalized.png', naive_equalized_img)
    cv2.imwrite(out_path + 'bi_equalized.png', bi_equalized)
    cv2.imwrite(out_path + 'equalized_local.png', equalized_local)
    cv2.imwrite(out_path + 'equalized_average.png', equalized_average)
    cv2.imwrite(out_path + 'alternative_equalized_average.png', alternative_equalized_average)


#experiments()