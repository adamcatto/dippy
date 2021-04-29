import os
from operator import xor

import cv2
import numpy as np
from skimage.morphology import dilation
from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio
from skimage.util import pad, view_as_windows, crop
from skimage.feature import canny
from skimage.filters import sobel, sobel_h, sobel_v, prewitt, prewitt_h, prewitt_v
from tqdm import tqdm

from color_to_gray_operations import luminosity_method as lm


def find_empty_rows_cols_indices(arr):
    """
    returns a dict with keys 'rows' and 'cols', with values corresponding to indices of empty rows and empty columns
    """
    rows = []
    cols = []
    for i in range(arr.shape[0]):
        row = arr[i, :]
        if np.all(row == 0):
            rows.append(i)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        if np.all(col == 0):
            cols.append(j)
    return {'rows': rows, 'cols': cols}

def weighted_median_linear_interpolation(img, edge_tube):
    print(edge_tube[262, 268])
    interpolated = np.zeros(edge_tube.shape)
    cv2.imwrite('../output_data/test_output/edge_tube.png', edge_tube)

    empty_rows_cols = find_empty_rows_cols_indices(edge_tube)
    empty_rows = empty_rows_cols['rows']
    empty_cols = empty_rows_cols['cols']
    
    # -- create lists to keep track of the uppermost and lowermost nonzero edge_tube pixel at each column
    # -- (indexed by `j` in inner for-loop), and nearest vertical-direction edge tube pixels at each column
    cols_uppermost = []
    cols_lowermost = []
    cols_upper = []
    cols_lower = []
    nearest_upper_vals = []
    nearest_lower_vals = []

    # -- get uppermost and lowermost values in each column

    for col in range(interpolated.shape[1]):
        if col in empty_cols:
            cols_uppermost.append(-1)
            continue
        for row in range(interpolated.shape[0]):
            val = edge_tube[row][col]
            if val != 0:
                cols_uppermost.append(row)
                nearest_lower_vals.append(val)
                break

    cols_upper = cols_uppermost
    cols_lower = cols_upper
    nearest_lower_vals = cols_upper
    nearest_upper_vals = nearest_lower_vals

    for col in range(interpolated.shape[1]):
        if col in empty_cols:
            cols_lowermost.append(-1)
        for row in range(1, interpolated.shape[0]):
            val = edge_tube[-(row)][col]
            if val != 0:
                cols_lowermost.append(interpolated.shape[0] - row + 1)
                break

    # -- now run the whole interpolation

    for i in tqdm(range(interpolated.shape[0])):
        if i not in empty_rows:
            # -- find leftmost and rightmost nonzero values in edge_tube to use in dynamic programming
            for left_min in range(interpolated.shape[1]):
                pixel_val = edge_tube[i][left_min]
                if pixel_val != 0:
                    leftmost_idx = left_min
                    leftmost_value = pixel_val
                    break
            for right_min in range(1, interpolated.shape[1]):
                pixel_val = edge_tube[i][-(right_min)]
                if pixel_val != 0:
                    rightmost_idx = interpolated.shape[1] - right_min + 1
                    rightmost_value = pixel_val
                    break
            
            # -- at start of new row, initialize nearest horizontal indices to 
            current_left_val = leftmost_value
            current_right_val = current_left_val
            left_dist, right_dist = leftmost_idx, leftmost_idx
        else:
            # -- handle cases in which edge tube row has no nonzero values
            leftmost_idx = -1
            leftmost_value = -1
            rightmost_idx = -1
            rightmost_value = -1
            left_dist, right_dist = leftmost_idx, rightmost_idx
            # -- the trick here is that later when we go to add distances and pixel values to target arrays, 
            # -- we check if left_dist is -1. if yes, then don't use it to calculate new pixel value.

        for j in range(interpolated.shape[1]):
            # -- handle cases in which edge tube column has no nonzero values
            left_dist += 1
            right_dist -= 1
            if j in empty_cols:
                if i in empty_rows:
                    interpolated[i, j] = np.random.randint(0,255)
                    continue
                else:
                    interpolated[i, j] = ((current_left_val / (left_dist + 1)) + (current_right_val / (right_dist + 1))) * (left_dist + right_dist + 2)
                    continue
            upper_dist = np.abs(i - cols_upper[j])
            lower_dist = np.abs(cols_lower[j] - i)
            
            current_val = edge_tube[i][j]

            if current_val == 0:
                if j <= leftmost_idx:
                    current_right_val = leftmost_value
                    current_left_val = current_right_val

                    if i < cols_uppermost[j]:
                        current_lower_value = nearest_lower_vals[j]
                        current_upper_value = current_lower_value
                    elif i > cols_lowermost[j]:
                        current_upper_value = nearest_upper_vals[j]
                        current_lower_value = current_upper_value
                    else:
                        current_upper_value = nearest_upper_vals[j]
                        current_lower_value = nearest_lower_vals[j]
                elif j >= rightmost_idx:
                    current_left_val = rightmost_value
                    current_right_val = current_left_val

                    if i < cols_uppermost[j]:
                        current_lower_value = nearest_lower_vals[j]
                        current_upper_value = current_lower_value
                    elif i > cols_lowermost[j]:
                        current_upper_value = nearest_upper_vals[j]
                        current_lower_value = current_upper_value
                    else:
                        current_upper_value = nearest_upper_vals[j]
                        current_lower_value = nearest_lower_vals[j]

                else:
                    if i < cols_uppermost[j]:
                        current_lower_value = nearest_lower_vals[j]
                        current_upper_value = current_lower_value
                    elif i > cols_lowermost[j]:
                        current_upper_value = nearest_upper_vals[j]
                        current_lower_value = current_upper_value
                    else:
                        current_upper_value = nearest_upper_vals[j]
                        current_lower_value = nearest_lower_vals[j]

            elif current_val != 0:
                current_left_val = current_val
                current_upper_value = current_val
                left_dist = 1
                upper_dist = 1
                # set new upper column index to the current row `i`
                cols_upper[j] = i
                # set new upper column pixel value to the current pixel value
                nearest_upper_vals[j] = current_val

                # -- if current pixel location is at rightmost index or lowermost index, don't search for new locations
                if j < rightmost_idx - 1:
                    # -- if current pixel is in edge tube, need to search for nearest edge tube pixel to the right
                    nearest_right_found = False
                    r = 1
                    while j + r <= rightmost_idx:
                        if edge_tube[i][j + r] != 0:
                            current_right_val = edge_tube[i][j + r]
                            nearest_right_found = True
                            right_dist = r
                            break
                        else:
                            r += 1
                if i < cols_lowermost[j] - 1:
                    # -- if current pixel is in edge tube, need to search for nearest edge tube pixel below
                    nearest_lower_found = False
                    l = 1

                    while not nearest_lower_found:
                        v = edge_tube[i + l, j]
                        if v == 0:
                            l += 1
                        else:
                            lower_idx = i + l
                            cols_lower[j] = lower_idx
                            current_lower_value = edge_tube[i + l][j]
                            nearest_lower_vals[j] = current_lower_value
                            lower_dist = l
                            
                            nearest_lower_found = True

            values = [current_upper_value, current_lower_value]
            distances = [upper_dist, lower_dist]
            if leftmost_idx != -1:
                values += [current_left_val, current_right_val]
                distances += [left_dist, right_dist]
            
            max_dist = np.max(distances)
            weights = (max_dist - distances) + 1
            median_array = []
            for c, w in enumerate(weights):
                median_array += [values[c]] * int(w)
            
            interpolated[i][j] = int(np.median(np.array(median_array)))
            
    return interpolated


def edge_detection_qual_non_ref(img, edge_map=None, edge_detector=None, similarity_measure='SSIM', weights=None):
    # -- initializations
    assert xor(bool(edge_detector), bool(edge_map))
    if edge_detector:
        print(edge_detector)
        if isinstance(edge_detector, str):
            edge_detector = edge_detector.lower()
            edge_detector_str = edge_detector
            if edge_detector == 'canny':
                edge_detector = canny
            elif edge_detector == 'prewitt':
                edge_detector = prewitt
            elif edge_detector == 'prewitt_h':
                edge_detector = prewitt_h
            elif edge_detector == 'prewitt_v':
                edge_detector = prewitt_v
            elif edge_detector == 'sobel':
                edge_detector = sobel
            elif edge_detector == 'sobel_h':
                edge_detector = sobel_h
            elif edge_detector == 'sobel_v':
                edge_detector = sobel_v
        
        edge_map = edge_detector(img)
        if edge_detector != canny:
            edge_map = np.where(edge_map > 15, 1, 0)

    if not isinstance(similarity_measure, str):
        raise AssertionError('similarity_measure must be a string denoting the similarity measure of your choice.')
    
    similarity_measure = similarity_measure.upper()
    
    if similarity_measure == 'SSIM':
        measure_similarity = structural_similarity
    elif similarity_measure == 'MSE':
        measure_similarity = mean_squared_error
    elif similarity_measure == 'PSNR':
        measure_similarity = peak_signal_noise_ratio

    if weights is None:
        weights = np.ones(8)

    # -- cast any possible bool arrays to False ~ 0 ; True ~ 255
    edge_map = edge_map.astype(int) * 255
    cv2.imwrite('../output_data/edge_detection/quality_measures/edge_maps/' + edge_detector_str + '.png', edge_map)

    dilated_edge_map = dilation(edge_map)
    cv2.imwrite('../output_data/edge_detection/quality_measures/dilated_edge_maps/' + edge_detector_str + '.png', dilated_edge_map)
    
    edge_tube = np.where(dilated_edge_map == 255, img, 0)
    cv2.imwrite('../output_data/edge_detection/quality_measures/edge_tubes/' + edge_detector_str + '.png', edge_tube)

    # -- perform weighted median linear interpolation to reconstruct image
    reconstructed_img = weighted_median_linear_interpolation(img, edge_tube=edge_tube)
    cv2.imwrite('../output_data/edge_detection/quality_measures/reconstructed/' + edge_detector_str + '.png', reconstructed_img)

    # -- compare original image with image reconstructed from edge map
    similarity_score = measure_similarity(img, reconstructed_img)

    return similarity_score


def compare_edge_detector_qualities(img, edge_detectors=None, measure='SSIM', weights=None):
    if edge_detectors is None:
        edge_detectors = ['canny', 'sobel', 'prewitt']
    if isinstance(edge_detectors, str):
        edge_detectors = [edge_detectors]

    scores = {d: None for d in edge_detectors}

    for d in edge_detectors:
        scores[d] = edge_detection_qual_non_ref(img=img, edge_detector=d, similarity_measure=measure, weights=weights)
    
    return scores


def run_edge_detector_quality_measure():
    img = lm(cv2.imread('../input_data/grayscale_tunnels/grayscale_tunnel_1.png'))
    qualities = compare_edge_detector_qualities(img)
    print(qualities)


def apply_frost_filter(window, tuning_factor=0.1):
    window_flat = np.ravel(window)
    sum_all_pixels = np.sum(window_flat)
    filter_val = 0
    mean = np.mean(window_flat)
    stdev = np.var(window_flat)
    var_coef = stdev / mean
    ms = []

    for i, val in enumerate(window_flat):
        position = (i // window.shape[0] / 2 , i % window.shape[0] // 2)
        dist = np.max(np.array(window.shape) // 2 - np.array(position))
        m = np.exp(var_coef ** 2 * (-tuning_factor) * dist)
        ms.append(m)
    
    ms = np.array(ms)
    return np.dot(window_flat, ms) / np.sum(ms)


def shearlet_transform(img, window_shape=17, shear=2, scale=1, translation=2):
    """
    # -- step 1:    frost filter preprocessing
                    this preserves details around edges
    # -- step 2: log transform
    # -- step 3: apply shearlet --> get road
    # -- step 4: apply exponential transform
    # -- step 5: binarize using threshold
    # -- step 6: apply morphological operation
    # -- step 7: return
    """
    if len(img.shape) == 3:
        img = lm(img)
    # -- step 1: frost filter preprocessing
    # -- initialize windows
    windowed_img = view_as_windows(arr_in=img, window_shape=window_shape)
    transformed_array = np.zeros(img.shape[0] * img.shape[1])
    for i, w in tqdm(enumerate(windowed_img)):
        transformed_array[i] = apply_frost_filter(window=w)

    transformed_array = np.log(transformed_array)
    transformed_img = transformed_array.reshape(img.shape)
    anisotropic_expansion_matrix = np.array([[1, k], [0, 1]])
    shear_matrix = np.array([[1, 1], [0, 1]])
    affine_system = np.linalg.norm(np.linalg.det(transformed_array)) ** (scale / 2)
    # todo: finish this function


