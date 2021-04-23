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


def get_value_dist_direction(arr, location, directions, k):
    values = {}
    #print(directions.shape)
    for d in directions:
        d = (d / k).astype(int)
        d = tuple(d)
        values[d] = arr[(location[0] + d[0], location[1] + d[1])]
    #print((location[0] + d[0], location[1] + d[1]))
    return values


def weighted_median_linear_interpolation(img, edge_tube, input_weights=np.ones(8), pad_width=500):
    #pad_width = max(edge_tube.shape)
    max_iters = 400
    if len(input_weights.shape) != 1:
        input_weights = np.ravel(input_weights)
    if input_weights.size == 9:
        input_weights = np.delete(input_weights, 4)
    interpolated = np.zeros(edge_tube.shape)
    edge_tube = pad(edge_tube, pad_width=pad_width)
    cv2.imwrite('../output_data/test_output/edge_tube.png', edge_tube)
    #edge_tube = view_as_windows(edge_tube, 3)
    for i in tqdm(range(interpolated.shape[0])):
        for j in range(interpolated.shape[1]):
            edge_pixels_found = False
            k = 0
            idx1 = pad_width + i
            idx2 = pad_width + j
            #distances = np.zeros(8)
            #values = np.zeros(8)
            #dists = [(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]]
            dists = np.array([(1,1), (1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0), (-1,-1)])
            #dists_dict = {x: None for x in dists}
            #values_dict = {x: None for x in dists}
            dists_dict = {}
            values_dict = {}
            
            #print(dists)
            if edge_tube[idx1, idx2] != 0:
                interpolated[i, j] = edge_tube[idx1, idx2]
                #print('in edge tube')
                continue # -- skip the search process
            
            while not edge_pixels_found:
                k += 1
                #print(k)
                if k == max_iters:
                    #print('not found' + str(k))
                    #print(i, j)
                    break
                
                if i - k == 0 or j - k == 0:
                    break

                new_pixels = get_value_dist_direction(edge_tube, (idx1, idx2), k * dists, k)
                #print(new_pixels)
                
                #b = 0
                for a, p in new_pixels.items():
                    #print(p)
                    if p != 0:
                        if not a in dists_dict.keys():
                            dists_dict[a] = k
                            values_dict[a] = p
                            #dists = np.delete(dists, a - b, 0)
                            #b += 1
                            
                if len(dists_dict) == 8:
                    edge_pixels_found = True

            if not dists_dict:
                interpolated[i][j] = img[i][j]
                continue
            distances = np.array(list(dists_dict.values()))
            values = np.array(list(values_dict.values()))
            max_dist = np.max(distances)
            min_dist = np.min(distances)
            weights = (max_dist - distances) + 1
            median_array = []
            for c, w in enumerate(weights):
                median_array += [values[c]] * int(w)
            
            #print(int(np.median(np.array(median_array))))
            interpolated[i][j] = int(np.median(np.array(median_array)))
    #print(interpolated)
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
    reconstructed_img = weighted_median_linear_interpolation(img, edge_tube=edge_tube, input_weights=weights)
    cv2.imwrite('../output_data/edge_detection/quality_measures/reconstructed/' + edge_detector_str + '.png', reconstructed_img)

    # -- compare original image with image reconstructed from edge map
    similarity_score = measure_similarity(img, reconstructed_img)

    return similarity_score


def compare_edge_detector_qualities(img, edge_detectors=None, measure='SSIM', weights=None):
    if edge_detectors is None:
        edge_detectors = ['canny', 'sobel', 'sobel_h', 'sobel_v', 'prewitt', 'prewitt_h', 'prewitt_v']
    if isinstance(edge_detectors, str):
        edge_detectors = [edge_detectors]

    scores = {d: None for d in edge_detectors}

    for d in edge_detectors:
        scores[d] = edge_detection_qual_non_ref(img=img, edge_detector=d, similarity_measure=measure, weights=weights)
    
    return scores


def main():
    img = lm(cv2.imread('../input_data/grayscale_tunnels/grayscale_tunnel_1.png'))
    qualities = compare_edge_detector_qualities(img)
    print(qualities)


main()
