from typing import Tuple
import os
from operator import xor

import cv2
import numpy as np
from skimage.morphology import dilation, erosion, opening, closing, disk
from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio
from skimage.util import pad, view_as_windows, crop, montage
from skimage.feature import canny
from skimage.filters import sobel, sobel_h, sobel_v, prewitt, prewitt_h, prewitt_v, difference_of_gaussians
from skimage.filters.rank import minimum
from skimage.measure import label
from tqdm import tqdm
#from findpeaks import frost_filter
#from findpeaks.filters.kuan import kuan_filter
#from scipy.fft import dctn, dstn, dct, dst, idctn, idstn
#from pywt import dwt2

from color_to_gray_operations import luminosity_method as lm
from segmentation import prepare_frame_segment, system_3


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
            elif edge_detector == 'lapgauss':
                edge_detector = difference_of_gaussians
        
        if edge_detector != difference_of_gaussians:
            edge_map = edge_detector(img)
        else:
            edge_map = edge_detector(img, 2)
        if edge_detector != canny:
            edge_map = np.where(edge_map > 1, 1, 0)

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
    edge_map = edge_map.astype(int)
    num_edge_pixels_predicted = np.sum(np.ravel(edge_map))
    edge_map *= 255
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
    similarity_score /= np.log(num_edge_pixels_predicted + np.e)

    return similarity_score


def compare_edge_detector_qualities(img, edge_detectors=None, measure='SSIM', weights=None):
    if edge_detectors is None:
        edge_detectors = ['canny', 'sobel', 'sobel_h', 'sobel_v', 'prewitt', 'prewitt_h', 'prewitt_v', 'lapgauss']
    if isinstance(edge_detectors, str):
        edge_detectors = [edge_detectors]

    scores = {d: None for d in edge_detectors}

    for d in edge_detectors:
        scores[d] = edge_detection_qual_non_ref(img=img, edge_detector=d, similarity_measure=measure, weights=weights)
    
    return scores


def run_edge_detector_quality_measure():
    img = lm(cv2.imread('../input_data/visolo_tunnel_gray.jpg'))
    qualities = compare_edge_detector_qualities(img)
    print(qualities)

    
def motion_edge_based_segmentation(filenames, out_dir, num_frames=150, num_prev_frames=10, blur=(7,7), as_numeric=True, stretched=True):
    """
    seed img ~ cumulative moving average
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if num_frames is None or num_frames > len(filenames):
        num_frames = len(filenames)

    edge_detector = canny

    initial_background_model = []
    for f in filenames[0:num_prev_frames]:
        seed_frame = lm(cv2.imread(f))
        if blur:
            seed_frame = cv2.blur(seed_frame, blur)
        seed_frame_edges = edge_detector(seed_frame).astype(int)
        #seed_frame_edges = seed_frame_edges
        initial_background_model.append(seed_frame_edges)

    initial_background_model = np.array(initial_background_model)
    initial_background_model = np.sum(initial_background_model, axis=0) / 10
    #seed_img = np.average(initial_background_model, axis=0)
    seed_img = initial_background_model
    seed_img_binary = np.where(initial_background_model >= 0.5, 1, 0)
    #print(seed_img_binary)
    
    
    #initial_background_model = np.array([lm(cv2.imread(f)) for f in filenames[0:num_prev_frames]])

    #if blur:
     #   seed_img = cv2.blur(seed_img, blur)
    counter = 10

    for i, f in tqdm(enumerate(filenames[num_prev_frames:num_frames])):
        img = lm(cv2.imread(f))
        if blur:
            img = cv2.blur(img, blur)
        img_edge_map = edge_detector(img).astype(int)
        seed_img = (counter * seed_img + img_edge_map) / (counter + 1)
        seed_img_binary = np.where(seed_img >= 0.5, 1, 0)
        #segmented = prepare_frame_segment(seed_img=seed_img_binary, img=img, blur=None)
        segmented = img_edge_map - seed_img_binary
        #segmented = binary_erosion(segmented, selem=np.ones((3,3)))
        counter += 1
        cv2.imwrite(os.path.join(out_dir, 'backgrounds', filenames[i + num_prev_frames][-8:]), seed_img_binary * 255)
        cv2.imwrite(os.path.join(out_dir, 'edge_maps', filenames[i + num_prev_frames][-8:]), img_edge_map * 255)
        cv2.imwrite(os.path.join(out_dir, 'segmented', filenames[i + num_prev_frames][-8:]), segmented * 255)



def apply_frost_filter(window, tuning_factor=0.1):
    """
    IMO this looks better than default findpeaks frost filter, but takes way longer.
    """
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


def compute_shearlet_at_point(point: Tuple[int,int], shear=2, scale=2, translation=(2,2)):
    shear_matrix = np.array([[1, shear], [0, 1]])
    anisotropic_expansion_matrix = np.array([[2**scale, 0], [0, 2**(scale/2)]])
    #even_shearlet = 2 ** (3 * scale / 2) * dwt2(shear_matrix * anisotropic_expansion_matrix * (np.array(point) - np.array(translation)), wavelet='haar')
    #odd_shearlet = 2 ** (3 * scale / 2) * _odd_ * shear_matrix * anisotropic_expansion_matrix * (np.array(point) - np.array(translation))


def shearlet_transform(img, window_shape=17, shear=1, scale=2, translation=(2,2)):
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
    #path = '/Users/adamcatto/SRC/dippy/output_data/edge_detection/shearlets/shearlets'
    if len(img.shape) == 3:
        img = lm(img)
    # -- step 1: frost filter preprocessing
    # -- initialize windows
    windowed_img = view_as_windows(arr_in=img, window_shape=window_shape)
    #transformed_array = np.zeros(img.shape[0] * img.shape[1])
    #for i, w in tqdm(enumerate(windowed_img)):
    #    transformed_array[i] = apply_frost_filter(window=w)

    #transformed_array = np.log(transformed_array)
    #transformed_img = transformed_array.reshape(img.shape)
    shear_matrix = np.array([[1, shear], [0, 1]])
    anisotropic_expansion_matrix = np.array([[2**scale, 0], [0, 2**(scale/2)]])
    dct_img = dctn(img)
    dst_img = dstn(img)
    #dct_img = dct(dct(img.T, norm='ortho').T, norm='ortho')
    #dst_img = dst(dst(img.T, norm='ortho').T, norm='ortho')
    
    #wav_img = np.dstack(dct_img, dst_img)
    
    dct_img *= 2**scale
    dst_img *= 2** (scale/2)

    new_dct_img = dct_img + shear * dst_img
    new_dct_img *= (2 ** (3*scale / 2))
    new_dst_img = dst_img
    new_dst_img *= (2 ** (3*scale / 2))


    path = '/Users/adamcatto/SRC/dippy/output_data/edge_detection/shearlets/shearlets'
    cv2.imwrite(os.path.join(path, 'even.png'), idctn(new_dct_img))
    cv2.imwrite(os.path.join(path, 'odd.png'), idctn(new_dst_img))

    #shearlet = compute_shearlet_at_point((i, j), shear, scale, translation)
    
    
    #affine_system = np.linalg.norm(np.linalg.det(transformed_array)) ** (scale / 2)

    # todo: finish this function


def motion_edge_based_segmentation_prev_frame(filenames, out_dir, num_frames=150, window_shape=9):
    """
    seed img ~ cumulative moving average
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if num_frames is None or num_frames > len(filenames):
        num_frames = len(filenames)

    edge_detector = sobel
    prev_frame = lm(cv2.imread(filenames[0]))
    prev_frame = kuan_filter(prev_frame, window_shape)
    #prev_frame = transformed_array.reshape(prev_frame.shape)
    prev_frame_edges = edge_detector(prev_frame).astype(bool).astype(int)
    prev_frame_edges_dilated = binary_dilation(prev_frame_edges).astype(int)

    for i, f in tqdm(enumerate(filenames[1:num_frames])):
        curr_frame = lm(cv2.imread(f))
        curr_frame = kuan_filter(curr_frame, window_shape)
        cv2.imwrite(os.path.join(out_dir, 'frost_filter', filenames[i + 1][-8:]), curr_frame)
        curr_frame_edges = edge_detector(curr_frame).astype(bool).astype(int)
        curr_frame_edges_dilated = binary_dilation(curr_frame_edges).astype(int)
        
        diff_dilated = curr_frame_edges_dilated - prev_frame_edges_dilated
        cv2.imwrite(os.path.join(out_dir, 'segmented', filenames[i + 1][-8:]), diff_dilated*255)
        prev_frame_edges_dilated = curr_frame_edges_dilated

def motion_segment_morphology(frame0, frame1):
    #img0 = lm(cv2.imread('../input_data/cleaned_gray_tunnel_sequence/0025.png'))
    #img1 = lm(cv2.imread('../input_data/cleaned_gray_tunnel_sequence/0026.png'))
    #canny_edges0 = canny(img0).astype(int) * 255
    #canny_edges1 = canny(img1).astype(int) * 255
    sobel_edges0 = sobel(frame0)
    sobel_edges1 = sobel(frame1)
    sobel_edges0 = np.where(sobel_edges0 > 10, 1, 0)
    sobel_edges1 = np.where(sobel_edges1 > 10, 1, 0)
    sobel_diff = sobel_edges1 - sobel_edges0

    sobel_diff = erosion(sobel_diff, np.ones((3,3)))
    #sobel_diff = erosion(sobel_diff, selem=np.ones((3,3)))
    sobel_diff = erosion(sobel_diff, selem=np.ones((5,5)))
    sobel_diff = dilation(sobel_diff, selem=np.ones((11,11)))
    sobel_diff = dilation(sobel_diff, selem=np.ones((11,11)))
    sobel_diff = dilation(sobel_diff, selem=np.ones((3,3)))
    sobel_diff = dilation(sobel_diff, selem=np.ones((5,5)))
    sobel_diff = dilation(sobel_diff, selem=np.ones((7,7)))
    sobel_diff = dilation(sobel_diff, selem=np.ones((9,9)))
    sobel_diff = closing(sobel_diff, selem=np.ones((11,11)))
    sobel_diff = closing(sobel_diff, np.ones((15,15)))
    r = sobel_diff.astype(int) * 255
    return r


def motion_segment_morphology_batch(img_dir, num_frames, num_prev_frames):
    files = os.listdir(img_dir)
    files = list(sorted(files))
    filenames = list(sorted([os.path.join(img_dir, f) for f in files]))

    diffs = []
    seg_diffs = []
    selem = np.ones((3, 3))
    img0 = lm(cv2.imread(filenames[0]))
    mask_zeros = np.zeros((int(img0.shape[0] / 3) + 12, img0.shape[1]))
    mask_ones = np.ones((img0.shape[0] - 12 - int(img0.shape[0] / 3), img0.shape[1]))
    mask = np.vstack((mask_zeros, mask_ones)).astype(int)
    print(mask.shape)
    #mask = np.dstack((mask, mask, mask))
    #motion_seg_images = system_3(img_dir, threshold=10)
    for i, f in tqdm(enumerate(filenames[num_prev_frames: num_frames])):
        img1 = lm(cv2.imread(f))
        img_stacked = np.dstack((img1, img1, img1))
        diff = motion_segment_morphology(img0, img1)
        diff *= mask
        #blobs_diff = label(diff, background=0)

        #motion_seg_img = motion_seg_images[i]
        #joined = diff + motion_seg_img
        #joined = np.dstack((joined, np.zeros(joined.shape), np.zeros(joined.shape)))
        #app = np.where(joined > 0, 255, img_stacked)
        #blobs_seg = label(motion_seg_img, background=0)
        #blobs_seg = label(erosion(motion_seg_img, selem=np.ones((5,5))), background=0)

        #intersection_blobs = np.where((blobs_seg != 0) & (diff != 0), blobs_seg, 0)
        #labels_to_include = np.unique(intersection_blobs)[1:]
        #joined_diff = np.where((blobs_seg in labels_to_include) | (diff > 0), 255, 0)
        #stack_diff = np.dstack((diff, np.zeros(diff.shape), np.zeros(diff.shape)))
        #diff = np.where(diff > 0, 255, img_stacked)
        #diff = np.dstack((joined_diff, np.zeros(joined_diff.shape), np.zeros(joined_diff.shape)))
        diff = np.dstack((diff, np.zeros(diff.shape), np.zeros(diff.shape)))
        diff = np.where(diff > 0, 255, img_stacked)
        
        seg_diffs.append(diff)
        img0 = img1
    
    for i, d in tqdm(enumerate(seg_diffs)):
        cv2.imwrite('../output_data/edge_detection/tunnel_sequence/morphology_diffs/' + files[i + num_prev_frames], d)
        cv2.imwrite('../output_data/edge_detection/tunnel_sequence/morphology_seg_combos/' + files[i + num_prev_frames], seg_diffs[i])



def make_montages(original_path, processed_path, out_dir):
    original_files = list(sorted(os.listdir(original_path)))
    processed_files = list(sorted(os.listdir(processed_path)))

    for i, f in tqdm(enumerate(processed_files)):
        original_img = lm(cv2.imread(os.path.join(original_path, f)))
        processed_img = lm(cv2.imread(os.path.join(processed_path, f)))
        arr = np.array([original_img, processed_img])
        mntg = montage(arr, grid_shape=(1, 2))
        out_file = os.path.join(out_dir, f)
        cv2.imwrite(out_file, mntg)


motion_segment_morphology_batch(img_dir='../input_data/cleaned_gray_tunnel_sequence/', num_frames=150, num_prev_frames=10)
make_montages('../input_data/cleaned_gray_tunnel_sequence/', 
    '../output_data/edge_detection/tunnel_sequence/morphology_diffs/', 
    '../output_data/edge_detection/tunnel_sequence/montages/')



"""
img = lm(cv2.imread('../output_data/deblur/local/gray_tunnel_1.png'))
bin_img = np.where(img > 180, 255, 0)

selem = np.array([
    [0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 0, 0],
    [1, 1, 0, 0, 0]
])
bin_img = erosion(bin_img, selem)

selem = np.array([
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]
])


#bin_img = closing(bin_img, selem)
#bin_img = dilation(bin_img, np.array([[1,1,1]]))
#bin_img = opening(image=bin_img, selem=np.ones((3,3)))
#print(bin_img)
#cv2.imwrite('../output_data/test_output/tunnel_deblur_edges.png', bin_img)
#img = lm(cv2.imread('../input_data/grayscale_tunnels/grayscale_tunnel_1.png'))
#shearlet_transform(img)
#run_edge_detector_quality_measure()

gray_tunnel_seq_names = list(sorted(os.listdir('../input_data/cleaned_gray_tunnel_sequence/')))
gray_tunnel_seq = list(sorted([os.path.join('../input_data/gray_tunnel_sequence/images/', t) for t in gray_tunnel_seq_names]))
edge_seg_dir = '../output_data/segmentation/edge_based'
motion_edge_based_segmentation_prev_frame(filenames=gray_tunnel_seq, out_dir=edge_seg_dir)
"""