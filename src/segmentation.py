from __future__ import annotations
import os
from typing import Tuple, Union
from collections import deque
from math import log


import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage.exposure import adjust_gamma
from scipy.stats import mode
from skimage.filters import threshold_niblack
from skimage.util import montage
from skimage.feature import canny, blob_doh


from color_to_gray_operations import luminosity_method as lm
from histogram_processing import stretch_histogram, equalize_histogram, compute_image_pdf
from decomposition import slice_bit_planes


# todo: create configs.py and name separate configs for separate files/experiments
config = {}
config['out_dir'] = '../output_data/segmentation/otsu'
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


def write_thresholds(img, out_dir):
    img = lm(img)
    for t in tqdm(range(256)):
        segmented = img <= t
        segmented = stretch_histogram(segmented.astype(int))
        cv2.imwrite(os.path.join(out_dir, str(t) + '.jpg'), segmented)


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


def pad_int(num, desired_length):
    num_str = str(num)
    num_str_len = len(num_str)
    assert num_str_len <= desired_length
    diff = desired_length - num_str_len
    zero_string = ''
    
    for i in range(diff):
        zero_string += '0'
    
    return zero_string + num_str


def test_abstract_painting():
    out_dir = '../output_data/abstract_painting_nopadding/'
    img = lm(cv2.imread(config['image_paths']['original']))
    start_num = 1
    for t in tqdm(range(11, 151, 2)):
        #padded_counter = pad_int(t, 3)
        thresholded = threshold_niblack(img, t)
        segmented = (img < thresholded).astype(int)
        segmented = stretch_histogram(segmented)
        cv2.imwrite(os.path.join(out_dir, str(start_num) + '.jpg'), segmented)
        start_num += 1


def prepare_frame_segment(seed_img, filename=None, img=None, blur=None, as_numeric=True, stretched=True):
    if filename:
        img = lm(cv2.imread(filename))
    elif img is not None:
        if len(img.shape) == 3:
            img = lm(img)
    else:
        raise Exception('Must specify either a path to the image file or a numpy-array image; got neither')
    
    if blur:
        if isinstance(blur, int):
            blur = (blur, blur)
        img = cv2.blur(img, blur)
        #seed_img = cv2.blur(seed_img, blur)

    img = slice_bit_planes(img, 4)
    seed_img = slice_bit_planes(seed_img, 4)
    diff_img = np.abs(img - seed_img)
    segmented = diff_img > threshold
    if as_numeric:
        segmented = segmented.astype(int)
        if stretched:
            segmented = stretch_histogram(segmented)
    return segmented


"""
def motion_segmentation(seed_img_file, in_dir, out_dir, threshold, num_frames, blur=None):
    # process the seed image
    if seed_img_file is not None:
        seed_img = lm(cv2.imread(seed_img_file))
        seed_img = slice_bit_planes(seed_img, 4)
"""


def _prepare_filenames(in_dir, num_frames:Union[None, int]):
    files = os.listdir(in_dir)
    num_files = len(files)
    if not num_frames:
        num_frames = num_files
    else:
        assert num_frames < num_files
    filenames = list(sorted([os.path.join(in_dir, x) for x in files][0:num_frames]))
    return filenames


def _prepare_frame_segment(seed_img, threshold=20, filename=None, img=None, blur=None, as_numeric=True, stretched=True):
    if filename:
        img = lm(cv2.imread(filename))
    elif img is not None:
        if len(img.shape) == 3:
            img = lm(img)
    else:
        raise Exception('Must specify either a path to the image file or a numpy-array image; got neither')
    
    if blur:
        if isinstance(blur, int):
            blur = (blur, blur)
        img = cv2.blur(img, blur)
        #seed_img = cv2.blur(seed_img, blur)

    img = slice_bit_planes(img, 4)
    seed_img = slice_bit_planes(seed_img, 4)
    diff_img = np.abs(img - seed_img)
    segmented = diff_img > threshold
    if as_numeric:
        segmented = segmented.astype(int)
        if stretched:
            segmented = stretch_histogram(segmented)
    return segmented

    
def system_1(seed_img_file, in_dir, out_dir, threshold, num_frames=150, as_numeric=True, blur=None, stretched=True, verbose=True):
    """
    seed img ~ user-specified
    """
    # process the seed image
    if seed_img_file is not None:
        seed_img = lm(cv2.imread(seed_img_file))
        seed_img = slice_bit_planes(seed_img, 4)
    filenames = _prepare_filenames(in_dir, num_frames=150)
    # process the frames in the directory
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if verbose:
        for i, f in tqdm(enumerate(filenames)):
            segmented = _prepare_frame_segment(seed_img, filename=f, blur=blur)
            cv2.imwrite(os.path.join(out_dir, 'segmented', filenames[i][-8:]), segmented)
    else:
        for i, f in enumerate(filenames):
            segmented = _prepare_frame_segment(seed_img)
            cv2.imwrite(os.path.join(out_dir, 'segmented', filenames[i][-8:]), segmented)


def system_2(in_dir, out_dir, threshold, num_frames=150, num_prev_frames=10, blur=None, as_numeric=True, stretched=True):
    """
    seed img ~ moving average
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    filenames = _prepare_filenames(in_dir, num_frames=150)
    initial_background_model = np.array([lm(cv2.imread(f)) for f in filenames[0:num_prev_frames]])
    seed_img = np.average(initial_background_model, axis=0)
    #print(initial_background_model.shape)
    if blur:
        seed_img = cv2.blur(seed_img, blur)
    previous_frames = deque(initial_background_model, maxlen=num_prev_frames)

    for i, f in tqdm(enumerate(filenames[num_prev_frames:])):
        img = lm(cv2.imread(f))
        if blur:
            img = cv2.blur(img, blur)
        seed_img = seed_img + ((img - previous_frames.popleft()) / num_prev_frames)
        previous_frames.append(img)
        segmented = _prepare_frame_segment(seed_img=seed_img, img=img, blur=None)
        cv2.imwrite(os.path.join(out_dir, 'segmented', filenames[i + num_prev_frames][-8:]), segmented)
        cv2.imwrite(os.path.join(out_dir, 'backgrounds', filenames[i + num_prev_frames][-8:]), seed_img)


def system_3(in_dir, threshold, num_frames=150, num_prev_frames=10, blur=None, as_numeric=True, stretched=True):
    """
    seed img ~ cumulative moving average
    """
    #if not os.path.exists(out_dir):
    #    os.mkdir(out_dir)
    filenames = _prepare_filenames(in_dir, num_frames=150)
    initial_background_model = np.array([lm(cv2.imread(f)) for f in filenames[0:num_prev_frames]])
    seed_img = np.average(initial_background_model, axis=0)
    if blur:
        seed_img = cv2.blur(seed_img, blur)
    counter = 10
    segmented_imgs = []

    for i, f in tqdm(enumerate(filenames[num_prev_frames:])):
        img = lm(cv2.imread(f))
        if blur:
            img = cv2.blur(img, blur)
        seed_img = (counter * seed_img + img) / (counter + 1)
        segmented = _prepare_frame_segment(seed_img=seed_img, img=img, blur=None)
        #segmented = blob_doh(segmented)
        counter += 1
        segmented_imgs.append(segmented)
        #cv2.imwrite(os.path.join(out_dir, 'segmented', filenames[i + num_prev_frames][-8:]), segmented)
        #cv2.imwrite(os.path.join(out_dir, 'backgrounds', filenames[i + num_prev_frames][-8:]), seed_img)
    return segmented_imgs


def system_3a(in_dir, out_dir, threshold, num_frames=150, num_prev_frames=10, blur=None, as_numeric=True, stretched=True):
    """
    seed img ~ cumulative moving average
    system 3 + edge information as regularization factor
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    filenames = _prepare_filenames(in_dir, num_frames=150)
    initial_background_model = np.array([lm(cv2.imread(f)) for f in filenames[0:num_prev_frames]])
    seed_img = np.average(initial_background_model, axis=0)
    if blur:
        seed_img = cv2.blur(seed_img, blur)
    counter = 10
    seed_edge = canny(seed_img, sigma=2)
    num_seed_edge = np.sum(np.ravel(seed_edge))

    for i, f in tqdm(enumerate(filenames[num_prev_frames:])):
        img = lm(cv2.imread(f))
        if blur:
            img = cv2.blur(img, blur)
        img_edge = canny(img, sigma=2)
        num_img_edge = np.sum(np.ravel(img_edge))
        diff_edge = np.where(img_edge != seed_edge, 1, 0)
        if np.sum(np.ravel(diff_edge)) / num_seed_edge < 0.5:
            segmented = np.zeros(img.shape)
        else:
            segmented = _prepare_frame_segment(seed_img=seed_img, img=img, blur=None)
        seed_img = (counter * seed_img + img) / (counter + 1)
        counter += 1
        cv2.imwrite(os.path.join(out_dir, 'segmented', filenames[i + num_prev_frames][-8:]), segmented)
        cv2.imwrite(os.path.join(out_dir, 'backgrounds', filenames[i + num_prev_frames][-8:]), seed_img)


def system_4(in_dir, out_dir, threshold, num_frames=150, num_prev_frames=10, blur=(7,7), as_numeric=True, stretched=True):
    """
    seed img ~ weighted cumulative moving average
    compute weights based on diff between current frame and previous frame
    thought:    add nonlinearities / weight-decay ~ additive temporal gradients?
                i.e.    first-order temporal gradient is just diff(img_t, img_t-1)
                        second-order additive temporal gradient is 
                        c1 * diff(t, t-1) + c2 * diff(t-1, t-2) + c3 * (diff(t, t-1) + diff(t-1, t-2))
                        and so forth.
                        then you can learn weights c_j from training images
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    filenames = _prepare_filenames(in_dir, num_frames=150)
    initial_background_model = np.array([lm(cv2.imread(f)) for f in filenames[0:num_prev_frames]])
    seed_img = np.average(initial_background_model, axis=0)
    if blur:
        seed_img = cv2.blur(seed_img, blur)
    # start off with initial background model as evenly weighted average
    total_weights = 10
    counter = 10
    average_diff = np.average(np.array([initial_background_model[i] - initial_background_model[i - 1] for i in range(1, num_prev_frames)]))
    previous_frame = initial_background_model[-1]
    # we are going to use previous-frame segmentation mask prediction later to constrain weights...
    # ... in cases where image doesn't change much but there are cars in the image; ideally we will segment them...
    # ... and so we can use segmentation mask to regularize the weight update
    previous_segmentation_prediction = _prepare_frame_segment(seed_img=seed_img, img = initial_background_model[-1], blur=None, stretched=False)

    for i, f in tqdm(enumerate(filenames[num_prev_frames:])):
        img = lm(cv2.imread(f))
        if blur:
            img = cv2.blur(img, blur)
        diff_img = np.abs(img - previous_frame)
        # cars might stop, but we don't want them to contribute much to background...
        # ... so constrain by segmentation mask prediction from previous frame
        constrained_diff_img = np.where(previous_segmentation_prediction > 0, 0, diff_img)
        weight = average_diff / (np.average(np.ravel(constrained_diff_img)) + 0.01)
        seed_img = (total_weights * seed_img + weight * img) / (total_weights + weight)
        segmented = _prepare_frame_segment(seed_img=seed_img, img=img, blur=None)
        total_weights += weight
        average_diff = (counter * average_diff + constrained_diff_img) / (counter + 1) # todo: make sure this works

        cv2.imwrite(os.path.join(out_dir, 'segmented', filenames[i + num_prev_frames][-8:]), segmented)
        cv2.imwrite(os.path.join(out_dir, 'backgrounds', filenames[i + num_prev_frames][-8:]), seed_img)
        previous_frame = img
        previous_segmentation_prediction = segmented # this is not necessary, but good practice ~ interpretability
    

def system_5(in_dir, out_dir, threshold, num_frames=150, num_prev_frames=10, blur=(3,3), as_numeric=True, stretched=True):
    """
    seed img ~ moving mode
    """
    filenames = _prepare_filenames(in_dir, num_frames=150)
    initial_background_model = np.array([cv2.imread(f) for f in filenames[0:num_prev_frames]])
    seed_img = mode(initial_background_model)
    previous_frames = deque(initial_background_model, maxlen=num_prev_frames)

    for i, f in tqdm(enumerate(filenames[num_prev_frames:])):
        img = lm(cv2.imread(f))



def system_6(in_dir, out_dir, threshold, num_frames, num_prev_frames, blend_coef, blur=(3,3), as_numeric=True, stretched=True):
    """
    seed img ~ blend moving average w/ moving mode
    hyperparemeter choices: `blend-coef` (optimize over these at, say, intervals of 0.05 from 0:1)
    """
    pass


def system_7(seed_img_file, in_dir, out_dir, threshold, num_frames, num_prev_frames, blend_coef, blur=(3,3), as_numeric=True, stretched=True):
    """
    seed img ~ blend( (blend moving average w/ moving mode) w/ (seed_img_file)
    hyperparemeter choices: `blend-coef` (optimize over these at, say, intervals of 0.05 from 0:1)
    """
    pass


"""
if seed_img_file is not None:
    print('system 1:')
    system_1(seed_img=seed_img, filenames=filenames, out_dir='../output_data/segmentation/system_1/', threshold=threshold, num_frames=num_frames, blur=blur)
print('system 2:')
system_2(filenames=filenames, out_dir='../output_data/segmentation/system_2/', threshold=threshold, blur=blur)
print('system 3:')
system_3(filenames=filenames, out_dir='../output_data/segmentation/system_3/', threshold=threshold, blur=blur)
print('system 3a:')
system_3a(filenames=filenames, out_dir='../output_data/segmentation/system_3a/', threshold=threshold, blur=blur)
print('system 4:')
system_4(filenames=filenames, out_dir='../output_data/segmentation/system_4/', threshold=threshold, blur=blur)
"""


def main():
    motion_segmentation(seed_img_file='../input_data/gray_tunnel_sequence/images/0023.png', 
                            in_dir='../input_data/cleaned_gray_tunnel_sequence/',
                            out_dir='/Users/adamcatto/SRC/dippy/output_data/segmentation/', 
                            threshold=40, num_frames=150, blur=None)


def make_montages():
    original_path = '/Users/adamcatto/SRC/dippy/input_data/gray_tunnel_sequence/images/'

    for i in tqdm(range(5)):
        if i != 4:
            seg_path = '/Users/adamcatto/SRC/dippy/output_data/segmentation/system_' + str(i+1) +'/segmented/'
        else:
            seg_path = '/Users/adamcatto/SRC/dippy/output_data/segmentation/system_3a/segmented/'
        seg_files = list(sorted(os.listdir(seg_path)))
        
        for j, f in tqdm(enumerate(seg_files)):
            seg_img = lm(cv2.imread(os.path.join(seg_path, f)))
            original_img = lm(cv2.imread(os.path.join(original_path, f)))
            arr = np.array([original_img, seg_img])
            mntg = montage(arr, grid_shape=(1, 2))
            if i != 4:
                out_file = os.path.join('/Users/adamcatto/SRC/dippy/output_data/segmentation/system_' + str(i+1), 'montages', f)
            else:
                out_file = os.path.join('/Users/adamcatto/SRC/dippy/output_data/segmentation/system_3a', 'montages', f)
            cv2.imwrite(out_file, mntg)


def rename_images_padded(in_dir):
    files = os.listdir(in_dir)
    filenames = list(sorted([os.path.join(in_dir, f) for f in files]))
    for i, f in tqdm(enumerate(filenames)):
        img = cv2.imread(f)
        n = int(files[i][0:-4])
        n = pad_int(num=n, desired_length=4)
        cv2.imwrite(os.path.join(in_dir, n + '.png'), img)
        os.remove(f)

#main()
#make_montages()
"""
img = lm(cv2.imread('../input_data/gray_tunnel_sequence/images/0023.png'))
img = stretch_histogram(np.log(img + 1))
cv2.imwrite('/Users/adamcatto/SRC/dippy/output_data/test_output/log_stretch_0023.png', img)
"""
# --

#rename_images_padded('/Users/adamcatto/SRC/dippy/input_data/gray_tunnel_sequence/images')

#test_abstract_painting()

#run_experiments(config)

# todo: make experiments function that loops over all images and applies all segmentation methods to each of them
