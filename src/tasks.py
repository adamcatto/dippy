import os

import numpy as np
import cv2
from tqdm import tqdm
from skimage.morphology import closing, opening, dilation, erosion
from skimage.filters import sobel

from edge_detection import dct_motion_blur_detection, motion_segment_morphology_batch, motion_segment_morphology
from color_to_gray_operations import luminosity_method as lm
from decomposition import slice_bit_planes
from histogram_processing import stretch_histogram
from utils import pad_int


def edge_based_segmentation(img_dir, num_frames, num_prev_frames):
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
    for i, f in tqdm(enumerate(filenames[num_prev_frames: num_frames])):
        img1 = lm(cv2.imread(f))
        img_stacked = np.dstack((img1, img1, img1))
        diff = motion_segment_morphology(img0, img1)
        diff *= mask
        diff = np.dstack((diff, np.zeros(diff.shape), np.zeros(diff.shape)))
        diff = np.where(diff > 0, 255, img_stacked)
        
        seg_diffs.append(diff)
        img0 = img1
    
    for i, d in tqdm(enumerate(seg_diffs)):
        cv2.imwrite('../output_data/final_systems/edge_based/' + files[i + num_prev_frames], d)
        # cv2.imwrite('../output_data/edge_detection/tunnel_sequence/morphology_seg_combos/' + files[i + num_prev_frames], seg_diffs[i])



def dct_blur_segmentation(in_dir, block_size, blur_kernel=None):
    """
    todo:   + make num_planes_to_keep a parameter, optimize over it
            + make multi-scale: incorporate info from multiple block sizes; see if it leads to performance improvement
            + make diff_blurmaps threshold a parameter; optimize over it to see if it leads to performance improvement
    """
    if isinstance(block_size, int):
        block_size_tuple = (block_size, block_size)
    else:
        block_size_tuple = block_size

    if isinstance(blur_kernel, int):
        blur_kernel = (blur_kernel, blur_kernel)

    files = list(sorted(os.listdir(in_dir)))
    filenames = list(sorted([os.path.join(in_dir, f) for f in files]))

    img0 = lm(cv2.imread(filenames[0]))
    if blur_kernel:
        img0 = cv2.blur(img0, blur_kernel)
    mask_zeros = np.zeros((int(img0.shape[0] / 3) + 12, img0.shape[1]))
    mask_ones = np.ones((img0.shape[0] - 12 - int(img0.shape[0] / 3), img0.shape[1]))
    mask = np.vstack((mask_zeros, mask_ones)).astype(int)
    # img0 = slice_bit_planes(img=img0, num_planes_to_keep=5)
    blurmap0 = stretch_histogram(dct_motion_blur_detection(img=img0, block_size=block_size))
    for j, f in tqdm(enumerate(filenames[1:150])):
        img1 = lm(cv2.imread(f))
        if blur_kernel:
            img1 = cv2.blur(img1, blur_kernel)
        img_stacked = np.dstack((img1, img1, img1))
        # img1 = slice_bit_planes(img=img1, num_planes_to_keep=5)
        blurmap1 = stretch_histogram(dct_motion_blur_detection(img=img1, block_size=block_size))
        diff_blurmaps = blurmap1 - blurmap0
        diff_blurmaps = np.where(diff_blurmaps > 20, 255, 0)
        # diff_blurmaps = closing(diff_blurmaps, selem=np.ones((15,15)))
        # diff_blurmaps = dilation(diff_blurmaps, selem=np.ones((11,11)))
        diff_blurmaps = closing(diff_blurmaps, selem=np.ones((21,21)))
        # diff_blurmaps = closing(diff_blurmaps, selem=np.ones((11,11)))
        # diff_blurmaps = closing(diff_blurmaps, selem=np.ones((17,17)))
        diff_blurmaps = opening(diff_blurmaps, selem=np.ones((19,19)))
        # diff_blurmaps = dilation(diff_blurmaps, selem=np.ones((11,11)))
        diff_blurmaps = closing(diff_blurmaps, selem=np.ones((27,27)))

        diff_blurmaps *= mask
        
        diff_blurmaps = np.dstack((diff_blurmaps, np.zeros(diff_blurmaps.shape), np.zeros(diff_blurmaps.shape)))
        diff_blurmaps = np.where(diff_blurmaps > 0, 255, img_stacked)
        cv2.imwrite('../output_data/final_systems/dct_blur/' + files[j + 1], diff_blurmaps)
        img0, blurmap0 = img1, blurmap1


def combined_edge_dct_blur_segmentation(in_dir, block_size, num_frames, num_prev_frames, blur_kernel=None):
    if isinstance(block_size, int):
        block_size_tuple = (block_size, block_size)
    else:
        block_size_tuple = block_size

    if isinstance(blur_kernel, int):
        blur_kernel = (blur_kernel, blur_kernel)

    files = list(sorted(os.listdir(in_dir)))
    filenames = list(sorted([os.path.join(in_dir, f) for f in files]))

    img0 = lm(cv2.imread(filenames[0]))
    if blur_kernel:
        img0 = cv2.blur(img0, blur_kernel)

    mask_zeros = np.zeros((int(img0.shape[0] / 3) + 12, img0.shape[1]))
    mask_ones = np.ones((img0.shape[0] - 12 - int(img0.shape[0] / 3), img0.shape[1]))
    mask = np.vstack((mask_zeros, mask_ones)).astype(int)

    blurmap0 = stretch_histogram(dct_motion_blur_detection(img=img0, block_size=block_size))
    sobel_edges0 = sobel(img0)
    sobel_edges0 = np.where(sobel_edges0 > 10, 255, 0)

    for j, f in tqdm(enumerate(filenames[num_prev_frames: num_frames])):
        img1 = lm(cv2.imread(f))
        if blur_kernel:
            img1 = cv2.blur(img1, blur_kernel)
        img_stacked = np.dstack((img1, img1, img1))
        blurmap1 = stretch_histogram(dct_motion_blur_detection(img=img1, block_size=block_size))
        diff_blurmaps = blurmap1 - blurmap0
        diff_blurmaps = np.where(diff_blurmaps > 30, 255, 0)

        # edge map component
        sobel_edges1 = sobel(img1)
        sobel_edges1 = np.where(sobel_edges1 > 10, 255, 0)
        sobel_diff = sobel_edges1 - sobel_edges0

        # combine
        diff = np.clip(diff_blurmaps + sobel_diff, 0, 255)
        diff = closing(diff, selem=np.ones((7,7)))
        diff = closing(diff, selem=np.ones((11,11)))
        diff = closing(diff, selem=np.ones((13,13)))
        diff = opening(diff, selem=np.ones((19,19)))
        diff = dilation(diff, selem=np.ones((21,21)))
        #diff = closing(diff, selem=np.ones((55,1)))
        #diff = closing(diff, selem=np.ones((55,1)))
        #diff = closing(diff, selem=np.ones((55,1)))
        #diff = closing(diff, selem=np.ones((71,1)))
        #diff = closing(diff, selem=np.ones((1,13)))
        # diff = erosion(diff, selem=np.ones((1,13)))
        # diff = erosion(diff, selem=np.ones((1,17)))
        diff *= mask

        diff = np.dstack((diff, np.zeros(diff.shape), np.zeros(diff.shape)))
        diff = np.where(diff > 0, 255, img_stacked)

        cv2.imwrite('../output_data/final_systems/combined/' + files[j + 1], diff)
        img0, blurmap0, sobel_edge0 = img1, blurmap1, sobel_edges1


def main():
    in_dir = '../input_data/cleaned_gray_tunnel_sequence/'
    out_dir = '../output_data/final_systems/'
    dct_blur_segmentation(in_dir, block_size=16)
    # combined_edge_dct_blur_segmentation(in_dir, block_size=16, num_frames=150, num_prev_frames=1)
    # edge_based_segmentation(in_dir, 150, 1)


main()