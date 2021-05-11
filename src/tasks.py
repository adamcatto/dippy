import os
import subprocess

import numpy as np
import cv2
from tqdm import tqdm
from skimage.morphology import closing, opening, dilation, erosion
from skimage.filters import sobel
from skimage.measure import label, regionprops
import pandas as pd

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
        # diff = np.where(diff > 0, 255, img_stacked)
        # diff = np.where(diff > 0, 255, 0)
        mask_diff = np.where(diff > 0, 255, 0)

        cv2.imwrite('../output_data/final_systems/edge_based/' + files[i + num_prev_frames], diff)
        cv2.imwrite('../output_data/final_systems/masks/edge_based/' + files[i + 1], mask_diff)
        # seg_diffs.append(diff)
        img0 = img1
    
    # for i, d in tqdm(enumerate(seg_diffs)):
        # cv2.imwrite('../output_data/final_systems/edge_based/' + files[i + num_prev_frames], d)
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
        diff_blurmaps = closing(diff_blurmaps, selem=np.ones((21,21)))
        # diff_blurmaps = closing(diff_blurmaps, selem=np.ones((15,15)))
        # diff_blurmaps = dilation(diff_blurmaps, selem=np.ones((11,11)))
        # diff_blurmaps = closing(diff_blurmaps, selem=np.ones((11,11)))
        # diff_blurmaps = closing(diff_blurmaps, selem=np.ones((17,17)))
        diff_blurmaps = opening(diff_blurmaps, selem=np.ones((19,19)))
        # diff_blurmaps = dilation(diff_blurmaps, selem=np.ones((11,11)))
        diff_blurmaps = closing(diff_blurmaps, selem=np.ones((27,27)))

        diff_blurmaps *= mask
        
        diff_blurmaps = np.dstack((diff_blurmaps, np.zeros(diff_blurmaps.shape), np.zeros(diff_blurmaps.shape)))
        # diff_blurmaps = np.where(diff_blurmaps > 0, 255, img_stacked)
        # mask_diff = np.where(diff_blurmaps > 0, 255, 0)
        
        cv2.imwrite('../output_data/final_systems/dct_blur/' + files[j + 1], diff_blurmaps)
        cv2.imwrite('../output_data/final_systems/masks/dct_blur/' + files[j + 1], mask_diff)
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
        diff = closing(diff, selem=np.ones((13,13)))
        diff = opening(diff, selem=np.ones((19,19)))
        diff = dilation(diff, selem=np.ones((21,21)))
        diff *= mask

        diff = np.dstack((diff, np.zeros(diff.shape), np.zeros(diff.shape)))
        diff = np.where(diff > 0, 255, img_stacked)
        mask_diff = np.where(diff > 0, 255, 0)

        cv2.imwrite('../output_data/final_systems/combined/' + files[j + 1], diff)
        cv2.imwrite('../output_data/final_systems/masks/combined/' + files[j + 1], mask_diff)
        img0, blurmap0, sobel_edge0 = img1, blurmap1, sobel_edges1


def fill_missing_annotation_files(annotation_dir, img_dir):
    annotation_files = list(sorted([f for f in os.listdir(annotation_dir) if f[0] != '.']))
    num_imgs = len(os.listdir(img_dir))
    for i in range(num_imgs):
        file_index = pad_int(i, 4)
        if file_index + '.txt' not in annotation_files:
            subprocess.call(['touch', os.path.join(annotation_dir, file_index + '.txt')])


def convert_segmentations_to_bboxes(seg_dir, out_dir):
    filenames = list(sorted([os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if f[0] != '.']))
    for _, f in tqdm(enumerate(filenames)):
        img = cv2.imread(f)
        seg_region = cv2.split(img)[0] # recall that segmentation mask is in blue channel
        labeled_seg_region = label(seg_region, background=0)
        labels = np.unique(labeled_seg_region)
        z = np.zeros(seg_region.shape)
        zero_matrix = np.zeros(seg_region.shape)
        if len(labels) > 0:
            labels = labels[1:]
        else:
            continue
        for lab in labels:
            region_img = np.where(labeled_seg_region == lab, 1, 0)
            region_props = regionprops(region_img)[0]
            min_row, min_col, max_row, max_col = region_props.bbox
            z = cv2.rectangle(z, (min_col, min_row), (max_col, max_row), 255, -1)
        z = np.dstack((z, zero_matrix, zero_matrix))
        cv2.imwrite(os.path.join(out_dir, f[-8:]), z)


def log_result_info(predicted_img, gt_annotation_file, img_number, log_file=None):
    num_true_pos = 0
    num_false_pos = 0
    num_false_neg = 0
    num_encapsulated = 0
    iou = 0
    if isinstance(img_number, int):
        img_number = pad_int(img_number, 4)
    if len(predicted_img.shape) == 3:
        predicted_img = cv2.split(predicted_img)[0] # get blue channel, i.e. bbox mask component of img

    z = np.zeros(predicted_img.shape)
    with open(gt_annotation_file) as annotation_file:
        lines = annotation_file.readlines()
        if len(lines) == 0:
            num_objects_pred = len(np.unique(label(predicted_img))) - 1
            num_false_pos = num_objects_pred
            iou = np.nan
            results = iou, num_true_pos, num_false_pos, num_false_neg, num_encapsulated
            if log_file:
                with open(log_file, 'a') as outfile:
                    outfile.write(img_number + ',' + str(results).replace(' ', '')[1: -1] + '\n')
            return results
        else:
            print(len(lines))
            predicted_components = label(predicted_img, background=0)
            false_pos_labels = set(np.unique(predicted_components))
            false_pos_labels.discard(0)
            num_objects_pred = len(false_pos_labels)
            for annotation_line in lines:
                annotation = np.array(annotation_line.split(' ')[1:]).astype(float)
                center_coords = z.shape[1] * annotation[0], z.shape[0] * annotation[1]
                width, height = z.shape[1] * annotation[2], z.shape[0] * annotation[3]
                bottom_left = int(center_coords[0] - width / 2), int(center_coords[1] - height / 2)
                top_right = int(center_coords[0] + width/2), int(center_coords[1] + height/2)
                rectangle_img = cv2.rectangle(z, bottom_left, top_right, 255, -1)
                intersection_img = (rectangle_img + predicted_img) // 2
                intersection_img = np.where(intersection_img == 255, 255, 0)

                label_overlap = predicted_components * rectangle_img / 255
                labels = list(sorted(np.unique(label_overlap)))
                if len(labels) > 1:
                    for pred_label in labels[1:]: # index-zero element will be 0; use next element
                        false_pos_labels.discard(pred_label)
                num_predicted_pixels = np.sum(np.ravel(predicted_img)) / 255
                num_gt_pixels = np.sum(np.ravel(rectangle_img)) / 255
                num_intersection_pixels = np.sum(np.ravel(intersection_img)) / 255
                num_union_pixels = num_predicted_pixels + num_gt_pixels - num_intersection_pixels

                iou = num_intersection_pixels / num_union_pixels
                if iou == 0:
                    num_false_neg += 1
                else:
                    num_true_pos += 1

        # summarize the results once we've gone through each ground-truth bbox
        # note that currently, `num_false_pos` can be negative – e.g. suppose one predicted box encapsulating three
        # ground-truth boxes. Then we have `num_objects_pred` = 1, `num_true_pos` = 3 ==> `num_false_pos` = -2.
        # to measure this sort of overshooting, we will track a variable called `num_encapsulated`, which is 0 if
        # `num_false_pos` >= 0 and `- num_false_pos` otherwise.
        num_encapsulated = num_objects_pred - num_true_pos
        if num_encapsulated < 0:
            num_encapsulated = - num_encapsulated
        else:
            num_encapsulated = 0

        num_false_pos = len(false_pos_labels)
        results = iou, num_true_pos, num_false_pos, num_false_neg, num_encapsulated
        if log_file:
                with open(log_file, 'a') as outfile:
                    outfile.write(img_number + ',' + str(results).replace(' ', '')[1: -1] + '\n')
        return results


def draw_bounding_boxes(img, yolo_annotation_file, thickness=-1, only_box=True):
    """
    `only_box` will filter out 
    """
    if only_box:
        # this is to correct for cases in which we have bright light ==> green pixel value = 255. 
        # don't want these pixels to be included as bbox regions, but our algorithm uses green pixel values of 255
        # after green filled box has been drawn. therefore ensure that no pixels = 255 by dividing original img by 2.
        rectangle_img = img // 2
    if thickness == 'fill':
        thickness = -1
    for annotation in yolo_annotation_file.readlines():
        try:
            annotation = np.array(annotation.split(' ')[1:]).astype(float)
            center_coords = img.shape[1] * annotation[0], img.shape[0] * annotation[1]
            width, height = img.shape[1] * annotation[2], img.shape[0] * annotation[3]
            bottom_left = int(center_coords[0] - width / 2), int(center_coords[1] - height / 2)
            top_right = int(center_coords[0] + width/2), int(center_coords[1] + height/2)
            rectangle_img = cv2.rectangle(rectangle_img, bottom_left, top_right, (0, 255, 0), thickness)
        except:
            continue
    
    if only_box:
        b, g, r = cv2.split(rectangle_img)
        g = np.where(g == 255, 255, 0)
        z = np.zeros(g.shape)
        rectangle_img = np.dstack((z, g, z))
    return rectangle_img
    

def write_bounding_boxes(in_dir, out_dir, annotation_dir):
    in_files = list(sorted([os.path.join(in_dir, f) for f in os.listdir(in_dir)]))
    annotation_files = list(sorted([os.path.join(annotation_dir, f) for f in os.listdir(annotation_dir) if f[0] != '.']))
    for img_file, annotation_file in tqdm(zip(in_files, annotation_files)):
        img = cv2.imread(img_file)
        with open(annotation_file, 'r') as annotations:
            bbox_img = draw_bounding_boxes(img, annotations)
            cv2.imwrite('../output_data/final_systems/ground_truth_bboxes/' + img_file[-8:], bbox_img)


def get_results(in_dir, bbox_annotation_dir, results_file):
    in_files = list(sorted([os.path.join(in_dir, f) for f in os.listdir(in_dir) if f[0] != '.']))
    # bbox_img_files = list(sorted([os.path.join(bbox_img_dir, f) for f in os.listdir(bbox_img_dir) if f[0] != '.']))
    bbox_annotation_files = list(sorted([os.path.join(bbox_annotation_dir, f) 
                                        for f in os.listdir(bbox_annotation_dir)[1:] if f[0] != '.']))
    logs = {}
    with open(results_file, 'w') as outfile:
        outfile.write('file_prefix,iou,num_true_pos,num_false_pos,num_false_neg,num_encapsulated\n')
    # for i, (predicted_file, ground_truth_annotation_file) in tqdm(enumerate(zip(in_files, bbox_annotation_files))):
    for i in tqdm(range(len(in_files))):
        str_i = pad_int(i, 4)
        predicted_file = os.path.join(in_dir, str_i + '.png')
        ground_truth_annotation_file = os.path.join(bbox_annotation_dir, str_i + '.txt')
        print(predicted_file, ground_truth_annotation_file)
        
        predicted_img = cv2.imread(predicted_file)
        z = np.zeros(predicted_img.shape)
        b_pred, g_pred, r_pred = cv2.split(predicted_img)
        print('\n\n')

        results = log_result_info(predicted_img, ground_truth_annotation_file, pad_int(i, 4), results_file)


def main():
    # in_dir = '../input_data/cleaned_gray_tunnel_sequence/'
    # out_dir = '../output_data/final_systems/'
    # annotation_dir = '../input_data/tunnel_labels_yolo/'
    in_dir = '../output_data/final_systems/edge_based/'
    predicted_bbox_dir = '../output_data/final_systems/predicted_bboxes/edge_based/'
    bbox_annotation_dir = '../input_data/tunnel_labels_yolo/'
    results_file = '../output_data/final_systems/logs/edge_based_results.txt'
    
    # edge_based_segmentation(in_dir, 856, 0)
    # write_bounding_boxes(in_dir, out_dir, annotation_dir)
    # fill_missing_annotation_files(annotation_dir, in_dir)
    # dct_blur_segmentation(in_dir, block_size=16)
    # combined_edge_dct_blur_segmentation(in_dir, block_size=16, num_frames=150, num_prev_frames=1)
    # edge_based_segmentation(in_dir, 150, 1)
    # convert_segmentations_to_bboxes(in_dir, predicted_bbox_dir)
    get_results(predicted_bbox_dir, bbox_annotation_dir, results_file)


main()


def test():
    img = cv2.imread('../output_data/final_systems/edge_based/0164.png')
    img = lm(img)
    labeled_img = label(img)
    print(len(np.unique(labeled_img)) - 1)
