from typing import List

import numpy as np
import cv2
from skimage import exposure

from base import make_bool
from color_to_gray_operations import luminosity_method
from histogram_processing import stretch_histogram


def convert_img_to_bit_planes(img):
    bit_planes = []

    for j in range(8):
        bit_plane = np.full((img.shape[0], img.shape[1]), 2 ** (7 - j), np.uint8)
        decomposition = cv2.bitwise_and(bit_plane.astype(int), img.astype(int))
        bit_planes.append(decomposition)

    return bit_planes


def convert_img_to_bit_planes_binary(img):
    bit_planes = convert_img_to_bit_planes(img)
    return [make_bool(img, threshold=0) for img in bit_planes]


def slice_bit_planes(img, num_planes_to_keep):
    if len(img.shape) == 3 and img.shape[2] == 3:
        bgr = cv2.split(img)
        sliced = [slice_bit_planes(x, num_planes_to_keep) for x in bgr]
        return np.dstack(sliced)

    bit_planes = convert_img_to_bit_planes(img)
    bit_planes = bit_planes[0: num_planes_to_keep]
    print(len(bit_planes))
    #print(bit_planes)
    #for j, bp in enumerate(bit_planes):
        #bit_planes[j] = bp * (2 ** (7 - j))
    img = sum(bit_planes)
    return img


class MovingObjectDetector:
    """
    This is a retrospective moving object detector. the frames are fixed, so we iterate over frames start to finish.
    """
    def __init__(self, frames: List[np.array]):
        if len(frames[0].shape) == 3:
            self.frames = [luminosity_method(f) for f in frames]
        else:
            self.frames = frames

        self.bit_planes_per_frame = np.array([convert_img_to_bit_planes(f) for f in self.frames])

    def model_background(self, frame_number, max_frames):
        start_frame = max(0, frame_number-max_frames)
        previous_bit_plane_list = self.bit_planes_per_frame[start_frame: frame_number]
        
        grouped_planes = list(zip(previous_bit_plane_list))
        
        average_previous_bit_plane_values = [sum(bp) / len(previous_bit_plane_list) for bp in grouped_planes]
        new_bps = [np.round(bp) for bp in average_previous_bit_plane_values]
        return new_bps

    def forward(self):
        pass



#/Users/adamcatto/SRC/dippy/input_data/motion/first_frame.png
f = cv2.imread('/Users/adamcatto/SRC/dippy/input_data/motion/first_frame.png')

f1 = (luminosity_method(f)).astype(int)
#f1 = exposure.adjust_gamma(f1, 1)
cv2.imwrite('/Users/adamcatto/SRC/dippy/input_data/motion/gray_first_frame.png', f1)
#f1 = luminosity_method(cv2.imread('/Users/adamcatto/SRC/dippy/input_data/motion/first_frame.png'))

f2 = cv2.imread('/Users/adamcatto/SRC/dippy/input_data/motion/second_frame.png')
f2 = (luminosity_method(f2)).astype(int)
#f2 = exposure.adjust_gamma(f1, 1)
cv2.imwrite('/Users/adamcatto/SRC/dippy/input_data/motion/gray_second_frame.png', f2)

#cv2.imwrite('/Users/adamcatto/SRC/dippy/output_data/decomposition/motion/subtract.png', f2-f1)
#print(f1)
detector = MovingObjectDetector([f1, f2])

bg = detector.model_background(2, max_frames=2)
zipped_bit_planes = list(zip(*bg))
compared = [cv2.bitwise_xor(x[0], x[1]) for x in zipped_bit_planes]
compared = compared[1]
cv2.imwrite('/Users/adamcatto/SRC/dippy/output_data/decomposition/motion/bitwise_xor.png', stretch_histogram(compared))


def compute_nms(ns, compared):
    """
    ns: neighborhood_size
    """
    nms = np.zeros(compared.shape)
    print(nms.shape)

    for _ in range(5):
        for i in range(ns, compared.shape[0] - ns):
            for j in range(ns, compared.shape[1] - ns):
                if compared[i, j] > 0:
                    neighborhood = [1 if compared[i + x, j + y] > 0 else 0 for x in [-ns, 0, ns] for y in [-ns, 0, ns] ]
                    print(neighborhood)
                    if sum(neighborhood) >= round(len(neighborhood) / 2) + 1:
                        nms[i, j] = 1
                    else:
                        nms[i, j] = 0
        compared = nms

    return nms

n = compute_nms(ns=1, compared=compared)


cv2.imwrite('/Users/adamcatto/SRC/dippy/output_data/decomposition/motion/nms.png', stretch_histogram(n))


"""
f = [make_bool(b, 0).astype(int) * 255 for b in bg]

for i, x in enumerate(f):
    for j, y in enumerate(x):
        
        cv2.imwrite('/Users/adamcatto/SRC/dippy/output_data/decomposition/motion/frame_' + str(i) + '_plane_' + str(7 - j) + '.png', y)
"""

"""
img = (cv2.imread('../output_data/smooth_segment_blend/overlay_smooth_rolling.png'))
bit_planes = slice_bit_planes(img, 3)
cv2.imwrite('../output_data/decomposition/bit_plane/tunnel_1.png', bit_planes)
print(bit_planes)
print(np.max(bit_planes))
    
"""