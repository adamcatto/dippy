import os

import base
import color_to_gray_operations
from histogram_processing import stretch_histogram
import cv2
import numpy as np


def main(PATH='../input_data/noisy_segments/'):
    #img = cv2.imread('../input_data/tunnel_1.png')
    for f in os.listdir(PATH):
        to_read = os.path.join(PATH, f)
        img = cv2.imread(to_read)
        grayscale = color_to_gray_operations.luminosity_method(img)
        cv2.imwrite('../input_data/noisy_segments/' + f, grayscale)

    #b = base.make_bool(b, 200)
    #b = np.where(b == True, 255, 0)
    #cv2.imwrite('../output_data/binarized/tunnel_1.png', b)

main()