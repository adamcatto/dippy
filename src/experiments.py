import base
import color_to_gray_operations
from histogram_processing import stretch_histogram
import cv2
import numpy as np


def main():
    img = cv2.imread('../input_data/tunnel_1.png')
    b = color_to_gray_operations.luminosity_method(img)
    b = base.make_bool(b, 200)
    b = np.where(b == True, 255, 0)
    cv2.imwrite('../output_data/binarized/tunnel_1.png', b)

main()