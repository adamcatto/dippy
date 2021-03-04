import numpy as np
import cv2
import color_to_gray_operations


def image_negative(img: np.array, max_intensity=255, convert_color_to_grayscale=False, grayscale_conversion=color_to_gray_operations.luminosity_method):
    assert not (convert_color_to_grayscale and len(img.shape) != 3)
    assert len(img.shape) in (2, 3)

    if convert_color_to_grayscale:
        img = grayscale_conversion(img)

    largest_value = np.max(img)
    img = (img - max_intensity) * -1
    return img
    

def log_transform(img:np.array):
    return np.log(img + 1)


def gamma_correction(img, gamma):
    pass


def normalize(img):
    max_value = np.max(img)
    min_value = np.min(img)
    img = 255 * (img - min_value) / (max_value - min_value)
    return img


def operations_tests(path='../nyc_tunnel/'):
    img_1 = cv2.imread(path + '3_nyc_tunnel.png')
    img_2 = cv2.imread(path + '4_nyc_tunnel.png')

    img_1 = color_to_gray_operations.luminosity_method(img_1)
    img_2 = color_to_gray_operations.luminosity_method(img_2)

    difference = normalize(np.clip(img_2 - img_1[0:239, 0:353], 0, 255)) ** 1.2
    
    cv2.imwrite('../output_data/test_output/tunnel_difference.png', difference)

#operations_tests()