import numpy as np
import cv2


def lightness_method(img):
    """
    Averages most- and least-prominent colors in the image.
    """
    max_channel_value = np.max(img, axis=2)
    min_channel_value = np.min(img, axis=2)
    gray_value = (max_channel_value + min_channel_value) / 2
    return gray_value


def average_method(img):
    """
    Average the values of the blue, green, and red channels
    """
    b, g, r = cv2.split(img)
    gray_value = (b + g + r) / 3
    return gray_value


def luminosity_method(img, blue_weight=0.07, green_weight=0.72, red_weight=0.21):
    """
    Weighted average of the three channel values; simulates human visual system (HVS) luminosity perception – humans
    are most sensitive to green light, then red, then blue.
    """
    b, g, r = cv2.split(img)

    weighted_blue = b * blue_weight
    weighted_green = g * green_weight
    weighted_red = r * red_weight
    
    gray_value = weighted_blue + weighted_green + weighted_red
    return gray_value


def photopic_luminance_method(img):
    gray_value = luminosity_method(img, blue_weight=0.065, green_weight=0.67, red_weight=0.256)
    return gray_value


def scotopic_luminance_method(img):
    gray_value = luminosity_method(img, blue_weight=4.33, green_weight=1.039, red_weight=-0.702)
    return gray_value


# todo: write decolorization controller method that selects optimal method based on some image quality metric


def color_to_gray_tests(in_path, out_path):
    #in_path='../input_data/tunnel_1.png'
    img = cv2.imread(in_path)
    print(img.shape)
    lightness_img = lightness_method(img)
    average_img = average_method(img)
    luminosity_img = luminosity_method(img)
    photopic_img = photopic_luminance_method(img)
    scotopic_img = scotopic_luminance_method(img)
    cv2.imwrite(out_path + 'lightness.jpg', lightness_img)
    cv2.imwrite(out_path + 'average.jpg', average_img)
    cv2.imwrite(out_path + 'luminosity.jpg', luminosity_img)
    cv2.imwrite(out_path + 'photopic.jpg', photopic_img)
    cv2.imwrite(out_path + 'scotopic.jpg', scotopic_img)
    """
    print('lightness method:\n\n' + str(lightness_method(img)))
    print('\n\naverage method:\n\n' + str(average_method(img)))
    print('\n\nluminosity method: ' + str(luminosity_method(img)))
    print('\n\nphotopic luminance method: ' + str(photopic_luminance_method(img)))
    print('\n\nscotopic luminance method: ' + str(scotopic_luminance_method(img)))
    """
