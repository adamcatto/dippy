from __future__ import annotations
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error as mse


######## helper functions ########

def make_bool(img_array, threshold=0):
        bool_array = img_array
        if img_array.dtype.name != 'bool':
            f = lambda x: False if x <= threshold else True
            bool_array = np.vectorize(f)(img_array).astype(bool)
        return bool_array


def compute_psnr(img: np.array, enhanced_img: np.array, dynamic_range=255):
    psnr = 10 * np.log10((dynamic_range - 1) ** 2 / mse(img, enhanced_img))
    return psnr


def crop_match(img_1, img_2):
    shape_1 = img_1.shape
    shape_2 = img_2.shape
    dims = tuple(min(a, b) for a, b in zip(shape_1, shape_2))
    print(dims)
    new_img_1 = img_1[
        (shape_1[0] - dims[0]) // 2 : ((shape_1[0] - dims[0]) // 2) + dims[0],
        (shape_1[1] - dims[1]) // 2 : ((shape_1[1] - dims[1]) // 2) + dims[1]
    ]
    new_img_2 = img_2[
        (shape_2[0] - dims[0]) // 2 : ((shape_2[0] - dims[0]) // 2) + dims[0],
        (shape_2[1] - dims[1]) // 2 : ((shape_2[1] - dims[1]) // 2) + dims[1]
    ]
    return new_img_1, new_img_2


def normalize_by_channel(img):
    b, g, r = cv2.split(img)
    min_b = np.min(b)
    min_g = np.min(g)
    min_r = np.min(r)

    max_b = np.max(b)
    max_g = np.max(g)
    max_r = np.max(r)

    expanded_b_min = np.ones((b.shape)) * min_b
    expanded_g_min = np.ones((g.shape)) * min_g
    expanded_r_min = np.ones((r.shape)) * min_r

    expanded_b_max = np.ones((b.shape)) * max_b
    expanded_g_max = np.ones((g.shape)) * max_g
    expanded_r_max = np.ones((r.shape)) * max_r

    normalized_b = 255 * ((b - expanded_b_min) / (expanded_b_max - expanded_b_min))
    normalized_g = 255 * ((g - expanded_g_min) / (expanded_g_max - expanded_g_min))
    normalized_r = 255 * ((r - expanded_r_min) / (expanded_r_max - expanded_r_min))

    return cv2.merge((normalized_b, normalized_g, normalized_r))



#######################################################################
#######################################################################
#######################################################################


class Image(object):
    def __init__(self, img_array: np.array):
        assert len(img_array.shape) in [2, 3]
        assert np.all(img_array == np.clip(img_array, 0, 255))
        self.array = img_array
        if len(img_array.shape) == 2:
            self.img_type = 'grayscale'
            self.num_channels = 1
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
            self.img_type = 'grayscale'
            self.num_channels = 1
        else:
            self.img_type = 'color'
            self.num_channels = img_array.shape[2]
        
    def __add__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __mult__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __floordiv__(self, other):
        raise NotImplementedError

    def __and__(self, other: Image, threshold_self=0, threshold_other=0):
        _bool_self = make_bool(self.array, threshold_self)
        _bool_other = make_bool(other.array, threshold_other)
        return np.logical_and(bool_self, bool_other)

    def __or__(self, other: Image, threshold_self=0, threshold_other=0):
        _bool_self = make_bool(self.array, threshold_self)
        _bool_other = make_bool(other.array, threshold_other)
        return np.logical_or(bool_self, bool_other)

    def __not__(self, threshold_self=0):
        _bool_self = make_bool(self.array, threshold_self)
        return np.logical_not(bool_self)

    def __xor__(self, other: Image, threshold_self=0, threshold_other=0):
        _bool_self = make_bool(self.array, threshold_self)
        _bool_other = make_bool(other.array, threshold_other)
        return np.logical_xor(bool_self, bool_other)

    def alpha_blend(self, other, alpha):
        assert all((type(alpha) == float, alpha >= 0, alpha <= 1))
        _first, _second = crop_match(self.array, other.array)
        _first = alpha * _first
        _second = (1 - alpha) * _second
        return _first + _second

    def rescale(self, scale_percent=200):
        image = self.array
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        new_size = (width, height)
        output = cv2.resize(image, new_size)
        return output

    

    def auto_alpha_blend(self, alpha=0.4, kernel_size=(25, 25)):
        img = self.array
        #smoothed_img = img

        b, g, r = cv2.split(img)
        min_b = np.min(b)
        min_g = np.min(g)
        min_r = np.min(r)
        min_img = np.fmin(b, np.fmin(g, r))
        channel_axis = [i for i in [0, 1, 2] if img.shape[i] == 3][0]

        # smooth image using Gaussian Blur five times
        for i in range(5):
            smoothed_img = cv2.GaussianBlur(min_img, (25, 25), cv2.BORDER_DEFAULT)
            min_img = smoothed_img

        smoothed_img = smoothed_img.astype(float) * 1
        smoothed_img = smoothed_img.astype(int)
        b_subtract = b - smoothed_img
        g_subtract = g - smoothed_img
        r_subtract = r - smoothed_img

        to_subtract = np.dstack([smoothed_img] * 3) * 3 / 4

        result_img = img - to_subtract
        normalized_result_img = normalize_by_channel(result_img)

        #new_img = img - smoothed_img
        return self.alpha_blend(ClassicalImage(normalized_result_img), alpha)

        
class ClassicalImage(Image):
    def __init__(self, img_array):
        Image.__init__(self, img_array)

    def __add__(self, other: ClassicalImage):
        return np.clip(self.array + other.array, 0, 255)

    def __sub__(self, other: ClassicalImage):
        return np.clip(self.array - other.array, 0, 255)

    def __mult__(self, other: float):
        return np.clip(self.array * other, 0, 255)

    def __truediv__(self, other: float):
        return np.clip(self.array / other, 0, 255)

    def __floordiv__(self, other: float):
        return np.clip(np.floor_divide(self.array, other), 0, 255)


#img_1 = ClassicalImage(cv2.cvtColor(cv2.imread('input_data/tunnel_1.png'), cv2.COLOR_BGR2GRAY))
#img_2 = ClassicalImage(cv2.cvtColor(cv2.imread('input_data/tunnel_2.jpg'), cv2.COLOR_BGR2GRAY))

img_1 = ClassicalImage(cv2.imread('input_data/tunnel_1.png'))

#new_img = img_1.alpha_blend(img_2, 0.5)
new_img = img_1.auto_alpha_blend(alpha=0.4)
cv2.imwrite('output_data/result.jpg', new_img)
