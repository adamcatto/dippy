import os
import numpy as np
import cv2
from tqdm import tqdm


def pad_int(num, desired_length):
    num_str = str(num)
    num_str_len = len(num_str)
    assert num_str_len <= desired_length
    diff = desired_length - num_str_len
    zero_string = ''
    
    for i in range(diff):
        zero_string += '0'
    
    return zero_string + num_str


def remove_duplicates(in_dir, test_dir):
    files = os.listdir(in_dir)
    filenames = list(sorted([os.path.join(in_dir, f) for f in files]))

    previous_frame = cv2.imread(filenames[0])
    cv2.imwrite(os.path.join(test_dir, pad_int(0, 4) + '.png'), previous_frame)
    filenames = filenames[1:]
    zero_arr = np.zeros(previous_frame.shape)
    counter = 1

    for i, f in tqdm(enumerate(filenames)):
        img = cv2.imread(f)
        #diff_img = img - previous_frame
        if (img != previous_frame).any():
            cv2.imwrite(os.path.join(test_dir, pad_int(counter, 4) + '.png') , img)
            counter += 1
        else:
            print(pad_int(i, 4))
        previous_frame = img
    
    print(len(os.listdir(in_dir)))
    print(len(os.listdir(test_dir)))
        #if diff_img == zero_arr:
            #os.remove(f)

    """    
    new_files = os.listdir(in_dir)
    new_filenames = [os.path.join(in_dir, f) for f in new_files]

    for i, f in tqdm(enumerate(new_filenames)):
        i = pad_int(i, 4)
        os.rename(f, os.path.join(test_dir, i + '.png'))
    """


def sliding_window(img, window):
    pass

#remove_duplicates(in_dir='/Users/adamcatto/SRC/dippy/input_data/gray_tunnel_sequence/images', test_dir='/Users/adamcatto/SRC/dippy/input_data/cleaned_gray_tunnel_sequence')
