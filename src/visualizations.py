import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

import color_to_gray_operations


VIZ_PATH = '../output_data/visualizations/gray_intensities/'


def visualize_grayscale_intensities(img, out_path):
    img_x, img_y = np.mgrid[0: img.shape[0], 0: img.shape[1]]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(img_x, img_y, img, rstride=1, cstride=1, cmap=plt.cm.jet,
            linewidth=0)
    plt.savefig(out_path + 'surface.png')
    plt.close()


def visualize_color_intensities(color_img, out_path):
    b, g, r = cv2.split(color_img)

    blue_x, blue_y = np.mgrid[0: b.shape[0], 0: b.shape[1]]
    green_x, green_y = np.mgrid[0: g.shape[0], 0: g.shape[1]]
    red_x, red_y = np.mgrid[0: r.shape[0], 0: r.shape[1]]

    fig_blue = plt.figure()
    ax_blue = fig_blue.gca(projection='3d')
    ax_blue.plot_surface(blue_x, blue_y, b ,rstride=1, cstride=1, cmap=plt.cm.jet,
                linewidth=0)
    plt.savefig(out_path + 'blue_surface.png')
    plt.close()

    fig_green = plt.figure()
    ax_green = fig_green.gca(projection='3d')
    ax_green.plot_surface(green_x, green_y, g ,rstride=1, cstride=1, cmap=plt.cm.jet,
                linewidth=0)
    plt.savefig(out_path + 'green_surface.png')
    plt.close()

    fig_red = plt.figure()
    ax_red = fig_red.gca(projection='3d')
    ax_red.plot_surface(red_x, red_y, r ,rstride=1, cstride=1, cmap=plt.cm.jet,
                linewidth=0)
    plt.savefig(out_path + 'red_surface.png')
    plt.close()


def visualize_histogram(img):
    if len(img.shape) == 2:
        hist, bins = np.histogram(img, bins=[x for x in range(0, 257)])
        fig = plt.figure()
        fig.plot(bins, hist)
        fig.show()


def visualization_tests(path='../input_data/tunnel_1.png'):
    path = '/Users/adamcatto/src/L0-Smoothing/src/output_tunnels/tunnel_1.png'
    img = cv2.imread(path)
    visualize_color_intensities(img, out_path=VIZ_PATH)


def experiments(path='../input_data/noisy_segments/honeycomb_1.png'):
    img = cv2.imread(path)
    img = color_to_gray_operations.luminosity_method(img)
    visualize_grayscale_intensities(img, out_path=VIZ_PATH)


experiments()
#visualization_tests()