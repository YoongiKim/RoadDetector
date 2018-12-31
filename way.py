import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import math

def frange(start, stop, step):
    x = start
    while x < stop:
        yield x
        x += step

def center_range(center, range, step):
    i = 0

    while math.fabs(i * step) < range:
        yield center + i * step
        yield center - i * step
        i += 1


def find_way(map):
    h, w = map.shape

    # x = a*y^2
    _max = 0
    best_a = 0

    for a in center_range(0, 2.0, 0.01):
        _sum = 0

        for y in frange(0, 1.0, 0.01):
            x = a * (1-y)**2 + 1.0

            y_pos, x_pos = int(y*h), int(x*w/2)
            if y_pos < 0 or y_pos >= h:
                continue
            if x_pos < 0 or x_pos >= w:
                continue

            if map[y_pos, x_pos] == 255:
                _sum += 1  # count intersection points

        print('a={} -> sum={}'.format(a, _sum))
        if _sum > _max:
            best_a = a
            _max = _sum

    for y in frange(0, 1.0, 0.01):
        x = best_a * (1 - y) ** 2 + 1.0

        y_pos, x_pos = int(y * h), int(x * w / 2)
        if y_pos < 0 or y_pos >= h:
            continue
        if x_pos < 0 or x_pos >= w:
            continue

        map[y_pos, x_pos] = 255

    plt.imshow(map)
    plt.show()

    return best_a

if __name__ == '__main__':
    find_way(imread('map.png'))