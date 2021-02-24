import numpy as np
from PIL import Image
import math
import random

np.random.seed(1234)

size = (1920, 1080)
count = 100
dropoff = 0.6 #greater values mean light has shorter reach

points = np.random.random_sample((count, 2)) * np.array(size)

xs = np.stack([np.arange(size[0], dtype=np.float64)] * count)
ys = np.stack([np.arange(size[1], dtype=np.float64)] * count)

xs = xs - points[:, :1]
ys = ys - points[:, 1:]

xs = np.power(xs, 2)
ys = np.power(ys, 2)

xs = np.stack([xs] * size[1])
ys = np.stack([ys] * size[0])

ys = ys.transpose((2, 1, 0))

arr = xs + ys

arr = np.power(arr, dropoff)

arr = 1 / arr

arr = np.sum(arr, axis=1)

arr *= 90 


img_arr = arr.astype(int)
img_arr = np.dstack([img_arr] * 3)
img = Image.fromarray(arr)
img = img.convert("RGB")
img.save("img2.png")