import numpy as np
from PIL import Image
import math
import random

np.random.seed(1234)
np.set_printoptions(threshold=np.inf)

size = (1920, 1080)
count = 50
dropoff = 0.6 #greater values mean light has shorter reach

points = np.random.random_sample((count, 2)) * np.array(size)
colors = np.random.random_sample((count, 3)) * np.array([255, 255, 255])
#colors = np.ones((count, 3), dtype=np.float64) * np.array([255, 0, 0])

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

arr = np.stack([arr] * 3)

arr = np.transpose(arr, axes=(1, 3, 2, 0))

arr = arr * colors

arr = np.sum(arr, axis=2)

arr = arr.astype(np.uint8)
img = Image.fromarray(arr)
img = img.convert("RGB")
img.save("img2.png")