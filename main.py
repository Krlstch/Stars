import numpy as np
from PIL import Image
import math
import random

np.random.seed(1234)

size = (1080, 1920)
count = 50
dropoff = 1/2.5 #greater values mean light has shorter reach

#points = np.random.random_sample((count, 2)) * np.array(size)
input_file = Image.open("input.png")
input_pix = input_file.load()

print(input_file.size)
black_pix = set()
for i in range(input_file.size[0]):
    for j in range(input_file.size[1]):
        if input_pix[i, j][0] == 0: #is black
            black_pix.add((4*j+0.1, 4*i+0.1))

chosen_pix = []
try:
    for _ in range(count):
        chosen = random.choice(list(black_pix))
        chosen_pix.append(chosen)
        for pix in [element for element in list(black_pix)]:
            if math.dist(chosen, pix) < 40:
                black_pix.remove(pix)
except IndexError:
    pass

points = np.array(chosen_pix)
print(len(chosen_pix))

arr = np.zeros(shape=size, dtype=np.float64)
for i in range(arr.shape[0]):
    print(i)
    for j in range(arr.shape[1]):
        dist = points - np.array([i, j])
        dist = dist ** 2
        dist = np.sum(dist, axis=1)
        dist = np.power(dist, dropoff)
        dist = 1 / dist
        dist = np.sum(dist)
        arr[i, j] += dist

arr = arr * (255 / np.amax(arr))

img_arr = arr.astype(int)
img_arr = np.dstack([img_arr] * 3)
img = Image.fromarray(arr)
img = img.convert("RGB")
img.save("img2.png")

