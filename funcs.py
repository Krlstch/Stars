from numba.core.decorators import jit
import numpy as np
import numba

np.set_printoptions(threshold=np.inf)

size = (200, 100)
count = 10
dropoff = 0.5 #greater values mean light has shorter reach    


@numba.jit(nopython=True)
def create_array_pure_numpy(size, count=10, dropoff=0.5, points=np.array([0, 0]), colors=np.array([0, 0, 0])):
    xs = np.zeros((count, size[0]), dtype=np.float64) + np.arange(size[0], dtype=np.float64)
    ys = np.zeros((count, size[1]), dtype=np.float64) + np.arange(size[1], dtype=np.float64)

    xs = xs - points[:, :1]
    ys = ys - points[:, 1:]

    xs = np.power(xs, 2)
    ys = np.power(ys, 2)

    xs = np.zeros((size[1], count, size[0]), dtype=np.float64) + xs
    ys = np.zeros((size[0], count, size[1]), dtype=np.float64) + ys

    ys = ys.transpose((2, 1, 0))

    arr = xs + ys

    arr = np.power(arr, dropoff)

    arr = 1 / arr

    arr = np.zeros((3, size[1], count,size[0])) + arr

    arr = np.transpose(arr, axes=(1, 3, 2, 0))

    arr = arr * colors

    arr = np.sum(arr, axis=2)
    arr = arr.astype(np.uint8)

    return arr

@numba.jit(nopython=True)
def color_stars(size, points, colors, count=10, dropoff=0.5):
    arr = np.zeros(shape=(size[1], size[0], 3), dtype=np.float64)
    for i in range(size[1]):
        for j in range(size[0]):
            for k in range(count):
                dist = (1/(((j-points[k, 0])**2 + (i-points[k, 1])**2)**dropoff))
                arr[i, j, 0] +=  dist * colors[k, 1]
                arr[i, j, 1] += dist * colors[k, 1]
                arr[i, j, 2] += dist * colors[k, 2]
    
    arr = np.where(arr > 255, 255, arr)
    arr = arr.astype(np.uint8)
    return arr

@numba.jit(nopython=True)
def grey_stars(size, points, brightness, count=10, dropoff=0.5):
    arr = np.zeros(shape=(size[1], size[0]), dtype=np.float64)
    for i in range(size[1]):
        for j in range(size[0]):
            for k in range(count):
                dist = (1/(((j-points[k, 0])**2 + (i-points[k, 1])**2)**dropoff))
                arr[i, j] += dist * brightness[k]
    
    arr = np.where(arr > 255, 255, arr)
    arr = arr.astype(np.uint8)
    return arr

@numba.jit(nopython=True)
def grey_extra_stars(size, points, brightness, count=10, dropoff=0.5):
    arr = np.zeros(shape=(size[1], size[0], 3), dtype=np.float64)
    for i in range(size[1]):
        for j in range(size[0]):
            for k in range(count):
                dist = (1/(((j-points[k, 0])**2 + (i-points[k, 1])**2)**dropoff))
                arr[i, j, 2] += 3 * dist * brightness[k]

    arr = np.where(arr > 765, 765, arr)
    arr = arr.astype(np.uint16)
    arr[:, :, 1] = arr[:, :, 2] // 3
    arr[:, :, 2] -= arr[:, :, 1]
    arr[:, :, 0] = arr[:, :, 2] // 2
    arr[:, :, 2] -= arr[:, :, 0]

    arr = arr.astype(np.uint8)
    return arr

@numba.jit(nopython=True)
def grey_extra_2_stars(size, points, brightness, count=10, dropoff=0.5):
    arr = np.zeros(shape=(size[1], size[0], 3), dtype=np.float64)
    for i in range(size[1]):
        for j in range(size[0]):
            for k in range(count):
                dist = (1/(((j-points[k, 0])**2 + (i-points[k, 1])**2)**dropoff))
                arr[i, j, 1] += 7 * dist * brightness[k]

    conv_arr = create_conv_arr()
    arr = np.where(arr > 1785, 1785, arr)
    arr = arr.astype(np.uint16)
    for i in range(size[1]):
        for j in range(size[0]):
            arr[i, j] = conv_arr[arr[i, j, 1]]

    arr = arr.astype(np.uint8)
    return arr

@numba.jit(nopython=True)
def create_conv_arr():
    r_to_B = (-1, 0, -1, 0, 1, 0, 1)
    r_to_R = (-1, -1, 0, 0, 0, 1, 1)

    convert_arr = np.empty(shape=(1786, 3), dtype=np.uint16)
    for i in range(1786):
        convert_arr[i, 1] = (i + 3) // 7
        convert_arr[i, 0] = convert_arr[i, 1] + r_to_R[(i + 3) % 7]
        convert_arr[i, 2] = convert_arr[i, 1] + r_to_B[(i + 3) % 7]

    return np.asarray(convert_arr, dtype=np.uint8)