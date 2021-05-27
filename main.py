from PIL import Image
import funcs
import timeit
import numpy as np
import numba

DIRECTORY = "imgs"  
    

if __name__ == "__main__":
    size = (1920, 1080)
    count = 50
    dropoff = 0.5

    np.random.seed(1234)
    points = np.random.random_sample((count, 2)) * np.array(size)
    colors = np.random.random_sample((count, 3)) * np.array([256, 256, 256])
    brightness = np.random.random_sample(count) * 255

 
    # arr = funcs.color_stars(size=size, count=count, dropoff=dropoff, points=points, colors=colors)
    # file_name = "colors0"

    arr = funcs.grey_stars(size=size, points=points, brightness=brightness, count=count, dropoff=dropoff)
    file_name = "grey0"
    img = Image.fromarray(arr)
    img.save(f"{DIRECTORY}\\{file_name}.png")


    arr = funcs.grey_extra_stars(size=size, points=points, brightness=brightness, count=count, dropoff=dropoff)
    file_name = "grey_extra_0"
    img = Image.fromarray(arr)
    img.save(f"{DIRECTORY}\\{file_name}.png")

    arr = funcs.grey_extra_2_stars(size=size, points=points, brightness=brightness, count=count, dropoff=dropoff)
    file_name = "grey_extra2_0"
    img = Image.fromarray(arr)
    img.save(f"{DIRECTORY}\\{file_name}.png")
