import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift

def gaussian(x, sigma):
    return np.exp(-x**2 / (2*sigma**2))

def create_smooth_filter(size, r, R, filter_type='low'):
    rows, cols = size
    center_row, center_col = rows // 2, cols // 2
    filter = np.zeros((rows, cols))

    for x in range(rows):
        for y in range(cols):
            distance = np.sqrt((x - center_row)**2 + (y - center_col)**2)
            if filter_type == 'low':
                filter[x, y] = gaussian(distance - r, sigma=10)
            elif filter_type == 'high':
                filter[x, y] = 1 - gaussian(distance - r, sigma=10)
            elif filter_type == 'band':
                filter[x, y] = gaussian(distance - r, sigma=10) * (1 - gaussian(distance - R, sigma=10))

    return filter

def load_and_resize_image(file_path, target_size=(512, 512)):
    with Image.open(file_path) as img:
        img_resized = img.resize(target_size)
        return np.array(img_resized.convert('L'))

def apply_filter(image, filter):
    fimage = fft2(image)
    fimage_shifted = fftshift(fimage)
    filtered_image = fimage_shifted * filter
    return ifft2(fftshift(filtered_image))

def display_images(images, titles):
    plt.figure(figsize=(15, 5))
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i+1)
        plt.imshow(np.abs(image), cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

folder_path = './imgs-P3/' 

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        file_path = os.path.join(folder_path, filename)
        image = load_and_resize_image(file_path)

        size = image.shape
        r, R = 30, 60

        smooth_low_pass = create_smooth_filter(size, r, R, 'low')
        smooth_high_pass = create_smooth_filter(size, r, R, 'high')
        smooth_band_pass = create_smooth_filter(size, r, R, 'band')

        filtered_low = apply_filter(image, smooth_low_pass)
        filtered_high = apply_filter(image, smooth_high_pass)
        filtered_band = apply_filter(image, smooth_band_pass)

        display_images([image, filtered_low, filtered_high, filtered_band],
                       ['Original', 'Smooth Low-Pass', 'Smooth High-Pass', 'Smooth Band-Pass'])
