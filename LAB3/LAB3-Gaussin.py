import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import os
from PIL import Image

def gaussianFilter(size, sigma):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

def convolution_spatial_domain(image, mask):
    return convolve2d(image, mask, mode='same')

def convolution_frequency_domain(image, mask):
    fimage = np.fft.fft2(image, s=image.shape)
    fmask = np.fft.fft2(mask, s=image.shape)
    fconvolution = fimage * fmask
    return np.fft.ifft2(fconvolution).real

def measure_time(func, image, mask):
    start_time = time.time()
    func(image, mask)
    return time.time() - start_time

def load_and_resize_images_from_folder(folder, target_size=(512, 512)):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder, filename)
            with Image.open(img_path) as img:
                img_resized = img.resize(target_size)
                images.append(np.array(img_resized.convert('L')))
    return images


folder_path = './imgs-P3/' 

# Загрузка и изменение размера изображений из папки Loading and resizing images from a folder
loaded_images = load_and_resize_images_from_folder(folder_path)

mask_sizes = [3, 5, 7, 10, 12, 14, 16, 32]  # Размеры масок для тестирования Mask sizes for testing
times_spatial = {mask_size: [] for mask_size in mask_sizes}
times_frequency = {mask_size: [] for mask_size in mask_sizes}

for image in loaded_images:
    for mask_size in mask_sizes:
        mask = gaussianFilter(mask_size, 1)  # Гауссовский фильтр Gaussian filter

        # Измерение времени Measuring time
        time_spatial = measure_time(convolution_spatial_domain, image, mask)
        time_frequency = measure_time(convolution_frequency_domain, image, mask)

        times_spatial[mask_size].append(time_spatial)
        times_frequency[mask_size].append(time_frequency)

# Вычисление среднего времени выполнения для каждого размера маски Calculating the average execution time for each mask size
avg_times_spatial = [np.mean(times_spatial[mask_size]) for mask_size in mask_sizes]
avg_times_frequency = [np.mean(times_frequency[mask_size]) for mask_size in mask_sizes]

# Построение графиков Graphing
plt.figure(figsize=(20, 12))
plt.plot(mask_sizes, avg_times_spatial, marker='o', label='Spatial Domain')
plt.plot(mask_sizes, avg_times_frequency, marker='x', label='Frequency Domain')
plt.xlabel('Mask Size')
plt.ylabel('Average Time (seconds)')
plt.title('Average Convolution Times for Different Mask Sizes')
plt.legend()
plt.show()