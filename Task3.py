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

def load_and_resize_images_from_folder(folder, target_size=(256, 256)):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder, filename)
            with Image.open(img_path) as img:
                img_resized = img.resize(target_size)
                images.append(np.array(img_resized.convert('L')))
    return images

# Путь к папке с изображениями
folder_path = './imgs-P3/'  # Замените на ваш путь

# Загрузка и изменение размера изображений из папки
loaded_images = load_and_resize_images_from_folder(folder_path)

mask_sizes = [3, 5, 7]  # Размеры масок для тестирования
times_spatial = []
times_frequency = []

for image in loaded_images:
    for mask_size in mask_sizes:
        mask = gaussianFilter(mask_size, 1)  # Гауссовский фильтр

        # Измерение времени
        time_spatial = measure_time(convolution_spatial_domain, image, mask)
        time_frequency = measure_time(convolution_frequency_domain, image, mask)

        times_spatial.append(time_spatial)
        times_frequency.append(time_frequency)

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(mask_sizes, times_spatial, marker='o', label='Spatial Domain')
plt.plot(mask_sizes, times_frequency, marker='x', label='Frequency Domain')
plt.xlabel('Mask Size')
plt.ylabel('Time (seconds)')
plt.title('Convolution Times for Different Mask Sizes')
plt.legend()
plt.show()
