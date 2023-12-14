import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import convolve2d
import os
from PIL import Image

def my_mask(n):
    matrix = -np.ones((n+1, n+1))
    for i in range(n+1):
        matrix[i, :i+1] = 1
    np.fill_diagonal(matrix, 0)
    return matrix

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
times_spatial = {mask_size: [] for mask_size in mask_sizes}
times_frequency = {mask_size: [] for mask_size in mask_sizes}

for image in loaded_images:
    for mask_size in mask_sizes:
        mask = my_mask(mask_size)

        # Измерение времени
        time_spatial = measure_time(convolution_spatial_domain, image, mask)
        time_frequency = measure_time(convolution_frequency_domain, image, mask)

        times_spatial[mask_size].append(time_spatial)
        times_frequency[mask_size].append(time_frequency)

# Вычисление среднего времени выполнения для каждого размера маски
avg_times_spatial = [np.mean(times_spatial[mask_size]) for mask_size in mask_sizes]
avg_times_frequency = [np.mean(times_frequency[mask_size]) for mask_size in mask_sizes]

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(mask_sizes, avg_times_spatial, marker='o', label='Spatial Domain')
plt.plot(mask_sizes, avg_times_frequency, marker='x', label='Frequency Domain')
plt.xlabel('Mask Size')
plt.ylabel('Average Time (seconds)')
plt.title('Average Convolution Times for Different Mask Sizes')
plt.legend()
plt.show()
