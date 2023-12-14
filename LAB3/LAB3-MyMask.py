import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import os
from PIL import Image

def my_mask(n):
    matrix = np.zeros((n + 1, n + 1))
    matrix[np.triu_indices_from(matrix, k=1)] = -1
    matrix[np.tril_indices_from(matrix, k=-1)] = 1
    return matrix

def my_filter(im, n):
    mask = my_mask(n)
    fimage = np.fft.fft2(im, s=im.shape)
    fmask = np.fft.fft2(mask, s=im.shape)
    fconvolution = fimage * fmask
    return np.fft.ifft2(fconvolution).real

def load_and_resize_images_from_folder(folder, target_size=(1024, 1024)):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder, filename)
            with Image.open(img_path) as img:
                img_resized = img.resize(target_size)
                images.append(np.array(img_resized.convert('L')))
    return images

def display_images(original, filtered, title):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')

    plt.suptitle(title)
    plt.show()

folder_path = './imgs-P3/'  

loaded_images = load_and_resize_images_from_folder(folder_path)

# Apply my_filter on all loaded images with a mask of size 4 and display them
if loaded_images:
    n = 4
    for i, original_image in enumerate(loaded_images):
        filtered_image = my_filter(original_image, n)
        display_images(original_image, filtered_image, f"Image {i+1}")
else:
    print("No images found in the specified directory.")
