import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy.fft import fft2, ifft2


def load_and_resize_images_from_folder(folder, target_size=(1024, 1024)):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder, filename)
            with Image.open(img_path) as img:
                img_resized = img.resize(target_size)
                images.append(np.array(img_resized.convert('L')))
    return images

def demonstrate_ft_linearity(image1, image2):
    lambda_values = [0, 0.25, 0.5, 0.75, 1]
    plt.figure(figsize=(15, 10))
    for i, lambda_val in enumerate(lambda_values):
        combined_image = lambda_val * fft2(image1) + (1 - lambda_val) * fft2(image2)
        combined_image = ifft2(combined_image)
        plt.subplot(2, 3, i+1)
        plt.imshow(np.abs(combined_image), cmap='gray')
        plt.title(f'λ={lambda_val}')
        plt.axis('off')
    plt.suptitle('Inverse Transform of Combined FT for Different λ Values')
    plt.show()


folder_path = './imgs-P3/'  

loaded_images = load_and_resize_images_from_folder(folder_path)


    # Demonstrate FT linearity with the first two images (if available)
    if len(loaded_images) >= 2:
        demonstrate_ft_linearity(loaded_images[0], loaded_images[1])
else:
    print("No images found in the specified directory.")
