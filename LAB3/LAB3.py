#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from scipy.ndimage import filters
import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob
import os
import sys

#sys.path.append("../../p1/code") # set the path for visualPercepUtils.py
import visualPercepUtilsModx3 as vpu


def gaussianFilter(size, sigma):
    """Создает гауссовский фильтр заданного размера и сигма."""
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)


def gaussianFilterFrequency(im, size, sigma):
    filterMask = gaussianFilter(size, sigma)
    
    # Расширение filterMask до размера изображения
    paddedFilter = np.zeros_like(im, dtype=np.float32)
    a, b = filterMask.shape
    y, x = im.shape
    startx = x//2 - a//2
    starty = y//2 - b//2  
    paddedFilter[starty:starty+b, startx:startx+a] = filterMask

    # Преобразование Фурье для фильтра и изображения
    ft_filter = fft.fft2(fft.ifftshift(paddedFilter))  # fftshift для центрирования фильтра
    ft_image = fft.fft2(im)

    # Применение фильтра в частотной области
    ft_filtered = ft_image * ft_filter
    filtered_im = np.absolute(fft.ifft2(ft_filtered))

    return filtered_im

def bandPassFilter(shape, r, R):
    n, m = shape
    m2, n2 = np.floor(m / 2.0), np.floor(n / 2.0)
    [vx, vy] = np.meshgrid(np.arange(-m2, m2 + 1), np.arange(-n2, n2 + 1))
    distToCenter = np.sqrt(vx ** 2.0 + vy ** 2.0)
    filter = np.logical_and(distToCenter < R, distToCenter > r)
    return filter.astype('float')

def bandPassFilterApply(im, r, R):
    filterFreq = bandPassFilter(im.shape, r, R)
    filterFreq = fft.ifftshift(filterFreq)
    ft_image = fft.fft2(im)
    ft_filtered = ft_image * filterFreq
    filtered_im = np.absolute(fft.ifft2(ft_filtered))
    return filtered_im

im = np.array(Image.open('./imgs-P3/einstein.jpg').convert('L')) 
size, sigma = 5, 1
gaussian_filtered_image = gaussianFilterFrequency(im, size, sigma)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(im, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(gaussian_filtered_image, cmap='gray')
plt.title('Gaussian Filter Applied')
plt.show()

# Применение band-pass фильтра
r, R = 5, 30
bandpass_filtered_image = bandPassFilterApply(gaussian_filtered_image, r, R)

# Отображение результата band-pass фильтра
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(gaussian_filtered_image, cmap='gray')
plt.title('After Gaussian Filter')
plt.subplot(1, 2, 2)
plt.imshow(bandpass_filtered_image, cmap='gray')
plt.title('Band Pass Filter Applied')
plt.show()

def testGaussianFilter(im, params=None):
    """Тестирование применения гауссовского фильтра."""
    if params is None:
        params = {'size': 5, 'sigma': 1}

    size = params['size']
    sigma = params['sigma']
    filtered_image = gaussianFilterFrequency(im, size, sigma)

    # Визуализация оригинального и отфильтрованного изображения
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray')
    plt.title('Test Plot filter Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Test Plot Filtered Gaussian ')
    plt.show()

    return [filtered_image]


# ----------------------
# Fourier Transform
# ----------------------

def FT(im):
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
    return fft.fftshift(fft.fft2(im))  # perform also the shift to have lower frequencies at the center


def IFT(ft):
    return fft.ifft2(fft.ifftshift(ft))  # assumes ft is shifted and therefore reverses the shift before IFT


def testFT(im, params=None):
    ft = FT(im)
    
    #print(ft.shape)
    
    phase = np.angle(ft)
    magnitude = np.log(np.absolute(ft) + 1)
        
    lowest_magnitude = np.min(magnitude)
    highest_magnitude = np.max(magnitude)
    print("Min Magnitude :", lowest_magnitude)
    print("Max Magnitude:", highest_magnitude)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.boxplot(magnitude.flatten())
    plt.title('Boxplot of Magnitude Values')

    plt.subplot(1, 2, 2)
    plt.hist(phase.flatten(), bins=50)
    plt.title('Histogram of Phase Values')
    plt.show()

    bMagnitude = True
    if bMagnitude:
        im2 = np.absolute(IFT(ft))  # IFT consists of complex number. When applied to real-valued data the imaginary part should be zero, but not exactly for numerical precision issues
    else:
        im2 = np.real(IFT(ft)) # with just the module we can't appreciate the effect of a shift in the signal (e.g. if we use fftshift but not ifftshift, or viceversa)
        # Important: one case where np.real() is appropriate but np.absolute() is not is where the sign in the output is relevant
    return [magnitude, phase, im2]
  

# -----------------------
# Convolution theorem
# -----------------------

# the mask corresponding to the average (mean) filter
def avgFilter(filterSize):
    mask = np.ones((filterSize, filterSize))
    return mask/np.sum(mask)

    # apply average filter in the spatial domain
def averageFilterSpace(im, filterSize):
    return filters.convolve(im, avgFilter(filterSize))


# apply average filter in the frequency domain
def averageFilterFrequency(im, filterSize):
    filterMask = avgFilter(filterSize)  # the usually small mask
    filterBig = np.zeros_like(im, dtype=float)  # as large as the image (dtype is important here!)

    # Now, place filter (the "small" filter mask) at the center of the "big" filter

    ## First, get sizes
    w, h = filterMask.shape
    w2, h2 = w / 2, h / 2  # half width and height of the "small" mask
    W, H = filterBig.shape
    W2, H2 = W / 2, H / 2  # half width and height of the "big" mask

    ## Then, paste the small mask at the center using the sizes computed before as an aid
    filterBig[int(W2 - w2):int(W2 + w2), int(H2 - h2):int(H2 + h2)] = filterMask

    # FFT of the big filter
    filterBig = fft.ifftshift(filterBig)  # shift origin at upper-left corner

    # Finally, IFT of the element-wise product of the FT's
    return np.absolute(IFT(FT(im) * FT(filterBig)))  # both '*' and multiply() perform elementwise product


def testConvTheo(im, params=None):
    filterSize = params['filterSize']

    # image filtered with a convolution in spatial domain
    imFiltSpace = averageFilterSpace(im, filterSize)

    # image filtered in frequency domain
    imFiltFreq = averageFilterFrequency(im, filterSize)

    # How much do they differ?
    # To quantify the difference, we use the Root Mean Square Measure (https://en.wikipedia.org/wiki/Root_mean_square)
    margin = 5  # exclude some outer pixels to reduce the influence of border effects
    rms = np.linalg.norm(imFiltSpace[margin:-margin, margin:-margin] - imFiltFreq[margin:-margin, margin:-margin], 2) / np.prod(im.shape)
    print("Images filtered in space and frequency differ in (RMS):", rms)

    return [imFiltSpace, imFiltFreq]


# -----------------------------------
# High-, low- and band-pass filters
# -----------------------------------

# generic band-pass filter (both, R and r, given) which includes the low-pass (r given, R not)
# and the high-pass (R given, r not) as particular cases
def bandPassFilter(shape, r=None, R=None):
    n, m = shape
    m2, n2 = np.floor(m / 2.0), np.floor(n / 2.0)
    [vx, vy] = np.meshgrid(np.arange(-m2, m2 + 1), np.arange(-n2, n2 + 1))
    distToCenter = np.sqrt(vx ** 2.0 + vy ** 2.0)
    if R is None:  # low-pass filter assumed
        assert r is not None, "at least one size for filter is expected"
        filter = distToCenter<r # same as np.less(distToCenter, r)
    elif r is None:  # high-pass filter assumed
        filter = distToCenter>R # same as np.greater(distToCenter, R)
    else:  # both, R and r given, then band-pass filter
        if r > R:
            r, R = R, r  # swap to ensure r < R (alternatively, warn the user, or throw an exception)
        filter = np.logical_and(distToCenter<R, distToCenter>r)
    filter = filter.astype('float')  # convert from boolean to float. Not strictly required

    bDisplay = True
    if bDisplay:
        plt.imshow(filter, cmap='gray')
        plt.show()
        plt.title("The filter in the frequency domain")
        # Image.fromarray((255*filter).astype(np.uint8)).save('filter.png')

    return filter


def testBandPassFilter(im, params=None):
    r, R = params['r'], params['R']
    filterFreq = bandPassFilter(im.shape, r, R)  # this filter is already in the frequency domain
    filterFreq = fft.ifftshift(filterFreq)  # shifting to have the origin as the FT(im) will be
    return [np.absolute(fft.ifft2(filterFreq * fft.fft2(im)))]  # the filtered image



# -----------------
# Test image files
# -----------------
path_input = './imgs-P3/'
path_output = './imgs-out-P3/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*.pgm")
else:
    files = [path_input + 'einstein.jpg']  # lena255, habas, mimbre

# --------------------
# Tests to perform
# --------------------
bAllTests = True
if bAllTests:
   tests = ['testFT', 'testConvTheo', 'testBandPassFilter', 'testGaussianFilter']
else:
    tests = ['testFT']
    tests = ['testConvTheo']
    tests = ['testBandPassFilter']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {
               'testFT': '2D Fourier Transform',
               'testConvTheo': 'Convolution Theorem (tested on mean filter)',
               'testBandPassFilter': 'Frequency-based filters ("high/low/band-pass")',
               'testGaussianFilter': 'Gaussian Filter Application'  # Добавление описания для testGaussianFilter
            }

bSaveResultImgs = False

testsUsingPIL = []  # which test(s) uses PIL images as input (instead of NumPy 2D arrays)


# -----------------------------------------
# Apply defined tests and display results
# -----------------------------------------

def doTests():
    print("Testing on", files)
    for imfile in files:
        im_pil = Image.open(imfile).convert('L')
        im = np.array(im_pil)  # from Image to array

        for test in tests:
            if test == "testGaussianFilter":
                params['size'] = 4  #  желаемый размер фильтра
                params['sigma'] = 2   # Зад

            elif test is "testFT":
                params = {}
                subTitle = ": I, |F|, ang(F), IFT(F)"
            elif test is "testConvTheo":
                params = {}
                params['filterSize'] = 7
                subTitle = ": I, I*M, IFT(FT(I).FT(M))"
            else:
                params = {}
                r,R = 5,30 # for low-pass filter
                # 5,30 for band-pass filter
                # None, 30 for high-pass filter
                params['r'], params['R'] = r,R
                # let's assume r and R are not both None simultaneously
                if r is None:
                    filter="high pass" + " (R=" + str(R) + ")"
                elif R is None:
                    filter="low pass" + " (r=" + str(r) + ")"
                else:
                    filter="band pass" + " (r=" + str(r) + ", R=" + str(R) + ")"
                subTitle = ", " + filter + " filter"

            if test in testsUsingPIL:
                outs_pil = eval(test)(im_pil, params)
                outs_np = vpu.pil2np(outs_pil)
            else:
                # Вызов тестовой функции
                outs_np = eval(test)(im, params)
            print("# images", len(outs_np))
            print(len(outs_np))

            vpu.showInGrid([im] + outs_np, title=nameTests[test] + subTitle)



if __name__ == "__main__":
    doTests()

