from skimage.io import imread, imshow 
from skimage import color, filters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

image = imread('./images/KIZIN_16.jpg')

gray_image=color.rgb2gray(image)
hsv_image = color.rgb2hsv(image)
imshow(image)

red_channel = image[:,:,0]
green_channel = image[:,:,1]
blue_channel = image[:,:,2]

mean = np.mean(image)
variance = np.mean([abs(p - mean)**2 for p in np.nditer(image)])
contrast = np.sqrt(variance)
size = image.shape[0]*image.shape[1]
###Feature 1

bpnum = np.sum(blue_channel>20)
rpnum = np.sum(red_channel>20)



feature1=[bpnum, rpnum, contrast]

print("Feature 1, Amount of blue pixels, red pixels, and contrast: ", feature1) ###To detect spiderman comics images
##Feature2
brightness = np.mean(gray_image) ##Brightness

img_threshold = 0.5 ##Amount of white pixels
wbinary_image = gray_image > img_threshold
wpixels = np.count_nonzero(wbinary_image)
whites = wpixels/ size

bbinary_image = gray_image <= img_threshold ##Amount of black pixels
bpixels = np.count_nonzero(bbinary_image)
blacks = bpixels / size
feature2 = [brightness, whites,blacks]
print("Feature 2, Brightness, whites and blacks:", feature2)

##Feature3
rpaverage=np.mean(red_channel) ###Saturation of red, green and blue channels
gpaverage=np.mean(green_channel)
bpaverage=np.mean(blue_channel)

feature3 = [rpaverage, gpaverage, bpaverage]

print("Feature 3, rgb channels", feature3)

##Feature4
bwthresh=filters.threshold_otsu(gray_image) ###Extracting the white and black pixels from an image
binary=image <= img_threshold
bapixels= np.sum(binary == 0)
wapixels= np.sum(binary == 1)

feature4=[bapixels, wapixels, contrast] ### Extract amount of white and black pixels and calculate contrast 

print("Feature 4, black pixels in an image, white pixels in an image and contrast: ", feature4)

##Feature5
bpnum = np.sum(blue_channel>20) 
bpavg = np.mean(blue_channel)
saturationb = np.mean(hsv_image[:, :, 1])

feature5 = [bpnum, bpavg, saturationb]
print("Feature 5, sum of blue pixels, average of blue pixels and saturation of blue channel", feature5) ##For detecting images with a lot of blue, as if they were taken at sea