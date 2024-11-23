import cv2
import matplotlib.pyplot as plt
import numpy as np

# Citește imaginea din folderul 'images'
image = cv2.imread('images/lena.tif')


# Afișează imaginea
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Imaginea Originală')
plt.axis('off')
plt.show()

#--------------1--------------

def simple_average(image):
    gray = np.mean(image, axis=2).astype(np.uint8)  # Average across the R, G, B channels
    return gray

def simple_average_manual(image):
    # Get image dimensions
    height, width, _ = image.shape

    # Create an empty grayscale image
    gray_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each pixel in the image
    for i in range(height):
        for j in range(width):
            # Get the RGB values
            R, G, B = image[i, j]

            # Calculate the average manually
            gray_value = R // 3 + G // 3 + B // 3

            # Assign the result to the grayscale image
            gray_image[i, j] = gray_value
    return gray_image


Simple_avarage_im1 = simple_average(image)
Simple_avarage_im2 = simple_average_manual(image)


#--------------2--------------

def weighted_average(image, weights=(0.299, 0.587, 0.114)):
    gray = np.dot(image[...,:3], weights).astype(np.uint8)  # Weighted sum of R, G, B
    return gray


def weighted_average_manual(image):
    # Get image dimensions
    height, width, _ = image.shape

    # Create an empty grayscale image
    gray_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each pixel in the image
    for i in range(height):
        for j in range(width):
            # Get the RGB values
            B, G, R= image[i, j]  # BGR FORMAT !!

            # Apply the weighted average formula
            gray_value = int(0.299 * R + 0.587 * G + 0.114 * B)

            # Assign the result to the grayscale image
            gray_image[i, j] = gray_value

    return gray_image

Weighted_average_im1 = weighted_average(image)
Weighted_average_im2 = weighted_average_manual(image)



plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(Simple_avarage_im1, cv2.COLOR_BGR2RGB))
plt.title('simple avarage')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(Simple_avarage_im2, cv2.COLOR_BGR2RGB))
plt.title('manual avarage')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(Weighted_average_im1, cv2.COLOR_BGR2RGB))
plt.title('weighted_average')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(Weighted_average_im2, cv2.COLOR_BGR2RGB))
plt.title('manual weighted_average')
plt.axis('off')

plt.show()


#--------------3--------------

def desaturation(image):
    max_val = np.max(image, axis=2)
    min_val = np.min(image, axis=2)
    gray = (max_val/2 + min_val / 2).astype(np.uint8)
    return gray


def desaturation_manual(image):
    # Get image dimensions
    height, width, _ = image.shape

    # Create an empty grayscale image
    gray_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate through each pixel
    for i in range(height):
        for j in range(width):
            # Extract the RGB values
            B, G, R = image[i, j] # BGR FORMAT !!

            # Manually calculate the min and max of R, G, B
            max_val = max(R, G, B)
            min_val = min(R, G, B)

            # Compute the desaturated grayscale value
            gray_value = int(max_val/2 + min_val / 2)

            # Assign the computed value to the grayscale image
            gray_image[i, j] = gray_value

    return gray_image


Desaturation_im1 = desaturation(image)
Desaturation_im2 = desaturation_manual(image)



plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(Desaturation_im1, cv2.COLOR_BGR2RGB))
plt.title('desaturation')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(Desaturation_im2, cv2.COLOR_BGR2RGB))
plt.title('manual desaturation')
plt.axis('off')

plt.show()

#--------------4--------------

def decomposition_max(image):
    gray = np.max(image, axis=2).astype(np.uint8)
    return gray

def decomposition_min(image):
    gray = np.min(image, axis=2).astype(np.uint8)
    return gray


def max_decomposition_manual(image):
    # Get image dimensions
    height, width, _ = image.shape

    # Create an empty grayscale image
    gray_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate through each pixel
    for i in range(height):
        for j in range(width):
            # Extract the RGB values
            B, G, R = image[i, j] #BGR FORMAT!!

            # Find the maximum of the R, G, and B values
            max_val = max(R, G, B)

            # Assign the maximum value to the grayscale image
            gray_image[i, j] = max_val

    return gray_image


def min_decomposition_manual(image):
    # Get image dimensions
    height, width, _ = image.shape

    # Create an empty grayscale image
    gray_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate through each pixel
    for i in range(height):
        for j in range(width):
            # Extract the RGB values
            B, G, R = image[i, j] #BGR FORMAT !!

            # Find the minimum of the R, G, and B values
            min_val = min(R, G, B)

            # Assign the minimum value to the grayscale image
            gray_image[i, j] = min_val

    return gray_image

Decomposition_im1 = decomposition_max(image)
Decomposition_im2 = decomposition_min(image)
Decomposition_im3 = max_decomposition_manual(image)
Decomposition_im4 = min_decomposition_manual(image)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(Decomposition_im1, cv2.COLOR_BGR2RGB))
plt.title('decomposition_max')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(Decomposition_im3, cv2.COLOR_BGR2RGB))
plt.title('max_decomposition_manual')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(Decomposition_im2, cv2.COLOR_BGR2RGB))
plt.title('decomposition_min')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(Decomposition_im4, cv2.COLOR_BGR2RGB))
plt.title('min_decomposition_manual')
plt.axis('off')

plt.show()

#--------------5--------------
def single_channel_red(image):
    return image[:, :, 2]  # Red channel

def single_channel_green(image):
    return image[:, :, 1]  # Green channel

def single_channel_blue(image):
    return image[:, :, 0]  # Blue channel


def single_channel(image, channel):

    height, width, _ = image.shape

    channel_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            channel_value = image[y, x, channel]  # channel: 0=Blue, 1=Green, 2=Red #BGR FORMAT !!!
            channel_image[y, x] = channel_value
    return channel_image

Single_channel_im1 = single_channel_red(image)
Single_channel_im2 = single_channel_green(image)
Single_channel_im3 = single_channel_blue(image)
Single_channel_im4 = single_channel(image, 0)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(Single_channel_im1, cv2.COLOR_BGR2RGB))
plt.title('single_channel_red')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(Single_channel_im2, cv2.COLOR_BGR2RGB))
plt.title('single_channel_green')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(Single_channel_im3, cv2.COLOR_BGR2RGB))
plt.title('single_channel_blue')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(Single_channel_im4, cv2.COLOR_BGR2RGB))
plt.title('single_channel blue (BGR FORMAT channel: 0=Blue, 1=Green, 2=Red) ')
plt.axis('off')

plt.show()

#--------------6--------------
def custom_gray_shades(image, num_shades=8): #actually 7 levels !!
    gray = weighted_average(image)  # Convert to grayscale
    levels = np.linspace(0, 255, num_shades)  # Define the gray levels (pentru 8 [  0,  36,  73, 109, 146, 182, 219, 255])
    gray_quantized = np.digitize(gray, levels, right=True)  # Quantize the grayscale image
    return ((gray_quantized - 1) * (255 // (num_shades - 1))).astype(np.uint8)  # Map to shades

Custom_gray_shades_im1 = custom_gray_shades(image, 9)

plt.imshow(cv2.cvtColor(Custom_gray_shades_im1, cv2.COLOR_BGR2RGB))
plt.title('custom_gray_shades 8 levels')
plt.axis('off')
plt.show()

#--------------7--------------

def floyd_steinberg_dithering(image, num_shades=2):
    gray = weighted_average(image)
    levels = np.linspace(0, 255, num_shades)
    err_diffusion = np.copy(gray).astype(float)  # Work on a float copy

    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            old_pixel = err_diffusion[y, x]
            new_pixel = np.round((old_pixel / 255) * (num_shades - 1)) * (255 // (num_shades - 1))
            err_diffusion[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            if x + 1 < gray.shape[1]:
                err_diffusion[y, x + 1] += quant_error * 7 / 16
            if y + 1 < gray.shape[0] and x > 0:
                err_diffusion[y + 1, x - 1] += quant_error * 3 / 16
            if y + 1 < gray.shape[0]:
                err_diffusion[y + 1, x] += quant_error * 5 / 16
            if y + 1 < gray.shape[0] and x + 1 < gray.shape[1]:
                err_diffusion[y + 1, x + 1] += quant_error * 1 / 16

    return err_diffusion.astype(np.uint8)

Floyd_steinberg_dithering_im1 = floyd_steinberg_dithering(image, 2)


def stucki_dithering(image, num_shades=2):
    gray = weighted_average(image)
    levels = np.linspace(0, 255, num_shades)
    err_diffusion = np.copy(gray).astype(float)  # Work on a float copy

    # Stucki diffusion mask weights
    stucki_weights = np.array([
        [0, 0, 0, 8 / 42, 4 / 42],
        [2 / 42, 4 / 42, 8 / 42, 4 / 42, 2 / 42],
        [1 / 42, 2 / 42, 4 / 42, 2 / 42, 1 / 42]
    ])

    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            old_pixel = err_diffusion[y, x]
            new_pixel = np.round((old_pixel / 255) * (num_shades - 1)) * (255 // (num_shades - 1))
            err_diffusion[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            # Distribute the error according to the Stucki matrix
            for dy in range(3):
                for dx in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < gray.shape[1] and 0 <= ny < gray.shape[0] and stucki_weights[dy, dx + 2] > 0:
                        err_diffusion[ny, nx] += quant_error * stucki_weights[dy, dx + 2]

    return err_diffusion.astype(np.uint8)

Stucki_dithering_im2 = stucki_dithering(image, 2)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(Floyd_steinberg_dithering_im1, cv2.COLOR_BGR2RGB))
plt.title('Floyd_steinberg_dithering')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(Stucki_dithering_im2, cv2.COLOR_BGR2RGB))
plt.title('Stucki_dithering')
plt.axis('off')

plt.show()

 # 0.299 * R + 0.587 * G + 0.114 * B
def grayscale_to_color(gray):

    color_image = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

    color_image[..., 0] = gray * 0.114  # Blue channel
    color_image[..., 1] = gray * 0.587  # Green channel BGR FORMAT !!
    color_image[..., 2] = gray * 0.299  # Red channel

    # Clip values to stay within valid range
    color_image = np.clip(color_image, 0, 255)

    return color_image.astype(np.uint8)

Gray_im1 = weighted_average(image)
Grayscale_to_color_im1 = grayscale_to_color(Gray_im1)



def simple_color_mapping(gray):
    # Initialize a color image with the same dimensions as the grayscale image
    color_image = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

    # Map gray values to color values
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            # Extract the gray value
            gray_value = gray[i, j]

            # Map the gray value to RGB colors
            if gray_value < 128:  # Darker shades
                color_image[i, j] = [gray_value, 0, 255 - gray_value]  # Blue to Purple gradient
            else:  # Lighter shades
                color_image[i, j] = [255 - gray_value, gray_value, 0]  # Yellow to Red gradient

    return color_image

Simple_color_mapping_im1 = simple_color_mapping(Gray_im1)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(Grayscale_to_color_im1, cv2.COLOR_BGR2RGB))
plt.title('Grayscale_to_color')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(Simple_color_mapping_im1, cv2.COLOR_BGR2RGB))
plt.title('Simple_color_mapping')
plt.axis('off')

plt.show()