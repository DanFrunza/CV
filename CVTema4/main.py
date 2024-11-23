import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np


#OCR
def ocr_image(image):
    
    if len(image.shape) == 2:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    extracted_text = pytesseract.image_to_string(gray_image)
    
    print(f"Text recognized:\n{extracted_text}")
    return extracted_text



#Plot
def plot_image_and_text(image, string1 , text):
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title(string1)
    axs[0].axis('off')  

    axs[1].text(0.1, 0.5, text, fontsize=12, wrap=True) 
    axs[1].axis('off')  
    
    plt.tight_layout()
    plt.show()



#Gaussian noise
def add_gaussian_noise(image, mean=0, sigma=25):
    
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)

    noisy_image = cv2.add(image, noise)

    return noisy_image



#Salt and pepper noise
def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    
    noisy_image = np.copy(image)
    
    #salt(white pixels)
    num_salt = int(salt_prob * image.size)
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    
    #pepper(black pixels)
    num_pepper = int(pepper_prob * image.size)
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_image



#Rotation
def rotate_image(image, angle):

    #Dimesions
    (h, w) = image.shape[:2]
    
    #Center
    center = (w // 2, h // 2)
    
    #RotationMatrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    #New dimensions
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

    return rotated_image



#Vertical shear
def shear_image_vertical(image, shear_factor=0.5):

    (h, w) = image.shape[:2]
    
    #New dimensions
    new_w = int(w + shear_factor * h)  # w >>> bigger
    
    
    shearing_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    
    
    sheared_image = cv2.warpAffine(image, shearing_matrix, (new_w, h))
    
    return sheared_image



#Horizontal shear
def shear_image_horizontal(image, shear_factor=0.5):

    (h, w) = image.shape[:2]
    
    #New dimensions
    new_h = int(h + shear_factor * w)  # h >>> bigger
    
    
    shearing_matrix = np.float32([[1, 0, 0], [shear_factor, 1, 0]])
    
    
    sheared_image = cv2.warpAffine(image, shearing_matrix, (w, new_h))
    
    return sheared_image



#Resize with ratio
def resize_image_aspect_ratio(image, scale_factor):

    (h, w) = image.shape[:2]
    
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    resized_image = cv2.resize(image, (new_w, new_h))
    
    return resized_image



#Resize without ratio
def resize_image_no_aspect_ratio(image, new_width, new_height):

    resized_image = cv2.resize(image, (new_width, new_height))
    
    return resized_image



#Average blur
def apply_average_blur(image, kernel_size):
    
    blurred_image = cv2.blur(image, (kernel_size, kernel_size))
    return blurred_image




#Gaussian blur
def apply_gaussian_blur(image, kernel_size, sigma):

    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred_image




#Sharpen
def sharpen_image(image):

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image



#Morphological operations
def morphological_operations(image):
    kernel = np.ones((3,3), np.uint8)  #5x5 kernel
    eroded_image = cv2.erode(image, kernel, iterations=1)  #Erode
    dilated_image = cv2.dilate(image, kernel, iterations=1)  #Dilate
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  #Opening
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  #Closing
    return eroded_image, dilated_image, opened_image, closed_image



#Global and adaptive thresholding
def thresholding(image):
    #Gray
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Global thresholding
    _, global_threshold = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    
    # Adaptive thresholding
    adaptive_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
    
    return global_threshold, adaptive_threshold




#---1---
image1 = cv2.imread('images/sample21.jpg')
plot_image_and_text(image1, "sample21", ocr_image(image1))
#---Teste---
new_image = sharpen_image(image1)
plot_image_and_text(new_image, "sample21 but sharpened", ocr_image(new_image))

image2 = cv2.imread('images/example_02.jpg')
plot_image_and_text(image2, "example_02", ocr_image(image2))

image3 = cv2.imread('images/testocr-gt.jpg')
plot_image_and_text(image3, "testocr-gt", ocr_image(image3))



#---2---
image_gaussian_noise = add_gaussian_noise(image3)
image_salt_and_pepper = add_salt_and_pepper_noise(image3)

plot_image_and_text(image_gaussian_noise, "testocr-gt Gaussian noise", ocr_image(image_gaussian_noise))
plot_image_and_text(image_salt_and_pepper, "testocr-gt Salt and pepper noise", ocr_image(image_salt_and_pepper))



#---3---
rotated_image1 = rotate_image(image3, 45)
rotated_image2 = rotate_image(image3, 72)
rotated_image3 = rotate_image(image3, -33)

plot_image_and_text(rotated_image1, "testocr-gt rotated 45", ocr_image(rotated_image1))
plot_image_and_text(rotated_image2, "testocr-gt rotated 72", ocr_image(rotated_image2))
plot_image_and_text(rotated_image3, "testocr-gt rotated -33", ocr_image(rotated_image3))

shear_image_h = shear_image_horizontal(image3)
shear_image_v = shear_image_vertical(image3)

plot_image_and_text(shear_image_h, "testocr-gt Horizontal shear 0.5", ocr_image(shear_image_h))
plot_image_and_text(shear_image_v, "testocr-gt Vertical shear 0.5", ocr_image(shear_image_v))



#---4---
resized_with_image1 = resize_image_aspect_ratio(image3, 0.2)
resized_with_image2 = resize_image_aspect_ratio(image3, 3)
resized_without_image1 = resize_image_no_aspect_ratio(image3, 100, 200)
resized_without_image2 = resize_image_no_aspect_ratio(image3, 700, 500)

plot_image_and_text(resized_with_image1, "testocr-gt resized 0,2", ocr_image(resized_with_image1))
plot_image_and_text(resized_with_image2, "testocr-gt resized 3", ocr_image(resized_with_image2))
plot_image_and_text(resized_without_image1, "testocr-gt resized 100 200", ocr_image(resized_without_image1))
plot_image_and_text(resized_without_image2, "testocr-gt resized 700 500", ocr_image(resized_without_image2))



#---5---
average_blur_image1 = apply_average_blur(image3, 3)
average_blur_image2 = apply_average_blur(image3, 7)
gaussian_blur_image1 = apply_gaussian_blur(image3, 3, 1)
gaussian_blur_image2 = apply_gaussian_blur(image3, 7, 3)

plot_image_and_text(average_blur_image1, "testocr-gt avrg blur 3", ocr_image(average_blur_image1))
plot_image_and_text(average_blur_image2, "testocr-gt avrg blur 7", ocr_image(average_blur_image2))
plot_image_and_text(gaussian_blur_image1, "testocr-gt gaussian blur 3 1", ocr_image(gaussian_blur_image1))
plot_image_and_text(gaussian_blur_image2, "testocr-gt gaussian blur 7 3", ocr_image(gaussian_blur_image2))




#---6---
sharpened_image = sharpen_image(image3)
eroded_image, dilated_image, opened_image, closed_image = morphological_operations(image3)
global_threshold, adaptive_threshold = thresholding(image3)

plot_image_and_text(sharpened_image, "testocr-gt sharpened", ocr_image(sharpened_image))
plot_image_and_text(eroded_image, "testocr-gt eroded", ocr_image(eroded_image))
plot_image_and_text(dilated_image, "testocr-gt dilated", ocr_image(dilated_image))
plot_image_and_text(opened_image, "testocr-gt opened", ocr_image(opened_image))
plot_image_and_text(closed_image, "testocr-gt closed image", ocr_image(closed_image))
plot_image_and_text(global_threshold, "testocr-gt global threshold", ocr_image(global_threshold))
plot_image_and_text(adaptive_threshold, "testocr-gt adaptive threshold", ocr_image(adaptive_threshold))








