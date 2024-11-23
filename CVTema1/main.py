import cv2
import matplotlib.pyplot as plt
import numpy as np

# Citește imaginea din folderul 'images'
image = cv2.imread('images/lena.tif')

if image is None:
    print("Eroare: Imaginea nu a fost găsită.")
else:
    # Afișează dimensiunea imaginii
    print(f"Dimensiunea imaginii: {image.shape[1]}x{image.shape[0]}")

    # Afișează imaginea
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Imaginea Originală')
    plt.axis('off')
    plt.show()

    # Aplică estomparea (Blur)
    blurred_image_1 = cv2.GaussianBlur(image, (5, 5), 0)
    blurred_image_2 = cv2.GaussianBlur(image, (15, 15), 0)

    # Aplică accentuarea (Sharpening)
    kernel_sharpening1 = np.array([[0, -2, 0],
                                   [-2, 8, -2],
                                   [0, -2, 0]])
    sharpened_image1 = cv2.filter2D(image, -1, kernel_sharpening1)
    kernel_sharpening2 = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    sharpened_image2 = cv2.filter2D(image, -1, kernel_sharpening2)

    # Afișează rezultatele
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(blurred_image_1, cv2.COLOR_BGR2RGB))
    plt.title('Estompare 5x5')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(blurred_image_2, cv2.COLOR_BGR2RGB))
    plt.title('Estompare 15x15')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(sharpened_image1, cv2.COLOR_BGR2RGB))
    plt.title('Accentuat1')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(sharpened_image2, cv2.COLOR_BGR2RGB))
    plt.title('Accentuat2')
    plt.axis('off')

    plt.show()

    # Aplică filtrul specificat
    kernel_custom = np.array([[0, 2, 0],
                               [2, 8, 2],
                               [0, 2, 0]]) / 16  # Normalizarea kernel-ului

    filtered_image_custom = cv2.filter2D(image, -1, kernel_custom)

    # Afișează imaginea filtrată
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(filtered_image_custom, cv2.COLOR_BGR2RGB))
    plt.title('Imagine cu Filtrul Personalizat')
    plt.axis('off')
    plt.show()

    # Funcție pentru rotirea imaginii
    def rotate_image(image, angle):
        # Obține dimensiunile imaginii
        (h, w) = image.shape[:2]
        # Obține matricea de rotație
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        # Rotirea imaginii
        rotated_image = cv2.warpAffine(image, M, (w, h))
        return rotated_image

    # Rotirea imaginii la 45 de grade
    rotated_image_45 = rotate_image(image, 45)

    # Rotirea imaginii la -45 de grade
    rotated_image_neg_45 = rotate_image(image, -45)

    # Afișează imaginile rotite
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(rotated_image_45, cv2.COLOR_BGR2RGB))
    plt.title('Rotire 45 de grade')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(rotated_image_neg_45, cv2.COLOR_BGR2RGB))
    plt.title('Rotire -45 de grade')
    plt.axis('off')

    plt.show()

    # Funcție pentru tăierea unei părți dreptunghiulare din imagine
    def crop_image(image, x, y, width, height):

        # Asigură-te că coordonatele sunt în limitele imaginii
        if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
            print("Eroare: Coordonatele depășesc dimensiunile imaginii.")
            return None

        # Taie imaginea
        cropped_image = image[y:y + height, x:x + width]
        return cropped_image

    # Exemplu de utilizare a funcției de cropping
    x, y = 200, 200  # Poziția pixelului din colțul stânga sus
    width, height = 200, 150  # Dimensiunile dreptunghiului
    cropped_image = crop_image(image, x, y, width, height)

    if cropped_image is not None:
        # Afișează imaginea tăiată
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title('Imagine Tăiată')
        plt.axis('off')
        plt.show()


    # Creează o imagine goală (albă)
    panda_emoji = np.ones((500, 500, 3), dtype="uint8") * 255

    # Setează culorile pentru elemente
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Desenează conturul
    cv2.circle(panda_emoji, (250, 300), 155, black, 5)  # Conturul faței panda

    # Desenează capul
    cv2.circle(panda_emoji, (250, 300), 150, white, -1)

    # Desenează urechile
    cv2.circle(panda_emoji, (150, 150), 50, black, -1)  # Urechea stângă
    cv2.circle(panda_emoji, (350, 150), 50, black, -1)  # Urechea dreaptă

    # Desenează ochii
    cv2.circle(panda_emoji, (200, 280), 40, black, -1)  # Ochiul stâng
    cv2.circle(panda_emoji, (300, 280), 40, black, -1)  # Ochiul drept

    cv2.circle(panda_emoji, (200, 280), 20, white, -1)  # Interiorul ochiului stâng
    cv2.circle(panda_emoji, (300, 280), 20, white, -1)  # Interiorul ochiului drept

    # Desenează nasul
    cv2.circle(panda_emoji, (250, 350), 20, black, -1)

    # Desenează gura
    cv2.ellipse(panda_emoji, (250, 380), (50, 20), 0, 0, 180, black, 3)

    # Afișează imaginea
    cv2.imshow('Panda Emoji', panda_emoji)
    cv2.imwrite('panda_emoji.jpg', panda_emoji)  # Save
    cv2.waitKey(0)
    cv2.destroyAllWindows()



