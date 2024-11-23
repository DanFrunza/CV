import cv2
import numpy as np
import matplotlib.pyplot as plt



#-------Skin detection functions-------



#RGB function
def detect_skin_rgb(image):

    #R, G, B channels
    R = image[:, :, 2]
    G = image[:, :, 1]
    B = image[:, :, 0]

    #Conditions
    cond1 = (R > 95)
    cond2 = (G > 40)
    cond3 = (B > 20)
    cond4 = ((np.maximum(np.maximum(R, G), B) - np.minimum(np.maximum(R, G), B)) > 15) 
    cond5 = (np.abs(R - G) > 15) 
    cond6 = (R > G)
    cond7 = (R > B)


    #debugging

    # plt.imshow(cond1, cmap='gray')
    # plt.title("Condiția 1: R > 95")
    # plt.show()

    # plt.imshow(cond2, cmap='gray')
    # plt.title("Condiția 2: G > 40")
    # plt.show()

    # plt.imshow(cond3, cmap='gray')
    # plt.title("Condiția 3: B > 20")
    # plt.show()

    # plt.imshow(cond4, cmap='gray')
    # plt.title("Condiția 4")
    # plt.show()

    # plt.imshow(cond5, cmap='gray')
    # plt.title("Condiția 5: |R - G| > 15")
    # plt.show()

    # plt.imshow(cond6, cmap='gray')
    # plt.title("Condiția 6: R > G")
    # plt.show()

    # plt.imshow(cond7, cmap='gray')
    # plt.title("Condiția 7: R > B")
    # plt.show()

    #Combine conditions with logical AND
    condition = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7

    #Skin mask
    skin_mask = np.zeros_like(R)
    skin_mask[condition] = 255  # Set skin pixels to white
    return skin_mask



#HSV function
def detect_skin_hsv(image):

    #BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_image)

    #Condtitions
    cond1 = (H >= 0) & (H <= 50)
    cond2 = (S >= 0.23 * 255) & (S <= 0.68 * 255)
    cond3 = (V >= 0.35 * 255) & (V <= 255)


    # Combine conditions with logical AND
    condition = cond1 & cond2 & cond3

    #Skin mask
    skin_mask = np.zeros_like(H)
    skin_mask[condition] = 255  # Set skin pixels to white
    return skin_mask



#YCBCR function
def detect_skin_ycbcr(image):

    #BGR to YCbCr
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycbcr_image)

    #Conditions
    cond1 = (Y > 80)
    cond2 = (Cb > 85) & (Cb < 135) 
    cond3 = (Cr > 135) & (Cr < 180) 

    #debugging

    # plt.imshow(cond1, cmap='gray')
    # plt.title("Condiția 1")
    # plt.show()

    # plt.imshow(cond2, cmap='gray')
    # plt.title("Condiția 2")
    # plt.show()

    # plt.imshow(cond3, cmap='gray')
    # plt.title("Condiția 3")
    # plt.show()

    #Combine conditions with logical AND
    condition = cond1 & cond2 & cond3

    #Skin mask
    skin_mask = np.zeros_like(Y)
    skin_mask[condition] = 255  # Set skin pixels to white
    return skin_mask



#Detect face function
def detect_face(image, string):

    #Method
    if(string == "RGB"):
        skin_mask = detect_skin_rgb(image)
    if(string == "HSV"):
        skin_mask = detect_skin_hsv(image)
    if(string == "YCBCR"):
        skin_mask = detect_skin_ycbcr(image)

    #Contours
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        #Get the bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        #Draw the bounding box on the original image
        result_image = cv2.rectangle(image.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
        return result_image

    #No contour
    return image



#Evaluate skin detection function
def evaluate_skin_detection(detected_mask, ground_truth_mask):

    #Mask conversion to binary
    detected_mask_bin = detected_mask // 255  # 255 -> 1
    ground_truth_mask_bin = ground_truth_mask // 255  # 255 -> 1

    #Confusion matrix's components
    TP = np.sum((detected_mask_bin == 1) & (ground_truth_mask_bin == 1))  # True Positive
    FN = np.sum((detected_mask_bin == 0) & (ground_truth_mask_bin == 1))  # False Negative
    FP = np.sum((detected_mask_bin == 1) & (ground_truth_mask_bin == 0))  # False Positive
    TN = np.sum((detected_mask_bin == 0) & (ground_truth_mask_bin == 0))  # True Negative

    #Acurracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return (TP, FN, FP, TN), accuracy



#Plot function
def plotimages1(original,string1,mask,string2,face,string3):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(string1)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    plt.title(string2)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    plt.title(string3)
    plt.axis('off')

    plt.show()



#Load image and truth function
def load_image_and_ground_truth(image_path, gt_path):
   
    image = cv2.imread(image_path)
    ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    return image, ground_truth



#Evaluate pratheepan function
def evaluate_pratheepan_dataset(method):
   
    #PATHS
    family_images = ["data/Pratheepan_Dataset/FamilyPhoto/2007_family.jpg",
                     "data/Pratheepan_Dataset/FamilyPhoto/f_family.jpg",
                     "data/Pratheepan_Dataset/FamilyPhoto/family.jpg"]

    family_gt = ["data/Ground_Truth/GroundT_FamilyPhoto/2007_family.png",
                 "data/Ground_Truth/GroundT_FamilyPhoto/f_family.png",
                 "data/Ground_Truth/GroundT_FamilyPhoto/family.png"]

    for image_path, gt_path in zip(family_images, family_gt):
        
        image, ground_truth = load_image_and_ground_truth(image_path, gt_path)

        #Mask
        if method == "RGB":
            skin_mask = detect_skin_rgb(image)
        elif method == "HSV":
            skin_mask = detect_skin_hsv(image)
        elif method == "YCBCR":
            skin_mask = detect_skin_ycbcr(image)

        confusion_mat, accuracy = evaluate_skin_detection(skin_mask, ground_truth)
        tp, fn, fp, tn = confusion_mat

        #Print
        print(f"Evaluare pentru imaginea: {image_path}")
        print(f"Matricea de confuzie: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
        print(f"Acuratețea: {accuracy:.4f}")
        print("-" * 50)

        face_detected_image = detect_face(image, method)
        
        ground_truth_colored = cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2BGR)  
        
        plotimages1(image, "Original", skin_mask, f"Mask {method}", ground_truth_colored, "Ground Truth")





test_image1 = cv2.imread('data/Tests/skin/3.jpg')
test_image2 = cv2.imread('data/Tests/skin/4.jpg')
test_group = cv2.imread('data/Tests/skin/group2.jpg')



test_image1_mask_HSV = detect_skin_hsv(test_image1)
test_image2_mask_HSV = detect_skin_hsv(test_image2)
test_group_mask_HSV = detect_skin_hsv(test_group)
test_image1_detectface_HSV = detect_face(test_image1, "HSV")
test_image2_detectface_HSV = detect_face(test_image2, "HSV")
test_group_detectface_HSV = detect_face(test_group, "HSV")

plotimages1(test_image1, "Original", test_image1_mask_HSV, "Mask HSV", test_image1_detectface_HSV, "DetectFace")
plotimages1(test_image2, "Original", test_image2_mask_HSV, "Mask HSV", test_image2_detectface_HSV, "DetectFace")
plotimages1(test_group, "Original", test_group_mask_HSV, "Mask HSV", test_group_detectface_HSV, "DetectFace")

test_image1_mask_RGB = detect_skin_rgb(test_image1)
test_image2_mask_RGB = detect_skin_rgb(test_image2)
test_group_mask_RGB = detect_skin_rgb(test_group)
test_image1_detectface_RGB = detect_face(test_image1, "RGB")
test_image2_detectface_RGB = detect_face(test_image2, "RGB")
test_group_detectface_RGB = detect_face(test_group, "RGB")

plotimages1(test_image1, "Original", test_image1_mask_RGB, "Mask RGB", test_image1_detectface_RGB, "DetectFace")
plotimages1(test_image2, "Original", test_image2_mask_RGB, "Mask RGB", test_image2_detectface_RGB, "DetectFace")
plotimages1(test_group, "Original", test_group_mask_RGB, "Mask RGB", test_group_detectface_RGB, "DetectFace")

test_image1_mask_YCBCR = detect_skin_ycbcr(test_image1)
test_image2_mask_YCBCR = detect_skin_ycbcr(test_image2)
test_group_mask_YCBCR = detect_skin_ycbcr(test_group)
test_image1_detectface_YCBCR = detect_face(test_image1, "YCBCR")
test_image2_detectface_YCBCR = detect_face(test_image2, "YCBCR")
test_group_detectface_YCBCR = detect_face(test_group, "YCBCR")

plotimages1(test_image1, "Original", test_image1_mask_YCBCR, "Mask YCBCR", test_image1_detectface_YCBCR, "DetectFace")
plotimages1(test_image2, "Original", test_image2_mask_YCBCR, "Mask YCBCR", test_image2_detectface_YCBCR, "DetectFace")
plotimages1(test_group, "Original", test_group_mask_YCBCR, "Mask YCBCR", test_group_detectface_YCBCR, "DetectFace")



# family_images = ["data/Pratheepan_Dataset/FamilyPhoto/2007_family.jpg",
#                  "data/Pratheepan_Dataset/FamilyPhoto/f_family.jpg",
#                  "data/Pratheepan_Dataset/FamilyPhoto/family.jpg"]

# family_gt = ["data/Ground_Truth/GroundT_FamilyPhoto/2007_family.png",
#              "data/Ground_Truth/GroundT_FamilyPhoto/f_family.png",
#              "data/Ground_Truth/GroundT_FamilyPhoto/family.png"]

# face_images = ["data/Pratheepan_Dataset/FacePhoto/06Apr03Face.jpg",
#                "data/Pratheepan_Dataset/FacePhoto/m_unsexy_gr.jpg"]

# face_gt = ["data/Ground_Truth/GroundT_FacePhoto/06Apr03Face.png",
#            "data/Ground_Truth/GroundT_FacePhoto/m_unsexy_gr.png"]

print("\nRGB \n")
evaluate_pratheepan_dataset("RGB")
print("\nHSV \n")
evaluate_pratheepan_dataset("HSV")
print("\nYCBCR \n")
evaluate_pratheepan_dataset("YCBCR")

