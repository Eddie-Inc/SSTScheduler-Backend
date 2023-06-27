#packages
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract #Google OCR

#Read file
#read the file
file=r'/Users/Ayaan/Downloads/2023 Term3-4 TimeTable_S308.png'
img = cv2.imread(file, 0)
img.shape

#Yeet image into binary format
thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

#Change CoLoR
img_bin = 255-img_bin
cv2.imwrite('/Users/Ayaan/Downloads/BNRY.png', img_bin)

print(0)

#plot the image using mat so we can test output
plotting = plt.imshow(img_bin, cmap='gray')
plt.show()

print("1")

#Detect boxes

#Width of kernel as 100th of toal (Lower resolution)
kernel_len = np.array(img).shape[1]//100

#Detect rows
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

#Detect cols
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

#Make 2x2 kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#MORPH_RECT defines it as a rect and a square


#Detect vertical lines

#Use vertical kernel to get cols in jpg
image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
cv2.imwrite('/Users/Ayaan/Downloads/vertical_lines.jpg', vertical_lines)

#Plot the mat
plotting = plt.imshow(image_1)
plt.show
print(2)


#Detect horizontal lines
print("here")
#Use horizontal kernel to get rows in jpg
image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
cv2.imwrite('/Users/Ayaan/Downloads/horizontal_lines.jpg', horizontal_lines)

#Plot the mat
plotting = plt.imshow(image_2)
plt.show

#How does it all work:
#Each thing either rows or col has the kernels set specifically
#Everything that then isn't vertical or horizontal respectively gets destroyed or "eroded"


#OMG WHAT IF WE COMBINE THE LINES TO GET BOXES
img_boxes = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)

#Erode the words
img_boxes = cv2.erode(~img_boxes, kernel, iterations=2)
thresh, img_boxes = cv2.threshold(img_boxes, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
cv2.imwrite('/Users/Ayaan/Downloads/img_boxes.jpg', img_boxes)

bitxor = cv2.bitwise_xor(img, img_boxes)
bitnot = cv2.bitwise_not(bitxor) #Ms Tang pls be proud


#Nows the fun part

def contours(cnts, method="left-to-right"):

    reverse = False
    i = 0

    #sort boxes
    sortedBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, sortedBoxes) = zip(*sorted(zip(cnts, sortedBoxes), key=lambda b:b[1][i], reverse=reverse))

    #return list of periods
    return(cnts, sortedBoxes)

#sort cnts by top to bottom
contours, sortedBoxes = contours(contours, method="left-to-right")

#