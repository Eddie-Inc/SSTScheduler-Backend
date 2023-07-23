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
pytesseract.pytesseract.tesseract_cmd = '/Users/Ayaan/homebrew/Cellar/tesseract/5.3.1_1/bin/tesseract'

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


# Detect contours for following box detection
contours, hierarchy = cv2.findContours(img_boxes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#Nows the fun part

def sort_contours(cnts, method="left-to-right"):

    reverse = False
    i = 0

    #sort boxes
    sortedBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, sortedBoxes) = zip(*sorted(zip(cnts, sortedBoxes), key=lambda b:b[1][i], reverse=reverse))

    #return list of periods
    return(cnts, sortedBoxes)

#sort cnts by top to bottom
contours, sortedBoxes = sort_contours(contours, method="left-to-right")

#Get day of period
heights = [sortedBoxes[i][3] for i in range(len(sortedBoxes))]

#get average height so we can figure out the rough amount for day, kinda like repeating a science experiment for accuracy
mean = np.mean(heights)

#list of boxes
boxAll = []

#Get position (x,y), width and height for every contour and show the contour on image
for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    if (w<1000 and 90<h):
        image = cv2.rectangle(img,(x,y),(x+w, y+h),(250, 10, 300), 2)
        print()
        boxAll.append([x, y, w, h])

plotting = plt.imshow(image, cmap='gray')
plt.show()

#Creating two lists to define row and coloumn in which cell is located for each
row = []
coloumn = []
j=0

#sorting the boxes to their respective row and coloumn (Can edit)

for i in range(len(boxAll)):
    if(i==0): #get where the coloumn starts
        coloumn.append(boxAll[i])
        previous = boxAll[i]
    else:
        if(boxAll[i][1]<=previous[1]+mean/2):
            coloumn.append(boxAll[i])
            previous=boxAll[i]

            if(i==len(boxAll)-1):
                row.append(coloumn)
        else:
            row.append(coloumn)
            coloumn=[]
            previous = boxAll[i]
            coloumn.append(boxAll[i])
print(coloumn)
print("row:")
print(row)

#get max cells

countcol = 0
for i in range(len(row)):
    countcol = len(row[i])
    if countcol > countcol:
        countcol = countcol

#get the center of each coloumn

center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]

center = np.array(center)
center.sort()

#arrange boxes by order by using center
finalboxes = []

for i in range(len(row)):
    arr=[]
    for k in range(countcol):
        arr.append([])
    for j in range(len(row[i])):
        diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
        minimum = min(diff)
        indexing = list(diff).index(minimum)
        arr[indexing].append(row[i][j])
    finalboxes.append(arr)

#Extract text from cells through Pytesseract
outer = []
for i in range(len(finalboxes)):
    for j in range(len(finalboxes[i])):
        inner=''
        if(len(finalboxes[i][j])==0):
            outer.append(" ")
        else:
            for k in range(len(finalboxes[i][j])):
                y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2], finalboxes[i][j][k][3]
                finalimg = bitnot[x:x+h, y:y+w]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                border = cv2.copyMakeBorder(finalimg,2,2,2,2,cv2.BORDER_CONSTANT, value=[255, 255])
                resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                dilation = cv2.dilate(resizing, kernel, iterations=1)
                erosion = cv2.erode(dilation, kernel, iterations=1)

                out = pytesseract.image_to_string(erosion)
                if len((out)==0):
                    out = pytesseract.image_to_string(erosion, config='-- psm 3')
                inner = inner+ " "+ out

            outer.append(inner)

#Create df of schedule
arr = np.array(outer)
dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
print("here")
print(dataframe)
data = dataframe.style.set_properties(align="left")

#Make into excel file
data.to_excel("/Users/Ayaan/Downloads/schedule.xlsx")