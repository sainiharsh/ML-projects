# simple program to read an Image 

import cv2

img = cv2.imread('cartoon1.jpg')
gray = cv2.imread('cartoon1.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('cartoon image',img)
cv2.imshow('graycartoon image',gray)

cv2.waitKey(0)
cv2.destroyALLWindows()