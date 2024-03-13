import cv2
import numpy as np


def img_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))

    final = cv2.cvtColor(limg,cv2.COLOR_LAB2BGR)

    return final


def image_clearing(image):
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    
    image_sharp = cv2.filter2D(image, -1, kernel)

    return image_sharp

def image_bilateral(image):
    dst = cv2.bilateralFilter(image, -1, sigmaColor=20, sigmaSpace=5) 
    # sigmaColor: 색 공간에서 필터의 표준편차(너무크면 엣시 부분을 구분못함) sigmaSpace: 좌표공간에서 필터의 표준편차(값이 클수록 연산시간이 길어짐)

    return dst

def image_histo(image):
    # image = cv2.imread(img_url)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channel = cv2.split(ycrcb)
    channel = list(channel)

    channel[0] = cv2.equalizeHist(channel[0])

    dst_ycrcb = cv2.merge(channel)
    dst = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)


    return dst