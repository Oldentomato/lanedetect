from tensorflow import keras
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import PIL
from PIL import ImageOps
from PIL import Image

import cv2




def get_mask(in_img,model_size,outsize):

    model_path = "models/pretrained_models/road_segmentation_160_160.h5"    # ex:"models/pretrained_models/road_segmentation_160_160.h5"
    model = keras.models.load_model(model_path, compile=False)

    img= cv2.resize(in_img, model_size)
    data=np.expand_dims(img, axis=0)
    #print(data.shape)
    #print("s:"+str(time.time()))
    val_pred = model.predict(data)
    #print("e:"+str(time.time()))
    mask = np.argmax(val_pred[0], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    #print(mask.shape)
    #print("done")
    result_img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    result_img=img_to_array(result_img)
    
    result_img = cv2.resize(np.array(result_img),model_size)
    result_img = np.repeat(result_img[:,:,np.newaxis],3,-1) # 1채널에서 3채널로 늘리기
    result_img = result_img.astype(np.uint8)

    dst = cv2.bitwise_and(img, result_img)

    # cv2.resize(img,outsize)
    return cv2.resize(in_img,outsize), cv2.resize(dst, outsize)




def run(image_path):
    img_size=(160, 160)
    out_size=(600,400)

    get_image = cv2.imread(image_path)
    img,mask=get_mask(get_image,img_size,out_size)  # Note that the model only sees inputs at 160x160.

    return mask
    # cv2.imshow('Input frame',img)
    # cv2.imshow('predicted mask',mask)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()