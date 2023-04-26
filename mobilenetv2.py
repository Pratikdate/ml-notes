import tensorflow as tf 
from keras.applications.mobilenet_v2 import decode_predictions,preprocess_input
import numpy as np 
from keras.preprocessing import image
import cv2

model=tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224,224,3),weights='mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5')



cap=cv2.VideoCapture(0)



while True:
    succ,img=cap.read()
    data = np.empty((1, 224, 224, 3))


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    x,y,w,h = cv2.boundingRect(thresh)
    ROI = img[y:y+h, x:x+w]



    img=cv2.resize(ROI,(224,224))
    cv2.imshow('img',img)
    data[0]=img
    img=data
    
    img=preprocess_input(img)
    p=model.predict(img)
    m=np.max(model.predict(img))
    if m>0.40:
        pre=decode_predictions(p,top=1)
        print(pre)
    
    if cv2.waitKey(1)== ord('q'):
        break