# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:04:50 2022

@author: e1805
"""
import cv2
import numpy as np
import os
from yolo_model import YOLO
    
try:
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        cv2.imshow('Input', frame)
        path = "images"
        cv2.imwrite(os.path.join(path , "yeni.jpg"), frame)
        #cv2.imwrite("yeni.jpg", frame)
        
        
        break     
      #  c = cv2.waitKey(1) 
    cap.release()
    cv2.destroyAllWindows()
    
    yolo = YOLO(0.6,0.5)
    file = "data/coco_classes.txt"
    
    with open(file) as f:
        class_name = f.readlines()
        
    all_classes =[c.strip() for c in class_name]
    
    f = "yeni.jpg"
    
    path = "images/"+f
    image = cv2.imread(path)
    cv2.imshow("image", image)
    
    pimage = cv2.resize(image, (416,416))
    pimage = np.array(pimage, dtype ="float32")
    pimage /= 255.0
    pimage = np.expand_dims(pimage, axis = 0) #YOLO 
    
    #yolo
    
    boxes, classes, scores = yolo.predict(pimage,image.shape)
    
    for box, score, cl in zip(boxes, scores, classes):
        
        x,y,w,h = box
        
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = max(0, np.floor(x + w + 0.5).astype(int))
        bottom = max(0, np.floor(y + h + 0.5).astype(int))
        
        
        cv2.rectangle(image, (top,left), (right,bottom),(255,0,0),2)
        cv2.putText(image, "{} {}".format(all_classes[cl],score),(top,left-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1,cv2.LINE_AA)     
    cv2.imshow("yolo", image)   
       
except Exception as e:
    print(e)