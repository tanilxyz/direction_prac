#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2


# In[2]:


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# In[3]:


model


# In[3]:


import uuid
import os
import time


# In[4]:


IMAGES_PATH = os.path.join("C:/Users/tanil/yolo/directions", "C:/Users/tanil/yolo/directions/images")
labels = ['right', 'left', 'up', 'down']
number_imgs = 5


# In[5]:


cap = cv2.VideoCapture(0)
while  cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        continue
    results = model(frame)
    if results is None:
        continue
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[6]:


cap = cv2.VideoCapture(0)
for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        cv2.imwrite(imgname, frame)
        cv2.imshow('Image Collection', frame)
        time.sleep(2)
        if cv2.waitKey(10) & 0xFF == ('q'):
            break
cap.release()
cv2.destroyAllWindows()


# In[7]:


get_ipython().system('pip install pyqt5 lxml --upgrade')
get_ipython().system('cd labelImg && pyrcc5 -o libs/resources.py resources.qrc')


# In[8]:


get_ipython().system('python.exe -m pip install --upgrade pip')


# In[9]:


get_ipython().system('pip install pyqt5 lxml --upgrade')
get_ipython().system('cd labelImg && pyrcc5 -o libs/resources.py resources.qrc')


# In[11]:


get_ipython().system('cd yolov5 && python train.py --img 320 --batch 16 --epochs 500 --data database.yaml --weights yolov5s.pt --workers 2')


# In[12]:


model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/last.pt', force_reload=True)


# In[13]:


img = os.path.join("C:Users/tanil/yolo/directions", "C:/Users/tanil/yolo/directions/images", "C:/Users/tanil/yolo/directions/images/down.979f895f-385b-11ef-bfba-e8d5ab0add49.jpg")


# In[14]:


results = model(img)


# In[15]:


results.print()


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.squeeze(results.render()))
plt.show


# In[18]:


cap = cv2.VideoCapture(0)
while  cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        continue
    results = model(frame)
    if results is None:
        continue
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

