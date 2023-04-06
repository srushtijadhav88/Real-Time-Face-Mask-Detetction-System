#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install opencv-python')


# In[3]:


import cv2


# In[4]:


img=cv2.imread('image1.jpg')


# In[5]:


img.shape


# In[6]:


img


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


plt.imshow(img)


# In[9]:


while True:
    cv2.imshow('Result',img)
    #27- ASCII of Escape
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()


# In[10]:


haar_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[11]:


haar_data.detectMultiScale(img)


# In[12]:


#cv2.rectangle(img,(x,y),(w,h),(b,g,r),border_thickness)


# In[13]:


while True:
    faces=haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
    cv2.imshow('Result',img)
    #27- ASCII of Escape
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()


# In[17]:


import numpy as np


# In[15]:


capture=cv2.VideoCapture(0)
data = []
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face = img[y:y+h , x:x+w , :]
            face = cv2.resize(face,(50,50))
            print(len(data))
            if len(data)<400:
                data.append(face)
    cv2.imshow('Result',img)
    #27- ASCII of Escape
    if cv2.waitKey(2) == 27 or len(data) >= 200:
        break
        
capture.release()
cv2.destroyAllWindows()


# In[18]:


np.save('without_mask.npy',data)


# In[19]:


capture=cv2.VideoCapture(0)
data = []
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face = img[y:y+h , x:x+w , :]
            face = cv2.resize(face,(50,50))
            print(len(data))
            if len(data)<400:
                data.append(face)
    cv2.imshow('Result',img)
    #27- ASCII of Escape
    if cv2.waitKey(2) == 27 or len(data) >= 200:
        break
        
capture.release()
cv2.destroyAllWindows()


# In[20]:


np.save('with_mask.npy',data)


# In[23]:


plt.imshow(data[0])


# In[ ]:




