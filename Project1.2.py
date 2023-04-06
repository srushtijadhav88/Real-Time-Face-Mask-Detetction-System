#!/usr/bin/env python
# coding: utf-8

# In[31]:

import streamlit as st
import cv2
from PIL import Image
import numpy as np
import json
from streamlit_lottie import st_lottie
from pygame import mixer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA



with_mask=np.load('with_mask.npy')
without_mask=np.load('without_mask.npy')

with_mask=with_mask.reshape(200, 50 * 50 * 3)
without_mask=without_mask.reshape(200, 50 * 50 * 3)



X = np.r_[with_mask,without_mask]





labels = np.zeros(X.shape[0])




labels[200:] = 1.0




names = { 0:'MASK' , 1:'NO MASK' }

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def mask():

    x_train, x_test, y_train, y_test = train_test_split(X,labels,test_size = 0.25)


    pca = PCA(n_components = 3)
    x_train=pca.fit_transform(x_train)



    x_train, x_test, y_train, y_test = train_test_split(X,labels,test_size = 0.30)




    svm = SVC()
    svm.fit(x_train , y_train)



    #x_test=pca.transform(x_test)
    y_pred=svm.predict(x_test)





    accuracy_score(y_test , y_pred)


    # #BEEP
    #pip install pygame

    mixer.init()
    sound= mixer.Sound(r'Beep Beep.mp3')



    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    # set the rectangle background to white
    rectangle_bgr = (255, 255, 255)

    # make a black image
    img = np.zeros((500, 500))

    # set some text
    text = "Some text in a box!"
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    # set the text start position
    text_offset_x = 10
    text_offset_y = img.shape[0] - 25
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

    font_scale = 1.5
        
    font = cv2.FONT_HERSHEY_PLAIN


    haar_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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
                face = face.reshape(1,-1)
                #face = pca.transform(face)
                pred = svm.predict(face)[0]
                n = names[int(pred)]
                print(n)
                
                
                if (int(pred)==1):
                    
            
                    
                    status = "No Mask"
                    
                    sound.play()
                    

                    x1,y1,w1,h1 = 0,0,175,75
                    #Draw black background rectangle
                    cv2.rectangle(img, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
                    #Add text
                    cv2.putText(img, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                    cv2.putText(img,status,(100, 150), font, 3,(0, 0, 255),2,cv2.LINE_4)

                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255))
                    
              
                    
                    

                
                else:
                    
                    status = "Face Mask"
                    
                    sound.stop()

                    x1,y1,w1,h1 = 0,0,175,75
                    # Draw black background rectangle
                    cv2.rectangle(img, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
                    # Add text
                    cv2.putText(img, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(img,status,(100, 150), font, 3,(0, 255,0),2,cv2.LINE_4)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255,0))
                
                
                
        cv2.imshow('Result',img)
        #27- ASCII of Escape
        if cv2.waitKey(2) == 27:
            break
            
    capture.release()
    cv2.destroyAllWindows()
    





def main_opration():
    lottie_coding = load_lottiefile("facemask.json")
    st_lottie(
    lottie_coding,
    speed=1,
    reverse=False,
    loop=True,
    quality="low", # medium ; high
    height=250,
    width=250,
    key=None,
)
    st.title("★Welcome To Realtime Face Mask Detection★")
    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        cursor: pointer;
            border: 1px solid #3498db;
            background-color: transparent;
            height: 50px;
            margin-left:200px;
            width: 200px;
            color: #3498db;
            font-size: 1.5em;
            box-shadow: 0 6px 6px rgba(0, 0, 0, 0.6);
    }
    </style>""", unsafe_allow_html=True)
    #st.button("Start",  type="primary", disabled=False)
    if st.button("Start"):
        mask()
        
    
    #st.text("_____________________________________________________________________________________________________________")

if __name__ == "__main__":
    main_opration()

# In[ ]:




