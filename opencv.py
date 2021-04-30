import streamlit as st
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def run_opencv_app():
    
    st.title('OPEN CV')  

    img1 = Image.open('opencv.png')
    st.write('')
    st.image(img1, width =300)
    
    st.write('OpenCV는 영상관련 처리할 수 있는 여러가지 API와 툴을 제공하고 있습니다. interactive art나 image stitching, 공장의 불량품 검출 시스템 및 로봇공학 등 다양한 영상처리 시스템에서 이용되고 있습니다.')
    
    st.write('이 section에서는 업로드한 사진을 바탕으로 OpenCV를 간접적으로 체험해보고자 합니다.')
    st.subheader('✋사진을 업로드해주세요.')
  
    uploaded_file = st.file_uploader("Choose a image file", type=["png","jpg",'jpeg'])

    
    obj = st.radio('choose the action',['HueHistogram','desaturation','brightness','darkness','gamma','gaussian(3X3)','gaussian(7X7)','sharpen','canny'])

    if uploaded_file is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Now do something with the image! For example, let's display it:
        st.write('👇원본사진')
        st.image(opencv_image, channels="BGR")

        opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_RGB2BGR)

        if obj == 'HueHistogram' :  
            hsv_img = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2HSV)
            st.write('👇처리 후 사진')
            st.image(hsv_img)
        elif obj == 'desaturation' :
            saturationScale = 0.01
            hsvImage = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2HSV)
            hsvImage = np.float32(hsvImage)
            H , S, V = cv2.split(hsvImage)
            S = np.clip(S * saturationScale , 0,255)
            hsvImage = cv2.merge([H,S,V])
            hsvImage = np.uint8(hsvImage)
            imgBgr = cv2.cvtColor(hsvImage,cv2.COLOR_HSV2BGR)
            st.write('처리 후 사진')
            st.image(imgBgr)



        elif obj == 'brightness' :
            beta = 100
            ycbImage = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2YCrCb)
            ycbImage = np.float32(ycbImage)
            Ychannel,Cr,Cb = cv2.split(ycbImage)
            Ychannel = np.clip(Ychannel + beta , 0 ,255)
            ycbImage = cv2.merge([Ychannel , Cr , Cb])
            ycbImage = np.uint8(ycbImage)
            ycbImage = cv2.cvtColor(ycbImage, cv2.COLOR_YCrCb2BGR)
            st.write('처리 후 사진')
            st.image(ycbImage)

        elif obj == 'darkness' :
            beta = -100
            ycbImage = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2YCrCb)
            ycbImage = np.float32(ycbImage)
            Ychannel,Cr,Cb = cv2.split(ycbImage)
            Ychannel = np.clip(Ychannel + beta , 0 ,255)
            ycbImage = cv2.merge([Ychannel , Cr , Cb])
            ycbImage = np.uint8(ycbImage)
            ycbImage = cv2.cvtColor(ycbImage, cv2.COLOR_YCrCb2BGR)
            st.write('처리 후 사진')
            st.image(ycbImage)

        elif obj == 'gamma' :
            gamma = 1.5

            fullRange = np.arange(0,256)

            lookupTable = np.uint8(255 * np.power((fullRange/255.0) , gamma))

            output = cv2.LUT(opencv_image, lookupTable)

            st.write('처리 후 사진')
            st.image(output)

        elif obj == 'gaussian(3X3)' :
            dst1 = cv2.GaussianBlur(opencv_image,(5,5),0)
            st.write('처리 후 사진')
            st.image(dst1)
        
        elif obj == 'gaussian(7X7)' :
            dst1 = cv2.GaussianBlur(opencv_image,(25,25),0)
            st.write('처리 후 사진')
            st.image(dst1)
        
        elif obj == 'sharpen':
            sharpen = np.array(
                                    [
                                        [0,-1,0],
                                        [-1,5,-1],
                                        [0,-1,0]
                                    ] ,dtype = 'int')
            
            result = cv2.filter2D(opencv_image,-1,sharpen)      
            
            st.write('처리 후 사진')
            st.image(result)

        elif obj == 'canny':
            threshold_1 =150 # high : 0~255
            threshold_2 = 100 # low :200이상

            result = cv2.Canny(opencv_image,threshold_1,threshold_2)
            st.write('처리 후 사진')
            st.image(result)


    else :
        "사진을 업로드 해주세요"

    return

    