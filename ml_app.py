import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import cv2 as cv
import tempfile


def run_ml_app():
    
    st.title('COCO dataset')
    st.subheader('자율주행 관련 데이터 셋')  

    st.write('COCO datset은 수많은 이미지를 저장하고 있으며 이를 기반으로 한 인공지능 모델을 제공하고 있습니다. 그 중에서 저는 ssd_mobilenet_v1_coco_2017_11_17 을 이용하여 동영상을 detection 해보았습니다.')

    st.write('sdd_mobilenet이 가지고 있는 class는 다음과 같습니다.')

    img = Image.open('coco.JPG')
    st.image(img)

    st.write('총 99개의 클래스를 사용합니다. 이 안에는 사람,자전거,자동차,오토바이,핸드백,넥타이,칫솔 등이 학습되어 있습니다.')

    st.write('streamlit에 모델링 구현은 EC2 (free tear사용)로는 구현이 어려웠습니다. local에서 진행한 예측모델을 자료로 첨부하고자 합니다.')

    st.write('다음의 비디오는 Pixabay에서 다운로드 받았습니다.')

    video_file = open('India.mp4', 'rb')
    video_bytes = video_file.read()
        
    st.video(video_bytes)

    st.write('다음은 odject-detection을 한 후의 영상입니다.')

    video_result = open('output.mp4','rb')

    video2 = video_result.read()
            
    st.video(video2)

    return