import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

from ml_app import run_ml_app
from coco import run_coco_app
from yolo import run_yolo_app
from Reference import run_ref_app
def main():
    
    # 사이드바 메뉴
    menu= ['Home','SDC','COCO DATASET','YOLO','Reference']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.title('Object detecion을 이용한 자율주행')
        
        st.write('인공지능의 발달로 인해 자율주행은 한발짝 나아갈 수 있게 되었습니다. 시각 정보에서 객체를 탐지하는 수준으로까지 발달하며 이를 토대로 차량의 이동 경로를 계획하고 차를 구동시키고 있습니다. 이 기술의 핵심은 바로 객체 인식 기술에 있습니다. 오늘은 객체인식 기술에 대해서 알아보고자 합니다. ')
        img = Image.open('data/car1.jpg')
        st.image(img)

        video_file = open('data/OD.mp4', 'rb')
        st.write('Object detection의 예시.')
        video_bytes = video_file.read()
            
        st.video(video_bytes)
        
    elif choice =='SDC':
        run_coco_app()

    elif choice =='COCO DATASET':
        run_ml_app()

    elif choice =='YOLO':
        run_yolo_app()
    
    elif choice =='Reference':
        run_ref_app()

if __name__ == '__main__':
    main()