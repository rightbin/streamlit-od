import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def run_yolo_app():
    
    st.title('YOLO(You Only Look Once)')
    st.subheader('YOLO모델')  

    st.write('Object detection 분야에서 쓰이는 모델로는, Faster-RCNN, MobileNet, SSD 등이 있습니다. 이 곳에서는 YOLO모델을 학습해보며 object-threshold와 nms-threshold에 대해 알아보도록 하겠습니다.')

    st.write('YOYO 모델에는 두개의 파라미터를 설정할 수 있습니다. 첫째는 object-threshold 두번째는 nms-threshold입니다.')
    st.subheader('object-threshold')
    st.write('object-threshold은 수많은 detection으로 나온 확률치를 기준으로 삼습니다. 예를 들어 0.3이라고 지정한다면 확률이 0.3이 안되는 box는 탈락이 됩니다.')
    st.subheader('nms-threshold')
    st.write('nms-threshold은 box들의 갯수를 가지고 판단합니다. 예를 들어 person이라고 판단한 box가 10가 있다고 가정하고 nms-threshold를 0.7로 선택한다면 상위 7개의 box를 모두 사용하는 것입니다.')
    st.write('라디오박스를 이용해 특정한 값을 지정해두었습니다. 비교해보세요.')

    obj = st.radio('obj_threshold',[0.05,0.1,0.3])
    if obj == 0.05 :
        st.write("obj_threshold : {} .".format(obj))
    elif obj == 0.1 :
        st.write("obj_threshold : {} .".format(obj))
    elif obj == 0.3 :
        st.write("obj_threshold : {} .".format(obj))

    box = st.radio('box_threshold',[0.3,0.5,0.7])
    if box == 0.05 :
        st.write("box_threshold : {} .".format(box))
    elif box == 0.1 :
        st.write("box_threshold : {} .".format(box))
    elif box == 0.3 :
        st.write("box_threshold : {} .".format(box))

    if obj == 0.05 and box == 0.3:
        img = Image.open('0.05, 0.3.jpg')
        st.image(img)
    elif obj == 0.05 and box == 0.5:
        img = Image.open('0.05, 0.5.jpg')
        st.image(img)
    elif obj == 0.05 and box == 0.7:
        img = Image.open('0.05, 0.7.jpg')
        st.image(img)
    elif obj == 0.1 and box == 0.3:
        img = Image.open('0.1, 0.3.jpg')
        st.image(img)
    elif obj == 0.1 and box == 0.5:
        img = Image.open('0.1, 0.5.jpg')
        st.image(img)
    elif obj == 0.1 and box == 0.7:
        img = Image.open('0.1, 0.7.jpg')
        st.image(img)
    elif obj == 0.3 and box == 0.3:
        img = Image.open('0.3, 0.3.jpg')
        st.image(img)
    elif obj == 0.3 and box == 0.5:
        img = Image.open('0.3, 0.5.jpg')
        st.image(img)
    elif obj == 0.3 and box == 0.7:
        img = Image.open('0.3, 0.7.jpg')
        st.image(img)

    return

    