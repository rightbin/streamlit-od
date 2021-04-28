import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def run_coco_app():
    
    st.title('SDC(self driving car)')
    st.subheader('자율주행에 사용되는 센서')  
    st.write('👇SDC에 대해 직접 정리한 블로그 링크입니다')
    st.write('https://rightbin.tistory.com/32')

    st.write('자율주행 자동차의 객체 인식 분야가 다른 점은 이질적인 센서를 동시에 사용한다는 점입니다. 동시 사용이 필요한 것은 하나의 센서의 정보만으로는 객체 인식이 이루어지지 않기 때문입니다. 아직까지는 센서의 인식률이 신뢰할 만큼 충분하지 않기 때문에 다양한 센서로 인식률을 높이는 작업이 중요한 단계입니다.')
    
    img = Image.open('sdc1.JPG')
    st.image(img)

    st.write('각광을 받고 있는 센서 중 하나는 LIDAR(라이다)입니다. 라이다는 3차원 정보를 가공해서 제공해주기 때문에 자율주행에 활용되기 용이합니다. 물론 라이다센서가 좋은 것 만은 아닙니다. 라이다와 가장 많이 비교되는 것은 (Radar)레이다센서입니다. 레이다는 2차원으로 제공되기 때문에 위험상황에서 빠르고 신뢰성 있게 돌발적인 상황을 감지할 수 있습니다. 급격한 도로위의 상황을 생각한다면 좀 더 처리가 빠른 레이다센서가 유용해보입니다. 이 외에도 초음파센서가 물체를 detection 하는데 사용이 되곤합니다')
    img2 = Image.open('sdc2.JPG')
    st.image(img2)

    return

    