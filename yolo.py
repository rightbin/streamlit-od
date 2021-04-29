import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def run_yolo_app():
    
    st.title('YOLO(You Only Look Once)')
    st.subheader('YOLO모델')  

    img1 = Image.open('yolo.JPG')
    st.image(img1)
    
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

    st.subheader('구현 영상 & 코드')
    
    video_yolo = open('YOLO.mp4', 'rb')
    video4 = video_yolo.read()
    st.video(video4)

    st.text(
                """
import os
import time
import cv2
import numpy as np 

def process_image(img) :

  image_org = cv2.resize(img , (416,416), interpolation = cv2.INTER_CUBIC)
  image_org = np.array(image_org , dtype = 'float32')
  image_org = image_org/255.0
  image_org =  np.expand_dims(image_org , axis=0)

  return image_org

def get_classes(file) :
    with open(file) as f:
        name_of_class = f.readlines()

    name_of_class = [ class_name.strip()   for class_name in name_of_class ]

    return name_of_class

def box_draw(image, boxes, scores, classes, all_classes):

  for box, score, cl in zip(boxes, scores, classes):
      x, y, w, h = box

      top = max(0, np.floor(x + 0.5).astype(int))
      left = max(0, np.floor(y + 0.5).astype(int))
      right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
      bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

      cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
      cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                  (top, left - 6),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.6, (0, 0, 255), 1,
                  cv2.LINE_AA)

      print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
      print('box coordinate x,y,w,h: {0}'.format(box))

  print()


def detect_image(image, yolo, all_classes) : 
  

  pimage = process_image(image)

  image_boxes , image_classes , image_scores = yolo.predict(pimage,image.shape)

  if image_boxes is not None :
    box_draw(image, image_boxes , image_scores, image_classes, all_classes)

  
  return image


##욜로 모델 import 하기
from yolo_model import YOLO

##욜로 모델 만들기
yolo = YOLO(0.3, 0.7)

all_classes = get_classes('yolo/data/coco_classes.txt')

image = cv2.imread('yolo/images/test/people.JPG')

result = detect_image(image, yolo , all_classes)

cv2.imshow("result", result)
cv2.imwrite('0.3, 0.7.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
            """
        )
        
    return

    return

    