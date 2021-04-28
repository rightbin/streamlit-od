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
  """ image : 오리지날 이미지
  yolo : 욜로 모델
  all_classes : 전체 클래스 이름.

  변환된 이미지 리턴! """

  pimage = process_image(image)

  image_boxes , image_classes , image_scores = yolo.predict(pimage,image.shape)

  if image_boxes is not None :
    box_draw(image, image_boxes , image_scores, image_classes, all_classes)

  
  return image


##욜로 모델 import 하기
from yolo.model.yolo_model import YOLO

##욜로 모델 만들기
yolo = YOLO(0.3, 0.7)

all_classes = get_classes('yolo/data/coco_classes.txt')

image = cv2.imread('yolo/images/test/people.JPG')

result = detect_image(image, yolo , all_classes)

cv2.imshow("result", result)
cv2.imwrite('0.3, 0.7.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()