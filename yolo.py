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


    if st.button('CLICK!'):
        video_yolo = open('YOLO.mp4', 'rb')
        video4 = video_yolo.read()
        st.video(video4)

        st.text(
                """
import tensorflow as tf
import os
import pathlib
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np 

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#버젼 호환을 위한 코드
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))

  return model

PATH_TO_LABELS = 'C:\\Users\\JB\\Documents\\rightbin\\Tensorflow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

PATH_TO_TEST_IMAGES_DIR = pathlib.Path('C:\\Users\\JB\\Documents\\rightbin\\Tensorflow\\models\\research\\object_detection\\test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))


# DEtection

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'

detection_model = load_model(model_name)

detection_model.signatures['serving_default'].output_dtypes

detection_model.signatures['serving_default'].output_shapes

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = tf.convert_to_tensor(output_dict['detection_masks'], dtype=tf.float32)
    output_dict['detection_boxes'] = tf.convert_to_tensor(output_dict['detection_boxes'], dtype=tf.float32)
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])  
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


def show_inference(model):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = frame
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      np.array(output_dict['detection_boxes']),
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed',None),
      use_normalized_coordinates=True,
      line_thickness=8)

  cv2.imshow("LIVE" , image_np)
  out.write(image_np)

cap = cv2.VideoCapture('tensorflow-object-detection\\data\\videos\\India.mp4')

# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter('output.mp4',0x7634706d , 20.0, (640,480))

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#재생할 파일의 높이 얻기
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#재생할 파일의 프레임 레이트 얻기
fps = cap.get(cv2.CAP_PROP_FPS)

#windows 계열 DIVX
#저장할 비디오 코덱
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fourcc = cv2.VideoWriter_fourcc(*'H264')
#저장할 파일 이름
filename = 'output.mp4'

#파일 stream 생성
out = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))
#filename : 파일 이름
#fourcc : 코덱
#fps : 초당 프레임 수
#width : 넓이
#height : 높이


if cap.isOpened() == False :
    print("Error opening video stream or file")

else :

    while(cap.isOpened()) :

        #사진을 1장씩 가져와서.
        ret , frame = cap.read()

        #제대로 사진 가져왔으면, 화면에 표시!
        if ret == True:

            show_inference(detection_model)


            # 키보드에서 esc키를 누르면 exit하라는 것.
            if cv2.waitKey(25) & 0xFF == 27 :
                break 

        else:
            break


cv2.waitKey(0)
out.release()
cv2.destroyAllWindows()
            """
        )
        
    else:
        st.write(' 👆 버튼을 클릭하시면 구동 영상과 코드를 확인하실수 있습니다.')

    return

    return

    