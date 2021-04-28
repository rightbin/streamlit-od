import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import cv2 as cv
import tempfile


def run_ml_app():
    
    st.title('COCO DATASET & SSD')
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

    if st.button('CLICK!'):
        video_ssd = open('SSD.mp4', 'rb')
        video3 = video_ssd.read()
        st.video(video3)

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
            import time

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
                base_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
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

            # http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz

            model_name = 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8'

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


                cap = cv2.VideoCapture('C:\\Users\\JB\\Documents\\rightbin\\Tensorflow\\tensorflow-object-detection\\data\\videos\\video.mp4')

                if cap.isOpened() == False :
                    print("Error opening video stream or file")

                else :

                    while(cap.isOpened()) :

                        #사진을 1장씩 가져와서.
                        ret , frame = cap.read()

                        #제대로 사진 가져왔으면, 화면에 표시!
                        if ret == True:
                            start_time = time.time()
                            show_inference(detection_model)
                            # 키보드에서 esc키를 누르면 exit하라는 것.
                            end_time = time.time()
                            print(end_time -start_time)
                            if cv2.waitKey(25) & 0xFF == 27 :
                                break 

                        else:
                            break

                # image_path = 'C:\\Users\\JB\\Documents\\rightbin\\Tensorflow\\tensorflow-object-detection\\data\\images'

            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
        )
        
    else:
        st.write(' 👆 버튼을 클릭하시면 구동 영상과 코드를 확인하실수 있습니다.')

    return
    