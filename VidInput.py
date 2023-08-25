import cv2
import os
import shutil
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'}
import tensorflow as tf
import tensorflow_hub as hub
import sys
import warnings
import time
import tf2onnx




#/home/orin_nx/Desktop/LTV/HighZoom1.ts
#global ProcessTimeList

def streamRead():

    #Reads user input for IP connection and captures video from it
    #192.168.1.184:1234
    ipadd = input("Enter UDP IP: ")
    portnum = input("Enter port number: ")
    cap = cv2.VideoCapture('udp://' + ipadd + ':' + portnum)

    if not cap.isOpened():
        print("Cannot display video")
        exit(-1)
    print('Video found')
    frames = 0
    '''
    shutil.rmtree("/home/orin_nx/Desktop/Frames")
    os.mkdir("/home/orin_nx/Desktop/Frames")
    '''
    i=0
    while True:
        ret, frame = cap.read()
        if not ret:
            print('frame empty')
            break
        resized_frame = resizeSquare(frame, 800, 1333)
        #modelDet(resized_frame)
        #cv2.imshow('image', resized_frame)
        modelDet(resized_frame)
        print(resized_frame.shape)
        i+=1
        if cv2.waitKey(34)&0XFF == ord('q'):
            print(processTimeList)
            break
        '''
        #if frames >=900:
            #continue
        out_path = "/home/orin_nx/Desktop/Frames"
        frame_name = 'Frame' + str(i) + '.jpg'
        #cv2.imwrite(os.path.join(out_path, frame_name), resized_frame)
        frames+=1
        '''
    cap.release()
    cv2.destroyAllWindows()

def fileRead():
    global processTimeList
    processTimeList = []
    pathtovid = input("Paste the path to the video you would like displayed: ")
    pathtovid = pathtovid.strip('"')
    print(pathtovid)
    fileCap = cv2.VideoCapture(pathtovid)
    # net = cv2.dnn.readNet("/home/orin_nx/Desktop/efficientdet_d0_coco17_tpu-32/saved_model/saved_model.pb")
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    if not fileCap.isOpened():
        print("Cannot display video")
        exit(-1)
    print('Video found')
    frames = 0
    
    shutil.rmtree("/home/orin_nx/Desktop/Frames")
    os.mkdir("/home/orin_nx/Desktop/Frames")
    frameSkip = 0
    i=0
    while True:
        ret, frame = fileCap.read()
        if not ret:
            print('frame empty')
            break
        resized_frame = resizeSquare(frame, 512, 512)
        #net.setInput(resized_frame)
        modelDet(resized_frame)
        #cv2.imshow('image', resized_frame)

        

        # 
        # if frames==0:
        #     odimg = modelDet(resized_frame)
        # elif frameSkip!=2:
        #     odimg = modelStat(resized_frame)
        #     frameSkip+=1
        # else:
        #     odimg = modelDet(resized_frame)
        #     frameSkip = 0
        #     print("DETECGTION")
        # 

        print(resized_frame.shape)
        i+=1
        if cv2.waitKey(34)&0XFF == ord('q'):
            print(processTimeList)
            break
        
        #if frames >=900:
            #continue
        out_path = "/home/orin_nx/Desktop/Frames"
        frame_name = 'Frame' + str(i) + '.jpg'
        #cv2.imwrite(os.path.join(out_path, frame_name), odimg)
        frames+=1
        
    fileCap.release()
    cv2.destroyAllWindows()

def resizeSquare(im, target_width, target_height):
    # Calculate the scaling factor based on the target dimensions and the larger dimension of the original image
    scale_factor_width = target_width / max(im.shape[1], im.shape[0])
    scale_factor_height = target_height / max(im.shape[0], im.shape[1])
    scale_factor = min(scale_factor_width, scale_factor_height)
    new_width = int(im.shape[1] * scale_factor)
    new_height = int(im.shape[0] * scale_factor)
    resized_im = cv2.resize(im, (new_width, new_height))
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    ax, ay = (target_width - new_width) // 2, (target_height - new_height) // 2
    canvas[ay:ay + new_height, ax:ax + new_width] = resized_im

    return canvas




def read_label_map(label_map_path):

  item_id = None
  item_name = None
  items = {}

  with open(label_map_path, "r") as file:
      for line in file:
          line.replace(" ", "")
          if line == "item{":
              pass
          elif line == "}":
              pass
          elif "id" in line:
              item_id = int(line.split(":", 1)[1].strip())
          elif "name" in line:
              item_name = line.split(":", 1)[1].replace("'", "").strip()

          if item_id is not None and item_name is not None:
              items[item_name] = item_id
              item_id = None
              item_name = None
  #print ([i for i in read_label_map("/home/orin_nx/Desktop/mscoco_label_map.pbtxt")])
  return items



def modelDet(img):
    global processTimeList
    global global_boxes, global_classes, global_scores
    # read image and preprocess
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # get height and width of image
    h, w, _ = img.shape

    # net.setInput(img)
    # resp = net.forward()
    input_tensor = np.expand_dims(img, 0)
    #predict from model
    resp = model(input_tensor)
    #save current boxes for static box
    # global_boxes = resp['detection_boxes']
    # global_classes = resp['detection_classes']
    # global_scores = resp['detection_scores']
    # iterate over boxes, class_index and score list
    for boxes, classes, scores in zip(resp['detection_boxes'].numpy(), resp['detection_classes'], resp['detection_scores'].numpy()):
        classes = np.vectorize(convert_to_string)(classes)
        for box, cls, score in zip(boxes, classes, scores): # iterate over sub values in list
            if score > 0.3: 
                ymin = int(box[0] * h)
                xmin = int(box[1] * w)
                ymax = int(box[2] * h)
                xmax = int(box[3] * w)
                # write classname for bounding box
                #cls = tf.cast(cls, tf.int32)
                cls = cls.astype(float)
                cls = cls.astype(int)
                #print(type(cls))
                #print(classes)
                cv2.putText(img, classes[cls], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                # draw on image
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (128, 0, 128), 4)
    # convert back to bgr and save image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("image", img)
    return img
    #processTime = time2 - time1
    #processTimeList.append(processTime)

def modelStat(img):
    global processTimeList
    global global_boxes, global_classes, global_scores
    # read image and preprocess
    #print(global_boxes, '\n' + global_classes, '\n' + global_scores)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #for box, classes, score in zip(global_boxes, global_classes, global_scores):
        #print(f"{box}\t{classes}\t{score}")
    # get height and width of image
    h, w, _ = img.shape

    for boxes, classes, scores in zip(global_boxes.numpy(), global_classes, global_scores.numpy()):
        classes = np.vectorize(convert_to_string)(classes)
        for box, cls, score in zip(boxes, classes, scores): # iterate over sub values in list
            if score > 0.3: 
                ymin = int(box[0] * h)
                xmin = int(box[1] * w)
                ymax = int(box[2] * h)
                xmax = int(box[3] * w)
                # write classname for bounding box
                #cls = tf.cast(cls, tf.int32)

                cls = cls.astype(float)
                cls = cls.astype(int)
                #print(type(cls))
                #print(classes)
                cv2.putText(img, classes[cls], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                # draw on image
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (128, 0, 128), 4)
                print("RECKTANGOW")
    # convert back to bgr and save image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("image", img)
    return img

def convert_to_string(x):
    return str(x)

def main():
    global model
    #model = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1")
    #model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_800x1333/1")
    #model = tf.saved_model.load("/home/orin_nx/Downloads/efficientdet_d1_coco17_tpu-32/saved_model")
    #model = hub.load("/home/orin_nx/Desktop/")
    #model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d7/1")
    #model = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite1/detection/1")
    #model = tf2onnx.convert.from_saved_model("/home/orin_nx/Desktop/")
    print("MODEL LOAD")
    read_label_map("/home/orin_nx/Desktop/mscoco_label_map.pbtxt")
    while True: 
        userchoice = input("Type stream to read from stream. Type local to read from local files: ")
        userchoice = userchoice.upper()
        if userchoice == 'STREAM':
            streamRead()
        if userchoice == 'LOCAL':
            fileRead()
        else:
            print('Input not understood. Try again.')

if __name__ == "__main__":
    main()
