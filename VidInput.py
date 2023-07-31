import cv2
import os
import shutil
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import sys
import warnings


#/home/orin_nx/Desktop/LTV/HighZoom.ts

def streamRead():

    #Reads user input for IP connection and captures video from it
    #192.168.1.184:1234
    ipadd = input("Enter UDP IP: ")
    portnum = input("Enter port number: ")
    cap = cv2.VideoCapture('udp://' + ipadd + ':' + portnum)

    if not cap.isOpened():
        print('No stream detected. Program exit')
        exit(-1)
    
    print('Stream found')

    i=0

    #Creates folder for frames to be stored in 
    frames = 0
    shutil.rmtree("/home/orin_nx/Desktop/Frames")
    os.mkdir("/home/orin_nx/Desktop/Frames")
    while True:
        ret, frame = cap.read()
        #cv2.imshow('image', frame)
        
        if not ret:
            print('frame empty')
            break

        cv2.imshow('image', frame)
        i+=1
        #cv2.waitKey(0) 
        if cv2.waitKey(34)&0XFF == ord('q'):
            break
        if frames >= 900:
            continue
        out_path = "/home/orin_nx/Desktop/Frames"
        frame_name = 'Frame' + str(i) + '.jpg'
        cv2.imwrite(os.path.join(out_path, frame_name), frame)
        frames+=1

    #destroy window
    cap.release()
    cv2.destroyAllWindows()

def fileRead():
    pathtovid = input("Paste the path to the video you would like displayed: ")
    pathtovid = pathtovid.strip('"')
    print(pathtovid)
    fileCap = cv2.VideoCapture(pathtovid)

    #fileCap = cv2.VideoCapture(r"C:\Users\Addic\Projh\Test Videos\HighZoom.ts")
    #fileCap = cv2.VideoCapture(0)
    if not fileCap.isOpened():
        print("Cannot display video")
        exit(-1)
    print('Video found')
    frames = 0
    shutil.rmtree("/home/orin_nx/Desktop/Frames")
    os.mkdir("/home/orin_nx/Desktop/Frames")
    i=0
    while True:
        ret, frame = fileCap.read()
        if not ret:
            print('frame empty')
            break
        resized_frame = resizeSquare(frame)
        #modelDet(resized_frame)
        #cv2.imshow('image', resized_frame)
        modelDet(resized_frame)
        print(resized_frame.shape)
        i+=1
        if cv2.waitKey(34)&0XFF == ord('q'):
            break
        #if frames >=900:
            #continue
        out_path = "/home/orin_nx/Desktop/Frames"
        frame_name = 'Frame' + str(i) + '.jpg'
        cv2.imwrite(os.path.join(out_path, frame_name), resized_frame)
        frames+=1
    fileCap.release()
    cv2.destroyAllWindows()

def resizeSquare(im):
  s = max(im.shape[0:2])
  f = np.zeros((s, s, 3), np.uint8)
  ax, ay = (s - im.shape[1])//2, (s - im.shape[0])//2
  f[ay:im.shape[0]+ay, ax:ax+im.shape[1]] = im
  #f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
  #rgb_tensor = tf.convert_to_tensor(f, dtype = tf.uint8)
  return f


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

  return items

  print ([i for i in read_label_map("/content/mscoco_label_map.pbtxt")])

def modelDet(img):

  # read image and preprocess

  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # get height and width of image
  h, w, _ = img.shape

  input_tensor = np.expand_dims(img, 0)

  # predict from model
  resp = model(input_tensor)

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
              '''
              classes[cls] = tf.cast(classes[cls], tf.int32)
              ymin = tf.cast(ymin, tf.int32)
              ymax = tf.cast(ymax, tf.int32)
              xmin = tf.cast(xmin, tf.int32)
              xmax = tf.cast(xmax, tf.int32)
              '''

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

def convert_to_string(x):
    return str(x)

def main():
    global model 
    warnings.filterwarnings("ignore", message="Importing a function (__inference___call___32344) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.", category=UserWarning)
    warnings.filterwarnings("ignore", message="Importing a function (__inference_EfficientDet-D0_layer_call_and_return_conditional_losses_97451) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.", category=UserWarning)
    warnings.filterwarnings("ignore", message="Importing a function (__inference_bifpn_layer_call_and_return_conditional_losses_77595) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.", category=UserWarning)
    warnings.filterwarnings("ignore", message="Importing a function (__inference_EfficientDet-D0_layer_call_and_return_conditional_losses_103456) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.", category=UserWarning)
    warnings.filterwarnings("ignore", message="Importing a function (__inference_EfficientDet-D0_layer_call_and_return_conditional_losses_93843) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.", category=UserWarning)
    warnings.filterwarnings("ignore", message="Importing a function (__inference_EfficientDet-D0_layer_call_and_return_conditional_losses_107064) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.", category=UserWarning)
    warnings.filterwarnings("ignore", message="Importing a function (__inference_bifpn_layer_call_and_return_conditional_losses_75975) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.", category=UserWarning)

    model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")
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
