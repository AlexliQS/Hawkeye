import cv2
import os
import shutil
import numpy
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'}
import tensorflow as tf
import tensorflow_hub as hub
import sys
import warnings
import time
import ffmpeg
import glob
import threading
import psutil
import subprocess
import tkinter as tk
from jtop import jtop
from tqdm import tqdm
from tensorflow.python.ops.numpy_ops import np_config 
np_config.enable_numpy_behavior()
#/home/orin_nx/Desktop/LTV/HighZoom1.ts
#global ProcessTimeList
#/home/orin_nx/Desktop/LTV/People.mp4


classificationSet = set()
scoreSet = set()

def streamRead():

    #Reads user input for IP connection and captures video from it
    #192.168.1.184:1234
    #192.168.1.197

    ipadd = input("Enter UDP IP: ")
    portnum = input("Enter port number: ")
    fileCap = cv2.VideoCapture('udp://' + ipadd + ':' + portnum)

    #Display whether or not a video can be played. Exit program if nothing is found. 

    if not fileCap.isOpened():
        print("Cannot display video")
        exit(-1)
    print('Video found')
    frames = 0
    
    #Make a local folder to store post-processed frames

    shutil.rmtree("/home/orin_nx/Desktop/Frames")
    os.mkdir("/home/orin_nx/Desktop/Frames")
    time1 = time.perf_counter()
    i=0
    print("Processing video...")

    #Main loop that reads in every frame. Checks if frame contains information. 

    while True:
        ret, frame = fileCap.read()
        if not ret:
            print('frame empty')
            break

        #Resize the frame and perform detection by calling detection function. 

        resized_frame = resizeSquare(frame, 800, 1333)
        odimg = modelDet(resized_frame)
        print(resized_frame.shape, frames)
        i+=1

        #Quit program if 'q' is pressed

        if cv2.waitKey(34)&0XFF == ord('q'):
            #print(processTimeList)
            break
        
        #Write processed frame to local folder

        out_path = "/home/orin_nx/Desktop/Frames"
        frame_name = 'Frame' + str(i) + '.jpg'
        cv2.imwrite(os.path.join(out_path, frame_name), odimg)
        frames+=1

    #Calculate processing time and print, then call vidPlay function which stitches frames in local folder together to form bounding box output video

    time2 = time.perf_counter()
    processTime = time2 - time1
    print(processTime)
    vidPlay()

#Reads input file from local files to perform processing and output bounding box video

def fileRead():

    
    global processTimeList
    global accumulated_scores, classCount
    accumulated_scores = []
    classCount = {}
    #User inputs path to local file to be analyzed, then take video from that file

    pathtovid = input("Paste the path to the video you would like displayed: ")
    pathtovid = pathtovid.strip('"')
    print(pathtovid)
    fileCap = cv2.VideoCapture(pathtovid)

    #Get total frames for future progress bar implementation

    #totFrames=int(fileCap.get(cv2.CAP_PROP_TOTAL_TIME))
    #totFrames = totFrames / 30
    totFrames = 1

    
    #If file cannot be found, exit the program. If found, send a notification

    if not fileCap.isOpened():
        print("Cannot display video")
        exit(-1)
    print('Video found')
    frames = 0
    
    #Delete old directory from previous program runs, and create new one. Start timer to keep track of program running time 

    if os.path.exists("/home/orin_nx/Desktop/Frames"):
        shutil.rmtree("/home/orin_nx/Desktop/Frames")
    os.mkdir("/home/orin_nx/Desktop/Frames")
    time1 = time.perf_counter()
    i=0
    print("Processing video...")

    #Create progress bar (unfinished)

    with tqdm(total=totFrames, desc="Processing Video", unit="frame") as pbar:

        #Main loop that reads in every frame. Checks if frame contains information.

        while True:
            ret, frame = fileCap.read()

            #Check if frame contains information

            if not ret:
                print('frame empty')
                break

            #Resize frame and call modelDet function to overlay bounding boxes and labels

            resized_frame = resizeSquare(frame, 800, 1333)
            odimg = modelDet(resized_frame)
            #print(resized_frame.shape, frames)
            i+=1

            #If 'q' is pressed, quit program

            if cv2.waitKey(34)&0XFF == ord('q'):
                #print(processTimeList)
                break
            
            #Save processed frame to local post-processed frame folder
            out_path = "/home/orin_nx/Desktop/Frames"
            frame_name = 'Frame' + str(i) + '.jpg'
            cv2.imwrite(os.path.join(out_path, frame_name), odimg)
            frames+=1

            #Update progress bar

            pbar.update(1)

        #Calculate and print total time taken for video to be processed
        time2 = time.perf_counter()
        processTime = time2 - time1
        #print(processTime)

        #call vidPlay function to stitch together frames in the folder into final output video

        vidPlay()

    


#Stitches frames in folder together to create output .avi video

def vidPlay():
    global boxes, classes, scores
    global accumulated_scores, classCount
    #Finds all jpg frames in folder and sorts them, then reads them into frame

    image_folder = "/home/orin_nx/Desktop/Frames/"
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x.replace("Frame", "").replace(".jpg", "")))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    #Calls for user input for what to save folder name as, then save output video under that name in folder as well as text files with tensors in them

    nameChoice = input("Enter the name of the folder you would like the output files to be saved as: ")
    os.mkdir('./' + nameChoice)
    video = cv2.VideoWriter('./' + nameChoice + '/' + nameChoice + '.avi', 0, 30, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    print("Video now available on local files.")

    #flattened_scores = [score for sublist in accumulated_scores for score in sublist]
    #scoreMean = sum(flattened_scores) / len(flattened_scores)
    with open('./' + nameChoice + '/' + 'Classifications', 'w') as file:
        file.write(' '.join(map(str, classes)))
        #file.write(' '.join(map(lambda x: x + ',', classes)))
    with open('./' + nameChoice + '/' + 'Scores', 'w') as file:
        file.write(' '.join(map(str, accumulated_scores)))
        #file.write('MEAN CONFIDENCE SCORE: '.join(map(str, scoreMean)))
        #file.write(' '.join(map(lambda x: x + ',', scores)))
    with open('./' + nameChoice + '/' + 'ClassificationMap', 'w') as file:
        # Iterate through the items in the classCount dictionary
        for key, value in classCount.items():
            # Convert the key and value to strings and write to the file
            file.write(f"{str(key)}: {str(value)}\n")
    #Destroy OpenCV Windows

    cv2.destroyAllWindows()
    video.release()


def resizeSquare(im, target_width, target_height):

    # Calculate the scaling factor based on the target dimensions and the larger dimension of the original image

    scale_factor_width = target_width / max(im.shape[1], im.shape[0])
    scale_factor_height = target_height / max(im.shape[0], im.shape[1])
    scale_factor = min(scale_factor_width, scale_factor_height)

    # Calculate the new dimensions and resize

    new_width = int(im.shape[1] * scale_factor)
    new_height = int(im.shape[0] * scale_factor)
    resized_im = cv2.resize(im, (new_width, new_height))

    # Create a canvas of the target dimensions, calculate the position to paste the resized image in the center of the canvas and put resized image onto it

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    ax, ay = (target_width - new_width) // 2, (target_height - new_height) // 2
    canvas[ay:ay + new_height, ax:ax + new_width] = resized_im
    return canvas


def read_label_map(label_map_path):

    #initialize storage variables

    item_id = None
    item_name = None
    items = {}

    #Filter out text file (remove spaces, skip beginnings and ends, extract information)

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

            #Store information into storage variables

            if item_id is not None and item_name is not None:
                items[item_name] = item_id
                item_id = None
                item_name = None
    return items

#Bounding box/labelling frame function

def modelDet(img):
    global processTimeList
    global global_boxes, global_classes, global_scores
    global classificationSet, scoreSet
    global boxes, classes, scores
    global accumulated_scores, classCount
    # read image and preprocess
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get height and width of image and resize tensor to proper model size

    h, w, _ = img.shape
    input_tensor = np.expand_dims(img, 0)
    #time1=time.perf_counter()

    #Run frame through pretrained model

    resp = model(input_tensor)
    accumulated_scores.append(resp['detection_scores'])
    #save current boxes for static box (part of attempted implementation for real-time where only one frame every second was actually processed
    #and bounding box coordinates were saved to be drawn on the following frames after the first during the one-second time interval)

    # global_boxes = resp['detection_boxes'].numpy()
    # global_classes = resp['detection_classes']
    # global_scores = resp['detection_scores'].numpy()

    #time2 = time.perf_counter()

    # classificationSet.add(resp['detection_classes'])
    # scoreSet.add(resp['detection_scores'])

    # iterate over boxes, class_index and score list

    for boxes, classes, scores in zip(resp['detection_boxes'].numpy(), resp['detection_classes'], resp['detection_scores'].numpy()):
        # print(type(resp['detection_boxes']))
        # print(type(resp['detection_classes']))
        # print(type(resp['detection_scores']))
        classes = np.vectorize(convert_to_string)(classes)
        #key = (tuple(classes),)
        
        for box, cls, score in zip(boxes, classes, scores): # iterate over sub values in list
            if score > 0.6: 
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

                #Uncomment following block of code and comment out next block to draw bounding box around anything detected above set confidence level:

                cv2.putText(img, classes[cls], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (128, 0, 128), 4)

                #Uncomment following block of code and comment out previous block to restrict detection to people/cars/trucks:
                # print(classes[cls])
                # if classes[cls]=="1.0":
                #     cv2.putText(img, classes[cls], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                # # draw on image
                #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (128, 0, 128), 4)
                # if classes[cls]=="8.0":
                #     cv2.putText(img, classes[cls], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                # # draw on image
                #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (128, 0, 128), 4)
                # if classes[cls]=="3.0":
                #     cv2.putText(img, classes[cls], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                # # draw on image
                #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (128, 0, 128), 4)
                 
                #Update hashmap for classification object count
                key=cls
                if key in classCount:
                    classCount[key]+=1
                else:
                    classCount[key]=1
    
    # convert back to bgr and save image

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #cv2.imshow("image", img)
    return img
    #processTime = time2 - time1
    #processTimeList.append(processTime)

def convert_to_string(x):
    return str(x)

#Main function

def main():

    global model, label, window
    print("Main Loop Begin")

    #Uncomment line to load in whichtever COCO pre-trained tensorflow model is desired:

    #model = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1")
    #model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    #model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_800x1333/1")
    #model = hub.load("https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_640x640/1")
    #model = hub.load("https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1")
    #model = hub.load("https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1")
    model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_1024x1024/1")
    #model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")
    #model = hub.load("/home/orin_nx/Desktop/")
    #model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d7/1")
    #model = tf.saved_model.load("/home/orin_nx/Downloads/efficientdet_d1_coco17_tpu-32/saved_model")
    print("MODEL LOAD SUCCESFUL")

    #Calls read label map function to filter out label text file

    read_label_map("/home/orin_nx/Desktop/mscoco_label_map.pbtxt")
    
    # window = tk.Tk()
    # label=tk.Label(text="Initializaing")
    
    #Start statistics thread and main application thread for simultaneous running

    stat_thread = threading.Thread(target=stat_cpu)
    stat_thread.start()
    hawk_thread = threading.Thread(target=hawkeye)
    hawk_thread.start()

    #Kills jtop terminal 

    #subprocess.run(['pkill', 'gnome-terminal'])
    #window.mainloop()

#Automatically opens new terminal window displaying jtop for jetson statistics

def stat_cpu():
    commands = ['jtop']
    subprocess.run(['gnome-terminal', '--', 'bash', '-c', ';'.join(commands)])

#Calls for user input in continous loop and runs the rest of the program according to how input is taken

def hawkeye():
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
