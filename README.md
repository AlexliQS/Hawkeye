# **Hawkeye - Summer '23 Intern Project by Alex Li**
---

Hawkeye is a video object detection and labeling model built to run on the NVIDIA Jetson Orin NX board and any COCO-pretrained Tensorflow 2 models. It takes either an input UDP stream or a video from local files and produces various object detection statistics as well as a bounding box video output after processing (complete and runs smoothly) and hopefully in real-time in the future(in progress/incomplete) while monitoring the Jetson's system usage. The primary use for this software is to benchmark typical object detection performance by using the various pretrained models, and to output statistics to come to a more comfortable conclusion about the performance. It can be an internal tool that acts as a reference for when in-house development begins for the company's own object detection model. Additionally, for in-house development, the progress on the real-time model can be used as groundwork for understanding or building the company's own program. [Github Link](https://github.com/AlexliQS/Hawkeye/blob/main/ImgVidReconstruct.py)

![F](./Images/F.png)

### System Requirements

- NVIDIA Jetson Orin NX
- cuDNN 8.6.0
- CUDA 11.4.19
- NVIDIA L4T 35.3.1
- NVIDIA Jetpack 5.1.1
- Tensorflow 2.11
- OpenCV 4.5.4
- Python 3.8
- Ubuntu 20.04
- Jetson-Stats 4.2.3

## Image Reconstruction (Output after complete processing)

Run the program with the terminal command:

`python ImgVidReconstruct.py`

User should see the following as the program and model load in; the warnings are normal:

![ProgramStart](./Images/ProgramStart.png)

After the model loads in, the user should see a success message: `MODEL LOADED SUCCESFULLY`. Additionally, a terminal window running jtop (Real-time system monitoring tool for the NVIDIA Jetson) will pop out, showing statistics for how the computer is handling the program, which looks like the following image:

![jtop](./Images/jtop.png)



The user will be prompted to indicate whether they would like to read the input video from a UDP stream or simply from local files. Local files give better quality and better performance. If the desired input is from a stream, ensure the stream is running properly. Upon selecting **stream**, user will enter in the IP address and port number to create the connection, and it will automatically start processing the video given that the stream is working normally.  If **local** is chosen instead, user will be asked to paste in the path to the video (or user can type in the file name and type if it is in the same folder as the program), after which the program will start processing. The process shown will be using the **local** approach, but the **stream** process is the same after entering in IP and port number.  

After an initial delay, the video will begin processing frame by frame, and the output will look similar to below (the red error messages are normal and can be ignored):

![G](./Images/Alfalfa.png)

Once it is done, the user will be asked to name the output folder. The folder will contain the output video with the same name as the folder (.avi formatted), as well as separate text files for each of the various tensors produced by the program and statistics relating to them. Additionally, the program creates a 'Frames' folder on the desktop that contains every frame from the video with the bounding boxes drawn around detected objects. I named the folder MDDemo and it can be found under the `cwd`, in this case, Home:

![e](./Images/folder.png)

After navigating into the folder, the user should see the output video as a **.avi file**, and then 4 text files. The **ClassificationMap** contains a hashmap of all the detection classes as keys and the total count of how many were detected (after non-max suppression, or NMS) for the duration of the entire input video. Here, we can see all the erroneous detections that occured. **Classifications** shows what was found on the last frame just as reference. **DetectionCountMap** is another hashmap which shows the count of how many objects were detected during every frame of the video, where the frame number is the key. The model's output for this value, however, seems to include all proposed detections pre-NMS, so it should always be an extremely high number, or the maximum which is 300. **Scores** contains a tensor for every processed frame that shows every confidence score of proposed detections, ordered from greatest to least. Only the ones above the threshold have bounding boxes on them, which is pre-set at 0.6. 

![rnc](./Images/rncnc.png)
![pl](./Images/insidefolder.png)

At this point, the program is ready for the next video and will ask the user to choose between stream or local input, and from there on the program runs the same as above. 

## Notable Weaknesses


## Learning Objectives

Speaking now from an intern perspective, I experienced an insane amount of growth in many areas over the past few months. I was admittedly very intimidated upon initially hearing about what my project would entail, especially since I would not be told much about how to go about completing the set requirements. However, I knew that doing my internship in this manner would allow me to learn as much as possible within the time frame. I knew I was interested in computer vision because of coursework and a previous experience at a manufacturer for autonomous vehicle cameras, but in terms of actual application in the world of object detection specifically, I did not know anything. 

I started off the first few weeks by researching general knowledge about object detection. This included learning all the frameworks (YOLO, RCNN, Centernet, SSD's, etc) and general knowledge on how the algorithms worked. I then dove into the basics of machine learning, neural networks, GPU's, and the overall state of AGI, as well as how to use OpenCV. Next, I researched the entirety of the tech stack that would make the program work, which meant researching the softwares that came with or were necessary for compatibility with the NVIDIA Jetson. I had to look into TensorFlow vs PyTorch, and ended up choosing TensorFlow. I became familiar with deep learning and computer vision terminology and developed a strong intuition as to how they worked within a larger ecosystem, as well as its relations to each other. 

After getting the green light from the team, I set off into writing the application after the Orin board came in. The largest struggle I had completing this project started here, which was installing everything I needed properly onto the Ubuntu system. Getting the version compatibility down as well as troubleshooting when something didn't install properly took a hefty amount of time. I would install one of the programs just to realize the version was wrong, and then run into issues completely uninstalling it, or I would reinstall just to realize another software is now imcompatible and I now had to reinstall that software which would cause yet another incompatibility, and was in a way stuck in this cycle. I had to ditch a few softwares because I realized that compatibility was simply impossible. 

However, I learned how to read technical documentation, how to troubleshoot/debug, became very comfortable with navigating Linux, how to build from source and what that meant, how to use virtual environments, choosing and connecting compatibility matrices, and more. I even broke the Jetson at one point, and had to have help to get it re-flashed. I went back and forth from running the program on Google Colab to my XPS to the Orin, depending on if I needed features that weren't installed yet properly on the Orin. This was ultimately quite inefficient of me, and I could have utilized time better to focus on installations (with the exception of when the Orin was running and unusable because of installing a large file); however on the positive side, I was able to learn even more because of this expansion (Colab notebooks, running programs on the cloud, etc). 

I followed countless online tutorials for the listed items in my tech stack, and also became much more proficient with Python and the many libraries I used. I ended up getting rid of many of them for various reason (better alternate, not necessary, too much overhead, incompatible) but still ultimately gained far more exposure than I would have in any university course due to the nature of me needing to define the entire stack and make it work. I got a grasp of the sheer amount of libraries, large or small, that could be used creatively to make any idea work, for example, Tkinter, tqdm, psutil, threading, warnings, pandas, numpy, etc. I never realized the extent of their power, as this is my first time really diving into software as well since I focused on hadware in the past. I also had to review networking and learn VLC in order to make the UDP input work.  Overall, my general software intuition has gotten *significantly* better and the value of the knowledge I have gained is invaluable.

## Experience

I have no complaints as to how my internship went, I thought it was smooth. Seeing that the company is growing quickly however, I do believe there are a few improvements that would be possible if the available resources and infrastructure setup will drastically increase. From a personal perspective, I have learned so much more from this experience than I would have if things were "set up" for me from the start, and talking to some of my peers that work at very large and established corporations in a wide variety of industries makes me really realize that. It was great for my growth. On the flip side, from the company's perspective, I believe I could have been of much better use directly to QS had things been more streamlined. I spent a lot of time on things such as researching and installing the tech stack (which again I don't mind), but had I been given strong pointers or set requirements on what to use, I would have been able to finish the project much faster, which would allow me to aid the company in another way with the leftover time (making large improvements such as real-time processing, starting another project, aiding full-time employees in miscellaneous duties, etc). Essentially, it is a trade-off between how deep the intern's learning experience is versus the intern's immediate value to QS, which is something I think to be worthy of a deep discussion when next summer's intern program is being established. 

Having co-interns would also be a good experience for peer learning/revision and for mitigating imposter syndrome, which, on a side note in case it sounds concerning, did not have too large of an impact on me since I could rationalize with YOE, degree level, and my major-switching background, but having someone with a similar experience level could help with confidence and speaking up. As for culture, I personally find the company culture to be exceptionally positive and uplifting, with gifted engineers always willing to try to help with any issue. The nature of the flexibility with starting and ending time, as well as the lunch breaks, makes it so that insignificant things of that nature won't cause unecessary stress which is amazing. I was also provided with state of the art technology (high-spec XPS, NVIDIA Jetson Orin, monitor, adjustable height desk, comfortable chair, etc etc) that further alleviated potential unecessary stress and allowed me to focus. 

