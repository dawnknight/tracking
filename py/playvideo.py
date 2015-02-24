import numpy as np
import cv2,pdb
import video



video_src = '/home/andyc/Videos/jayst.mp4'
cap = cv2.VideoCapture(video_src)

cam = video.create_capture(video_src)
ret = True

plt.figure(1,[10,12])
axL = plt.subplot(111)
im = plt.imshow(np.zeros([612,512,3]))


if cap.isOpened():                                                                                                                
    rval,frame = cap.read()                                                                                                     
else:                                                                                                                             
    rval = False                                                                                                                
while rval:                                                                                                                     
    rval,frame = cap.read() 
    im.set_data(frame[:,:,::-1])
    plt.draw()


