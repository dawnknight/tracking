


####### read by img ########### 
import os, glob,pickle
import scipy as sp
import numpy as np
from PIL import Image
import scipy.ndimage as nd

path = '/home/andyc/image/jpg/jayst'
imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))
im = nd.imread(imlist[0])
bg = np.zeros(im.shape)
#a = [[78,131],[79,141]]
#k = []

#for i in range(len(imlist)):
for i in range(200): 
    print i
    im0 = nd.imread(imlist[i]).astype(float)
    #k.append(im0[a[0][1]:a[1][1],a[0][0]:a[1][0]].mean()) 


    bg[:] =bg+im0
bg = bg/(i+1)



imshow(uint8(bg))
pickle.dump(bg,open("./BG/2015-jayst.pkl","wb"),True) 

'''
#####  read by video #####
import cv2,pickle
import numpy as np



print('reading video...')
cap = cv2.VideoCapture('/home/andyc/test video/TLC00005.AVI')
vid = []
if cap.isOpened():
    rval,frame = cap.read()
else:
    rval = False
while rval:
    rval,frame = cap.read()
    if rval:
        vid.append(frame)
print('done reading video...')

start =15000
end   =16000

bg = np.zeros(vid[0].shape)

#for i in range(start,end):
#    print i
#    bg[:] =bg+vid[i] 

#bg = bg/(end-start)

bg[:] = vid[start:end].mean(0)

imshow(uint8(bg)) 
pickle.dump(bg[:,:,::-1],open("./BG/TLC0005(15000-16000).pkl","wb"),True)  
'''
