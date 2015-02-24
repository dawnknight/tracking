import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.ndimage.morphology as ndm
from scipy.ndimage.filters import median_filter as mf


vid = np.zeros([15350,480,640,3],dtype = uint8)

cap = cv2.VideoCapture('/home/andyc/Videos/TLC00005.AVI')






for ii in range(15350):
    print(ii)
    rval, vid[ii] = cap.read()


mu       = np.zeros(vid[0].shape,dtype=float)

plt.figure(1,figsize=[20,10])
axL = plt.subplot(121)
left = plt.imshow(vid[0][:,:,::-1])
axis('off')
axR = plt.subplot(122)
right = plt.imshow(mu,clim=[0,1],cmap = 'gist_gray',interpolation='nearest')
axis('off')



#vid = vid.astype()
BG = np.zeros([480,640])
trun = 50
start = 15225
end = 15240
buf = double(vid[start-trun/2:start+trun/2+1])
ind = range(trun+1)
ind.pop(trun/2)
th =40

#build tree filter
Tf = (vid[0:100,:,:,1].mean(0)>100) & (vid[0:100,:,:,0].mean(0)<100)
Tf = ndm.binary_closing(Tf,structure=np.ones((4,4)))
Tf = ~ndm.binary_fill_holes(Tf)


for ii in range(start,end):
    print(ii)
    left.set_data(vid[ii][:,:,::-1])
    #pdb.set_trace()
     
    BG[:] = np.abs(buf[ind]-buf[trun/2]).mean(3).mean(0)>40. 
    #fg = np.abs(buf[ind].mean(3)-buf[trun/2].mean(2)).mean(0)>40.
    
    fgo = ndm.binary_opening(BG*Tf)
    fgf = ndm.binary_fill_holes(fgo)

    right.set_data(mf(fgf,5))
    plt.draw()
     
    buf = np.roll(buf,-1,0)
    buf[-1] = vid[ii+trun/2+1]

