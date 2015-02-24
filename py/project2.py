import os
import sys,os
import cv2
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.ndimage.filters import median_filter as mf
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import binary_fill_holes as fh
from scipy.ndimage.morphology import binary_dilation as bd
from scipy.ndimage.morphology import binary_erosion as be
from scipy.ndimage.filters import median_filter as mf
import scipy.ndimage as nd



# -- set the video name                                                                                  
dpath = '/home/andyc/Videos/'
fname = 'TLC00000.AVI'
vidname = os.path.join(dpath,fname)
nrow = 480
ncol = 640

fig, ax = plt.subplots(figsize=[10,7.5])
fig.subplots_adjust(0,0,1,1)
ax.axis('off')
im = ax.imshow(np.zeros([nrow,ncol,3]),clim=[0,1],cmap = 'gist_gray')
fig.canvas.draw()






N = 100
trun = N*2+1

buf =[]

# initial buffer

cap = cv2.VideoCapture(vidname)
for ii in range(trun):
    print("creating buffer....at frame {0}\r".format(ii)),
    sys.stdout.flush()
    # -- open the video capture
    rval,frame = cap.read()                                                                              
    buf.append(frame)

print("\n end of buffer....\n")

cap = cv2.VideoCapture(vidname)
for ii in range(1000):
    print("frame {0}\r".format(ii)),
    sys.stdout.flush()
    rval,frame = cap.read()

    BG = np.abs(buf-frame).mean(3).mean(0)
    fg = (BG>40.)

    if ii>=trun:
        buf=np.roll(buf,-1,0)
        buf[-1] = frame
    im.set_data(fg)
    fig.canvas.draw()
