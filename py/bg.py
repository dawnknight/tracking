import os, glob,pickle
import scipy as sp
import numpy as np
from PIL import Image
import scipy.ndimage as nd

path = '/home/andyc/image/IR/EVENING/'
imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))
im = nd.imread(imlist[0])[:,:,0]
bg = np.zeros(im.shape)
w  = np.ones(im.shape)*10**(-6)
one = np.ones(im.shape)
Th =6
for i in range(3600):#len(imlist)-1):
      
    tmp = np.zeros(im.shape)
    wtmp = np.zeros(im.shape)
    print i 
    im0 = nd.imread(imlist[i]).mean(2).astype(float)
    im1 = nd.imread(imlist[i+1]).mean(2).astype(float)

    diff = abs(im1-im0)

    tmp[diff<Th] = im0[diff<Th]     

    wtmp[diff<Th]=one[diff<Th] 

    w = w + wtmp

    bg[:,:] =bg+tmp

bg = bg/w  
imshow(bg,cmap ='gray')

pickle.dump(bg,open("bg_all.pkl","wb"),True)
