import os, glob,pickle
import scipy as sp
import numpy as np
from PIL import Image
import scipy.ndimage as nd

path = '/home/andyc/image/IR/EVENING/'
imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))
im = nd.imread(imlist[0])[:,:,0]
a = [[78,131],[79,141]]
k = []

for i in range(3600): 
    print i
    im0 = nd.imread(imlist[i]).mean(2).astype(float)
    k.append(im0[a[0][1]:a[1][1],a[0][0]:a[1][0]].mean()) 


#    bg[:,:] =bg+im0
#bg = bg/(len(imlist))
#pickle.dump(bg,open("bg_all.pkl","wb"),True) 
