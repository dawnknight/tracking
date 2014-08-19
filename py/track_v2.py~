import numpy as np
import scipy as sp
import os,glob,cv2,pickle
import scipy.ndimage as nd
import scipy.ndimage.morphology as ndm
from PIL import Image

#def main():                                                                    
if 1:
    path ='/home/andyc/image/park/1/'
    imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))

    im = nd.imread(imlist[0])[:,:,0]
    H,W = im.shape
    bg = np.zeros(im.shape)
    w  = np.ones(im.shape)*10**(-6)
    one = np.ones(im.shape)
    bg_Th =6
    th = 30
    
    ori1 = np.zeros([H,W,3])
    ori2 = np.zeros([H,W,3])
    f1 = np.zeros(im.shape)
    f2 = np.zeros(im.shape)

    need_filte =1
    adp_BG =1
    for i in range(3600):#len(imlist)-1):
       tmp = np.zeros(im.shape)
       wtmp = np.zeros(im.shape)
       print i
       if i ==0:
           ori1[:,:,:] = nd.imread(imlist[i])
           f1[:,:] = ori1.mean(2).astype(float)
       else:
           ori1[:,:,:] = ori2
           f1[:,:] = f2
 
       ori2[:,:,:] = nd.imread(imlist[i+1])
       f2[:,:] = ori2.mean(2).astype(float)

       if adp_BG == 1:
           diff = abs(f2-f1)
           tmp[diff<bg_Th]  = f1[diff<bg_Th]
           wtmp[diff<bg_Th] = one[diff<bg_Th]
           w = w + wtmp
           bg[:,:] =bg+tmp
           BG = bg/w
       else:
           BG = pickle.load(open("bg_all.pkl","rb"))


       diff = (f1-BG)
       result = np.zeros(diff.shape)
       result[abs(diff)>th]=255

       r_norm = result/255
       rc = ndm.binary_closing(r_norm,structure=np.ones((2,2)))
       ro = ndm.binary_opening(rc,structure=np.ones((2,2)))
       label_im, nb_labels = nd.label(ro)
       sizes = nd.sum(ro, label_im, range(nb_labels + 1))

       if need_filte:
           mask_size = (sizes < 5) #| (sizes > 500)
           remove_pixel = mask_size[label_im]
           label_im[remove_pixel] = 0
 
       tmp = np.zeros([H,W])
       tmp[label_im!=0] = 255

       sx = nd.sobel(tmp, axis=0, mode='constant')
       sy = nd.sobel(tmp, axis=1, mode='constant')
       edge = np.hypot(sx, sy)

       tmp = ori1[:,:,0]
       tmp[edge!=0] = 0       
       ori1[:,:,0] = tmp
       
       tmp = ori1[:,:,1]
       tmp[edge!=0] = 136 
       ori1[:,:,1] = tmp    
 
       tmp = ori1[:,:,2]
       tmp[edge!=0] = 0
       ori1[:,:,2] = tmp



       fig = Image.fromarray(ori1.astype(np.uint8))
       fig.save('/home/andyc/tracking_proj/output/park/%.5d.jpg'%i)
