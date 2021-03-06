import numpy as np
import scipy as sp
import os,glob,cv2,pickle
import scipy.ndimage as nd
import scipy.ndimage.morphology as ndm
from PIL import Image

#def main():
if 1:
    path ='/home/andyc/image/IR/EVENING/'
    imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))
    mask = pickle.load(open("Feb11_mask2.pkl","rb"))
    BG = pickle.load(open("bg_all.pkl","rb"))
  
    for i in range(len(imlist)-1): 
  
        print i
         
        ori = np.array(Image.open(imlist[i])
        f = ori.convert('L')).astype(np.float)
        diff = (f-BG)*mask
  
        th = 10
        result = np.zeros(diff.shape)
        result[abs(diff)>th]=255

        r_norm = result/255
        rc = ndm.binary_closing(r_norm,structure=np.ones((2,2)))
        ro = ndm.binary_opening(rc,structure=np.ones((2,2)))
        label_im, nb_labels = nd.label(ro)
        sizes = nd.sum(ro, label_im, range(nb_labels + 1))

        mask_size = (sizes < 200) | (sizes > 3000)
        remove_pixel = mask_size[label_im]
        label_im[remove_pixel] = 0

   #     mask_size = sizes > 3000
   #     remove_pixel = mask_size[label_im]
   #     label_im[remove_pixel] = 0

        tmp = np.zeros([1080,1920]) 
        tmp[label_im!=0] = 255

        sx = nd.sobel(tmp, axis=0, mode='constant')
        sy = nd.sobel(tmp, axis=1, mode='constant')
        edge = np.hypot(sx, sy)
        tmp = ori[:,:,1]
        tmp[edge!=0] = 255

        ori[:,:,1] = tmp
    
        im = Image.fromarray(ori.astype(np.uint8)) 
        im.save('/home/andyc/tracking_proj/output/Nomask_%.5d.png'%i)
    
#main()        

