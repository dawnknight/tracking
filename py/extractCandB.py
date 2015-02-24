import os,glob,cv2
import numpy as np
import scipy.ndimage as nd


path ='/home/andyc/image/tra/caraccident/'
imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))

Cpath = path + 'color/'
Bpath = path + 'blob/' 

Cx = [326,929]
Cy = [365,703]
Bx = [1104,1811]
By = [308,705]

color = np.zeros([Cy[1]-Cy[0],Cx[1]-Cx[0],3])
Blob  = np.zeros([By[1]-By[0],Bx[1]-Bx[0],3])

for i in range(len(imlist)):
    print i
    frame = nd.imread(imlist[i])
    color[:] = frame[Cy[0]:Cy[1],Cx[0]:Cx[1],:]
    Blob[:]  = frame[By[0]:By[1],Bx[0]:Bx[1],:]
    Cname = Cpath +'c'+str(i).zfill(4)+'.jpg' 
    Bname = Bpath +'b'+str(i).zfill(4)+'.jpg'
    cv2.imwrite(Cname,uint8(color[:,:,::-1]))
    cv2.imwrite(Bname,uint8(Blob))
