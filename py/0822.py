import numpy as np
import os,glob,cv2,pickle,cv
import scipy.ndimage as nd


path = '/home/andyc/image/tracking VIDEO0004/binary/'
imlist = sorted(glob.glob( os.path.join(path, '*.bmp')))

i = 21

f = imread(imlist[i])


s = nd.generate_binary_structure(2,2)

labeled_array, num_features = nd.measurements.label(np.around(f[:,:,0]/255), structure=s)
coor = []
cnt = []

for i in range(1,num_features+1):
    coor.append(np.where(labeled_array==i))
    cnt.append(len(np.where(labeled_array==i)[1]))

lth = 200
Lth = 6500

cnt = array(cnt)
idx = arange(num_features)
idx = idx[(cnt<Lth)&(cnt>lth)]

if len(idx) == 0:
    idx = []
    print("no label is qualified !!")
  
else:
    idx = [idx[cnt[idx].argmax()]] 

    print("index number is")
    print(idx[0])
    print("pixels number it has is")
    print(cnt[idx[0]])

    ff = zeros(f[:,:,0].shape)
    ff[coor[idx[0]][0],coor[idx[0]][1]]=255

    plt.figure(1,figsize=[20,10])
    plt.subplot(122)
    imshow(ff[::-1,:],clim=[0,1],cmap = 'gist_gray',interpolation='nearest')
    plt.subplot(121)
    imshow(f[::-1,:,:],clim=[0,1],cmap = 'gist_gray',interpolation='nearest')
