import numpy as np
import cv2
import scipy.ndimage.morphology as ndm
import scipy.ndimage as nd
from random import randint

try:
    vid
except:
    print('reading video...')
    cap = cv2.VideoCapture('/home/andyc/VIDEO0004.mp4')
    vid = []

    for _ in range(600):
        ret, frame = cap.read()
        frame = cv2.resize(frame,(0,0),fx = 0.25,fy=0.25)  
        vid.append(frame)
    print('done reading video...')
alpha    = 0.01
lmc      = 15.
bmc      = 25.
mu       = np.zeros(vid[0].shape,dtype=float)
mu_old   = np.zeros(vid[0].shape,dtype=float)
sig2     = np.zeros(vid[0].shape,dtype=float)
sig2_old = np.zeros(vid[0].shape,dtype=float)
conf     = np.zeros(vid[0].shape,dtype=float)
tmp      = np.zeros(vid[0].shape,dtype=float)
plt.figure(1,figsize=[20,10])

plt.subplot(221)
left = plt.imshow(vid[0])
plt.subplot(222)
right = plt.imshow(conf,clim=[0,1],cmap = 'gist_gray',interpolation='nearest')
plt.subplot(223)
right2 = plt.imshow(conf,clim=[0,1],cmap = 'gist_gray',interpolation='nearest')
plt.subplot(224)
right3 = plt.imshow(conf,clim=[0,1],interpolation='nearest')



for frame in vid:
    mu[:]       = alpha*frame + (1.0-alpha)*mu_old
    mu_old[:]   = mu
    sig2[:]     = alpha*(1.0*frame-mu)**2 + (1.0-alpha)*sig2_old
    sig2_old[:] = sig2

    sig = sig2**0.5

    lmcs = lmc*sig
    bmcs = bmc*sig
    
    fg= np.abs(1.0*frame-mu)[:,:,0]-2*sig[:,:,0]>0.0
    fgo = ndm.binary_opening(fg) 
    fgc = ndm.binary_closing(fg)
    fgf = ndm.binary_fill_holes(fgo)



    s = nd.generate_binary_structure(2,2)
    labeled_array, num_features = nd.measurements.label(fgf, structure=s)  
    coor = []
    cnt = []
    lth = 200
    Lth = 250000
    for i in range(1,num_features+1):
        coor.append(np.where(labeled_array==i))
        cnt.append(len(np.where(labeled_array==i)[1]))
    idx = list(set(np.where(asarray(cnt)<Lth)[0]).intersection\
                  (np.where(asarray(cnt)>lth)[0]))    
    #idx = np.where(asarray(cnt)>30)[0]
    tmp[:,:,:] = frame
    print(len(idx))
    if (len(idx)>2 & len(idx)<50):
        for i in range(len(idx)):
            ulx = asarray(coor[idx[i]][1]).min()
            lrx = asarray(coor[idx[i]][1]).max()
            uly = asarray(coor[idx[i]][0]).min()
            lry = asarray(coor[idx[i]][0]).max()
        
            R = randint(0,255)        
            G = randint(0,255)
            B = randint(0,255)

            tmp[uly,ulx:lrx,0] = R
            tmp[uly,ulx:lrx,1] = G 
            tmp[uly,ulx:lrx,2] = B

            tmp[lry,ulx:lrx,0] = R
            tmp[lry,ulx:lrx,1] = G
            tmp[lry,ulx:lrx,2] = B

            tmp[uly:lry,ulx,0] = R 
            tmp[uly:lry,ulx,1] = G
            tmp[uly:lry,ulx,2] = B
            
            tmp[uly:lry,lrx,0] = R
            tmp[uly:lry,lrx,1] = G
            tmp[uly:lry,lrx,2] = B


    
    #conf[:] = (100.*((1.0*frame - mu) - lmcs)/(bmcs-lmcs)).clip(0,100)
    left.set_data(frame[:,:,::-1])

    right.set_data(uint8(tmp[:,:,::-1]))    

    right2.set_data(fgf)

    right3.set_data(fgo)

    #right.set_data(np.abs(1.0*frame-mu).mean(2)-1*sig.mean(2)>0.0)
    #right.set_data(np.abs(1.0*frame-mu).max(2)-1*sig.max(2)>0.0)
    #right.set_data(fgo)
    #right2.set_data(fgo)
    #right3.set_data(uint8(labeled_array*255/num_features))
    #right2.set_data(np.abs(1.0*frame-mu)[:,:,1]-1*sig[:,:,1]>0.0)
    #right3.set_data(np.abs(1.0*frame-mu)[:,:,2]-1*sig[:,:,2]>0.0)
    plt.draw()
