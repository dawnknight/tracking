
#kalman tracking

import numpy as np
import cv2,pickle,time
import scipy.ndimage.morphology as ndm
import scipy.ndimage as nd
from random import randint
from PIL import Image

#bg parameters
alpha    = 0.01
lmc      = 15.
bmc      = 25.
    
#kalman parameters

R = 0.01*np.eye(4) #noise
dt = 1
A = asarray([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
H = asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
Q = 0.1*np.eye(4) #noise
P = 100*np.eye(4)
B = asarray([[0.5*dt**2,0,0,0],[0,0.5*dt**2,0,0],[0,0,dt,0],[0,0,0,dt]])
u = asarray([[0,0,0,0]]).T     #[ax,ay,ax,ay] accelation

kfinit=0
x = np.zeros([2204,4])
vid_idx=0

ori = asarray([[0,0,0,0]]) 
ori_old = asarray([[0,0,0,0]]) 
len_old = [0,0]

flag = 1
I = np.eye(4)

Trj = {}   # trajectory
Trj['0'] = [[0,0]]
imsave = 0

def coor_extract(idx,lab_mtx,frame,coor):
    global ori 
    global len_old

    if len(idx)==0:        
        left.set_data(frame[:,:,::-1])
        plt.draw()
        ori = ori_old 
        length = len_old 
        flag = 0
    else: 
        flag = 1
        #for i in range(len(idx)):
        
        ulx = asarray(coor[idx[0]][1]).min()
        lrx = asarray(coor[idx[0]][1]).max()
        uly = asarray(coor[idx[0]][0]).min()
        lry = asarray(coor[idx[0]][0]).max()
        
        #centroid = [(uly+lry)/2,(ulx+lrx)/2]
        #length   = [lry-uly,lrx-ulx]
       
        ori[:,0:2] = [uly,ulx]
        ori[:,2:]  = ori[:,0:2]-ori_old[:,0:2] #velocity
        length   = [lry-uly,lrx-ulx]     
            
        #print("Bbox size")
        #print(length[0],length[1])

        #draw
        cv2.rectangle(frame,(ulx,uly),(lrx,lry),(0,255,0),1)
            
        left.set_data(frame[:,:,::-1])
        plt.draw()
        ori_old[:] = ori
        len_old = length

    Kalman_update(ori,length,frame,flag)
  
def Kalman_update(ori,length,frame,flag):

    global kfinit
    global P
    global PP
    global K
    global xp
    global x
    global vid_idx
    global u

    if kfinit == 0 :        
        xp = np.array([[0,0,0,0]]).T                   #predict x(initial)
        kfinit =1        
    else: 
        xp = np.dot(A,np.array([x[vid_idx-1,:]]).T)+np.dot(B,u) #predict x

    ulx = float(round(xp[1]))
    lrx = float(round(xp[1]+length[1]))
    uly = float(round(xp[0]))
    lry = float(round(xp[0]+length[0]))

    # draw predict

    if ((ulx>=0.0) & (uly>=0.0) & (lrx<frame.shape[1]) & (lry<frame.shape[0])):

        cv2.rectangle(frame,(int(ulx),int(uly)),(int(lrx),int(lry)),(0,0,255),1)
        trj_tmp = Trj['0']
        trj_tmp.append([round((uly+lry)/2),round((ulx+lrx)/2)]) 
        Trj['0'] = trj_tmp
    else:
        trj_tmp = Trj['0']
        trj_tmp.append(trj_tmp[-1])
        Trj['0'] = trj_tmp

    if flag ==1:


        PP = np.dot(np.dot(A,P),A.T)+Q                    #covariance  
        Y = ori.T-np.dot(H,xp)                            #residual 
        S = np.dot(np.dot(H,PP),H.T)+R                    #covariance
        K = np.dot(np.dot(PP,H.T),np.linalg.inv(S))       #kalman gain     
        #update  state & covariance
        x[vid_idx,:] = (xp+np.dot(K,Y)).T
        P = np.dot((I-np.dot(K,H)),PP)
        #update velocity & accelation
        if vid_idx>0:

            x[vid_idx,2] = x[vid_idx,0]-x[vid_idx-1,0]
            x[vid_idx,3] = x[vid_idx,1]-x[vid_idx-1,1]
            u = asarray([[x[vid_idx,2]-x[vid_idx-1,2],\
                          x[vid_idx,3]-x[vid_idx-1,3],\
                          x[vid_idx,2]-x[vid_idx-1,2],\
                          x[vid_idx,3]-x[vid_idx-1,3]]]).T 

    else: # if measument  can not obtain

        x[vid_idx,:] = xp.T
        if vid_idx>0:
            x[vid_idx,2] = x[vid_idx-1,2]
            x[vid_idx,3] = x[vid_idx-1,3]
            u = asarray([[x[vid_idx-1,2]-x[vid_idx-2,2],\
                          x[vid_idx-1,3]-x[vid_idx-2,3],\
                          x[vid_idx-1,2]-x[vid_idx-2,2],\
                          x[vid_idx-1,3]-x[vid_idx-2,3]]]).T


    left.set_data(frame[:,:,::-1])

    print(x[vid_idx,:])


    plt.draw()
        
    if imsave:
        imc = Image.fromarray(frame[:,:,::-1].astype(np.uint8))
        imc.save('/home/andyc/image/tracking VIDEO0004/color/c%.3d.jpg'%vid_idx)


def Fg_extract(frame): #extract foreground    

    mu[:]       = alpha*frame + (1.0-alpha)*mu_old
    mu_old[:]   = mu
    sig2[:]     = alpha*(1.0*frame-mu)**2 + (1.0-alpha)*sig2_old
    sig2_old[:] = sig2

    sig = sig2**0.5

    lmcs = lmc*sig
    bmcs = bmc*sig

    fg= np.abs(1.0*frame-mu)[:,:,0]-2*sig[:,:,0]>0.0
    fgo = ndm.binary_opening(fg)
    fgf = ndm.binary_fill_holes(fgo)
    right.set_data(fgf)
    plt.draw()

   
    if imsave:
        im = zeros(frame.shape)
        a = fgf.astype(np.uint8)*255
        im[:,:,0]=a
        im[:,:,1]=a
        im[:,:,2]=a
        
        im = Image.fromarray(im.astype(np.uint8))
        im.save('/home/andyc/image/tracking VIDEO0004/binary/b%.3d.bmp'%vid_idx)



    return fgf
        
def objextract(Fg):

    s = nd.generate_binary_structure(2,2)
    labeled_array, num_features = nd.measurements.label(Fg, structure=s)
    coor = []
    cnt = []

    if num_features == 0:
       idx = []
    else:    
        lth = 200   # label pixel number less than lth will be removed
        Lth = 6500
        for i in range(1,num_features+1):
            coor.append(np.where(labeled_array==i))
            cnt.append(len(np.where(labeled_array==i)[1]))

        #idx = list(set(np.where(asarray(cnt)<Lth)[0]).intersection\
        #              (np.where(asarray(cnt)>lth)[0]))
        cnt = array(cnt)
        idx = arange(num_features)
        idx = idx[(cnt<Lth)&(cnt>lth)]
      
        if len(idx)==0:
            idx = []
        else:
            idx = [idx[cnt[idx].argmax()]]

            '''   
            idx_max = 0
            for ii in range(len(idx)):
                if cnt[idx[ii]]>idx_max:
                    idx_max = idx[ii]
            if idx_max == 0:
                idx = [] 
            else:
                idx = [idx_max]
            ''' 
    return idx,labeled_array,coor,cnt

try:
    vid
except:
    print('reading video...')
    cap = cv2.VideoCapture('/home/andyc/VIDEO0004.mp4')
    vid = []

    if cap.isOpened():
        rval,frame = cap.read()
    else:
        rval = False
    while rval:
        rval,frame = cap.read()
        if rval:
            frame = cv2.resize(frame,(0,0),fx = 0.25,fy=0.25)
            vid.append(frame)

    print('done reading video...')


mu       = np.zeros(vid[0].shape,dtype=float)
mu_old   = np.zeros(vid[0].shape,dtype=float)
sig2     = np.zeros(vid[0].shape,dtype=float)
sig2_old = np.zeros(vid[0].shape,dtype=float)

plt.figure(1,figsize=[20,10])
plt.subplot(121)
left = plt.imshow(vid[0][:,:,::-1])
plt.subplot(122)
right = plt.imshow(mu,clim=[0,1],cmap = 'gist_gray',interpolation='nearest')


for frame in vid:
    print("frame no")
    print(vid_idx)
    Fg = Fg_extract(frame)
    idx,lab_mtx,coor,cnt = objextract(Fg)
    coor_extract(idx,lab_mtx,frame,coor)
    vid_idx = vid_idx+1


