
#kalman tracking

import numpy as np
import cv2,pickle,time
import scipy.ndimage.morphology as ndm
import scipy.ndimage as nd
from random import randint


#bg parameters
alpha    = 0.01
lmc      = 15.
bmc      = 25.
    
#kalman parameters

R = asarray([[0.2845,0.0045],[0.0045,0.0455]])

H = asarray([[1,0,0,0],[0,1,0,0]])
#Q = 0.01*np.eye(4)
Q = 0.1*np.eye(4)
P = 100*np.eye(4)
dt = 1
A = asarray([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
g = 6
Bu = np.array([[0,0,0,g]]).T
kfinit=0
x = np.zeros([2204,4])
idx = 0
vid_idx=0
'''
def Main():
    try:
        vid
    except:
        print('reading video...')
        cap = cv2.VideoCapture('/home/andyc/K_test.avi')
        vid = []

        for _ in range(200):
            ret, frame = cap.read()
            #frame = cv2.resize(frame,(0,0),fx = 0.25,fy=0.25)
            vid.append(frame)
        print('done reading video...')

        mu       = np.zeros(vid[0].shape,dtype=float)
        mu_old   = np.zeros(vid[0].shape,dtype=float)
        sig2     = np.zeros(vid[0].shape,dtype=float)
        sig2_old = np.zeros(vid[0].shape,dtype=float)
#        conf     = np.zeros(vid.shape,dtype=float)

        plt.figure(1,figsize=[10,5])
        left = plt.imshow(vid[0])


        for frame in vid:
             Fg = Fg_extract(frame)
             idx,lab_mtx = objextract(Fg)
             coor_extract(idx,lab_mtx,frame)
             left.set_data(frame[:,:,::-1])
             plt.draw()
'''

def coor_extract(idx,lab_mtx,frame,coor):

    for i in range(len(idx)):
        ulx = asarray(coor[idx[i]][1]).min()
        lrx = asarray(coor[idx[i]][1]).max()
        uly = asarray(coor[idx[i]][0]).min()
        lry = asarray(coor[idx[i]][0]).max()
           
        #centroid = [(uly+lry)/2,(ulx+lrx)/2]
        #length   = [lry-uly,lrx-ulx]
       
        ori = [uly,ulx]
        length   = [lry-uly,lrx-ulx]     

        #draw

        frame[uly,ulx:lrx,0] = 0
        frame[uly,ulx:lrx,1] = 176
        frame[uly,ulx:lrx,2] = 0

        frame[lry,ulx:lrx,0] = 0
        frame[lry,ulx:lrx,1] = 176
        frame[lry,ulx:lrx,2] = 0

        frame[uly:lry,ulx,0] = 0
        frame[uly:lry,ulx,1] = 176
        frame[uly:lry,ulx,2] = 0

        frame[uly:lry,lrx,0] = 0
        frame[uly:lry,lrx,1] = 176
        frame[uly:lry,lrx,2] = 0
      
        left.set_data(frame[:,:,::-1])
        plt.draw()
        Kalman_update(ori,length,frame)

  
def Kalman_update(ori,length,frame):

    global kfinit
    global P
    global PP
    global K
    global xp
    global x
    global vid_idx

    if kfinit == 0 :
       #xp = np.array([[frame.shape[1]/2,frame.shape[0]/2,0,0]]).T
       xp = np.array([[0,0,0,0]]).T                   #predict x(initial)
    else: 
       xp = np.dot(A,np.array([x[vid_idx-1,:]]).T)+Bu #predict x
 
    ulx = float(round(xp[1]))
    lrx = float(round(xp[1]+length[1]))
    uly = float(round(xp[0]))
    lry = float(round(xp[0]+length[0]))


    '''
    kfinit = 1
    PP = np.dot(np.dot(A,P),A.T)+Q
    K = np.dot(np.dot(PP,H.T),np.linalg.inv(np.dot(np.dot(H,PP),H.T)+R)) 
    x[vid_idx,:] = (xp+np.dot(K,(np.array([[ori[0],ori[1]]]).T-np.dot(H,xp)))).T   
    P = np.dot((np.eye(4)-np.dot(K,H)),PP)
    '''

    # draw predict

    if ((ulx>=0.0) & (uly>=0.0) & (lrx<frame.shape[1]) & (lry<frame.shape[0])):

        frame[uly,ulx:lrx,0] = 0
        frame[uly,ulx:lrx,1] = 0
        frame[uly,ulx:lrx,2] = 255

        frame[lry,ulx:lrx,0] = 0
        frame[lry,ulx:lrx,1] = 0
        frame[lry,ulx:lrx,2] = 255

        frame[uly:lry,ulx,0] = 0
        frame[uly:lry,ulx,1] = 0
        frame[uly:lry,ulx,2] = 255

        frame[uly:lry,lrx,0] = 0
        frame[uly:lry,lrx,1] = 0
        frame[uly:lry,lrx,2] = 255

        left.set_data(frame[:,:,::-1])
        plt.draw()
         

    kfinit = 1
    PP = np.dot(np.dot(A,P),A.T)+Q                    #covariance  
    Y = (np.array([[ori[0],ori[1]]]).T-np.dot(H,xp))  #residual 
    S = np.dot(np.dot(H,PP),H.T)+R                    #covariance
    K = np.dot(np.dot(PP,H.T),np.linalg.inv(S))       #kalman gain     
    #update
    x[vid_idx,:] = (xp+np.dot(K,Y)).T
    P = np.dot((np.eye(4)-np.dot(K,H)),PP)
  
    print(xp[0:2].T)
    print(np.array([[ori[0],ori[1]]]))
    print(Y)
      
    print(x[vid_idx,:])


    #pickle.dump(PP,open("PP.pkl","wb"),True)
    #pickle.dump(K,open("K.pkl","wb"),True)
    #pickle.dump(x[vid_idx,:],open("x.pkl","wb"),True)
    #pickle.dump(P,open("P.pkl","wb"),True)

def Fg_extract(frame): #extract foreground

    
    mu[:]       = alpha*frame + (1.0-alpha)*mu_old
    mu_old[:]   = mu
    sig2[:]     = alpha*(1.0*frame-mu)**2 + (1.0-alpha)*sig2_old
    sig2_old[:] = sig2

    sig = sig2**0.5

    lmcs = lmc*sig
    bmcs = bmc*sig

    fg= np.abs(1.0*frame-mu)[:,:,0]-1*sig[:,:,0]>0.0
    fgo = ndm.binary_opening(fg)
    fgf = ndm.binary_fill_holes(fgo)

    return fgf
        
def objextract(Fg):

    s = nd.generate_binary_structure(2,2)
    labeled_array, num_features = nd.measurements.label(Fg, structure=s)
    coor = []
    cnt = []
    lth = 30   # label pixel number less than lth will be removed
    for i in range(1,num_features+1):
        coor.append(np.where(labeled_array==i))
        cnt.append(len(np.where(labeled_array==i)[1]))

    idx = np.where(asarray(cnt)>lth)[0]
    
    return idx,labeled_array,coor

try:
    vid
except:
    print('reading video...')
    cap = cv2.VideoCapture('/home/andyc/ball.avi')
    vid = []

    for _ in range(2204):
        ret, frame = cap.read()
        #frame = cv2.resize(frame,(0,0),fx = 0.25,fy=0.25)                                           
        vid.append(frame)
    print('done reading video...')

    mu       = np.zeros(vid[0].shape,dtype=float)
    mu_old   = np.zeros(vid[0].shape,dtype=float)
    sig2     = np.zeros(vid[0].shape,dtype=float)
    sig2_old = np.zeros(vid[0].shape,dtype=float)
    #conf     = np.zeros(vid.shape,dtype=float)                                                                                            

    plt.figure(1,figsize=[20,10])
    left = plt.imshow(vid[0][:,:,::-1])

    for frame in vid:
        print(vid_idx)
        Fg = Fg_extract(frame)
        idx,lab_mtx,coor = objextract(Fg)
        coor_extract(idx,lab_mtx,frame,coor)
        #left.set_data(frame[:,:,::-1])
        #plt.draw()
        #time.sleep(1)
        vid_idx = vid_idx+1
