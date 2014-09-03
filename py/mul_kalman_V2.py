#kalman tracking
import numpy as np
import cv2,pickle,time
import scipy.ndimage.morphology as ndm
import scipy.ndimage as nd
import numpy.random as rand
from PIL import Image
from munkres import Munkres
import pdb

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
P ={}
B = asarray([[0.5*dt**2,0,0,0],[0,0.5*dt**2,0,0],[0,0,dt,0],[0,0,0,dt]])
u = {}   # asarray([[0,0,0,0]]).T     #[ax,ay,ax,ay] accelation

kfinit=0
xp ={}
x = {}
x_old={}
vid_idx=0

ori = {} 
ori_old = {}
length ={}
len_old = {}

flag = 1
I = np.eye(4)

Trj = {}   # trajectory
Trj[0] = [[0,0]]
imsave = 1

def coor_extract(idx,lab_mtx,frame,coor):
    global ori 
    global ori_old
    global length
    global len_old

    if len(idx)==0:        
        left.set_data(uint8(frame))
        plt.draw()
        
        try:
            ori_old[0]
        except:
            ori_old[0] = [0,0,0,0]
            len_old[0] = [0,0]

        ori = ori_old 
        length = len_old
        flag = 0

    else: 

        flag = 1
        for i in range(len(idx)):
            #for i in range(len(idx)):
        
            ulx = asarray(coor[idx[i]][1]).min()
            lrx = asarray(coor[idx[i]][1]).max()
            uly = asarray(coor[idx[i]][0]).min()
            lry = asarray(coor[idx[i]][0]).max()
        
            #centroid = [(uly+lry)/2,(ulx+lrx)/2]
            #length   = [lry-uly,lrx-ulx]
                    
            try:
                ori[i]
            except:
                ori[i] = [0,0,0,0]
                ori_old[i] = [0,0,0,0]
                length[i] = [0,0]

            ori[i][0:2] = [uly,ulx]
            ori[i][2:]  = [p-q for p,q in zip(ori[i][0:2],ori_old[i][0:2])] #velocity
            length[i]   = [lry-uly,lrx-ulx]     

            #draw
            cv2.rectangle(frame,(ulx,uly),(lrx,lry),(0,255,0),1)
            
            left.set_data(uint8(frame))
            plt.draw()

            ori_old[i] = ori[i]
            len_old[i] = length[i]

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

    xcor = []
    ycor = []

    for i in range(len(ori)): 
        if kfinit == 0:
            xp[i] = np.array([[randint(0,frame.shape[0]),randint(0,frame.shape[1]),0,0]]).T               #predict x(initial) 
        else: 
            try:
                u[i]   
            except:
                u[i] = array([[0,0,0,0]]).T 
        if i < len(x):
            xp[i] = np.dot(A,x[i].T)+np.dot(B,u[i]) #predict x
        else :   #there is new obj appear
            xp[i] = np.array([[randint(0,frame.shape[0]),randint(0,frame.shape[1]),0,0]]).T 

        xcor.append(xp[i][1])
        ycor.append(xp[i][0])  

    if len(ori)>1: 
        order = Objmatch(ycor,xcor,ori,len(ori))
    else:
        order = [(0,0)]

    for i in range(len(ori)):                  
        #pdb.set_trace()
        ref = array([ori[order[i][1]]])
        box_len = length[order[i][1]]
 
        ulx = float(round(xp[i][1]))
        lrx = float(round(xp[i][1]+box_len[1]))
        uly = float(round(xp[i][0]))
        lry = float(round(xp[i][0]+box_len[0]))

        # draw predict
        if ((ulx>=0.0) & (uly>=0.0) & (lrx<frame.shape[1]) & (lry<frame.shape[0])): # inside the boundary              
            if i ==0:
                cv2.rectangle(frame,(int(ulx),int(uly)),(int(lrx),int(lry)),(0,0,255),1)
            else:
                cv2.rectangle(frame,(int(ulx),int(uly)),(int(lrx),int(lry)),(255,0,0),1)
             
            try:
               trj_tmp = Trj[i]
               trj_tmp.append([round((uly+lry)/2),round((ulx+lrx)/2)])
            except:
               trj_tmp =[[round((uly+lry)/2),round((ulx+lrx)/2)]]
            Trj[i] = trj_tmp 
        else:
            try:
               trj_tmp = Trj[i]
               trj_tmp.append(trj_tmp[-1])
            except:
               trj_tmp =[[round((uly+lry)/2),round((ulx+lrx)/2)]]
            Trj[i] = trj_tmp

        if flag ==1:
            try:
                P[i]   
            except:    
                P[i] = 100*np.eye(4)

            PP = np.dot(np.dot(A,P[i]),A.T)+Q                    #covariance  
            Y = ref.T-np.dot(H,xp[i])                            #residual 
            S = np.dot(np.dot(H,PP),H.T)+R                    #covariance
            K = np.dot(np.dot(PP,H.T),np.linalg.inv(S))       #kalman gain     
            #update  state & covariance
            x[i] = (xp[i]+np.dot(K,Y)).T
            #x[vid_idx,:] = (xp+np.dot(K,Y)).T
            P[i] = np.dot((I-np.dot(K,H)),PP)
        else: # if measument  can not obtain
            x[i] = xp[i].T
        #update velocity & accelation 
        if vid_idx>0:
            try :                                 
                x[i][0][2] = x_old[i][0][2]
                x[i][0][3] = x_old[i][0][3]
                u[i] = asarray([[x[i][0][2]-x_old[i][0][2],\
                              x[i][0][3]-x_old[i][0][3],\
                              x[i][0][2]-x_old[i][0][2],\
                              x[i][0][3]-x_old[i][0][3]]]).T
            except:
                x[i][0][2] = x[i][0][0]
                x[i][0][3] = x[i][0][1]
                u[i] = asarray([[x[i][0][2],\
                                 x[i][0][3],\
                                 x[i][0][2],\
                                 x[i][0][3]]]).T 
        x_old[i]=x[i]   

    left.set_data(uint8(frame))
    plt.draw()       
    kfinit = 1
 
    if imsave:
        imc = Image.fromarray(frame[:,:,::-1].astype(np.uint8))
        imc.save('/home/andyc/image/tra/2 balls/result/c%.3d.jpg'%vid_idx)

def Objmatch(objy,objx,ref,L):
   
    cmtx = np.zeros((L,L))
    #pdb.set_trace()
    for i in range(L):
        cmtx[i,:] =((objy - ref[i][0])**2+(objx - ref[i][1])**2).T 
        
    m = Munkres()
    indexes = m.compute(cmtx)  
    return indexes

def Fg_extract(frame,type = 1): #extract foreground    

    if type ==1:
        mu[:]       = alpha*frame + (1.0-alpha)*mu_old
        mu_old[:]   = mu
        sig2[:]     = alpha*(1.0*frame-mu)**2 + (1.0-alpha)*sig2_old
        sig2_old[:] = sig2
        
        sig = sig2**0.5
        
        lmcs = lmc*sig
        bmcs = bmc*sig
        
        fg= np.abs(1.0*frame-mu)[:,:,0]-1*sig[:,:,0]>0.0
    elif type == 2:
        try:
            fg = np.abs(1.0*frame.mean(2)-BG)>50.0
        except:
            BG = pickle.load(open("bg13-19.pkl","rb"))
            BG = cv2.resize(BG,(0,0),fx = 0.5,fy=0.5)
            fg = np.abs(1.0*frame.mean(2)-BG)>50.0
            

    fgo = ndm.binary_opening(fg)
    fgf = ndm.binary_fill_holes(fgo)
    right.set_data(fgf)
    plt.draw()
       
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

        cnt = array(cnt)
        idx = arange(num_features)
        idx = idx[(cnt<Lth)&(cnt>lth)]
      
        if len(idx)==0:
            idx = []
        elif len(idx)>1:              
            #idx = [idx[cnt[idx].argmax()]]
            idx = sorted(range(len(cnt)),key=lambda x:cnt[x])[::-1][0:2]

    return idx,labeled_array,coor,cnt
'''
try:
    vid
except:
    print('reading video...')
    cap = cv2.VideoCapture('/home/andyc/image/tra/ball10.avi')
    vid = []

    if cap.isOpened():
        rval,frame = cap.read()
    else:
        rval = False
    while rval:
        rval,frame = cap.read()
        if rval:
            #frame = cv2.resize(frame,(0,0),fx = 0.25,fy=0.25)
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
'''


import os,glob

path ='/home/andyc/Videos/Crowd_PETS09/S0_BG/Crowd_PETS09/S0/Background/View_001/Time_13-19/'
imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))
im = nd.imread(imlist[0])
im = cv2.resize(im,(0,0),fx = 0.5,fy=0.5)
mu       = np.zeros(im.shape,dtype=float)
mu_old   = np.zeros(im.shape,dtype=float)
sig2     = np.zeros(im.shape,dtype=float)
sig2_old = np.zeros(im.shape,dtype=float)
frame    = np.zeros(im.shape,dtype=float)

plt.figure(1,figsize=[20,10])
plt.subplot(121)
left = plt.imshow(im)
plt.subplot(122)
right = plt.imshow(mu,clim=[0,1],cmap = 'gist_gray',interpolation='nearest')

for ii in range(len(imlist)):
    print("frame no")
    print(ii)
    frame = nd.imread(imlist[ii])
    frame = cv2.resize(frame,(0,0),fx = 0.5,fy=0.5)
    Fg = Fg_extract(frame,2)
    idx,lab_mtx,coor,cnt = objextract(Fg)
    coor_extract(idx,lab_mtx,frame,coor)
    vid_idx = vid_idx+1
