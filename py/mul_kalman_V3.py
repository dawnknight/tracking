#kalman tracking
import numpy as np
import cv2,pickle,time,pdb,copy
import scipy.ndimage.morphology as ndm
import scipy.ndimage as nd
import numpy.random as rand
from PIL import Image
from munkres import Munkres


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
B = asarray([[0.5*dt**2,0,0,0],[0,0.5*dt**2,0,0],[0,0,dt,0],[0,0,0,dt]])

vid_idx=0

ori = {} 
ori_old = {}
length ={}
#len_old = {}

flag = 1
I = np.eye(4)

imsave = 0

blobs = {}
obj_idx = 0
exist_blobs = 0

class Blob(): #create a new blob                                                                                                          

    def __init__(self,fshape):
        self.x = []
        self.x_old = []
        self.xp = np.array([[randint(0,fshape[0]),randint(0,fshape[1]),0,0]]).T
        self.u  = array([[0,0,0,0]]).T
        self.P = 100*np.eye(4)
        self.Trj = []
        self.len = [0,0]                #Bbox size                                                                                       
        self.ref = array([[0,0,0,0]])   #Bbox position and velocity x,y,vx,vy                                                          
        self.ori = []                   #Bbox measurment position
        self.status = 0                 # 0 : as initial  1 : as live  2 : as dead     
        self.dtime = 0                  # obj dispear period                                                                              

    def Color(self):    #initial Blob color                                                                                               
        try:
            self.R
        except:
            self.R = randint(0,255)
            self.G = randint(0,255)
            self.B = randint(0,255)
        return (self.R,self.G,self.B)


def Fg_Extract(frame,type = 1): #extract foreground    

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
        
def Objextract(Fg):

    s = nd.generate_binary_structure(2,2)
    labeled_array, num_features = nd.measurements.label(Fg, structure=s)
    coor = []
    cnt = []
    if num_features == 0:
       idx = []
    else:    
        lth = 150   # label pixel number less than lth will be removed
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
            idx = sorted(range(len(cnt)),key=lambda x:cnt[x])[::-1][0:len(idx)]
    
    return idx,labeled_array,coor,cnt

def Coor_Extract(idx,lab_mtx,frame,coor):
    global ori 
    global ori_old
    global length
    global len_old

    #if len(idx)<len(blobs):
    ori = {}

    if len(idx)==0:        
        flag = 0
    else: 
        flag = 1
        for i in range(len(idx)):
        
            ulx = int(array(coor[idx[i]][1]).min())
            lrx = int(array(coor[idx[i]][1]).max())
            uly = int(array(coor[idx[i]][0]).min())
            lry = int(array(coor[idx[i]][0]).max())
                        
            ori[i] = array([0,0,0,0])
            ori_old[i] = array([0,0,0,0])
            length[i] = [0,0]

            ori[i][0:2] = [uly,ulx]
            #ori[i][2:]  = [p-q for p,q in zip(ori[i][0:2],ori_old[i][0:2])] #velocity
            length[i]   = [lry-uly,lrx-ulx]     

            #draw
            #pdb.set_trace()
            cv2.rectangle(frame,(ulx,uly),(lrx,lry),(0,255,0),1)
            
            #ori_old[i] = copy.deepcopy(ori[i])
            #len_old[i] = length[i]

    left.set_data(frame[:,:,::-1])
    plt.draw()
  
    print('idx number is {0} blobs number is {1}'.format(len(idx),len(blobs)))
    print('There are {0} live blobs'.format(exist_blobs))
    if (len(idx)>0 or len(blobs)>0):
        Kalman_update(ori,length,frame,flag)
    
def Kalman_update(ori,length,frame,flag):

    #global kfinit
    global P
    global PP
    global K
    global xp
    global x
    global vid_idx
    global u
    global blobs
    global obj_idx
    global exist_blobs

    xcor = []
    ycor = []              
    blob_idx = [] 

    if vid_idx == 89:
       pdb.set_trace()    

    for i in range(len(ori)-exist_blobs):           #if new objs come in  
        blobs[obj_idx] = Blob(frame.shape)         #initialize new blob                                                             
        exist_blobs+=1
        obj_idx += 1   

    for i in range(len(blobs)): 
        if blobs[i].dtime <5: 
            if blobs[i].status ==0: 
                blobs[i].status = 1
            else:
                blobs[i].xp = np.dot(A,blobs[i].x.T)+np.dot(B,blobs[i].u)         
            xcor.append(blobs[i].xp[1])
            ycor.append(blobs[i].xp[0])
            blob_idx.append(i)
        else:
           #print(blobs[i].x)
           #print(blobs[i].xp)
           #print(ori)
           '''
           xori = []   
           yori = []
           for jj in range(len(ori)):
               xori.append(ori[jj][1])    
               yori.append(ori[jj][0])
           Objmatch(yori,xori,blob[i].x[0][0:2],1,len(yori))
           '''  
           blobs[i].status = 2  
     


    if len(ori)>1: 
        order = Objmatch(ycor,xcor,ori,len(ori),len(blob_idx))
    elif len(ori) ==1: 
        order = [(0,0)]
    #if vid_idx ==71:
    #   pdb.set_trace() 

    if (len(ori)<len(blob_idx)) & (len(ori)!=0):
        blob_idx = list(set([a[1] for a in order]) & set(blob_idx))      #find common term between 2 lists 
    
  

    for i,j in zip(blob_idx,range(len(blob_idx))):                  
        #pdb.set_trace()
        if len(ori)!=0:
            try:
                ori[order[j][1]][2:] = array([ori[order[j][1]][0:2]])-blobs[i].ref[0:2]  #measure the velocity
            except:
                ori[order[j][1]][2:] = array([ori[order[j][1]][0:2]])

            if len(ori) == len(blob_idx):
                blobs[i].ref = array([ori[order[j][1]]])
                blobs[i].len = length[order[j][1]]
            elif len(ori) < len(blob_idx) :          #objs outside the view 

                if i in array(order)[:,1]:
                    blobs[i].ref = array([ori[order[j][1]]])   
                    blobs[i].len = length[order[j][1]]   
                else:
                    blobs[i].dtime += 1  
                    if blobs[i].dtime ==5:
                        exist_blobs -=1 

        ulx = int(round(blobs[i].xp[1]))
        lrx = int(round(blobs[i].xp[1]+blobs[i].len[1]))
        uly = int(round(blobs[i].xp[0]))
        lry = int(round(blobs[i].xp[0]+blobs[i].len[0]))

        # draw predict
                
        if ((lrx>=0) & (lry>=0) & (ulx<frame.shape[1]) & (uly<frame.shape[0])): #at least part of the obj inside the view 
            #pdb.set_trace() 
            cv2.rectangle(frame,(max(ulx,0) ,max(uly,0)),\
                                (min(lrx,frame.shape[1]),min(lry,frame.shape[0])),\
                                 blobs[i].Color(),1)             
            try:
               trj_tmp = blobs[i].Trj
               trj_tmp.append( [ int(min(max(round((uly+lry)/2),0),frame.shape[0])),frame.shape[0],\
                                 int(min(max(round((ulx+lrx)/2),0),frame.shape[1])),frame.shape[1]])
            except: 
                trj_tmp = [[int(min(max(round((uly+lry)/2),0),frame.shape[0])),frame.shape[0],\
                            int(min(max(round((ulx+lrx)/2),0),frame.shape[1])),frame.shape[1]]]
            blobs[i].Trj = trj_tmp 
        else:      #obj outside the view
            trj_tmp = blobs[i].Trj
            trj_tmp.append(trj_tmp[-1])
            blobs[i].Trj = trj_tmp
            blobs[i].dtime +=1 



        if flag ==1:
        
            PP = np.dot(np.dot(A,blobs[i].P),A.T)+Q                    #covariance  
            Y = blobs[i].ref.T-np.dot(H,blobs[i].xp)                            #residual 
            S = np.dot(np.dot(H,PP),H.T)+R                    #covariance
            K = np.dot(np.dot(PP,H.T),np.linalg.inv(S))       #kalman gain     
            #update  state & covariance
            blobs[i].x = (blobs[i].xp+np.dot(K,Y)).T
            #x[vid_idx,:] = (xp+np.dot(K,Y)).T
            blobs[i].P = np.dot((I-np.dot(K,H)),PP)
        else: # if measument  can not obtain
            blobs[i].x = blobs[i].xp.T
    
        #update velocity & accelation 
        if vid_idx>0:
            try :                                 
                blobs[i].x[0][2] = blobs[i].x_old[0][2]
                blobs[i].x[0][3] = blobs[i].x_old[0][3]
                blobs[i].u = asarray([[blobs[i].x[0][2]-blobs[i].x_old[0][2],\
                                       blobs[i].x[0][3]-blobs[i].x_old[0][3],\
                                       blobs[i].x[0][2]-blobs[i].x_old[0][2],\
                                       blobs[i].x[0][3]-blobs[i].x_old[0][3]]]).T
            except:
                blobs[i].x[0][2] = blobs[i].x[0][0]
                blobs[i].x[0][3] = blobs[i].x[0][1]
                blobs[i].u = asarray([[blobs[i].x[0][2],\
                                       blobs[i].x[0][3],\
                                       blobs[i].x[0][2],\
                                       blobs[i].x[0][3]]]).T 
        blobs[i].x_old=blobs[i].x   

    left.set_data(frame[:,:,::-1])
    plt.draw()       
    #kfinit = 1
 
    if imsave:
        imc = Image.fromarray(frame[:,:,::-1].astype(np.uint8))
        imc.save('/home/andyc/image/tra/2 balls/result/c%.3d.jpg'%vid_idx)



def Objmatch(objy,objx,ref,L,W):

    cmtx = np.zeros((L,W))
    #pdb.set_trace()                                                                                                                       
    for i in range(L):
        cmtx[i,:] =((objy - ref[i][0])**2+(objx - ref[i][1])**2).T
    if vid_idx == 74:
        pdb.set_trace()
    m = Munkres()
    indexes = m.compute(cmtx)
    return indexes



try:
    vid
except:
    print('reading video...')
    cap = cv2.VideoCapture('/home/andyc/crowd2.avi')
    vid = []

    if cap.isOpened():
        rval,frame = cap.read()
    else:
        rval = False
    while rval:
        rval,frame = cap.read()
        if rval:
            frame = cv2.resize(frame,(0,0),fx = 0.5,fy=0.5)
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
    print("frame no : {0}".format(vid_idx))
    #frame = frame[:,:,::-1]
    Fg = Fg_Extract(frame,2)
    idx,lab_mtx,coor,cnt = Objextract(Fg)
    Coor_Extract(idx,lab_mtx,frame,coor)
    vid_idx = vid_idx+1
