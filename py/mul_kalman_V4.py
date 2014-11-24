#kalman tracking
import numpy as np
import cv2,pickle,pdb,copy
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

flag = 1
I = np.eye(4)

imsave = 0
draw_Trj = 1
kill_Trj = 1
Trj_type = 0   # 0 : from measurment Bbox  # 1 :from prediction Bbox 



scale = 0.5
maskon = 1

blobs = {}
obj_idx = 0
live_blobs = []




class Blob(): #create a new blob                                                                                                          

    def __init__(self,fshape,ini_x,ini_y):
        self.x = []                #current status
        self.x_old  = []           #previous  status
        self.xp = np.array([[ini_y,ini_x,0,0]]).T
        self.u  = array([[0,0,0,0]]).T
        self.P = 100*np.eye(4)
        self.Trj = {}
        self.len = [0,0]                #Bbox size                                                                                       
        self.ref = array([[0,0,0,0]])   #Bbox position and velocity x,y,vx,vy                                                          
        self.ori = []                   #Bbox measurment position
        self.status = 0                 # 0 : as initial  1 : as live  2 : as dead     
        self.dtime = 0                  # obj dispear period                                                                              
        self.ivalue = []                # store intensity value in Bbox

        self.Trj['x'] = []
        self.Trj['y'] = []
        self.Trj['frame'] = []

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
            BG = pickle.load(open("jstr.pkl","rb"))
            BG = cv2.resize(BG,(0,0),fx = scale,fy=scale)
            fg = np.abs(1.0*frame.mean(2)-BG)>50.0
    if maskon:
       mask = pickle.load(open("jstr_mask.pkl","rb"))     
       mask = cv2.resize(mask,(0,0),fx = scale,fy=scale)
       fg = fg*mask

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
        lth = 120   # label pixel number less than lth will be removed
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
            idx = sorted(range(len(cnt)),key=lambda x:cnt[x])[::-1][0:len(idx)]
    
    return idx,labeled_array,coor,cnt

def Coor_Extract(idx,lab_mtx,frame,coor): 
    length = {}
    ori = {}
    ori_color ={} 
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
            length[i] = [0,0]

            ori[i][0:2] = [uly,ulx]
            length[i]   = [lry-uly,lrx-ulx]     
            
            ori_color[i] = array([frame[uly:lry,ulx:lrx,0].flatten().mean(),\
                                  frame[uly:lry,ulx:lrx,1].flatten().mean(),\
                                  frame[uly:lry,ulx:lrx,2].flatten().mean()])

            cv2.rectangle(frame,(ulx,uly),(lrx,lry),(0,255,0),1)

    left.set_data(frame[:,:,::-1])
    plt.draw()
  
    print('  measurment number is {0} \n  There are {1} blobs'.format(len(idx),len(blobs)))
    print('  {0} of them are alive\n\n'.format(len(live_blobs) ))
    if (len(idx)>0 or len(blobs)>0):
        Kalman_update(ori,ori_color,length,frame,flag)
    
def Kalman_update(ori,ori_color,length,frame,flag):

    global P
    global PP
    global K
    global xp
    global x
    global vid_idx
    global u
    global blobs
    global obj_idx
    global live_blobs
    global order
    
     
    xcor = []
    ycor = []              
    blob_idx = [] 
    lines ={}
    NB_idx =[]
    ini_x = []
    ini_y = [] 
    order = [] 
    line_exist = 0

    for _,i in enumerate(live_blobs):
        if blobs[i].dtime <5:
            if blobs[i].status ==0:
                blobs[i].status = 1
            else:
                blobs[i].xp = np.dot(A,blobs[i].x_old.T)+np.dot(B,blobs[i].u)
            xcor.append(round(blobs[i].xp[1]))
            ycor.append(round(blobs[i].xp[0]))
            blob_idx.append(i)
        else:
            blobs[i].status = 2
            live_blobs = [a for a in live_blobs if a!=i]


    if (len(ori)!=0) & (len(live_blobs)!=0):                      
        order = Objmatch(ycor,xcor,ori,ori_color,len(ori),len(blob_idx),frame,length)   
        # order = [(A1,B1),....,(An,Bn)] An : idx of ori,Bn : idx of blob_idx   
        if len(ori)>len(live_blobs):                                                 # new blobs come in
            NB_idx = list(set(range(len(ori))).difference([aa[0] for aa in order]))  # new blobs ori idx 
            ini_y = [ori[i][0] for i in NB_idx]
            ini_x = [ori[i][1] for i in NB_idx]
    elif (len(ori)>0) & (len(live_blobs)==0):        # multiple new objs come in  
          order = [(i,i) for i in range(len(ori))]
          ini_y = [ori[i][0] for i in range(len(ori))]
          ini_x = [ori[i][1] for i in range(len(ori))]
    else:   # case 1 : (len(ori)==0) & (len(blob_idx)>0)              
            # case 2 : (len(ori)==0) & (len(blob_idx)=0)
          print("do nothing") 

    for i in range(len(ori)-len(live_blobs)):           #if new objs come in  
        blobs[obj_idx] = Blob(frame.shape,ini_x[i],ini_y[i])              #initialize new blob             
        if len(NB_idx):                  
            order.append((NB_idx[i],len(live_blobs)))
        live_blobs.append(obj_idx)
        blob_idx.append(obj_idx)
        obj_idx += 1   

    if (len(ori)<len(blob_idx)) & (len(ori)!=0):
        #o_idx = [a[1] for a in order]
        blob_idx = [blob_idx[i] for i in [a[1] for a in order] ]          #find the remaining blobs between 2 lists

    oidx = [k[1] for k in order]
    for _,i in enumerate(live_blobs):  
        if blobs[i].dtime<5:
            if len(ori)!=0:
                         
                if i in blob_idx:
                    jj = oidx.index(live_blobs.index(i))                                          
                    ori[order[jj][0]][2:] = array([ori[order[jj][0]][0:2]])-blobs[i].ref[0][0:2]  #measure the velocity
                    #if len(ori) == len(blob_idx): 
                    blobs[i].ref = array([ori[order[jj][0]]])
                    blobs[i].len = length[order[jj][0]]
                    #blobs[i].dtime = max(0,blobs[i].dtime - 1)
                else:
                    blobs[i].ref = blobs[i].x            # objs which are temporally hided
                    blobs[i].dtime += 1
            else:
                blobs[i].dtime += 1

            ulx = int(round(blobs[i].xp[1]))
            lrx = int(round(blobs[i].xp[1]+blobs[i].len[1]))
            uly = int(round(blobs[i].xp[0]))
            lry = int(round(blobs[i].xp[0]+blobs[i].len[0]))
            # draw predict
            if ((lrx>=0) & (lry>=0) & (ulx<frame.shape[1]) & (uly<frame.shape[0])): #at least part of the obj inside the view 
                cv2.rectangle(frame,(max(ulx,0) ,max(uly,0)),\
                                    (min(lrx,frame.shape[1]),min(lry,frame.shape[0])),\
                                     blobs[i].Color(),1)
                blobs[i].ivalue.append(frame[max(uly,0):min(lry,frame.shape[0]),\
                                             max(ulx,0):min(lrx,frame.shape[1]),:].flatten().mean())  #avarge itensity value in Bbox   
                #try:
                trj_tmpx = copy.deepcopy(blobs[i].Trj['x'])
                trj_tmpy = copy.deepcopy(blobs[i].Trj['y'])
                if Trj_type == 1:  # from prediction 
                    trj_tmpy.append( [ int(min(max(round((uly+lry)/2),0),frame.shape[0]))])
                    trj_tmpx.append( [ int(min(max(round((ulx+lrx)/2),0),frame.shape[1]))])
                else:    #from measurment          
                    trj_tmpy.append( [ int(min(max(round(blobs[i].ref[0][0]+blobs[i].len[0]/2),0),frame.shape[0]))])
                    trj_tmpx.append( [ int(min(max(round(blobs[i].ref[0][1]+blobs[i].len[1]/2),0),frame.shape[1]))]) 
                ''' 
                except:
 
                    if Trj_type== 1:
                        trj_tmpy = [[int(min(max(round((uly+lry)/2),0),frame.shape[0]))]]
                        trj_tmpx = [[int(min(max(round((ulx+lrx)/2),0),frame.shape[1]))]] 
                    else:
                        trj_tmpy = [[int(min(max(round(blobs[i].ref[0][0]+blobs[i].len[0]/2),0),frame.shape[0]))]]
                        trj_tmpx = [[int(min(max(round(blobs[i].ref[0][1]+blobs[i].len[1]/2),0),frame.shape[1]))]]
                ''' 
                #blobs[i].Trj['x'] = trj_tmpx
                #blobs[i].Trj['y'] = trj_tmpy
            else:      #obj outside the view
                trj_tmpx = copy.deepcopy(blobs[i].Trj['x'])
                trj_tmpy = copy.deepcopy(blobs[i].Trj['y'])
                trj_tmpx.append(trj_tmpx[-1])
                trj_tmpy.append(trj_tmpy[-1])
                #blobs[i].Trj['x'] = trj_tmpx
                #blobs[i].Trj['y'] = trj_tmpy
                blobs[i].dtime +=1
            blobs[i].Trj['x'] = trj_tmpx
            blobs[i].Trj['y'] = trj_tmpy
 
            f_no = copy.deepcopy(blobs[i].Trj['frame'])
            f_no.append(vid_idx)
            blobs[i].Trj['frame'] = f_no

            # Draw Trj
            if draw_Trj :
                if (len(blobs[i].Trj)>4 & blobs[i].dtime<5):
                    plt.subplot(121)
                    lines = axL.plot(blobs[i].Trj['x'][4:],blobs[i].Trj['y'][4:],color = array(blobs[i].Color())[::-1]/255.,linewidth=2)
                line_exist = 1   
            if flag ==1:        
                PP = np.dot(np.dot(A,blobs[i].P),A.T)+Q                    #covariance  
                Y = blobs[i].ref.T-np.dot(H,blobs[i].xp)                            #residual 
                S = np.dot(np.dot(H,PP),H.T)+R                    #covariance
                K = np.dot(np.dot(PP,H.T),np.linalg.inv(S))       #kalman gain     
                #update  state & covariance
                blobs[i].x = (blobs[i].xp+np.dot(K,Y)).T
                blobs[i].P = np.dot((I-np.dot(K,H)),PP)
            else: # if measument  can not obtain
                blobs[i].x = blobs[i].xp.T
            #update velocity & accelation 
            if vid_idx>0:
                try :                                 
                    blobs[i].x[0][2] = blobs[i].x[0][0]-blobs[i].x_old[0][0]
                    blobs[i].x[0][3] = blobs[i].x[0][1]-blobs[i].x_old[0][1]
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

    # erase the Trj    
    while line_exist & kill_Trj:
        try:
            axL.lines.pop(0)
            plt.show()
        except:
            line_exist = 0
            #print('No Trj need to erase ~~~')

  
    if imsave:
        imc = Image.fromarray(frame[:,:,::-1].astype(np.uint8))
        imc.save('/home/andyc/image/tra/2 balls/result/c%.3d.jpg'%vid_idx)

def Objmatch(objy,objx,ref,refc,L,W,im,length):
    dmtx = np.zeros((L,W))
    cmtx = np.zeros((L,W))
    Wd = 0.5   #weight of distance                                                                                                    
    Wc = 1-Wd  #weight of color                                                                                                       

    for i in range(L):
        dmtx[i,:] =((objy - ref[i][0])**2+(objx - ref[i][1])**2).T
        cmtx[i,:] = Color_Dif(im,objx,objy,refc[i],length[i])
    dmtx[dmtx>400] = 10**6
    cmtx = cmtx*Wc + dmtx*Wd
    m = Munkres()
    if L<=W:
        indexes = m.compute(cmtx)
    else:     # len(ori) > # live_blobs                                                                                               
        indexes = m.compute(cmtx.T)
        indexes = [(s[1],s[0]) for s in indexes]
    return indexes

def Color_Dif(im,Cx,Cy,color,Blen):
    result = []
    for ii in range(len(Cx)):
        result.append( (( im[max(Cy[ii],0) : min(Cy[ii]+Blen[0],im.shape[0]),
                             max(Cx[ii],0) : min(Cx[ii]+Blen[1],im.shape[1]),:].mean(1).mean(0)-color)**2).mean())
    result = array(result)
    result[np.isnan(result)] = 10**6
    return result


try:
    vid
except:
    print('reading video...')
    cap = cv2.VideoCapture('/home/andyc/jaystr.avi')
    vid = []

    if cap.isOpened():
        rval,frame = cap.read()
    else:
        rval = False
    while rval:
        rval,frame = cap.read()
        if rval:
            frame = cv2.resize(frame,(0,0),fx = scale,fy=scale)
            vid.append(frame)

    print('done reading video...')

mu       = np.zeros(vid[0].shape,dtype=float)
mu_old   = np.zeros(vid[0].shape,dtype=float)
sig2     = np.zeros(vid[0].shape,dtype=float)
sig2_old = np.zeros(vid[0].shape,dtype=float)

plt.figure(1,figsize=[20,10])
axL = plt.subplot(121)
left = plt.imshow(vid[0][:,:,::-1])
axis('off')
axR = plt.subplot(122)
right = plt.imshow(mu,clim=[0,1],cmap = 'gist_gray',interpolation='nearest')
axis('off')

for frame in vid[:]:
    print("frame no : {0}".format(vid_idx))
    Fg = Fg_Extract(frame,2)
    idx,lab_mtx,coor,cnt = Objextract(Fg)
    Coor_Extract(idx,lab_mtx,frame,coor)
    vid_idx = vid_idx+1

