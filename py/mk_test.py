#kalman tracking
import numpy as np
import cv2,pickle,pdb,copy
import scipy.ndimage.morphology as ndm
import scipy.ndimage as nd
import numpy.random as rand
from PIL import Image
from munkres import Munkres
from scipy.ndimage.filters import gaussian_filter as gf

#bg parameters
alpha    = 0.01
lmc      = 15.
bmc      = 25.
    
#kalman parameters
R = 0.01*np.eye(4) #noise
dt = 1
A = array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
H = array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
Q = 0.1*np.eye(4) #noise
B = array([[0.5*dt**2,0,0,0],[0,0.5*dt**2,0,0],[0,0,dt,0],[0,0,0,dt]])

vid_idx=0

flag = 1
I = np.eye(4)

imsave = 1
draw_Trj = 1
kill_Trj = 1

scale = 0.75
maskon = 1

blobs = {}
obj_idx = 0
live_blobs = []


F_case = 0
M_case = 0
B_case = 0


#constant
fps = 4
SL  = 40        #speed limit km/h
C_len_p = 45*scale    #lenth of car in pixel #Day : 45(scale:1) #Feb11 : 182.4(scale:1)
C_wid_p = 34*scale    #Width of car in pixel #Day : 34(scale:1) #Feb11 : 106(scale:1) 
C_len_r = 3     #avg lenth of car in real world (unit in meter) 
D_limit_s = (SL*1000/3600/fps*C_len_p/C_len_r)**2 #(distance(unit in pixels) which car at most can move per frame) ^2

C_size = C_len_p*C_wid_p  #area of car in pixel
Bs_lb = C_size/4          #low bound of blobs size
Bs_factor = 6             #Bus size factor (area of common bus/area of common car)
Bs_ub = Bs_lb*Bs_factor   #upper bound of blobs size 


mask = pickle.load(open("Dayresize_mask.pkl","rb"))
mask = cv2.resize(mask,(0,0),fx = scale,fy=scale)


#AA = pickle.load(open("AA.pkl","rb"))

class Blob(): #create a new blob                                                                                                          

    def __init__(self,fshape,ini_x,ini_y):
        self.x = []                #current status
        self.x_old  = []           #previous  status
        self.xp = np.array([[ini_y,ini_x,0,0]]).T
        self.u  = array([[0,0,0,0]]).T
        self.P = 100*np.eye(4)
        self.Trj = {}
        self.len = [0,0]                #Bbox size                                                                                       
        self.ref = array([[ini_y,ini_x,0,0]])   #Bbox position and velocity x,y,vx,vy                                                          
        self.ori = []                   #Bbox measurment position
        self.status = 0                 # 0 : as initial  1 : as live  2 : as dead     
        self.dtime = 0                  # obj dispear period                                                                              
        self.ivalue = []                # store intensity value in Bbox
        self.RGB = []
 
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


def Fg_Extract(frame,type = 1,trun = 100): #extract foreground    
    
    global BG_old
    global F_case
    global M_case
    global B_case


    if type ==1:
        mu[:]       = alpha*frame + (1.0-alpha)*mu_old
        mu_old[:]   = mu
        sig2[:]     = alpha*(1.0*frame-mu)**2 + (1.0-alpha)*sig2_old
        sig2_old[:] = sig2        
        sig = sig2**0.5
        lmcs = lmc*sig
        bmcs = bmc*sig       
        sig_factor = 1
        #pdb.set_trace() 
        fg= (np.abs(1.0*frame-mu)[:,:,0]-sig_factor*sig[:,:,0]>0.0) +\
            (np.abs(1.0*frame-mu)[:,:,1]-sig_factor*sig[:,:,1]>0.0) +\
            (np.abs(1.0*frame-mu)[:,:,2]-sig_factor*sig[:,:,2]>0.0)
    elif type == 2:
        try:
            fg = np.abs(1.0*frame.mean(2)-BG)>50.0
        except:
            BG = pickle.load(open("Feb11_bg.pkl","rb"))
            BG = cv2.resize(BG,(0,0),fx = scale,fy=scale)
            fg = np.abs(1.0*frame.mean(2)-BG)>30.0

    elif type == 3 : #runningmean
        if len(vid)>trun:
            if (vid_idx-round(trun/2))<0:
                if F_case == 1: 
                    BG = BG_old
                else:
                    LB = 0
                    UB = trun-1
                    BG = array(vid[LB:UB+1]).mean(0)    
                    F_case = 1

            elif len(vid)-vid_idx<=(trun-1):
                if B_case == 1:
                    BG = BG_old
                else: 
                    LB = len(vid)-(trun-1)
                    UB = len(vid)
                    BG = array(vid[LB:UB+1]).mean(0)
                    B_case = 1                    
            else:
                if M_case == 1:
                    BG = BG_old + array(vid[vid_idx+int(trun/2)])/trun\
                                - array(vid[vid_idx-int(trun/2)+1])/trun   
                else:
                    LB = vid_idx-int(trun/2)+1
                    UB = vid_idx+int(trun/2)
                    #pdb.set_trace()
                    BG = array(vid[LB:UB+1]).mean(0)
                    M_case = 1

            fg = (np.abs(1.0*frame[:,:,0]-BG[:,:,0])>28.0)+\
                 (np.abs(1.0*frame[:,:,1]-BG[:,:,1])>28.0)+\
                 (np.abs(1.0*frame[:,:,2]-BG[:,:,2])>28.0)
        else:
            print('select truncation is larger then the sequence....')
            BG = array(vid).mean(0)
            fg = (np.abs(1.0*frame[:,:,0]-BG[:,:,0])>30.0)+\
                 (np.abs(1.0*frame[:,:,1]-BG[:,:,1])>30.0)+\
                 (np.abs(1.0*frame[:,:,2]-BG[:,:,2])>30.0)

    elif type==4: 
        if len(vid)>trun:

            if len(vid)-vid_idx<=(trun-1):
                if B_case == 1:
                    BG = BG_old
                else:
                    LB = len(vid)-(trun-1)
                    UB = len(vid)
                    BG = array(vid[LB:UB+1]).mean(0)
                    B_case = 1
            else:             
                if F_case == 1:
                    BG = BG_old + array(vid[vid_idx+trun-1])/trun\
                                - array(vid[vid_idx-1])/trun
                else:
                    LB = 0
                    UB = trun-1
                    BG = array(vid[LB:UB+1]).mean(0)
                    F_case = 1

            fg = (np.abs(1.0*frame[:,:,0]-BG[:,:,0])>30.0)+\
                 (np.abs(1.0*frame[:,:,1]-BG[:,:,1])>30.0)+\
                 (np.abs(1.0*frame[:,:,2]-BG[:,:,2])>30.0)
        else:
            print('select truncation is larger then the sequence....')
            BG = array(vid).mean(0)
            fg = (np.abs(1.0*frame[:,:,0]-BG[:,:,0])>30.0)+\
                 (np.abs(1.0*frame[:,:,1]-BG[:,:,1])>30.0)+\
                 (np.abs(1.0*frame[:,:,2]-BG[:,:,2])>30.0)


    if maskon:
       fg = fg*mask

    fgo = ndm.binary_opening(fg)
    fgf = ndm.binary_fill_holes(fgo)
    right.set_data(fgf)
    plt.draw()
    BG_old = BG       
    return fgf
        
def Objextract(Fg):

    s = nd.generate_binary_structure(2,2)
    labeled_array, num_features = nd.measurements.label(Fg, structure=s)
    coor = []
    cnt = []
    if num_features == 0:
       idx = []
    else:    
        for i in range(1,num_features+1):
            coor.append(np.where(labeled_array==i))
            cnt.append(len(np.where(labeled_array==i)[1]))
        cnt = array(cnt)
        idx = arange(num_features)
        idx = idx[(cnt<Bs_ub)&(cnt>Bs_lb)]
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

            length[i]   = [lry-uly,lrx-ulx]      
            #ori[i][0:2] = [uly,ulx]  
            ori[i][0:2] = [uly+int(length[i][0]/2),ulx+int(length[i][1]/2)]


            ori_color[i] = array([frame[uly:lry,ulx:lrx,0].flatten().mean(),\
                                  frame[uly:lry,ulx:lrx,1].flatten().mean(),\
                                  frame[uly:lry,ulx:lrx,2].flatten().mean()]) 
            
            cv2.rectangle(frame,(ulx,uly),(lrx,lry),(0,255,0),1)

    left.set_data(frame)
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
    ori_idx = []
    line_exist =0

   
    for _,i in enumerate(live_blobs):
        if blobs[i].dtime <8:
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
            #live_blobs = list(set(live_blobs).difference([i]))
 
    if (len(ori)!=0) & (len(live_blobs)!=0):                      
        # order = [(A1,B1),....,(An,Bn)] An : idx of ori,Bn : idx of blob_idx    
        order,ori_idx = Objmatch(ycor,xcor,ori,ori_color,len(ori),len(blob_idx),frame,length) 
        if (len(ori)>len(live_blobs)) or (len(ori_idx)>0):                           # new blobs come in
            NB_idx = list(set(range(len(ori))).difference([aa[0] for aa in order]))  # new blobs ori idx 
            ini_y = [ori[i][0] for i in NB_idx]
            ini_x = [ori[i][1] for i in NB_idx]
    elif (len(ori)>0) & (len(live_blobs)==0):        # multiple new objs come in  
          order = [(i,i) for i in range(len(ori))]
          ini_y = [ori[i][0] for i in range(len(ori))]
          ini_x = [ori[i][1] for i in range(len(ori))]
    else:   # case 1 : (len(ori)==0) & (len(blob_idx)>0)              
            # case 2 : (len(ori)==0) & (len(blob_idx)=0)
          print(" ") 

    for i in range(len(ori)-len(live_blobs)+len(ori_idx)):                #if new objs come in  
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
        if blobs[i].dtime<8:
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

            #ulx = int(round(blobs[i].xp[1]))
            #lrx = int(round(blobs[i].xp[1]+blobs[i].len[1]))
            #uly = int(round(blobs[i].xp[0]))
            #lry = int(round(blobs[i].xp[0]+blobs[i].len[0]))
 
            ulx = int(blobs[i].xp[1]-blobs[i].len[1]/2)
            lrx = int(blobs[i].xp[1]+blobs[i].len[1]/2)
            uly = int(blobs[i].xp[0]-blobs[i].len[0]/2)
            lry = int(blobs[i].xp[0]+blobs[i].len[0]/2)


            # draw predict
            if ((lrx>=0) & (lry>=0) & (ulx<frame.shape[1]) & (uly<frame.shape[0])): #at least part of the obj inside the view 
                #cv2.rectangle(frame,(max(ulx,0) ,max(uly,0)),\
                #                    (min(lrx,frame.shape[1]),min(lry,frame.shape[0])),\
                #                     blobs[i].Color(),1)
                blobs[i].ivalue.append(frame[max(uly,0):min(lry,frame.shape[0]),\
                                             max(ulx,0):min(lrx,frame.shape[1]),:].flatten().mean())  #avarge itensity value in Bbox
                trj_tmpx = copy.deepcopy(blobs[i].Trj['x'])
                trj_tmpy = copy.deepcopy(blobs[i].Trj['y'])
                     
                #trj_tmpy.append( [ int(min(max(round(blobs[i].ref[0][0]+blobs[i].len[0]/2),0),frame.shape[0]))])
                #trj_tmpx.append( [ int(min(max(round(blobs[i].ref[0][1]+blobs[i].len[1]/2),0),frame.shape[1]))])
                trj_tmpy.append( [ int(min(max(blobs[i].ref[0][0],0),frame.shape[0]))])                          
                trj_tmpx.append( [ int(min(max(blobs[i].ref[0][1],0),frame.shape[1]))]) 

            else:      #obj outside the view
                trj_tmpx = copy.deepcopy(blobs[i].Trj['x'])
                trj_tmpy = copy.deepcopy(blobs[i].Trj['y'])
                trj_tmpx.append(trj_tmpx[-1])
                trj_tmpy.append(trj_tmpy[-1])
                blobs[i].dtime +=1 

            blobs[i].Trj['x'] = trj_tmpx
            blobs[i].Trj['y'] = trj_tmpy
            f_no = copy.deepcopy(blobs[i].Trj['frame'])
            f_no.append(vid_idx)
            blobs[i].Trj['frame'] = f_no

            # Draw Trj
            if draw_Trj :
                if ((len(blobs[i].Trj)>0) & (blobs[i].dtime<8) ):# & (i in [23,24])):
                    plt.subplot(121)
                    lines = axL.plot(blobs[i].Trj['x'][0:],blobs[i].Trj['y'][0:],color = array(blobs[i].Color())[::-1]/255.,linewidth=2)
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
    left.set_data(frame)
    plt.draw()       


    if imsave:
        savename = '/home/andyc/image/tra/Day/c'+repr(vid_idx).zfill(3)+'.jpg'
        savefig(savename)

    # erase the Trj    
    while line_exist & kill_Trj:  
        try: 
            axL.lines.pop(0)
            plt.show()
        except:
            line_exist = 0
            #print('No more Trj in the image~~~')

def Objmatch(objy,objx,ref,refc,L,W,im,length):
    global D_limit_s
    dmtx = np.zeros((L,W))
    cmtx = np.zeros((L,W))
    Wd = 0.5   #weight of distance
    Wc = 1-Wd  #weight of color
    
    for i in range(L):
        dmtx[i,:] =((objy - ref[i][0])**2+(objx - ref[i][1])**2).T
        cmtx[i,:] = Color_Dif(im,objx,objy,refc[i],length[i]) 
    dmtx[dmtx>D_limit_s] = 10**6    
    cmtx = cmtx*Wc + dmtx*Wd
    tmp = copy.deepcopy(cmtx)     
    m = Munkres()
    if L<=W:
        indexes = m.compute(cmtx)
    else:     # len(ori) > # live_blobs
        indexes = m.compute(cmtx.T)
        indexes = [(s[1],s[0]) for s in indexes]
    
    D_idx = []
    if vid_idx>=0:
        for i in range(len(indexes))[::-1]:
            if tmp[indexes[i][0],indexes[i][1]]>10**5:
                D_idx.append(i)
                indexes.pop(i)

    return indexes,D_idx  

def Color_Dif(im,Cx,Cy,color,Blen):
    result = []
    for ii in range(len(Cx)):
        result.append( (( im[max(Cy[ii],0) : min(Cy[ii]+Blen[0],im.shape[0]),\
                             max(Cx[ii],0) : min(Cx[ii]+Blen[1],im.shape[1]),:].mean(1).mean(0)-color)**2).mean())
    result = array(result)
    result[np.isnan(result)] = 10**6
    return result

'''
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

for frame in vid:
    print("frame no : {0}".format(vid_idx))
    Fg = Fg_Extract(frame,2)
    idx,lab_mtx,coor,cnt = Objextract(Fg)
    Coor_Extract(idx,lab_mtx,frame,coor)
    vid_idx = vid_idx+1


'''
import os,glob

path ='/home/andyc/image/Day resize/'
#path ='/home/andyc/image/park/bip roof/'
imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))
im = nd.imread(imlist[0])
im = cv2.resize(im,(0,0),fx = scale,fy=scale)
mu       = np.zeros(im.shape,dtype=float)
mu_old   = np.zeros(im.shape,dtype=float)
sig2     = np.zeros(im.shape,dtype=float)
sig2_old = np.zeros(im.shape,dtype=float)
frame    = np.zeros(im.shape,dtype=float)
vid = []

plt.figure(1,figsize=[20,10])                                                                                                            
axL = plt.subplot(121)                                                                                                                  
left = plt.imshow(im[:,:,::-1])                                                                                                    
axis('off')                                                                                                                               
axR = plt.subplot(122)                                                                                                            
right = plt.imshow(mu,clim=[0,1],cmap = 'gist_gray',interpolation='nearest')
axis('off')   

print('reading video...')
 
for ii in range(len(imlist)):
    frame = nd.imread(imlist[ii])
    frame = cv2.resize(frame,(0,0),fx = scale,fy=scale)
    vid.append(frame)
print('done reading frame...')

for frame in vid[0:175]:
    print("frame no")
    print(vid_idx)

    #frame = cv2.resize(frame,(0,0),fx = scale,fy=scale)
    Fg = Fg_Extract(frame,4)
    idx,lab_mtx,coor,cnt = Objextract(Fg)
    Coor_Extract(idx,lab_mtx,frame,coor)
    vid_idx = vid_idx+1

