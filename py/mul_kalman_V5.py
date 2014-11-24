#kalman tracking
import numpy as np
import cv2,pickle,pdb,copy,time
import scipy.ndimage.morphology as ndm
import scipy.ndimage as nd
import numpy.random as rand
from PIL import Image
from munkres import Munkres
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.ndimage.filters import median_filter as mf
from numpy.linalg import pinv as pinv
from scipy.spatial.distance import mahalanobis as mdist
from skimage.color.colorconv import *



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

imsave = 0
draw_Trj = 1
kill_Trj = 1

scale = 1
maskon = 1

blobs = {}
obj_idx = 0
live_blobs = []


F_case = 0
M_case = 0
B_case = 0


#constant
fps = 25
SL  = 100        #speed limit km/h
C_len_p = 82*scale    #lenth of car in pixel #Day : 45 #Feb11 : 182.4 #crash : 82
C_wid_p = 31*scale    #Width of car in pixel #Day : 34 #Feb11 : 106 #crash : 31 
C_len_r = 3     #avg lenth of car in real world (unit in meter) 
#D_limit_s = ((SL*1000/3600/fps*C_len_p/C_len_r)**2*3)/diag  #(distance(unit in pixels) which car at most can move per frame) ^2
                                                          # normalize the distance by divid diag 
C_size = C_len_p*C_wid_p  #area of car in pixel
Bs_lb = C_size/4.5          #low bound of blobs size
Bs_factor = 6             #Bus size factor (area of common bus/area of common car)
Bs_ub = Bs_lb*Bs_factor*2   #upper bound of blobs size 




#mask = pickle.load(open("/home/andyc/tracking/py/mask/TLC0005_mask.pkl","rb"))
mask = pickle.load(open("./mask/caraccident_mask.pkl","rb"))
mask = cv2.resize(mask,(0,0),fx = scale,fy=scale)
Rmask = mask==0  #inversed mask
mask2 = pickle.load(open("./mask/caraccident_mask_1.pkl","rb"))
mask2 = cv2.resize(mask2,(0,0),fx = scale,fy=scale)


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

def ShadowRm(fg,frame,bg,N,h,w,th):
    NCC= np.zeros([h,w])
    BG = np.zeros([h+2*N,w+2*N,3])
    FG = np.zeros([h*2*N,w+2*N,3])
    BG[N:h+N,N:w+N,:]=bg
    FG[N:h+N,N:w+N,:]=frame
    for ii in range(h):
        for jj in range(w):
            if fg[ii,jj]==1:
                ER = sum((BG[ii-N:ii+N,jj-N:jj+N,:]*FG[ii-N:ii+N,jj-N:jj+N,:]).flatten())
                EB = (sum((BG[ii-N:ii+N,jj-N:jj+N,:]**2).flatten()))**0.5
                ET = (sum((FG[ii-N:ii+N,jj-N:jj+N,:]**2).flatten()))**0.5
                if (ER/EB/ET) >th:
                    NCC[ii,jj] = 1
    return NCC


def Fg_Extract(frame,type = 1,trun = 100): #extract foreground    
    
    global BG_old
    global F_case
    global M_case
    global B_case


    if type ==1:    # training BG
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

    elif type == 2: # avg total seq (need saved file)
            print('bg')
            BG = pickle.load(open("./BG/20141031.pkl","rb"))
            BG = cv2.resize(BG,(0,0),fx = scale,fy=scale)
            fg = (np.abs(1.0*frame[:,:,0]-BG[:,:,0])>50.)+\
                 (np.abs(1.0*frame[:,:,1]-BG[:,:,1])>50.)+\
                 (np.abs(1.0*frame[:,:,2]-BG[:,:,2])>50.)
    elif type == 2.5 : #avg total seq (need saved file) normalized image

        BG = pickle.load(open("./BG/car accident.pkl","rb"))
        BG = cv2.resize(BG,(0,0),fx = scale,fy=scale)
        
        fcal = np.zeros(BG.shape)     

        BG_r_avg = BG[:,:,0][Rmask].mean()
        BG_g_avg = BG[:,:,1][Rmask].mean()
        BG_b_avg = BG[:,:,2][Rmask].mean()

        BG_r_std = BG[:,:,0][Rmask].std()
        BG_g_std = BG[:,:,1][Rmask].std()
        BG_b_std = BG[:,:,2][Rmask].std()

        f_r_avg = frame[:,:,0][Rmask].mean()
        f_g_avg = frame[:,:,1][Rmask].mean()
        f_b_avg = frame[:,:,2][Rmask].mean()

        f_r_std = frame[:,:,0][Rmask].std()
        f_g_std = frame[:,:,1][Rmask].std()
        f_b_std = frame[:,:,2][Rmask].std()
       
        fcal[:,:,0] = (frame[:,:,0]-f_r_avg)/f_r_std*BG_r_std+BG_r_avg
        fcal[:,:,1] = (frame[:,:,1]-f_g_avg)/f_g_std*BG_g_std+BG_g_avg
        fcal[:,:,2] = (frame[:,:,2]-f_b_avg)/f_b_std*BG_b_std+BG_b_avg

        dif = abs(fcal-BG)
        fg  = ((dif[:,:,1]>20.)|(dif[:,:,2]>30.)) 
  



    elif type == 3 : #truncation mean 1. 
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

            fg = (np.abs(1.0*frame[:,:,0]-BG[:,:,0])>25.0)+\
                 (np.abs(1.0*frame[:,:,1]-BG[:,:,1])>25.0)+\
                 (np.abs(1.0*frame[:,:,2]-BG[:,:,2])>25.0)
        else:
            print('select truncation is larger then the sequence....')
            BG = array(vid).mean(0)
            fg = (np.abs(1.0*frame[:,:,0]-BG[:,:,0])>30.0)+\
                 (np.abs(1.0*frame[:,:,1]-BG[:,:,1])>30.0)+\
                 (np.abs(1.0*frame[:,:,2]-BG[:,:,2])>30.0)

    elif type==4:  #truncation mean 2.
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

    elif type == 5:
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

            fg = (np.abs(1.0*frame[:,:,0]-BG[:,:,0])>25.0)+\
                 (np.abs(1.0*frame[:,:,1]-BG[:,:,1])>25.0)+\
                 (np.abs(1.0*frame[:,:,2]-BG[:,:,2])>25.0)  

            #fg = fg * ShadowRm(fg,frame,BG,5,frame.shape[0],frame.shape[1],20)

    elif type == 6:
        global buf,Tf 
        N = 25
        trun = N*2+1
        ind =range(trun)
        ind.pop(trun/2)
        buf[-1] = vid[vid_idx+N+1]  
        
        #pdb.set_trace()
        if len(vid)>trun:

            BG = np.abs(buf[ind]-buf[trun/2]).mean(3).mean(0)
            fg = (BG>40.)*Tf                     
            buf=np.roll(buf,-1,0)
        else:
            print('select truncation is larger then the sequence....')
        #pdb.set_trace()

    if maskon:
       fg = fg*mask*mask2


    #fgo = ndm.binary_opening(fg,np.ones([5,5]))
    fgo = ndm.binary_dilation(fg,np.ones([2,2]))
    fgf = ndm.binary_fill_holes(fgo)
    right.set_data(mf(fgf,5))
    plt.draw()
    if (type!=1)&(type!=5):
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
        # order = [(A1,B1),....,(An,Bn)] An : idx of ori(Measurement),Bn : idx of blob_idx(predict)    
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
                if ((len(blobs[i].Trj)>0) & (blobs[i].dtime<8)  & (i in [0,3])):
                    plt.subplot(121)
                    lines = axL.plot(blobs[i].Trj['x'][0:],blobs[i].Trj['y'][0:],\
                                     color = array(blobs[i].Color())[::-1]/255.,linewidth=2)
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
        savename = '/home/andyc/image/tra/caraccident/c'+repr(vid_idx).zfill(3)+'.jpg'
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
    #if vid_idx >= 138 :  
    #    pdb.set_trace()
    dmtx = dmtx/diag # normalize the distance by divid diag
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
        pc = im[max(Cy[ii]-int(Blen[0]/2),0) : min(Cy[ii]+int(Blen[0]/2),im.shape[0]),\
                max(Cx[ii]-int(Blen[1]/2),0) : min(Cx[ii]+int(Blen[1]/2),im.shape[1]),:].mean(1).mean(0)
        result.append( (( (pc - pc.min()) - (color -color.min()) )**2).mean()/255)            
    result = array(result)
    result[result>0.7] = 10**6
    result[np.isnan(result)] = 10**6
    return result

def Mdist(pc,mc):
    Imtx = eye(3)*10**(-6)
    A = np.cov(array([pc,mc]).T)
    return mdist(pc,mc,np.dot(inv(np.dot(A.T,A)+Imtx),A.T))     


'''
fno = 0

try:
    vid
except:
    print('reading video...')
    cap = cv2.VideoCapture('/home/andyc/TLC00005.AVI')
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
            fno+=1

    print('done reading video...')

    print('There are {0} frames in this video...'.format(fno))

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

diag = (vid[0].shape[0]**2+vid[0].shape[1]**2)**0.5
D_limit_s = ((SL*1000/3600/fps*C_len_p/C_len_r)**2*3)/diag  

start = time.clock()

vid_idx = 15220
buf = double(array(vid[vid_idx -25:vid_idx +26]))

Tf = (vid[0][:,:,1]>100) & (vid[0][:,:,0]<100)
Tf = ndm.binary_closing(Tf,structure=np.ones((4,4)))
Tf = ~ndm.binary_fill_holes(Tf)


for frame in vid[15220:]:
    print("frame no : {0}".format(vid_idx))
    Fg = Fg_Extract(frame,6)
    idx,lab_mtx,coor,cnt = Objextract(Fg)
    Coor_Extract(idx,lab_mtx,frame,coor)
    vid_idx = vid_idx+1
end = time.clock()


print("Total computation time is {0}".formate(start-end))

'''
import os,glob

path ='/home/andyc/image/car accident/'
#path ='/home/andyc/image/park/bip roof/'
imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))
im = nd.imread(imlist[0])
im = cv2.resize(im,(0,0),fx = scale,fy=scale)
diag = (im.shape[0]**2+im.shape[1]**2)**0.5
mu       = np.zeros(im.shape,dtype=float)
mu_old   = np.zeros(im.shape,dtype=float)
sig2     = np.zeros(im.shape,dtype=float)
sig2_old = np.zeros(im.shape,dtype=float)
frame    = np.zeros(im.shape,dtype=float)
vid = []

plt.figure(1,figsize=[20,10])                                                                                                            
axL = plt.subplot(121)                                                                                                                  
left = plt.imshow(im)                                                                                                    
axis('off')                                                                                                                               
axR = plt.subplot(122)                                                                                                            
right = plt.imshow(mu,clim=[0,1],cmap = 'gist_gray',interpolation='nearest')
axis('off')   

print('reading video...')

D_limit_s = ((SL*1000/3600/fps*C_len_p/C_len_r)**2*3)/diag
 
for ii in range(len(imlist)):
    frame = nd.imread(imlist[ii])
    frame = cv2.resize(frame,(0,0),fx = scale,fy=scale)
    vid.append(frame)
print('done reading frame...')


for frame in vid[200:270]:
    print("frame no")
    print(vid_idx+15)
    Fg = Fg_Extract(frame,2.5)    
    idx,lab_mtx,coor,cnt = Objextract(Fg)
    Coor_Extract(idx,lab_mtx,frame,coor)
    #if imsave:
    #    savename = '/home/andyc/image/tra/Daycut/c'+repr(vid_idx).zfill(3)+'.jpg'
    #    savefig(savename)
    vid_idx = vid_idx+1


