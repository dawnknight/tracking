import numpy as np
import cv2,pdb,pickle,sys,copy
import video
from common import anorm2, draw_str
from time import clock
from scipy.stats import mode


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )




video_src = '/home/andyc/Videos/jayst.mp4'

viewmask = pickle.load(open("./mask/jayst_mask.pkl","rb"))




track_len = 100
detect_interval = 5
tracks = []
cam = video.create_capture(video_src)
fno = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
nrow  = cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
ncol  = cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)

frame_idx = 0

groups={}
ptsidx = 0
objidx = 0
pts = {}
obj = {}
tnum = 0

class Objects():
    def __init__(self):
        self.ptsTrj= {}
        self.pts = []
        self.Trj = []
        self.xTrj = []
        self.yTrj = []
        self.frame = []
        self.vel = []
        self.pos = []
        self.status = 1   # 1: alive  2: dead

    def Color(self):
        try:
            self.R
        except:
            self.R = randint(0,255)
            self.G = randint(0,255)
            self.B = randint(0,255)
        return (self.R,self.G,self.B)


class Objectpts():
    def __init__(self):
        self.Trj= []
        self.vel = []
        self.pos = []
        self.pa  = 0
        self.color = []
        self.status = 1   # 1: alive  2: dead

def errcheck():
    err  = []
    err2 = []
    for i in range(len(pts)):
        if i not in obj[pts[i].pa].pts:
            err.append(i)
    for i in range(len(obj)):
        for j in obj[i].pts:
            if pts[j].pa != i:
                err2.append([i,j])        
    return err,err2




def Initpts(tracks,tnum,del_pts,nobj=0,dth=30,vth = 2):
    global ptsidx,objidx

    if ptsidx ==0:
        for i in range(len(tracks)):
            pts[i] = Objectpts()
            pts[i].Trj[:] = tracks[i]
            pts[i].vel[:] = array(tracks[i][1:])-array(tracks[i][0:-1])
        ptsidx += len(tracks)
    else:


        if (tnum == len(tracks)) & (len(del_pts)==0) : # no pts change => simple update trj                               
            for i in range(len(obj)):
                if obj[i].status == 1:
                    Trj_idx = 0
                    for nn in obj[i].pts:
                        obj[i].ptsTrj[Trj_idx] = tracks[nn]
                        pts[nn].Trj = tracks[nn]
                        obj[i].pos.append(pts[nn].Trj[-1])
                        Trj_idx+=1
                    obj[i].Trj.append(mean(obj[i].pos,0))
                    obj[i].xTrj.append(int(mean(obj[i].pos,0)[0]))
                    obj[i].yTrj.append(int(mean(obj[i].pos,0)[1]))
                    obj[i].pos = []
                    obj[i].frame.append(frame_idx)
                    

        else:   # somes pts change     

            Npts = len(del_pts)+len(tracks)-tnum  #number of the new pts           
            clist = np.zeros(len(tracks)+len(del_pts)).astype(int)
        
            # -- delete missing points' info -- #
            mobj = []  #modified objs list
            
            mobj = [pts[i].pa for i in del_pts]

            #if frame_idx == 7:
            #    pdb.set_trace()

            for i in del_pts[::-1]:

                clist[i:]+=1
                j = pts[i].pa # parent idx
                del_idx = obj[j].pts.index(i)  #del idx in obj                                                              
                obj[j].pts.pop(del_idx)

                for oidx in set(mobj):  # update pts number in obj which has pts been deleted
                    for pidx in range(len(obj[oidx].pts)):
                        if obj[oidx].pts[pidx]>i:
                            obj[oidx].pts[pidx] -= 1

                for kk in range(del_idx,len(obj[j].ptsTrj)-1):
                    obj[j].ptsTrj[kk]=obj[j].ptsTrj[kk+1]
             
                if range(del_idx,len(obj[j].ptsTrj)-1) != []:
                    del obj[j].ptsTrj[kk+1]
                else: # del_idx is the rightest element in the list                                                        
                    del obj[j].ptsTrj[del_idx]
                    
                for jj in range(i,len(pts)-1):
                    pts[jj]=pts[jj+1]    

                try:
                    del pts[jj+1]
                except: # i is the rightest element in the list  
                    del pts[i]
                 
            # -- updata group information -- #                                                                              
            for i in range(len(obj)):

                Trj_idx = 0
                if len(obj[i].pts) == 0:  # check if obj still alive
                    obj[i].status = 0
                else:    
                    if i not in mobj:
                        offset = []
                        for k in obj[i].pts:
                            #if ((frame_idx == 2) & (i == 14) & (k==186)):
                            #    pdb.set_trace()

                            tr = k-clist[k]     # maping old idx to new idx   

                            obj[i].ptsTrj[Trj_idx] = tracks[tr]
                            
                            pts[tr].Trj = tracks[tr] 
                            obj[i].pos.append(pts[tr].Trj[-1])
                            Trj_idx+=1
                            offset.append(clist[k])                        
                        # -- group pts index update -- #         
                        obj[i].pts[:] = list(array(obj[i].pts)-array(offset))

                    else:
                       
                        for ll in obj[i].pts:  #may modify => only consider number larger than del_pts

                            obj[i].ptsTrj[Trj_idx] = tracks[ll]   
                            pts[ll].Trj = tracks[ll]
                            obj[i].pos.append(pts[ll].Trj[-1])
                            Trj_idx+=1
                 
            # -- initial new pts -- #                                                                                          
            if Npts > 0:
       
                for i in range(len(tracks)-Npts,len(tracks)):
                    pts[i] = Objectpts()
                    pts[i].Trj[:] = tracks[i]
                    pts[i].vel[:] = array(tracks[i][1:])-array(tracks[i][0:-1])

            # -- try assign new pts to existing group -- #                                                                      
                vel = [(array(tracks[i][1:])-array(tracks[i][0:-1])).mean(0) for i in range(len(tracks))]
                pos = array([tracks[i][-1] for i in range(len(tracks))])
                idx = np.ones(len(tracks))
        
                for i in range(len(tracks)-Npts,len(tracks)):
                 
                    if idx[i] == 1:
                        posvec = (sum((pos - pos[i])**2,1)**0.5) < dth
                        velvec = (sum((vel - vel[i])**2,1)**0.5) < vth
                        idxpos = velvec & posvec
                 
                        if sum(idxpos[:len(tracks)-Npts]):  #new pts belong to old groups
                                
                            palist = [pts[j].pa for j in np.arange(len(idxpos))[idxpos]] #parent list
                            pa = int(mode(palist)[0][0]) # most common parent ....                                               
                            Trj_idx = len(obj[pa].ptsTrj)

                            for mm in np.arange(len(idxpos))[idxpos]:
                                if mm >= (len(tracks)-Npts):
                                    if idx[mm] == 1:

                                        obj[pa].ptsTrj[Trj_idx] = pts[mm].Trj
                                        obj[pa].pos.append(pts[mm].Trj[-1])
                                        obj[pa].pts.append(mm)
                                        pts[mm].pa = pa
                                        Trj_idx += 1

                        else:       # -- generating new group -- #              
                            Trj_idx = 0
                            for mm in np.arange(len(idxpos))[idxpos]:
                                if Trj_idx == 0:
                                    obj[objidx] = Objects()
                                if idx[mm]!=0:    
                                    obj[objidx].ptsTrj[Trj_idx] = pts[mm].Trj
                                    obj[objidx].pos.append(pts[mm].Trj[-1])
                                    obj[objidx].pts.append(mm)
                                    pts[mm].pa = objidx
                                    Trj_idx += 1
                                
                            obj[objidx].Trj.append( mean(obj[objidx].pos,0))
                            obj[objidx].xTrj.append(int(mean(obj[objidx].pos,0)[0]))
                            obj[objidx].yTrj.append(int(mean(obj[objidx].pos,0)[1]))
                            obj[objidx].pos = []
                            obj[objidx].frame.append(frame_idx)
                            objidx +=1

                        idx[velvec & posvec] = 0


            for i in range(nobj): #update exsiting obj info without new adding obj
                if len(obj[i].pts)==0:
                        obj[i].status =0
                else:
                    #if (frame_idx == 17) & (i ==13) :
                    #    pdb.set_trace()
                        
                    obj[i].Trj.append( mean(obj[i].pos,0))
                    obj[i].xTrj.append(int(mean(obj[i].pos,0)[0]))
                    obj[i].yTrj.append(int(mean(obj[i].pos,0)[1]))
                    obj[i].pos = []
                    obj[i].frame.append(frame_idx)
    #if frame_idx:
    #    pdb.set_trace()




def InitGrouping(tracks,pts,dth=30,vth=2):
    global objidx
    # calculate velocity                                                                                                                  
    vel = [(array(tracks[i][1:])-array(tracks[i][0:-1])).mean(0) for i in range(len(tracks))]
    pos = array([tracks[i][-1] for i in range(len(tracks))])
    idx = np.ones(len(tracks))

    for i in range(len(tracks)):
        if idx[i] == 1:
           
           posvec = (sum((pos - pos[i])**2,1)**0.5) < dth
           velvec = (sum((vel - vel[i])**2,1)**0.5) < vth

           obj[objidx] = Objects()
           Trj_idx = 0

           for ii in [i for i,j in enumerate((velvec & posvec)) if j ==True ]:

               if idx[ii] ==  1:
                   obj[objidx].ptsTrj[Trj_idx] = pts[ii].Trj
                   obj[objidx].pos.append(pts[ii].Trj[-1])
                   obj[objidx].pts.append(ii)
                   pts[ii].pa = objidx
                   Trj_idx+=1
 
           idx[velvec & posvec] = 0 

           obj[objidx].Trj.append( mean(obj[objidx].pos,0))
           obj[objidx].xTrj.append(int(mean(obj[objidx].pos,0)[0]))
           obj[objidx].yTrj.append(int(mean(obj[objidx].pos,0)[1]))
           #pdb.set_trace()
           obj[objidx].pos = []
           obj[objidx].frame.append(frame_idx)
           objidx +=1

        else:
            continue

plt.figure(1,figsize=[10,12])
axL = plt.subplot(1,1,1)

frame = np.zeros([nrow,ncol,3]).astype('uint8')
im = plt.imshow(np.zeros([nrow,ncol,3]))


axis('off')

tdic = [0]
ret = 'True'

import time
Tstart = time.time()



while (frame_idx<fno):
    print('frame {0}\r'.format(frame_idx)),
    sys.stdout.flush()
    
    ret, frame[:] = cam.read()

    #frame[:] = (((frame/255.)**0.5)*255).astype('uint8') 
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#*viewmask 

    vis = frame.copy()

    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1

        tmppts = p1.reshape(-1, 2)
        allpts = array([[int(tmppts[i][1]),int(tmppts[i][0])] for i in range(len(tmppts))]).T
        allpts[0][((allpts[0]>=nrow) | (allpts[0]<0))] = 0
        allpts[1][((allpts[1]>=ncol) | (allpts[1]<0))] = 0
        allpts = allpts.T 

        inside = array([viewmask[allpts[i][0],allpts[i][1]] for i in range(len(allpts))])
        goodinside = inside*good


        new_tracks = []
        del_pts = []
        idx = -1

        for tr, (x,y), good_flag in zip(tracks, tmppts, goodinside):
            idx += 1

            if not good_flag:  # feature gone
                #pdb.set_trace()
                del_pts.append(idx)                
                continue
            
            if not ((y >=ncol) or (y <0) or (y >=nrow) or (y <0)):
                if viewmask[int(y),int(x)]:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)


            tr.append((x, y))
            if len(tr) > track_len:
                del tr[0]
            new_tracks.append(tr)        

        tmp = []


        for i in del_pts:
            if i <=(tnum-1):
                tmp.append(i)
        del_pts[:] = tmp        

        tracks = new_tracks
           
        Initpts(tracks,tnum,del_pts,len(obj))

        tnum = len(tracks)

        if frame_idx<2:
            InitGrouping(tracks,pts,dth=30,vth=2)       
        
        im.set_data(frame[:,:,::-1])
 
        
        for d in range(len(obj)):
            if ((obj[d].status ==1)):# & (d  in [12,16])):  
                lines = axL.plot(obj[d].xTrj,obj[d].yTrj,color = array(obj[d].Color()[::-1])/255.,linewidth=2) 
                line_exist = 1   
        plt.draw()        

        name = '/home/andyc/image/jayst/group/'+str(frame_idx).zfill(5)+'.jpg'
        savefig(name)
            
        while line_exist :    
            try:
                axL.lines.pop(0)
                plt.show()
            except:
                line_exist = 0
        plt.show()
        

        print('\nframe   {0}'.format(frame_idx))
        print('# pts   {0}'.format(len(pts)))
        print('#delp   {0}'.format(len(del_pts)))
        print('#tracks {0}'.format(len(tracks)))
        print('err  {0}'.format(errcheck()))
        print('{0} objs'.format(len(obj)))        
        print('######################\n')



    if frame_idx % detect_interval == 0:

        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)

        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                #if viewmask[int(y),int(x)]:
                    tracks.append([(x, y)])
    #pdb.set_trace()


    tdic.append(len(tracks))

    frame_idx += 1
    prev_gray = frame_gray
    #cv2.imshow('lk_track', vis)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break


Tend = time.time()
print('\ncomputation time is {0} sec ...\n'.format(Tend-Tstart))
cv2.destroyAllWindows()
#pickle.dump(obj,open('./pkl/obj_jayst.pkl','wb'),True)
