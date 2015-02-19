import numpy as np
import cv2,pdb,pickle,sys
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

viewmask = pickle.load(open("./mask/20150115-jayst_mask.pkl","rb"))




track_len = 100
detect_interval = 5
tracks = []
cam = video.create_capture(video_src)
frame_idx = 0
traceidx = 0



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
    global ptsidx,objidx,traceidx

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
                    for ii in obj[i].pts:
                        obj[i].ptsTrj[Trj_idx] = tracks[ii]
                        pts[ii].Trj = tracks[ii]
                        obj[i].pos.append(pts[ii].Trj[-1])
                        Trj_idx+=1
                    obj[i].Trj.append(mean(obj[i].pos,0))
                    obj[i].xTrj.append(int(mean(obj[i].pos,0)[0]))
                    obj[i].yTrj.append(int(mean(obj[i].pos,0)[1]))
                    #pdb.set_trace()
                    obj[i].pos = []
                    obj[i].frame.append(frame_idx)
                    

        else:   # somes pts change     

            Npts = len(del_pts)+len(tracks)-tnum  #number of the new pts           
            clist = np.zeros(len(tracks)+len(del_pts)).astype(int)
        
            # -- delete missing points' info -- #
            mobj = []  #modified objs list
            
            mobj = [pts[i].pa for i in del_pts]

            for i in del_pts[::-1]:
                clist[i:]+=1
                j = pts[i].pa # parent idx
                del_idx = obj[j].pts.index(i)  #del idx in obj                                                              
                obj[j].pts.pop(del_idx)

                for oidx in set(mobj):
                    for pidx in range(len(obj[oidx].pts)):
                        if obj[oidx].pts[pidx]>i:
                            obj[oidx].pts[pidx] -= 1

                for ii in range(del_idx,len(obj[j].ptsTrj)-1):
                    obj[j].ptsTrj[ii]=obj[j].ptsTrj[ii+1]
             
                try:
                    del obj[j].ptsTrj[ii+1]
                except: # del_idx is the rightest element in the list                                                        
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

                            ii = k-clist[k]     # maping old idx to new idx   

                            obj[i].ptsTrj[Trj_idx] = tracks[ii]
                            
                            pts[ii].Trj = tracks[ii] 
                            obj[i].pos.append(pts[ii].Trj[-1])
                            Trj_idx+=1
                            offset.append(clist[k])                        
                        # -- group pts index update -- #         
                        obj[i].pts[:] = list(array(obj[i].pts)-array(offset))
                        if traceidx ==1:
                            pdb.set_trace()

                    else:
                       
                        for ii in obj[i].pts:

                            obj[i].ptsTrj[Trj_idx] = tracks[ii]
                            pts[ii].Trj = tracks[ii]
                            obj[i].pos.append(pts[ii].Trj[-1])
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

                            for ii in np.arange(len(idxpos))[idxpos]:
                                if ii >= (len(tracks)-Npts):
                                    if idx[ii] == 1:

                                        obj[pa].ptsTrj[Trj_idx] = pts[ii].Trj
                                        obj[pa].pos.append(pts[ii].Trj[-1])
                                        obj[pa].pts.append(ii)
                                        pts[ii].pa = pa
                                        Trj_idx += 1

                        else:       # -- generating new group -- #              
                            Trj_idx = 0
                            for ii in np.arange(len(idxpos))[idxpos]:
                                if Trj_idx == 0:
                                    obj[objidx] = Objects()
                                if idx[ii]!=0:    
                                    obj[objidx].ptsTrj[Trj_idx] = pts[ii].Trj
                                    obj[objidx].pos.append(pts[ii].Trj[-1])
                                    obj[objidx].pts.append(ii)
                                    pts[ii].pa = objidx
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
im = plt.imshow(np.zeros([512,612,3]))
axis('off')



while True:
    print('frame {0}\r'.format(frame_idx)),
    sys.stdout.flush()
    ret, frame = cam.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)*viewmask
    vis = frame.copy()

    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        del_pts = []
        idx = -1
        #pdb.set_trace()
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            idx += 1
            if not good_flag:  # feature gone
                #pdb.set_trace()
                del_pts.append(idx)                
                continue

            tr.append((x, y))
            if len(tr) > track_len:
                del tr[0]
            new_tracks.append(tr)
            #cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        #pdb.set_trace()

        
        tmp = []

        for i in del_pts:
            if i <=tnum:
                tmp.append(i)
        del_pts[:] = tmp        
            
         
        tracks = new_tracks        
        Initpts(tracks,tnum,del_pts,len(obj))

        tnum = len(tracks)

        if frame_idx<2:
            InitGrouping(tracks,pts,dth=30,vth=2)

        #pdb.set_trace()
        #cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0,255 , 0))
       
       
        
        im.set_data(frame[:,:,::-1]) 
        plt.draw()
        for d in range(len(obj)):
            if obj[d].status ==1:  
                lines = axL.plot(obj[d].xTrj,obj[d].yTrj,color = array(obj[d].Color()[::-1])/255.,linewidth=2) 
        
        name = '/home/andyc/image/jayst/group/'+str(frame_idx).zfill(5)+'.jpg'
        savefig(name)

        #cor = [np.int32(tr) for tr in tracks]    

        #for jj in range(len(tracks)):
        #    line = axL.plot(cor[jj],color = array(blobs[i].Color())[::-1]/255.,linewidth=2)
            
        ''' 
        try:
            axL.lines.pop(0)
            plt.show()
        except:
            print('clear')
        '''

        #pdb.set_trace()
        #draw_str(vis, (20, 20), 'track count: %d' % len(tracks))

        
        print('\nframe   {0}'.format(frame_idx))
        print('# pts   {0}'.format(len(pts)))
        print('#delp   {0}'.format(len(del_pts)))
        print('#tracks {0}'.format(len(tracks)))
        print('err  {0}'.format(errcheck()))
        print('######################\n')
      
        
        #if frame_idx == 210:
        #    pdb.set_trace()
        
        
        #pdb.set_trace()
        


    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        #for x, y in [np.int32(tr[-1]) for tr in tracks]:
            #cv2.circle(mask, (x, y), 5, 0, -1)
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y)])


    frame_idx += 1
    prev_gray = frame_gray
    #cv2.imshow('lk_track', vis)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

#cv2.destroyAllWindows()

