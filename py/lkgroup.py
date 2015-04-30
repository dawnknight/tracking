
import numpy as np
import cv2,pdb,pickle
import video
from common import anorm2, draw_str
from time import clock
from sklearn.cluster import MeanShift,estimate_bandwidth
from munkres import Munkres

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


#plt.figure(1,figsize=[10,12])
#axL = plt.subplot(1,1,1)
#im = plt.imshow(np.zeros([512,612,3]))
#axis('off')

idx = 0
tdic = [0]
obj = {}

def Meanshiftcluster(pts,bandwidth):
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(pts)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    return labels,cluster_centers


def Drawgroup(f,tracks,idx):
    from itertools import cycle
    bw = 20
    vxth = 5
    vyth = 5

    pts = np.float32([[tr[-1][0]/bw,tr[-1][1]/bw,(tr[-1][1]-tr[-2][1])/vxth,\
                  (tr[-1][0]-tr[-2][0])/vyth] \
                   if tr[-1][1] != -100 else [-100,-100,-100,-100] \
                   for tr in tracks  ])

    ms = MeanShift(bandwidth=1, bin_seeding=True)
    ms.fit(pts)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    plt.figure(1,figsize=[15,12])
    plt.clf()
    imshow(f[:,:,::-1])
    axis('off')

    for k in range(n_clusters_):
        col = array([randint(0,255),randint(0,255),randint(0,255)])/255.
        my_members = labels == k
        cluster_center = cluster_centers[k][:2]
        if not (pts[my_members, 0][0] <= -5):
            plt.plot(pts[my_members, 0]*bw, pts[my_members, 1]*bw,'o', color = col )
            plt.plot(cluster_center[0]*bw, cluster_center[1]*bw, 'o', color = col, markersize=10)
        else:
            print(k)
    plt.show()

    
    name = '/home/andyc/image/AIG/gp/'+str(idx).zfill(4)+'.jpg'                                                
    savefig(name) 

    

def Objmatch(pre,cur,th): 
    #predict : predict cluster centers
    #current : current cluster centers

    cmtx = np.zeros((len(pre),len(cur)))

    for i in range(len(pre)):
        cmtx[i,:] = (((cur.T[0]-pre[i][0])**2+(cur.T[1]-pre[i][1])**2)**0.5)

    cmtx[cmtx>th] = 10**6

    m = Munkres()

    if len(pre)==len(cur):
        idxs = m.compute(cmtx)
        Type = 0
    elif len(pre)<len(cur):
        idxs = m.compute(cmtx)
        Type = 1
    else:     # previous clusters > current clusters => there r New clusters 
        idxs = m.compute(cmtx.T)
        idxs = [(s[1],s[0]) for s in idxs]
        Type = 2

    if Type ==1: #there are new clusters                       
        a = range(len(cur))
        b = (array(idxs).T)[1]
        res= list(set(a).difference(b))

    elif Type ==2: #some clusters disappear                  
        a = range(len(pre))
        b =(array(idxs).T)[0]
        res= list(set(a).difference(b))
    else: # status = 0
        res = []

    #pdb.set_trace()

    return idxs,Type,res




class Objects():
    def __init__(self):
        self.center = []
        self.ptsidx = []
        self.frame = []

        #self.ptsTrj= {}
        #self.xTrj = []
        #self.yTrj = []
        #self.vel = []
        #self.pos = []
        self.status = 1   # 1: alive  2: dead  
    def Color(self):
        try:
            self.R
        except:
            self.R = randint(0,255)/255.
            self.G = randint(0,255)/255.
            self.B = randint(0,255)/255.
        return (self.R,self.G,self.B)


class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.nrows = self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        self.ncols = self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        self.fnum  = int(self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0
        self.pregood = []
        self.mask  = pickle.load(open("./mask/jayst_mask.pkl","rb"))
        self.inigroup = 0

        plt.figure(1,figsize=[10,12])
        self.axL = plt.subplot(1,1,1)
        self.frame = np.zeros([self.nrows,self.ncols,3]).astype('uint8')
        self.im = plt.imshow(np.zeros([self.nrows,self.ncols,3]))
        axis('off')




    def run(self,bw=20,vxth=5,vyth=5):
        global idx
        while (self.frame_idx <self.fnum):
            ret, self.frame = self.cam.read()
            frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)#*self.mask
            vis = self.frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1][:2] for tr in self.tracks  ]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                
                #pdb.set_trace()                 

                pts = p1.reshape(-1, 2)
                #inframe = array([True if ((i[0]<self.nrows)&(i[1]<self.cols)&(i[0]>=0)&(i[1]>=0)) else False for i in pts])
                pts[pts.T[0]>self.ncols-1,0] = self.ncols-1
                pts[pts.T[0]<0,0] = 0
                pts[pts.T[1]>self.nrows-1,1] = self.nrows-1
                pts[pts.T[1]<0,1] = 0
                inroi   = array([True if (self.mask[ int(i[1]),int(i[0]) ] == 1) else False for i in pts])
                

                if (len(self.pregood)>0):
                    good[:len(self.pregood)] = good[:len(self.pregood)]&self.pregood

                good = (good & inroi)
                self.pregood = good

                for (x, y), good_flag, idx in zip(p1.reshape(-1, 2), good,range(len(self.tracks))):
                    if not good_flag:
                        self.tracks[idx].append((-100., -100.))
                        continue

                    self.tracks[idx].append((x, y))


                #feature pts contains normalized x,y,vx,vy info
                fpts = np.float32([[tr[-1][0]/bw,tr[-1][1]/bw,(tr[-1][1]-tr[-2][1])/vxth,(tr[-1][0]-tr[-2][0])/vyth]\
                                      if tr[-1][1] != -100 else [-100,-100,-100,-100] for tr in self.tracks  ])
                
                labels,cluster_center = Meanshiftcluster(fpts,1)
                
                # initial Group
                if self.inigroup ==0:
                    objlist = []
                    objidx = 0
                    self.inigroup = 1
                    for i in range(labels.max()+1):
                        members = (labels ==i)
                        if cluster_center[i][0] == -100 :
                            offset = array([1,1,1,1])
                        else:
                            offset = array([bw,bw,vxth,vyth])
                        obj[i] = Objects()
                        obj[i].center.append(cluster_center[i]*offset)
                        obj[i].ptsidx.append(np.where(members == True)[0])
                        obj[i].frame.append(self.frame_idx)
                        objlist.append(i)
                        objidx += 1

                    #predcenter = array([cluster_center.T[0]*offset[0]+cluster_center.T[2]*offset[2],\
                    #                    cluster_center.T[1]*offset[1]+cluster_center.T[3]*offset[3] ])                    
                    #predcenter[1][predcenter[1] <= -100] = -100             
                    #predcenter[0][predcenter[0] <= -100] = -100                
                    #predcenter = predcenter.T 

                else :
                    matchlab,status,rest= Objmatch(predcenter,cluster_center*offset,50)

                    #pdb.set_trace()

                    for (i,j) in matchlab: #update obj info
                        if cluster_center[j][0] == -100 :
                            offset = array([1,1,1,1])
                        else:
                            offset = array([bw,bw,vxth,vyth])

                        obj[objlist[i]].center.append(cluster_center[j]*offset)
                        obj[objlist[i]].ptsidx.append(np.where(labels==j)[0])
                        obj[objlist[i]].frame.append(self.frame_idx)

                    if status == 1:#new obj initial
                        for i in rest: # initialize new objs

                            members = (labels ==i)
                            offset = array([bw,bw,vxth,vyth])
                            obj[objidx] = Objects()
                            obj[objidx].center.append(cluster_center[i]*offset)
                            obj[objidx].ptsidx.append(np.where(members == True)[0])
                            obj[objidx].frame.append(self.frame_idx)
                            #pdb.set_trace()
                            objlist.append(objidx)
                            objidx += 1
                    elif status == 2 :#some clusters disappear    
                        for i in rest[::-1]:
                            
                            #pdb.set_trace()

                            obj[objlist[i]].status = 0
                            objlist.pop(objlist.index(objlist[i]))
                    
                    #xpre = [obj[i].center[-1][0]+obj[i].center[-1][2] for i in objlist]
                    #ypre = [obj[i].center[-1][1]+obj[i].center[-1][3] for i in objlist]
                    #predcenter  = array([xpre,ypre]).T
                
                #pdb.set_trace()
                #Drawgroup(frame,self.tracks,self.frame_idx)

                xpre = [obj[i].center[-1][0]+obj[i].center[-1][2] for i in objlist]
                ypre = [obj[i].center[-1][1]+obj[i].center[-1][3] for i in objlist]
                predcenter  = array([xpre,ypre]).T

                #Draw
         
                self.im.set_data(self.frame[:,:,::-1])

                #pdb.set_trace()

                text = []
                for i in objlist:
                    x = (array(obj[i].center).T)[0].astype('int')
                    y = (array(obj[i].center).T)[1].astype('int')
                    if x[-1] != -100:
                        #cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                        #print(x)
                        #lines = axL.plot(x,y,color = obj[i].Color()[::-1],linewidth=2)
                        #line_exist = 1
                        text.append( self.axL.annotate(str(i),xy=(x[-1],y[-1]),color = 'red',fontsize = 10) )
                        id_exist = 1

                plt.draw()


                name = '/home/andyc/image/AIG/number/'+str(self.frame_idx).zfill(4)+'.jpg'
                savefig(name)
                
                #pdb.set_trace()
                '''
                while line_exist :    
                    try:
                        axL.lines.pop(0)
                        plt.show()
                    except:
                        line_exist = 0
                
                '''        
                for i in range(len(text)):
                    self.axL.texts.remove(text[i])
                    plt.show()



            if self.frame_idx % self.detect_interval == 0:

                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1][:2]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)

                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
          
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y,self.frame_idx)])



            print('{0} - {1}'.format(self.frame_idx,len(self.tracks)))

            self.frame_idx += 1
            self.prev_gray = frame_gray
            #cv2.imshow('lk_track', vis)
            
            #name = '/home/andyc/image/AIG/lk/'+str(idx).zfill(5)+'.jpg'
            #cv2.imwrite(name,vis)
            idx += 1
            
            
            tdic.append(len(self.tracks))
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
        return tdic,self.tracks


def main():
    
    #viewmask = './mask/20150115-jayst_mask.pkl'

    



    video_src = '/home/andyc/Videos/jayst.mp4'
    print __doc__
    ans,tracks=App(video_src).run()
    #cv2.destroyAllWindows()
    return ans,tracks 

if __name__ == '__main__':
    ans,tracks = main()
