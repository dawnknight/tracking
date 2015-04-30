#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

import numpy as np
import cv2,pdb,pickle
import video
from common import anorm2, draw_str
from time import clock
from scipy.io import savemat

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

idx = 0
tdic = [0]

def Lcn(img,blksize):
# img is an M-by-N size image 
    m,n = img.shape
    mr  = m%blksize
    nr  = n%blksize
    img = img[:m-mr,:n-nr]
    m,n = img.shape
    imgr = np.zeros(img.shape)
    avg = np.zeros(img.shape)
    sigma = np.zeros(img.shape)
    result = np.zeros(img.shape)

    for i in range(blksize):
        for j in range(blksize):
            imgr[:] = np.roll(img,-i,0)
            imgr[:] = np.roll(imgr,-j,1)
            avg[i::blksize,j::blksize] = imgr.reshape(m/blksize,blksize,n/blksize,blksize).mean(1).mean(-1)
            sigma[i::blksize,j::blksize] = imgr.reshape(m/blksize,blksize,n/blksize,blksize).std(1).std(-1)
    sigma[:] = sigma+1
    result[:] = (img-avg)/sigma
    result[:] = (result-min(result.flatten()))*1./max(result.flatten())*255
    return result.astype('uint8')


alpha = 0.461

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.nrows = self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        self.ncols = self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        self.frame_idx = 0
        self.pregood = []
        self.mask  = pickle.load(open("./mask/jayst_mask.pkl","rb"))
        

    def run(self):
        global idx
        while (self.frame_idx <2400):
            ret, frame = self.cam.read()
            #frame = (((frame/255.)**0.5)*255).astype('uint8')  
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#*self.mask
            #frame_gray = frame[:,:,1]*1./sum(frame,2) 
            #mm = np.min(frame_gray.flatten())
            #MM = np.max((frame_gray-mm).flatten())
            #frame_gray = ((frame_gray-mm)/MM*255).astype('uint8')
            #frame_gray = Lcn(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),11)
            #pdb.set_trace()
            #frame_gray = ((0.5 +log(frame[:,:,1])-alpha*\
            #                    log(frame[:,:,2])-(1-alpha)*log(frame[:,:,0]))*255).astype(uint8)

            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1][:2] for tr in self.tracks  ]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                
                '''
                pts = p1.reshape(-1, 2)
                #inframe = array([True if ((i[0]<self.nrows)&(i[1]<self.cols)&(i[0]>=0)&(i[1]>=0)) else False for i in pts])
                pts[pts.T[0]>self.nrows-1,0] = self.nrows-1
                pts[pts.T[0]<0,0] = 0
                pts[pts.T[1]>self.ncols-1,1] = self.ncols-1
                pts[pts.T[1]<0,1] = 0
                inroi   = array([True if (self.mask[ int(i[0]),int(i[1]) ] == 1) else False for i in pts])

                #pdb.set_trace()
                '''
                if (len(self.pregood)>0):
                    good[:len(self.pregood)] = good[:len(self.pregood)]&good
                    #good = (good & inroi)
                    self.pregood = good


                    
                #pdb.set_trace()
                for (x, y), good_flag, idx in zip(p1.reshape(-1, 2), good,range(len(self.tracks))):
                    if not good_flag:
                        self.tracks[idx].append((0., 0.))#, self.frame_idx))
                        continue

                    self.tracks[idx].append((x, y ))#,self.frame_idx))
                    
                    cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
                #cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:

                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                #pdb.set_trace()
                for x, y in [np.int32(tr[-1][:2]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)    
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
          
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])#,self.frame_idx)])

            
            print('{0} - {1}'.format(self.frame_idx,len(self.tracks)))

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            
            #name = '/home/andyc/image/AIG/lking/'+str(self.frame_idx).zfill(5)+'.jpg'
            #cv2.imwrite(name,vis)
            
            
            
            tdic.append(len(self.tracks))
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

        #pdb.set_trace()

        return tdic,self.tracks


def main():
    #import sys
    #try: video_src = sys.argv[1]
    #except: video_src = 0
    
    #viewmask = './mask/20150115-jayst_mask.pkl'

    video_src = '/home/andyc/Videos/jayst.mp4'
    print __doc__
    ans,tracks=App(video_src).run()
    cv2.destroyAllWindows()

    return ans,tracks


if __name__ == '__main__':
    ans,tracks = main()
    trk ={}
    trk['tracks'] = tracks
    savemat('./mat/ptsTrjori',trk)
