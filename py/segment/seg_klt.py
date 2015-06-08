import numpy as np
import cv2,pdb,pickle
#from common import anorm2, draw_str
from time import clock
from scipy.io import savemat
from scipy.sparse import csr_matrix


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


video_src = '/home/andyc/Videos/jayst.mp4'

idx = 0
tdic = [0]

track_len = 10
detect_interval = 5
tracks = []
Ttracks = []
cam = cv2.VideoCapture(video_src)
nframe = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
nrows = cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
ncols = cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
pregood = []

timeseg = 600 #timesegment every 600 framese 
oldtrklen  = 0
saveidx = 1

for frame_idx in range(nframe):
    ret, frame = cam.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()

    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1][:2] for tr in tracks  ]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1

        if (len(pregood)>0):
            good[:len(pregood)] = good[:len(pregood)]&good
            pregood = good

        for (x, y), good_flag, idx in zip(p1.reshape(-1, 2), good,range(len(tracks))):
            if not good_flag:
                tracks[idx].append((-100., -100.,frame_idx))
                continue

            tracks[idx].append((x, y,frame_idx))
            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
            

    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1][:2]) for tr in tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)

        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y,frame_idx)])

    print('{0} - {1}'.format(frame_idx,len(tracks)))

    prev_gray = frame_gray
    cv2.imshow('lk_track', vis)
    tdic.append(len(tracks))

    #pdb.set_trace()

    if ((frame_idx+1)%timeseg) ==0:
        Xtracks = np.zeros([len(tracks)-oldtrklen,timeseg])
        Ytracks = np.zeros([len(tracks)-oldtrklen,timeseg])
        for segidx,i in enumerate(range(oldtrklen,len(tracks))):
            k = array(tracks[i]).T
            #kidx = list(k[2][k[0]!=-100]-timeseg*(saveidx-1))
            #Xtracks[segidx,:][kidx] = k[0]
            #Ytracks[segidx,:][kidx] = k[1]
            sidx = timeseg-k.shape[1]
            Xtracks[segidx,sidx:] = k[0]
            Ytracks[segidx,sidx:] = k[1]


        oldtrklen = len(tracks)
        trk={}
        trk['xtracks'] = csr_matrix(Xtracks)
        trk['ytracks'] = csr_matrix(Ytracks)
        savename = './mat/ptsTrj_part'+str(saveidx) 
        savemat(savename,trk)
        saveidx = saveidx + 1




Xtracks = np.zeros([len(tracks),frame_idx+1])
Ytracks = np.zeros([len(tracks),frame_idx+1])

fbr = pickle.load(open('./mask/forbiddenregion_mask.pkl','rb'))
for i in range(len(tracks)):
    k = array(tracks[i]).T
    #kidx = list(k[2][k[0]!=-100])
    sidx = frame_idx+1-k.shape[1]
    xx = max(min(k[0][k[0]!=-100][-1],(ncols-1)),0)
    yy = max(min(k[1][k[1]!=-100][-1],(nrows-1)),0)
    if fbr[yy,xx] == 1:
        #Xtracks[i,:][kidx] = k[0]
        #Ytracks[i,:][kidx] = k[1]
        Xtracks[i,:][sidx:] = k[0]    
        Ytracks[i,:][sidx:] = k[1]

trk ={}
trk['xtracks'] = csr_matrix(Xtracks)
trk['ytracks'] = csr_matrix(Ytracks)
savemat('./mat/sparse_ptsTrj',trk)
