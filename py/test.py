from scipy.io import loadmat
import cv2,pickle,pdb

tracks = loadmat('./mat/ptsTrj.mat')['jayst'][0]
trk = np.zeros([14880,2400,3])
for i in range(14880):
    trk[i,:,:] = tracks[i]

idx = []
for i in range(14880):
    if trk[i,:,1][38] !=-100:  
        if sum(trk[i,:,1]!=-100)>25:
            idx.append(i)

############################################################
def abbrev(name):

    abrvname = name[0]+'.'+name.split(' ')[-1]
    return abrvname



#############################################################
import pickle
from munkres import Munkres

cur = pickle.load(open('gcur.pkl','rb'))
pre = pickle.load(open('gpre.pkl','rb'))

cmtx = np.zeros((len(pre),len(cur)))
for i in range(len(pre)):
    cmtx[i,:] = (((cur.T[0]-pre[i][0])**2+(cur.T[1]-pre[i][1])**2)**0.5)


m = Munkres()
if len(pre)==len(cur):
    idxs = m.compute(cmtx)
    satus = 0
elif len(pre)<len(cur):
    idxs = m.compute(cmtx)
    satus = 1
else:     # previous clusters > current clusters => there r New clusters 
    idxs = m.compute(cmtx.T)
    idxs = [(s[1],s[0]) for s in idxs]
    satus = 2

if status ==1: #there are new clusters
    a = range(len(cur))
    b = (array(idxs).T)[1]
    res= list(set(a).difference(b))

elif status ==2 #some clusters disappear
    a = range(len(pre))
    b =(array(idxs).T)[0]
    res= list(set(a).difference(b))




############################################################################
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.cluster import MeanShift
import pickle

tracks = pickle.load(open('trk.pkl','rb'))
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
n_clusters_ = len(labels_unique)plt.figure(1)
plt.clf()
f = imread('/home/andyc/image/AIG/frame/00001.jpg')
imshow(f[::-1,:,:])

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k][:2]
    if not (pts[my_members, 0][0] <= -5):
        plt.plot(pts[my_members, 0]*bw, pts[my_members, 1]*bw, col + '.')
        plt.plot(cluster_center[0]*bw, cluster_center[1]*bw, 'o', markerfacecolor=col,\
                     markeredgecolor='k', markersize=5)
    else:
        print(k)
plt.show()

###################################################################


from scipy.io import loadmat
import cv2,pickle,pdb
import video

tracks = loadmat('./mat/ptsTrj.mat')['jayst'][0]
trk = np.zeros([14880,2400,3])
for i in range(14880):
    trk[i,:,:] = tracks[i]
mask = loadmat('./mat/manifold_labels.mat')['mask'][0]
mask = mask -1  # modify index from matlab to python
labels = loadmat('./mat/manifold_labels.mat')['labels'][0]
mlabels = np.ones(14880)*-1
for idx,i in enumerate(mask):
    mlabels[i] = labels[idx]

video_src = '/home/andyc/Videos/jayst.mp4'

cam = video.create_capture(video_src)
nrows = cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
ncols = cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
fnum  = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
frame_idx = 750

plt.figure(1,figsize=[10,12])
axL = plt.subplot(1,1,1)
frame = np.zeros([nrows,ncols,3]).astype('uint8')
im = plt.imshow(np.zeros([nrows,ncols,3]))
axis('off')
dot_exist = 0
color = array([random.randint(0,255) \
               for _ in range(3*len(set(labels)))])\
               .reshape(len(set(labels)),3)

cam.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frame_idx)

while (frame_idx <fnum):

    print frame_idx
    #frame = np.zeros([nrows,ncols,3]).astype('uint8')
    ret, frame[:] = cam.read()
    #im.set_data(frame[:,:,::-1])
    plt.draw()
    pts = trk[:,:,0].T[frame_idx]!=-100
    labinf = list(set(mlabels[pts])) # label in current frame
    for k in labinf:
        if k !=-1:
            x = trk[:,:,0].T[frame_idx][(mlabels==k)&pts]
            y = trk[:,:,1].T[frame_idx][(mlabels==k)&pts]

            dots = axL.scatter(x, y, s=20, color=(color[k].T)/255.,edgecolor='none')
            #dot_exist = 1
    im.set_data(frame[:,:,::-1])
    plt.draw()
    plt.show()
    #pdb.set_trace()

    #while dot_exist :
    #    try:
    #        dots.pop(0)
    #        plt.show()
    #    except:
    #        dot_exist = 0

    if frame_idx%50 == 0:
        clf()
        plt.figure(1,figsize=[10,12])
        axL = plt.subplot(1,1,1)
        frame = np.zeros([nrows,ncols,3]).astype('uint8')
        im = plt.imshow(np.zeros([nrows,ncols,3]))
        axis('off')


    frame_idx += 1



##################################################################3

plt.figure(1,figsize=[10,12])
axL = plt.subplot(1,1,1)
frame = np.zeros([500,500,3]).astype('uint8')
im = plt.imshow(np.ones([500,500,3]))
axis('off')

dots = []
dots.append(axL.scatter(250, 250, s=20, color=(0,1,0),edgecolor='none'))
dots.append(axL.scatter(200, 200, s=20, color=(1,0,0),edgecolor='none'))
plt.show()


for i in dots:
    i.remove()

plt.show()





####################################################################
for idx,subtrk in enumerate(tracks):
    print idx
    if len(subtrk)<len(tracks[0]):
        ext = len(tracks[0])-len(subtrk)
        tmp = np.ones([3,ext])*-100
        tmp[2] = np.arange(ext)
        tracks[idx] = [tuple(a) for a in tmp.T]+subtrk

for idx, i in enumerate(tracks):
    if len(i)!=len(tracks[0]):
        print idx




####################################################################





#create adjacent matrix
from scipy.io import loadmat,savemat

trk = loadmat('./mat/ptsTrj')
tracks = trk['jayst'][0]

nsmp = len(tracks)/10
#sample = sorted(random.sample(range(0, len(tracks)), nsmp))
smp = loadmat('./mat/sample_list')
sample = smp['list'][0]

k = 5
sigma = 2
adj = np.ones([nsmp,nsmp])*10000
chk = {}

for iidx,i in enumerate(sample):
    if iidx == 0:
        ichk =  (array(tracks[i]).T[0]!=-100)*1.0
        chk[i] = ichk
    else:
        ichk = chk[i]
    KNN = np.zeros(len(sample))

    for jidx,j in enumerate(sample[iidx:]):
        print('{0:.02f}% complete, {1} of {2}\r'.format(((nsmp+nsmp-iidx)*iidx/2+jidx)*100./(nsmp*nsmp-1)*2,\
                                                       (nsmp+nsmp-iidx)*iidx/2+jidx,\
                                                       nsmp*nsmp/2)),
        sys.stdout.flush()
        if i!=j:
            if iidx == 0:
                jchk = (array(tracks[j]).T[0]!=-100)*1.0  
                chk[j] = jchk
            else:
                jchk = chk[j]

            dis = sum((sum((array(tracks[i])-array(tracks[j]))**2,1)**0.5)*ichk*jchk)/(sum(ichk*jchk)+10**-6)    

            cost = exp(-dis/sigma)
            adj[iidx,jidx] = cost
    #preserve KNN pts       

    knnlist = np.argsort(adj[iidx,:])
    for kidx in knnlist[:k]:
        #KNN[kidx] = 1
        KNN[kidx] = adj[iidx,kidx]    

    adj[iidx,:] = KNN
    adj.T[iidx,:] = adj[iidx,:]

feature_mtx = {}
feature_mtx['feature'] = adj
savename = 'feature_matrix'
savemat(savename,feature_mtx)       


adj[adj>0]=1
feature_mtx = {}
feature_mtx['feature'] = adj
savename = 'feature_matrix(binary)'
savemat(savename,feature_mtx)



################################################################################
                                                                                                 
from scipy.io import loadmat,savemat
trk = loadmat('./mat/ptsTrj')
tracks = trk['jayst'][0]


nsmp = len(tracks)/10
smp = loadmat('./mat/sample_list')
sample = smp['list'][0]
sigma = 2

adjx = np.zeros([nsmp,len(tracks)])
for idx,i in enumerate(sample):
    ichk =  (array(tracks[i]).T[0]!=-100)*1.0
    jidx = 0
    for j in range(len(tracks)):
        if j not in sample:
            print('{0:.02f}% complete, {1} of {2}\r'.format( (idx*100.*len(tracks)+j)/nsmp/len(tracks),\
                                                              idx*len(tracks)+j,nsmp*len(tracks))),
            sys.stdout.flush()
            jchk = (array(tracks[j]).T[0]!=-100)*1.0
            dis = sum((sum((array(tracks[i])-array(tracks[j]))**2,1)**0.5)*ichk*jchk)/(sum(ichk*jchk)+10**-6)

            cost = exp(-dis/sigma)
            adjx[idx,jidx] = cost
            jidx+=1

restadj = {}
restadj['adj'] = adjx
savemat('restadj.mat',restadj)








======================================================
fig, ax = plt.subplots(figsize=[10,7.5])
lins = ax.plot(x,x*2)

for ii in range(100):
    lins[0].set_data(x,x*ii)
    fig.canvas.draw()




cap = cv2.VideoCapture(vidname)
fig, ax = plt.subplots(figsize=[10,7.5])
lins = ax.plot([],[])

for ii in range(1000):
    print("frame {0}\r".format(ii)),
    sys.stdout.flush()
    chk, frame = cap.read()

    if ii>0:
        result = hist(abs(frame[:,:,0]-frame_old[:,:,0]).flatten(),bins=255)
        lins[0].set_data(result[0]) 
    fig.canvas.draw()
    frame_old = frame
    


'''
import lktrack2
import os,glob,cv2,pickle,cv

path = '/home/andyc/image/Feb11/'
imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))
imnames = imlist[0:100]
# create tracker object
lkt = lktrack2.LKTracker(imnames)
# detect in first frame, track in the remaining

lkt.detect_points()
lkt.draw()
for i in range(len(imnames)-1):
    print i
    lkt.track_points()
    lkt.draw()
'''

'''
for im,ft in lkt.track():
    print 'tracking %d features' % len(ft)
# plot the tracks
figure()
imshow(im)
for p in ft:
    plot(p[0],p[1],'bo')
for t in lkt.tracks:
    plot([int(p[0][0]) for p in t],[int(p[0][1]) for p in t])
axis('off')
show()
'''

'''
import numpy as np
import cv2

cap = cv2.VideoCapture('/home/andyc/TLC00005.AVI')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff

cap.release()
cv2.destroyAllWindows()
'''
 
'''
import numpy as np
import cv2,pickle,os,glob
import scipy.ndimage as nd
from skimage.color import rgb2hsv


path ='/home/andyc/image/20141031/'
imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))

i = 177

frame = nd.imread(imlist[i])
fcal  = np.zeros(frame.shape)

BG = pickle.load(open("./BG/20141031.pkl","rb"))
mask = pickle.load(open("./mask/20141031_mask.pkl","rb"))
Rmask = mask==0

mask2 = pickle.load(open("./mask/20141031_mask_1.pkl","rb"))



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
'''

'''
fg = (np.abs(1.0*frame[:,:,0]-BG[:,:,0])>50.)+\
     (np.abs(1.0*frame[:,:,1]-BG[:,:,1])>50.)+\
     (np.abs(1.0*frame[:,:,2]-BG[:,:,2])>50.)

fg_gray = np.abs(1.0*frame.mean(2)-BG.mean(2))>50.0
'''
'''
BG_hsv = rgb2hsv(BG)
f_hsv  = rgb2hsv(frame)
'''
'''
fg = ((np.abs(1.0*f_hsv[:,:,0]-BG_hsv[:,:,0])>55.) &  (np.abs(1.0*f_hsv[:,:,0]-BG_hsv[:,:,0])<100.)) &\ 
     (np.abs(1.0*frame[:,:,1]-BG[:,:,1])>90./255.)
'''       



'''
figure(1),imshow(uint8(BG))
title('background')
figure(2),
subplot(2,2,1),imshow(uint8(abs(frame-BG)))
title('color')
subplot(2,2,2),imshow(uint8(abs(frame-BG))[:,:,0]*mask*mask2,'gray')
title('R')
subplot(2,2,3),imshow(uint8(abs(frame-BG))[:,:,1]*mask*mask2,'gray')
title('G')
subplot(2,2,4),imshow(uint8(abs(frame-BG))[:,:,2]*mask*mask2,'gray')
title('B')


figure(3),
subplot(2,2,1),imshow(uint8(abs(fcal-BG)))
title('color')
subplot(2,2,2),imshow(uint8(abs(fcal-BG))[:,:,0]*mask*mask2,'gray')
title('R')
subplot(2,2,3),imshow(uint8(abs(fcal-BG))[:,:,1]*mask*mask2,'gray')
title('G')
subplot(2,2,4),imshow(uint8(abs(fcal-BG))[:,:,2]*mask*mask2,'gray')
title('B')



'''

'''
figure(3),
subplot(2,2,1),imshow(frame)
title('color')
subplot(2,2,2),imshow(uint8(abs(f_hsv-BG_hsv)*255)[:,:,0],'gray')
title('H')
subplot(2,2,3),imshow(uint8(abs(f_hsv-BG_hsv)*255)[:,:,1],'gray')
title('S')
subplot(2,2,4),imshow(uint8(abs(f_hsv-BG_hsv))[:,:,2],'gray')
title('V')
'''



'''
figure(3),imshow(BG.mean(2),'gray')
title('background--gray')
figure(4),imshow(abs(frame.mean(2)-BG.mean(2)),'gray')
title('difference--gray')
'''
'''
clf()
th = 40
r = abs(vid[15231][:,:,0].astype('float')-vid[15232][:,:,0].astype('float'))
g = abs(vid[15231][:,:,1].astype('float')-vid[15232][:,:,1].astype('float'))
b = abs(vid[15231][:,:,2].astype('float')-vid[15232][:,:,2].astype('float'))

df = abs(vid[15231].astype('float')-vid[15232].astype('float'))

r[r<=th]=0
r[r>th]=1

g[g<=th]=0
g[g>th]=1

b[b<=th]=0
b[b>th]=1

d = (r==1)+(g==1)+(b==1)


df = (abs(1.0*vid[15231][:,:,0]-1.0*vid[15232][:,:,0])>th)+\
     (abs(1.0*vid[15231][:,:,1]-1.0*vid[15232][:,:,1])>th)+\
     (abs(1.0*vid[15231][:,:,2]-1.0*vid[15232][:,:,2])>th)



figure(1),
subplot(2,2,1)
imshow(r,'gray')
subplot(2,2,2)
imshow(g,'gray')
subplot(2,2,3)
imshow(b,'gray')
subplot(2,2,4)
imshow(d,'gray')


frame = vid[15232]
'''
