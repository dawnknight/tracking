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
