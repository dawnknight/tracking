import os
import sys
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.ndimage.measurements import label

# -- set the video name
#dpath = '/home/andyc/Videos/'
#fname = 'TLC00000.AVI'

vidname = '/home/andyc/Videos/jayst.mp4'
 
# --                                                                                                                                0
mask = pickle.load(open("./mask/jayst_mask.pkl","rb"))
#mask = pickle.load(open('./mask/20150115-jayst_mask.pkl','rb'))

# -- utilities
skip  = 0
nrow  = 512
ncol  = 612
nbuff = 241
tind  = nbuff//2
fac   = 4
frame = np.zeros([nrow,ncol,3])
img   = np.zeros([nrow,ncol,3])
bkg   = np.zeros([nrow,ncol,3],dtype=float)
buff  = np.zeros([nbuff,nrow,ncol,3])
diff  = np.zeros([nrow,ncol,3])
diffb = np.zeros([nrow/fac,ncol/fac])
rows  = np.arange(nrow*ncol/(fac*fac)).reshape(nrow/fac,ncol/fac) // (ncol/fac)
cols  = np.arange(nrow*ncol/(fac*fac)).reshape(nrow/fac,ncol/fac) % (ncol/fac)


# -- initialize the display
fig, ax = plt.subplots(2,2,figsize=[10,7.5])
fig.subplots_adjust(0,0,1,1,0,0)
[j.axis('off') for i in ax for j in i]
im = [ax[0][0].imshow(frame,), ax[0][1].imshow(frame),
      ax[1][0].imshow(frame,'gray'), ax[1][1].imshow(diffb,'gray')]
im[2].set_clim([20,128])
im[3].set_clim([15,25])
fig.canvas.draw()


# -- open the video capture and skip the initial frames
cap = cv2.VideoCapture(vidname)

#for ii in range(skip):
    # read the frame
#    chk, frame[:,:,:] = cap.read()

cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,skip);


# -- loop through frames
for ii in range(nbuff+2000):

    print("frame {0}\r".format(ii)),
    sys.stdout.flush()

    # read the frame
    chk, frame[:,:,:] = cap.read()

    # load buffer and update background
    if ii<nbuff:
        buff[ii,:,:,:] = frame[:,:,:]
        continue
    elif ii==nbuff:
        bkg[:,:,:] = buff.mean(0)
        cnt = 0
        continue
    else:
        bind = cnt % nbuff
        img[:,:,:]  = buff[(bind+tind)%nbuff]
        bkg[:,:,:] -= buff[bind]/float(nbuff)
        bkg[:,:,:] += frame/float(nbuff)
        buff[bind]  = frame
        cnt += 1


    # calculate the difference image then rebin and smooth
    diff[:,:,:] = img-bkg
    diffb[:,:]  = gf((np.abs(diff).mean(2)*mask).reshape(nrow/fac,fac,ncol/fac,fac
                                                  ).mean(1).mean(-1),[1,1])
    #diffb[:,:]  = gf(np.abs(diff).mean(2).reshape(nrow/fac,fac,ncol/fac,fac
    #                                              ).mean(1).mean(-1),[1,1]) 

    
    # update plots by removing labels then updating
    try:
        [i.remove() for i in recs]
    except:
        pass

    labcnt = 0
    labs = label(diffb>30)
    nlab = labs[1]


    if nlab>0:
        rcen = np.zeros(nlab)
        ccen = np.zeros(nlab)
        rmm  = np.zeros([nlab,2])
        cmm  = np.zeros([nlab,2])
        recs = []

        for jj in range(nlab):
            if sum((labs[0]==(jj+1)).flatten())>0: 
                labcnt += 1
                tlab = jj+1
                lind = labs[0]==tlab
                trow = rows[lind]
                tcol = cols[lind]
                rcen[jj] = trow.mean()*fac
                ccen[jj] = tcol.mean()*fac
                rmm[jj]  = [trow.min()*fac,trow.max()*fac]
                cmm[jj]  = [tcol.min()*fac,tcol.max()*fac]    
                recs.append(ax[0][0].add_patch(plt.Rectangle((cmm[jj,0],rmm[jj,0]),
                                                             cmm[jj,1]-cmm[jj,0],
                                                             rmm[jj,1]-rmm[jj,0],
                                                             facecolor='none',
                                                             edgecolor='orange',
                                                             lw=2)))   





    text = ax[0][0].annotate(str(labcnt),xy=(580,430),color = 'red',fontsize = 20)
    im[0].set_data(img.astype(np.uint8)[:,:,::-1])
    im[1].set_data(bkg.astype(np.uint8)[:,:,::-1])
    im[2].set_data(np.abs(diff).mean(2))
    im[3].set_data(diffb)
    fig.canvas.set_window_title('Frame {0}'.format(ii))

    #fig.savefig('../output/011714/utilization_pilot_example_'
    #            '{0:05}.jpg'.format(cnt-1))
    
    fig.canvas.draw()

    ax[0][0].texts.remove(text)


'''
# -- create a video
cmd = "ffmpeg -r 5 -i ../output/011714/utilization_pilot_example_%05d.jpg -qscale 24 -vcodec mpeg4 ../output/011714/utilization_pilot_example.mp4"#; rm ../output/011714/*.jpg"
print("writing video...")
os.system(cmd)
'''
