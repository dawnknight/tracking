import os
import sys,os
import cv2
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.ndimage.filters import median_filter as mf
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import binary_fill_holes as fh
from scipy.ndimage.morphology import binary_dilation as bd
from scipy.ndimage.morphology import binary_erosion as be
from scipy.ndimage.filters import median_filter as mf
import scipy.ndimage as nd



# -- set the video name
dpath = '/home/andyc/Videos/'
fname = 'TLC00000.AVI'
vidname = os.path.join(dpath,fname)

# -- utilities
nrow = 480
ncol = 640
rvec = np.zeros(nrow*ncol*256,dtype=int)
gvec = np.zeros(nrow*ncol*256,dtype=int)
bvec = np.zeros(nrow*ncol*256,dtype=int)

mask = np.zeros([nrow,ncol,3],dtype=int)

rind = np.zeros(nrow*ncol,dtype=int)
gind = np.zeros(nrow*ncol,dtype=int)
bind = np.zeros(nrow*ncol,dtype=int)

# -- set the kernel
ker = np.zeros([75,75],dtype=int)
ker[:,25:50] = 1
ker -= ker.sum()
st = np.ones([6,3])

# -- initialize the base indices
ind_base = np.arange(nrow*ncol)*256

# -- open the video capture
cap = cv2.VideoCapture(vidname)

# -- initialize the display
fig, ax = plt.subplots(figsize=[10,7.5])
fig.subplots_adjust(0,0,1,1)
ax.axis('off')
im = ax.imshow(mask,clim=[0,1],cmap = 'gist_gray')
fig.canvas.draw()

fig2,ax2 = plt.subplots(figsize=[10,7.5])
fig2.subplots_adjust(0,0,1,1)
ax2.axis('off')
im2 = ax2.imshow(mask[:,:,::-1])
fig2.canvas.draw()


for ii in range(200):
    print("frame {0}\r".format(ii)),
    sys.stdout.flush()
 
    chk, frame = cap.read()
 
    if ii>=0:
    # -- read the frame
        #chk, frame = cap.read()

    # -- set the indices
        rind[:] = ind_base+frame[:,:,0].flatten()
        gind[:] = ind_base+frame[:,:,1].flatten()
        bind[:] = ind_base+frame[:,:,2].flatten()

    # -- check if the value at those indices is unlikely
        mask[:,:,0] = (rvec[rind]<1e-3*(ii+1)).reshape(nrow,ncol)
        mask[:,:,1] = (gvec[gind]<1e-3*(ii+1)).reshape(nrow,ncol)
        mask[:,:,2] = (bvec[bind]<1e-3*(ii+1)).reshape(nrow,ncol)

    # -- update the histograms
        rvec[rind] += 1
        gvec[gind] += 1
        bvec[bind] += 1

    # -- update the display
#    im.set_data(mask.sum(2)>0)
#    im.set_data(bd(be(mask.sum(2)>0,np.ones([3,3])),np.ones([3,3])))

        result =  be(fh(be(bd(mask.sum(2)>0,iterations=1),st,iterations=2)),st,iterations=1)
    #result = mf(mask.sum(2)>1,2)

    
        if ii >= 0:
            s = nd.generate_binary_structure(2,2)
            labeled_array, num_features = nd.measurements.label(result, structure=s)  
            cnt = 0
            for i in range(1,num_features):
                if (((labeled_array==i).sum() >300) & ((labeled_array==i).sum()<ncol*nrow/15) ): 
                    cnt+=1
                    if ii>80: 
                        coor = np.where(labeled_array==i)
                        ulx = int(array(coor[1]).min())
                        lrx = int(array(coor[1]).max())
                        uly = int(array(coor[0]).min())
                        lry = int(array(coor[0]).max())
                        cv2.rectangle(frame,(ulx,uly),(lrx,lry),(0,0,255),1)
 
            text2=ax2.annotate(str(cnt),xy=(600,472),color = 'red',fontsize = 20)
  
           
        
        im.set_data(result)
#    im.set_data(gf(1.0*bd(be(mask.sum(2)>0,iterations=2),iterations=2),[9,3]))
#    im.set_data(be(bd(mask.sum(2)>0),st))
#    im.set_data(gf(1.0*(mask.sum(2)>0),[10,3]))
#    im.set_data(gf(1.0*bd(be(mask.sum(2)>0,iterations=2),iterations=2),[9,3]))
#    im.set_data(mf(mask.sum(2),[6,2]))
#    im.set_data(convolve(mask.sum(2),ker))
#        fig.canvas.set_window_title('Frame {0}'.format(ii))
#        fig.canvas.draw()
#        savename = '/home/andyc/image/project/sub/'+repr(ii).zfill(5)+'.jpg' 
#        savefig(savename)

        im2.set_data(frame[:,:,::-1]) 
        fig2.canvas.set_window_title('Frame {0}'.format(ii))
        fig2.canvas.draw() 
#        text2=ax2.annotate(str(cnt),xy=(600,472),color = 'red',fontsize = 20)
        savename = '/home/andyc/image/project/color/'+repr(ii).zfill(5)+'.jpg'
        savefig(savename)

#        ax.texts.remove(text)
        ax2.texts.remove(text2)



print("")
