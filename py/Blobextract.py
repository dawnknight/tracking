
def BloBextract(vidname,ith=30,pth=0,skip = 0,maskpsth = 0,fnum = 500 ,nbuff = 241,fac=8):

    #ith : intensity differece 
    #pth : threshold for number of pixels in certain label
    #skip: skip first N frame
    #maskpath : 0 :no mask , others : mask path 
    #fnum : how many frame you want to process
    #nbuff : buffer size
    #fac : downsampling scale

    import os
    import sys
    import cv2
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import gaussian_filter as gf
    from scipy.ndimage.measurements import label
    
    
    if os.path.isfile(vidname):
        pass
    else:
        print("Error!! Can not find the file...")
        exit
    
    cap = cv2.VideoCapture(vidname)
    chk, f = cap.read()

    nFrame = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

    nrow  = f.shape[0]
    ncol  = f.shape[1] 
    tind  = nbuff//2
    frame = np.zeros([nrow,ncol,3])
    img   = np.zeros([nrow,ncol,3])
    bkg   = np.zeros([nrow,ncol,3],dtype=float)
    buff  = np.zeros([nbuff,nrow,ncol,3])
    diff  = np.zeros([nrow,ncol,3])
    diffb = np.zeros([nrow/fac,ncol/fac])
    rows  = np.arange(nrow*ncol/(fac*fac)).reshape(nrow/fac,ncol/fac) // (ncol/fac)
    cols  = np.arange(nrow*ncol/(fac*fac)).reshape(nrow/fac,ncol/fac) % (ncol/fac)


    if not maskon:
        mask = np.ones([nrows,ncol])
    else:
        import pickle
        pickle.load(open(maskpath,"rb"))

     
    # -- initialize the display                                                                                                           
    fig, ax = plt.subplots(2,2,figsize=[10,7.5])
    fig.subplots_adjust(0,0,1,1,0,0)
    [j.axis('off') for i in ax for j in i]
    im = [ax[0][0].imshow(frame,), ax[0][1].imshow(frame),
          ax[1][0].imshow(frame,'gray'), ax[1][1].imshow(diffb,'gray')]
    im[2].set_clim([20,128])
    im[3].set_clim([15,25])
    fig.canvas.draw()


    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,skip);


# -- loop through frames                                                                                                              
    for ii in range(min(nFrame-skip,fnum)):    
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
                                                          ).mean(1).mean(-1),[3,1])                                              
            
            try:
                [i.remove() for i in recs]
            except:
                pass

                labcnt = 0
                labs = label(diffb>ith)
                nlab = labs[1]
                if nlab>0:
                    rcen = np.zeros(nlab)
                    ccen = np.zeros(nlab)
                    rmm  = np.zeros([nlab,2])
                    cmm  = np.zeros([nlab,2])
                    recs = []

                    for jj in range(nlab):
                        if sum((labs[0]==(jj+1)).flatten())>pth:
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
                            labcnt += 1

                    text = ax[0][0].annotate(str(labcnt),xy=(ncol-40,nrow-50),color = 'red',fontsize = 20)
                    im[0].set_data(img.astype(np.uint8)[:,:,::-1])
                    im[1].set_data(bkg.astype(np.uint8)[:,:,::-1])
                    im[2].set_data(np.abs(diff).mean(2))
                    im[3].set_data(diffb)
                    fig.canvas.set_window_title('Frame {0}'.format(ii))
                    fig.canvas.draw()
                    ax[0][0].texts.remove(text)
