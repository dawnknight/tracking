import matplotlib.pyplot as plt


def BlobExtract(vidname,ith=30,pth=0,skip = 0,fnum = 500 ,nbuff = 241,fac=8):

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
    from scipy.ndimage.filters import gaussian_filter as gf
    from scipy.ndimage.measurements import label
    
    
    if os.path.isfile(vidname):
        cap = cv2.VideoCapture(vidname)
    else:
        print("Error!! Can not find the file...")
        exit

    nFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    nrow  = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    ncol  = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)

    tind  = nbuff//2
    frame = np.zeros([nrow,ncol,3])
    img   = np.zeros([nrow,ncol,3])
    bkg   = np.zeros([nrow,ncol,3],dtype=float)
    buff  = np.zeros([nbuff,nrow,ncol,3])
    diff  = np.zeros([nrow,ncol,3])
    diffb = np.zeros([nrow/fac,ncol/fac])
    rows  = np.arange(nrow*ncol/(fac*fac)).reshape(nrow/fac,ncol/fac) // (ncol/fac)
    cols  = np.arange(nrow*ncol/(fac*fac)).reshape(nrow/fac,ncol/fac) % (ncol/fac)

    timestamp = {}

    maskpath = Choosemask()

    if not maskpath:
        mask = np.ones([nrow,ncol])
    else:
        import pickle
        mask = pickle.load(open(maskpath,"rb"))

    # -- initialize tblob dictionary
    Blobdict = {}
    
     
    # -- initialize the display                                                                                                           
    fig, ax = plt.subplots(2,2,figsize=[10,7.5])
    fig.subplots_adjust(0,0,1,1,0,0)
    [j.axis('off') for i in ax for j in i]
    im = [ax[0][0].imshow(frame,), ax[0][1].imshow(frame),
          ax[1][0].imshow(frame,'gray'), ax[1][1].imshow(diffb,'gray')]
    im[2].set_clim([20,128])
    im[3].set_clim([15,25])
    fig.canvas.draw()


    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,skip)


# -- loop through frames                                                                                                              
    if fnum == -1:
        pfnum = nFrame-skip #process frame number
    else:
        pfnum = min(nFrame-skip,fnum)


    for ii in range(pfnum):    
        print("frame {0}\r".format(ii+skip)),
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
                blobs = {}
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
                            blobs[labcnt] = [[cmm[jj,0],rmm[jj,0]],[cmm[jj,1]-cmm[jj,0],rmm[jj,1]-rmm[jj,0]]]        
                                              
                    text = ax[0][0].annotate(str(labcnt),xy=(ncol-40,nrow-50),color = 'red',fontsize = 20)
                    im[0].set_data(img.astype(np.uint8)[:,:,::-1])
                    im[1].set_data(bkg.astype(np.uint8)[:,:,::-1])
                    im[2].set_data(np.abs(diff).mean(2))
                    im[3].set_data(diffb)
                    fig.canvas.set_window_title('Frame {0}'.format(ii+skip))
                    fig.canvas.draw()
                    ax[0][0].texts.remove(text)
                blobs[0] = labcnt




        #Blobdict[ii] = blobs
        #timestamp[ii]= TimeStamp(img) 
                    
     
    #return Blobdict,timestamp

def DrawBlobcnt(bdict,timestamp):
    import scipy as sp
    idx = bdict.keys()[0]
    sidx = timestamp[idx][::-1].index('-')
    titlename = timestamp[idx][:-sidx-1]
    
   
    c = 0
    trunc = 1000
    per = 1800 #period: 1hr : 3600 
    num = []
    xlab = []

    for i in bdict.keys():
        num.append(bdict[i][0])
        if sp.mod(c,per)==0:
            xlab.append(timestamp[i][-sidx:])
        else:                                                                                                                          
            xlab.append('')                                                                                                            
        c+=1   
    plt.figure()     
    plt.plot(range(len(num)),num)
    plt.xticks(range(len(xlab)),xlab)
    plt.title(titlename)

    '''
    for j in range(int(ceil(len(bdict)/float(trunc)))):
        num = []
        xlab = []
        for i in bdict.keys()[j:(j+1)*trunc]:
            num.append(bdict[i][0])
            if sp.mod(c,per)==0:
                xlab.append(timestamp[i][-sidx:])
            else:
                xlab.append('')
            c+=1    

        plt.figure()
        plt.plot(range(trunc),num[:trunc])
        plt.xticks(range(trunc), xlab[:trunc])
        plt.title(titlename)
        plt.xlabel('Time')
        plt.ylabel('number')
     '''

def TimeStamp(img):
    Time = {}

    Time['YY'] = [[[256,464],[270,478]],[[272,464],[286,478]]]
    Time['MM'] = [[[304,464],[318,478]],[[320,464],[334,478]]]
    Time['DD'] = [[[352,464],[366,478]],[[368,464],[382,478]]]
    Time['hh'] = [[[400,464],[414,478]],[[416,464],[430,478]]]
    Time['mm'] = [[[448,464],[462,478]],[[464,464],[478,478]]]
    Time['ss'] = [[[496,464],[510,478]],[[512,464],[526,478]]]

    keys = ['YY','MM','DD','hh','mm','ss']
    sybl  =['/','/','-',':',':','']

    xofst = 3
    yofst = 4

    numdict = {}
    numdict[82]  = '0'
    numdict[44]  = '1'
    numdict[52]  = '2'
    numdict[42]  = '3' 
    numdict[74]  = '4'
    numdict[56]  = '5'
    numdict[62]  = '6'
    numdict[54]  = '7'
    numdict[66]  = '8'
    numdict[58]  = '9'
    
    trsl = '20' 

    for idx in range(len(keys)):

        posinfo = Time[keys[idx]]
  
        for i in range(len(posinfo)):
            #pdb.set_trace()
            num = sum(1.*(img[posinfo[i][0][1]:posinfo[i][1][1]-yofst,\
                              posinfo[i][0][0]:posinfo[i][1][0]-xofst,:].mean(2)>128).flatten())

            if num in numdict.keys():
                c = numdict[num]
            else:
                print('Error!! can not identify!!')

            trsl+=c    
        trsl+= sybl[idx] 
  
    return trsl


def Choosemask():
    from tkFileDialog import askopenfilename
    ans = raw_input("choose mask file? Yes or No ? \n").upper()
    if ans == 'Y' or ans =='YES':
        maskpath = askopenfilename(initialdir = '/home/andyc/tracking/py/mask/')
    else:
        maskpath = 0
    return maskpath



'''
for i in range(1,14):
    for j in range(1,14):
        chk = []  
        for k in range(10):
            if sum(NP[:i,:j,k].flatten()) in chk:
                continue;
            else:
                chk.append(sum(NP[:i,:j,k].flatten()))
        if len(chk) == 10 :
           print(i,j)
                    
'''
