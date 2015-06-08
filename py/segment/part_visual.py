from scipy.io import loadmat
import cv2,pickle,pdb
from scipy.sparse import csr_matrix

#tracks = loadmat('./mat/ptsTrj.mat')['jayst'][0]
#sample = len(tracks)
#fnum   = len(tracks[0])

pidx = 4
timeseg = 600
ptstrjname = './mat/ptsTrj_part'+str(pidx)+'.mat'
labelsname = './mat/label'+str(pidx)+'.mat'

ptstrj = loadmat(ptstrjname)
mask = loadmat(labelsname)['mask'][0]
labels = loadmat(labelsname)['label'][0]


xtrj = csr_matrix(ptstrj['xtracks'], shape=ptstrj['xtracks'].shape).toarray()
ytrj = csr_matrix(ptstrj['ytracks'], shape=ptstrj['ytracks'].shape).toarray()
sample = ptstrj['xtracks'].shape[0] # Trj number
fnum   = ptstrj['xtracks'].shape[1] # frame number
seidx  = ptstrj['startend'][0] 


trk = np.zeros([sample,fnum,3])
for i in range(sample):
    trk[i,:,0] = xtrj[i]
    trk[i,:,1] = ytrj[i]
    trk[i,:,2] = arange(fnum)

trk[trk<0]=0

fofst = (pidx-1)*timeseg #frame idx offset
IOI = 10930 #klt idx of interested
IOIf = np.where(xtrj[IOI-seidx[0]]>0)[0]
print('ptslen : {0}').format(len(IOIf)) 
IOIfs = IOIf[0]  #starting frame
IOIfe = IOIf[-1]  #end frame
IOIL  = labels[np.where(mask==IOI)[0]]  #label of klt idx 


mlabels = np.ones(sample)*-1
#build pts trj labels (-1 : not interested pts)

for idx,i in enumerate(mask): #map mask(interested idx) to mlabel(both not/interested and )
    mlabels[i-seidx[0]] = labels[idx]

vcxtrj = {}
vcytrj = {}

for i in np.unique(mlabels):
    vcxtrj[i]=[]
    vcytrj[i]=[]

video_src = '/home/andyc/Videos/jayst.mp4'

cam = cv2.VideoCapture(video_src)
nrows = cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
ncols = cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
fnum  = int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

plt.figure(1,figsize=[10,12])
axL = plt.subplot(1,1,1)
frame = np.zeros([nrows,ncols,3]).astype('uint8')
im = plt.imshow(np.zeros([nrows,ncols,3]))
axis('off')
color = array([random.randint(0,255) \
               for _ in range(3*len(set(labels)))])\
               .reshape(len(set(labels)),3)

cam.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,IOIfs+fofst)
frame_idx = IOIfs


def Virctr(x,y):
    '''
    calculate virtual center, and remove out lier

    '''    
    if len(x)<3:
        vcx = mean(x)
        vcy = mean(y)
    else:
        mx = np.mean(x)
        my = np.mean(y)
        sx = np.std(x)
        sy = np.std(y)
        idx = ((x-mx)<2*sx)&((y-my)<2*sy)
        vcx = np.mean(x[idx])
        vcy = np.mean(y[idx])
    return vcx,vcy




while (frame_idx < (IOIfe+1)):

    print (frame_idx+fofst)
    ret, frame[:] = cam.read()
    plt.draw()
    pts = trk[:,:,0].T[frame_idx]!=0 # pts appear in this frame



    #labinf = list(set(mlabels[pts])) # label in current frame
    dots = []
    #for k in labinf:
    #    if k !=-1:
    #pdb.set_trace()
    for k in IOIL:
            x = trk[:,:,0].T[frame_idx][(mlabels==k)&pts]
            y = trk[:,:,1].T[frame_idx][(mlabels==k)&pts]
            vx,vy = Virctr(x,y) # find virtual center
            
            vcxtrj[k].append(vx) 
            vcytrj[k].append(vy)
            #lines = axL.plot(vcxtrj[k],vcytrj[k],color = (0,1,0),linewidth=2)
            #lines = axL.plot(vcxtrj[k],vcytrj[k],color = (color[k-1].T)/255.,linewidth=2)
            line_exist = 1
            #dots.append(axL.scatter(vx, vy, s=50, color=(color[k-1].T)/255.,edgecolor='black')) 
            dots.append(axL.scatter(x, y, s=50, color=(color[k-1].T)/255.,edgecolor='none')) 
                      
    im.set_data(frame[:,:,::-1])
    plt.draw()
    plt.show()

    name = './img/test3/'+str(frame_idx).zfill(4)+'.jpg'
    savefig(name)
   
    
    while line_exist :
        try:
            axL.lines.pop(0)
        except:
            line_exist = 0
    plt.show()
    

    for i in dots:
        i.remove()
        plt.show()

    frame_idx = frame_idx+1
  
    #pdb.set_trace()

