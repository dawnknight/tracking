import os, glob,cv2
import scipy as sp
import numpy as np
from PIL import Image
import scipy.ndimage as nd

#path = '/home/andyc/image/Day/'
path = '/home/andyc/image/Feb11 resize/'
imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))

bg = np.zeros(nd.imread(imlist[0]).shape)
sig = np.ones(nd.imread(imlist[0]).shape)*10**(-6)
Ve = np.zeros(nd.imread(imlist[0]).shape)
He = np.zeros(nd.imread(imlist[0]).shape)
Ve_bg = np.zeros(nd.imread(imlist[0]).shape)
He_bg = np.zeros(nd.imread(imlist[0]).shape)

Cc = np.zeros(nd.imread(imlist[0]).shape) 
Ce = np.zeros(nd.imread(imlist[0]).shape)

Gs = np.zeros(nd.imread(imlist[0]).shape)

result = np.zeros(nd.imread(imlist[0]).shape)

mc = 15
Mc = 25

me = 3
Me = 9

for i in range(len(imlist)):
    print i
    im = nd.imread(imlist[i]).astype(float)
    Ve[:,:,:] = nd.sobel(im, 1)
    He[:,:,:] = nd.sobel(im, 0)

    if i is 0:
        bg[:,:,:] = im  
    else:
        bg[:,:,:] = (bg*i+im)/(i+1)        
        Ve_bg[:,:,:] = nd.sobel(bg, 1)
        He_bg[:,:,:] = nd.sobel(bg, 0)

        sig[:,:,:] = ((im-bg)**2+sig*i)/(i+1)

        Cc[:,:,:] = ((im - bg)-mc*sig**0.5)/((Mc-mc)*sig**0.5)*100        
        Cc[Cc<0] = 0
        Cc[Cc>100] = 100
        Cs = Cc.max(axis=2)  ##check

        dVe = abs(Ve - Ve_bg);
        dHe = abs(He - He_bg);

        dG = dVe+dHe
        G  = abs(He)+abs(Ve)
        G_bg = abs(He_bg)+abs(Ve_bg)
        
        Gs[:,:,0] = asarray([G[:,:,0],G_bg[:,:,0]]).max(axis=0)
        Gs[:,:,1] = asarray([G[:,:,1],G_bg[:,:,1]]).max(axis=0)
        Gs[:,:,2] = asarray([G[:,:,2],G_bg[:,:,2]]).max(axis=0)        
        Gs[Gs==0] = 10**(-6)
        R = dG/Gs
    
        Ve_sig = (Ve - Ve_bg)**2
        He_sig = (He - He_bg)**2
        sig_e =Ve_sig+He_sig
        sig_e[sig_e==0] =10**(-6)

        Ce[:,:,:] = (R*dG-me*sig_e**0.5)/((Me-me)*sig_e**0.5)*100        
        Ce[Ce<0] = 0
        Ce[Ce>100] = 100
        Cs_e = Ce.max(axis=2)  ##check

        C = asarray([Cs,Cs_e]).max(axis=0)

        name = '/home/andyc/image/Aug 06 (resize)/'+str(i).zfill(3)+'.jpg'

        result[:,:,0] = (C-C.min())/(C.max()-C.min())*255
        result[:,:,1] = result[:,:,0]
        result[:,:,2] = result[:,:,0]
        cv2.imwrite(name,uint8(result))
    
