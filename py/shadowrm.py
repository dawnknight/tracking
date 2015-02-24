import pickle
import numpy as np

bg = pickle.load(open("BG.pkl","rb"))
fg = pickle.load(open("fg.pkl","rb"))
frame = pickle.load(open("frame.pkl","rb"))
mask = pickle.load(open("./mask/TLC0005_mask.pkl","rb"))


h = frame.shape[0]
w = frame.shape[1] 
N = 5
th = 0.995

#def ShadowRm(fg,frame,bg,N,h,w,th):
if 1 :
    NCC= np.zeros([h,w])
    BG = np.zeros([h+2*N,w+2*N])
    FG = np.zeros([h*2*N,w+2*N])
    BG[N:h+N,N:w+N]=bg.mean(2)
    FG[N:h+N,N:w+N]=frame.mean(2)
    for ii in range(h):
        for jj in range(w):
            if fg[ii,jj]==1:
                ER = sum((BG[ii-N:ii+N,jj-N:jj+N]*FG[ii-N:ii+N,jj-N:jj+N]).flatten())
                EB = (sum((BG[ii-N:ii+N,jj-N:jj+N]**2).flatten()))**0.5+10**(-6)
                ET = (sum((FG[ii-N:ii+N,jj-N:jj+N]**2).flatten()))**0.5+10**(-6)
                #print(ER/EB/ET)
                if (ER/EB/ET) >th:
                    NCC[ii,jj] = 1
    #return NCC
   
plt.figure(1,figsize=[10,5])
plt.subplot(121)
imshow(fg*mask,'gray')
plt.subplot(122)
imshow(NCC*mask,'gray')
