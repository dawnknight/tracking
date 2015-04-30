import cv2,time
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
import matplotlib.pyplot as plt
from matplotlib.pylab import * 
import numpy as np



def Io():
  
    ans = raw_input("choose file type ... 1.Video 2.Image 3.others : ")
    if ans == '1':
        print('choose video file ...')
        files = askopenfilename(initialdir = '/home/andyc/Videos/')
        Type ='Video'
    elif ans == '2':
        print('choose image file ...')
        files = askopenfilename(initialdir = '/home/andyc/image/')
        Type = 'Image'
    else:
        print('choose file ...')
        files = askopenfilename(initialdir = '/home/andyc/')
        Type = "Others"
    
    return files,Type


def Readfile(files,Type):
    
    if Type == 'Video':
        cap = cv2.VideoCapture(files)
        _, frame = cap.read()
        cap.release()
    elif Type == 'Image':
        frame = imread(files)
    else:
        print('Not a image or video files ...\n')
    return frame


def Createmask(frame,files):
    import pickle

    nrows,ncols,nd = frame.shape
    tmp = np.zeros([nrows+100,ncols+100,nd])
    mask = np.zeros([nrows,ncols])
    tmp[50:nrows+50,50:ncols+50,:] = frame
    print('select the region pts ....')
    plt.figure(1),plt.imshow(uint8(tmp)[:,:,::-1])
    again = 1
    cnt = 1
    alpha = 0.5 #transparncy
    tmpmap = np.zeros(tmp.shape)

    sidx = files[::-1].index('/')
    eidx = files[::-1].index('.')+1
    mdname = files[-sidx:-eidx]     #mask default name

    while again:
        
        print("please select region {0}...".format(cnt))
        pts = np.round(ginput(0,0))
        cv2.fillPoly(tmpmap, pts =[pts.astype(int)], color=(0,255,0))
        plt.figure(2),plt.imshow(uint8(tmpmap*alpha+tmp*(1-alpha))[:,:,::-1])  
        ans = raw_input("Is the region right ? Yes or No ? \n").upper()
        if ans == 'Y' or ans =='YES' or ans =='':
            print('saving the setting....\n\n')
            cnt += 1
        else:
            print('reselect....\n\n')
            cv2.fillPoly(tmpmap, pts =[pts.astype(int)], color=(0,0,0))
            close(figure(2))
            continue

        ans = raw_input("Finish ? Yes or No ? \n").upper()

        if ans == 'Y' or ans =='YES' or ans =='':
            close('all')
            again = 0
            mask[:] = 1.0*(tmpmap[50:nrows+50,50:ncols+50,1]<200)
            print("where do you want to save?(choose folder)\n")
            savepath = askdirectory(initialdir = '/home/andyc/tracking/py/mask/')
            name = raw_input("Input name of the mask(without extension file name)\n... default name will be "+mdname+"_mask.pkl ... \n")
            if name == '':
                name = mdname          
            savename = savepath+'/'+name+'_mask.pkl'
            print(savename)
            pickle.dump(uint8(mask),open(savename,"wb"),True)
            figure(),plt.imshow(mask,'gray')
            plt.title('mask')
            print('\n\n mask pkl file is already created and saved in '+savepath)
            
            time.sleep(6)
            close('all')


        else:
            close(figure(2))

def run():

    name,types = Io()
    f = Readfile(name,types)
    Createmask(f,name)
