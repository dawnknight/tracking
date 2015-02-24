# compare with ground truth

import pickle
import matplotlib.pyplot as plt

#### re-arrange data ####

gt =pickle.load(open("/home/andyc/tracking/output/ca_coords.pkl","rb"))
GT={}
car1 = {}
car2 = {}
x1 = []
x2 = []
y1 = []
y2 = []

x ={}
y ={}


ino = 200  #initial frame number
eno = 257  #end frame number


frame = arange(ino,eno+1)

for i in frame:
    x2.append(gt['1_179_C_s_sb'][i][4])
    y2.append(gt['1_179_C_s_sb'][i][5])
    x1.append(gt['2_188_C_r_wb'][i][4])
    y1.append(gt['2_188_C_r_wb'][i][5])

x[0] = array(x1).T
x[1] = array(x2).T

y[0] = array(y1).T
y[1] = array(y2).T


GT['x'] = x
GT['y'] = y
GT['frame'] = frame

#### compare processing #####

class Blob(): #create a new blob                                                                                                      
    def __init__(self,fshape,ini_x,ini_y):
        self.x = []                #current status                                                                                       
        self.x_old  = []           #previous  status                                                                                    
        self.xp = np.array([[ini_y,ini_x,0,0]]).T
        self.u  = array([[0,0,0,0]]).T
        self.P = 100*np.eye(4)
        self.Trj = {}
        self.len = [0,0]                #Bbox size                                                                                  
        self.ref = array([[ini_y,ini_x,0,0]])   #Bbox position and velocity x,y,vx,vy                                                   
        self.ori = []                   #Bbox measurment position                                                                       
        self.status = 0                 # 0 : as initial  1 : as live  2 : as dead                                               
        self.dtime = 0                  # obj dispear period                                                                         
        self.ivalue = []                # store intensity value in Bbox                                                               
        self.RGB = []

        self.Trj['x'] = []
        self.Trj['y'] = []
        self.Trj['frame'] = []

    def Color(self):    #initial Blob color                                                                                       
        try:
            self.R
        except:
            self.R = randint(0,255)
            self.G = randint(0,255)
            self.B = randint(0,255)
        return (self.R,self.G,self.B)

blobs = pickle.load(open("./info/caraccident.pkl","rb"))

Bidx = [0,3]  #target blob idx  0:normal car # ploice car

clf()
x = {}
y = {}
frame ={}
color = {}
fnum = eno-ino  #total frame number                                                                       
#ino = 200 #initial frame number 



for i,_ in enumerate(Bidx):
    x[i] = array(blobs[Bidx[i]].Trj['x']).T
    y[i] = array(blobs[Bidx[i]].Trj['y']).T
    frame[i] = array(blobs[Bidx[i]].Trj['frame']).T
    color[i] = array(blobs[Bidx[i]].Color())


figure(1,figsize=[7.5,7.5])                                                                                                         
plt.grid(b=1,lw =2)                                                                                                                   
title('car acident (compare with manual label)')                                                                                       
plt.xlabel('Frame No.')                                                                                                                  
plt.ylabel('Horizontal position [pixels]')   

plot(GT['frame'],GT['x'][0],color = array([1.,0.,1.]),label = "$Red$"+" "+"$Car$"+"   "+"$(manual)$")
plot(GT['frame'],GT['x'][1],color = array([0.8,0.5,0.6]),label = "$Police$"+" "+"$Car$"+" "+"$(manual)$")

plot(frame[0][:fnum],x[0][0][:fnum],color = array([0.6,0.85,0.1]),label = "$Red$"+" "+"$Car$"+"   "+"$(auto)$")
plot(frame[1][:fnum],x[1][0][:fnum],color = array([0.,0.,1.]),label = "$Police$"+" "+"$Car$"+" "+"$(auto)$")

legend(loc=1)

namex = '/home/andyc/cfx.jpg'                                                            
savefig(namex)                                          

figure(2,figsize=[7.5,7.5])
plt.grid(b=1,lw =2)
title('car acident (compare with manual label)')
plt.xlabel('Frame No.')
plt.ylabel('vertical position [pixels]')

plot(GT['frame'],GT['y'][0],color = array([1.,0.,1.]),label = "$Red$"+" "+"$Car$"+"   "+"$(manual)$")
plot(GT['frame'],GT['y'][1],color = array([0.8,0.5,0.6]),label = "$Police$"+" "+"$Car$"+" "+"$(manual)$")

plot(frame[0][:fnum],y[0][0][:fnum],color = array([0.6,0.85,0.1]),label = "$Red$"+" "+"$Car$"+"   "+"$(auto)$")
plot(frame[1][:fnum],y[1][0][:fnum],color = array([0.,0.,1.]),label = "$Police$"+" "+"$Car$"+" "+"$(auto)$")

legend(loc=1)

namex = '/home/andyc/cfy.jpg'
savefig(namex)
