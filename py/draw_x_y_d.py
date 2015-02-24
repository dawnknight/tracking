import matplotlib.pyplot as plt
import pickle

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
A = [0,3]

clf()
x = {}
y = {}
frame ={} 
color = {}
fnum = 58 #total frame number
ino = 200 #initial frame number 

for i,_ in enumerate(A):
    x[i] = array(blobs[A[i]].Trj['x']).T
    y[i] = array(blobs[A[i]].Trj['y']).T
    frame[i] = array(blobs[A[i]].Trj['frame']).T
    color[i] = array(blobs[A[i]].Color())

xlabel = range(ino,ino+fnum)


figure(1,figsize=[7.5,7.5])
plt.grid(b=1,lw =2)
title('car acident')
plt.xlabel('Frame No.')
plt.ylabel('Horizontal position [pixels]')
plot(frame[1],x[1][0],'w')

clabel = ["$Red$"+" "+"$Car$"+"   "+"$(manual)$","$Police$"+" "+"$Car$"]

legend(loc=1)
for i in range(ino,ino+fnum):
    for idx,j in enumerate(A):
        if i in frame[idx]:
            
            plot(       frame[idx][0:i-frame[idx][0]+1],x[idx][0][0:i-frame[idx][0]+1],color = array(blobs[j].Color())[::-1]/255.\
                            ,label = clabel)
            plt.scatter(frame[idx][0:i-ino+1],x[idx][0][0:i-ino+1],color = array(blobs[j].Color()[::-1])/255.) 
    plt.xlim(ino,ino+fnum-1)
    namex = '/home/andyc/image/tra/caraccident/x/'+repr(i).zfill(3)+'.jpg'   
    savefig(namex)

'''

figure(2,figsize=[7.5,7.5])
plt.grid(b=1,lw =2)
title('car acident')
plt.xlabel('Frame No.')
plt.ylabel('vertical position [pixels]')
#plot(xlabel,x[0][0][0:fnum],'w')
plot(frame[1],y[1][0],'w')
#plot(frame[18],y[18][0],'w')


for i in range(ino,ino+fnum):
    for idx,j in enumerate(A):
        #print idx,j
        #pdb.set_trace()
        if i in frame[idx]:
            plot(frame[idx][0:i-frame[idx][0]+1],y[idx][0][0:i-frame[idx][0]+1],color = array(blobs[j].Color())[::-1]/255.)
            plt.scatter(xlabel[0:i-ino+1],y[idx][0][0:i-ino+1],color = array(blobs[j].Color())[::-1]/255.)
    plt.xlim(ino,ino+fnum-1)
    namey = '/home/andyc/image/tra/caraccident/y/'+repr(i).zfill(3)+'.jpg'
    savefig(namey)



figure(3,figsize=[7.5,7.5])
plt.grid(b=1,lw =2)
title('car acident')
plt.xlabel('Frame No.')
plt.ylabel('Distance [pixels]')
plot(xlabel[0:fnum],((x[0][0][0:fnum]-x[1][0][0:fnum])**2+(y[0][0][0:fnum]-y[1][0][0:fnum])**2)**0.5,'w')

for i in range(ino,ino+fnum):
    for idx,j in enumerate(A):
        if i in frame[idx]:
            if idx !=0:   
                plot(frame[idx][0:i-frame[idx][0]+1],((x[0][0][0:i-frame[idx][0]+1]-x[idx][0][0:i-frame[idx][0]+1])**2+\
                                                (y[0][0][0:i-frame[idx][0]+1]-y[idx][0][0:i-frame[idx][0]+1])**2)**0.5,\
                                                 color = 'b')
                plt.scatter(xlabel[0:i-ino+1],((x[0][0][0:i-ino+1]-x[idx][0][0:i-ino+1])**2+\
                                              (y[0][0][0:i-ino+1]-y[idx][0][0:i-ino+1])**2)*0.5,color = 'b')

            plt.xlim(ino,ino+fnum-1)
            named = '/home/andyc/image/tra/caraccident/d/'+repr(i).zfill(3)+'.jpg'
            savefig(named)


'''

###### calculate TTC (Time to crash) ######
fps = 25

D = ((x[0][0][0:257-frame[1][0]+1]-x[1][0][0:257-frame[1][0]+1])**2+\                   
     (y[0][0][0:257-frame[1][0]+1]-y[1][0][0:257-frame[1][0]+1])**2)**0.5

a = x[0][0][:58]
b = y[0][0][:58]

V0x =(a-np.roll(a,1))*fps
V0y =(b-np.roll(b,1))*fps

a = x[1][0][:58]
b = y[1][0][:58]

V1x =(a-np.roll(a,1))*fps
V1y =(b-np.roll(b,1))*fps



TTC = D/(((V0x-V1x)**2+(V0y-V1y)**2)**0.5+10**-6)
TTC[0]=TTC[1]


figure(4,figsize=[7.5,7.5])                                                                                                             
plt.grid(b=1,lw =2)                                                                                                                   
title('TTC')                                                                                                      
plt.xlabel('Frame No.')                                                                                                  
plt.ylabel('Time to crash')    

plot(range(200,258),TTC)

named = '/home/andyc/TTC.jpg'

savefig(named)
