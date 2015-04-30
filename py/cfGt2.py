import pickle
import matplotlib.pyplot as plt

class Objects():
    def __init__(self):
        self.ptsTrj= {}
        self.pts = []
        self.Trj = []
        self.xTrj = []
        self.yTrj = []
        self.frame = []
        self.vel = []
        self.pos = []
        self.status = 1   # 1: alive  2: dead                                                                                        

    def Color(self):
        try:
            self.R
        except:
            self.R = randint(0,255)
            self.G = randint(0,255)
            self.B = randint(0,255)
        return (self.R,self.G,self.B)

obj =pickle.load(open("/home/andyc/tracking/py/pkl/obj_jayst80.pkl","rb"))


x12 = obj[12].xTrj[:70]
y12 = obj[12].yTrj[:70]
x16 = obj[16].xTrj[:70]
y16 = obj[16].yTrj[:70]

d = ((array(x12)-array(x16))**2 + (array(y12)-array(y16))**2 )**0.5

figure(1,figsize=[7.5,7.5])
plt.grid(b=1,lw =2)
                                                                                                                 
title('Jay street')                                                                                    
plt.xlabel('Frame No.')                                                                                                             
plt.ylabel('Distance [pixels]')
xlabel = range(70)

plot(xlabel,d,'w')

for i in range(70):
    print i
    plot(xlabel[0:i+1],d[0:i+1],color = 'b')
    plt.scatter(xlabel[0:i+1],d[0:i+1],color = 'b')

    named = '/home/andyc/image/AIG/d/'+repr(i).zfill(3)+'.jpg'
    savefig(named)
