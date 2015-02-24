
i = 3
x = array(blobs[i].Trj['x']).T[0][2:]
y = array(blobs[i].Trj['y']).T[0][2:]


A =  np.vstack([x**3,x**2,x, np.ones(len(x))]).T
a,b,c,d = np.linalg.lstsq(A, y)[0]

#plot(x,y,'o', label='Original data', markersize=5)
plt.plot(x,a*x**3+b*x**2+c*x+d , color =array(blobs[i].Color())[::-1]/255., label='Fitted line',linewidth = 3)

#############################################################

figure()
imshow(frame[:,:,::-1])
axis('off')
for i in range(30):
       if len(blobs[i].Trj['x'])>20:
               x = array(blobs[i].Trj['x']).T[0][4:]
               y = array(blobs[i].Trj['y']).T[0][4:]
               A =  np.vstack([x**3,x**2,x, np.ones(len(x))]).T
               a,b,c,d = np.linalg.lstsq(A, y)[0]
               plot(x,y,'o', color =array(blobs[i].Color())[::-1]/255., markersize=5)
               plt.plot(x,a*x**3+b*x**2+c*x+d , color =array(blobs[i].Color())[::-1]/255.,linewidth = 3)




######################################################
def Ttrj(i):
       x = array(blobs[i].Trj['x']).T[0]
       y = array(blobs[i].Trj['y']).T[0]
 
       if Ttype(x,y) ==0:    
           A =  np.vstack([y**3,y**2,y, np.ones(len(y))]).T
           a,b,c,d = np.linalg.lstsq(A, x)[0]
           plt.plot(a*y**3+b*y**2+c*y+d ,y, color =array(blobs[i].Color())[::-1]/255.,linewidth = 3)
       else:
           A =  np.vstack([x**3,x**2,x, np.ones(len(x))]).T
           a,b,c,d = np.linalg.lstsq(A, y)[0]
           plt.plot(x,a*x**3+b*x**2+c*x+d , color =array(blobs[i].Color())[::-1]/255.,linewidth = 3)
      


def Ttype(x,y):
       d = (y[0]-y[-1])/(x[0]-x[-1])
       s = abs(math.atan(d)/pi*180)
       if s>=45.:
           r = 0
       else:
           r = 1
       return r    






#############################################################
import cmath
i = 5
x = array(blobs[i].Trj['x']).T[0][4:]
y = array(blobs[i].Trj['y']).T[0][4:]

p = x+y*1j

R =[]
T = []
for i in range(len(p)):
   R.append(cmath.polar(p[i])[0])
   T.append(cmath.polar(p[i])[1])

R = array(R)
T = array(T)

A =  np.vstack([R**3,R**2,R, np.ones(len(R))]).T
a,b,c,d = np.linalg.lstsq(A, T)[0]

tp = a*R**3+b*R**2+c*R+d

xp = R*cos(tp)
yp = R*cos(tp)

plot(x,y,'o', label='Original data', markersize=5)
plt.plot(xp, yp, 'r', label='Fitted line')
