import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage,\
                                 AnnotationBbox
import pickle
from scipy.ndimage import median_filter as mf

bdict = pickle.load(open('blob.pkl','rb'))
timestamp = pickle.load(open('time.pkl','rb'))

import scipy as sp
idx = bdict.keys()[0]
sidx = timestamp[idx][::-1].index('-')
titlename = timestamp[idx][:-sidx-1]

c = 0
trunc = 1000
per = 1800 #period: 1hr : 3600                                              

num = []
xlab = []
lab = []
for i in bdict.keys():
    num.append(bdict[i][0])
    lab.append(timestamp[i][-sidx:])
    if sp.mod(c,per)==0:
        xlab.append(timestamp[i][-sidx:])
    else:                                                                 
        xlab.append('')                                                    
    c+=1

fig, ax = plt.subplots()
plt.plot(range(len(num)),mf(num,10))



impath = '/home/andyc/tracking/py/179.jpg'
f = imread(impath)
imagebox = OffsetImage(f[::-1,:,:], zoom=0.4)

ab = AnnotationBbox(imagebox, [179,8],
                        xybox=(120., 180.),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.5,
                        arrowprops=dict(arrowstyle="->"))

ax.add_artist(ab)
#plt.draw()




impath = '/home/andyc/tracking/py/6065.jpg'
f = imread(impath)
imagebox = OffsetImage(f[::-1,:,:], zoom=0.4)

ab = AnnotationBbox(imagebox, [5823,11],
                        xybox=(160., 30.),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.5,
                        arrowprops=dict(arrowstyle="->"))

ax.add_artist(ab)
#plt.draw()


impath = '/home/andyc/tracking/py/10791.jpg'
f = imread(impath)
imagebox = OffsetImage(f[::-1,:,:], zoom=0.4)

ab = AnnotationBbox(imagebox, [10549,4],
                        xybox=(60., 150.),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.5,
                        arrowprops=dict(arrowstyle="->"))

ax.add_artist(ab)
#plt.draw()


impath = '/home/andyc/tracking/py/20185.jpg'
f = imread(impath)
imagebox = OffsetImage(f[::-1,:,:], zoom=0.4)

ab = AnnotationBbox(imagebox, [19943,6],
                        xybox=(-80., 150.),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.5,
                        arrowprops=dict(arrowstyle="->"))

ax.add_artist(ab)


plt.draw()

plt.xticks(range(len(xlab)),xlab)
plt.title(titlename)
plt.xlabel('Time') 
plt.ylabel('number of people') 


ax.set_xlim(-100, 21000)
ax.set_ylim(0, 14)
