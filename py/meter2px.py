import math
from math import pi
import numpy as np

def Meter2px(h,vd,vd1,hd,nrows,ncols):
    ### real scale (unit : meter)
    #h  : building size
    #vd : distance between buliding to the bottom of the image
    #vd1: distance between bottom to up of the image
    #hd : horizontal distancd
    
    ### pixels based
    # nrows
    # ncols


    theta = math.atan(1.*vd/h) #radius 
    z=(h**2+vd**2)**0.5
    alpha = pi/2-theta
    beta = math.atan((vd+vd1)*1./h)-theta

    ydp = beta/ncols
    xdp = beta/nrows

    ylen = np.zeros(ncols)
    xlen = np.ones(nrows)*hd/nrows

    distance = vd
    for i in range(ncols):
        
        gamma = (theta+ydp*(1+i))
        pylen = h*math.tan(gamma)-distance # pixel length
        ylen[i] = pylen
        distance = distance+pylen

    print ylen[-1]/ylen[0]
    return xlen,ylen[::-1]
