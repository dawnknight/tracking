'''
import numpy as np
import pickle,pdb

pts = pickle.load(open('pts.pkl','rb'))
pts=pts.astype('int')

bandwidth = 20
itr= 1
ini = 1
th = 0.5
#def Meanshift(pts,bandwidth,th = 0.5,ini = 1,itr = 1):


if 1:
                
    titr = 0         
    while itr:
        chk = np.zeros(len(pts))
        ctr = [] #center coordinate
        idx = 1
    
        if ini==1:
            ini =0
            for i in range(len(pts)):
                if chk[i] ==0:
                    pos = (sum((pts - pts[i])**2,1)**0.5)<bandwidth # boolean array
                    ctr.append(pts[pos].mean(0))    #group cneter
                    chk[pos] = idx
                    idx += 1
            oldctr = ctr
            itr = 1
        else:
            pdb.set_trace()


            for i,ctrpt in enumerate(oldctr):
                if i == 0:
                    disvec = (sum((pts - array(ctrpt[i]))**2,1)**0.5)
                    chk[disvec<bandwidth] = idx
                else:
                    dis = (sum((pts - array(ctrpt[i]))**2,1)**0.5)
                    disvec[dis<disvec] = dis[dis<disvec]
                    chk[dis<disvec] = idx

            chk 
            







            ctr.append(pts[pos].mean(0))




                chk[pos] = idx
                idx += 1



            pdb.set_trace()        



            itr = sum((sum((array(ctr)-array(oldctr))**2,1)**0.5>th)*1.)
            oldctr = ctr

        titr += 1
        print('{0} iteration: there are {0} clusters'.format(titr,len(ctr)))

    print('During {0} iterations, there are {0} clusters'.format(titr,len(ctr)))

    #return chk,ctr
'''

# Import stuff
import numpy as _n
#import pyublas as _p
import libmeanshift

def find(X, dim=0):
    return (X.ravel().nonzero())[dim]

def meanShift(X, radius, rate=.2, maxIter=100, minCsize=1, blur=0):
    '''MeanShift clustering algorithm.
 
    Alvaro Collet ported this to Python
    Based on Piotr's Image & Video Toolbox
    Based on code from Sameer Agarwal <sagarwal-at-cs.ucsd.edu>.

    For a broad discussion see:
    Y. Cheng, Mean-shift, mode seeking, and clustering, IEEE Transactions on
    Pattern Analysis and Machine Intelligence, Vol.17, 1995, pp. 790-799
 
    The radius or bandwidth is tied to the 'width' of the distribution and is
    data dependent.  Note that the data should be normalized first so that
    all the dimensions have the same bandwidth.  The rate determines how
    large the gradient decent steps are.  The smaller the rate, the more
    iterations are needed for convergence, but the more likely minima are not
    overshot.  A reasonable value for the rate is .2.  Low value of the rate
    may require an increase in maxIter.  Increase maxIter until convergence
    occurs regularly for a given data set (versus the algorithm being cut off
    at maxIter).
 
    Note the cluster means M do not refer to the actual mean of the points
    that belong to the same cluster, but rather the values to which the
    meanShift algorithm converged for each given point (recall that cluster
    membership is based on the mean converging to the same value from
    different points).  Hence M is not the same as C, the centroid of the
    points [see kmeans2 for a definition of C].
 
    USAGE
    Labels, Means = meanShift(X, radius, [rate], [maxIter], [minCsize], [blur] )
 
    INPUTS
     X           - Numpy array column vector of data - N vectors of dim p (X is
                   N-by-p). Warning, this array must be stored like this in the
                   underlying storage. To do so, you could do a deep copy of it
                   by doing: X = numpy.array(some_array)
     radius      - the bandwidth (radius of the window)
     rate        - [] gradient descent proportionality factor in (0,1]
     maxIter     - [] maximum number of iterations
     minCsize    - [] min cluster size (smaller clusters get eliminated)
     blur        - [] if blur then at each iter data is 'blurred', ie the
                   original data points move (can cause 'incorrect' results)
 
    OUTPUTS
     LABELS      - cluster membership
     MEANS       - cluster means
 
    EXAMPLE
 
    See also MEANSHIFTIM, DEMOCLUSTER
 
    Piotr's Image&Video Toolbox      Version 2.0
    Copyright 2008 Piotr Dollar.  [pdollar-at-caltech.edu]
    Please email me if you find bugs, or have suggestions or questions!
    Licensed under the Lesser GPL [see external/lgpl.txt]

    Ported to Python by Alvaro Collet <acollet@cs.cmu.edu>
    '''

    if rate < 0 or rate > 1:
        print "Rate must be between 0 and 1"
    
    # MeanShift class
    m = libmeanshift.MeanShift()

    # C++ code does the work 
    m.cluster(X, radius, rate, maxIter, blur)

    # calculate final cluster means per cluster
    N, p = X.shape
    k = int( m.labels.max() )

    Means = _n.zeros( (k,p) )
    
    for i in xrange(k):
        idx = find(m.labels == i+1)
        Means[i, :] = _n.mean(m.means[idx, :], 0)

    # Sort in descending order and reshape to have 1-d array
    order = _n.argsort(-m.ccounts, 0).reshape(m.ccounts.size)
    Means = Means[order]
    
    Labels = _n.zeros(m.labels.shape)
    
    for i in xrange(k):
        Labels[m.labels == order[i]+1] = i+1

    if minCsize > 1:
        ccounts = ccounts[order]
        ccounts = ccounts[ccounts >= minCsize]
        Means = Means[ccounts.size, :]
        Labels[Labels >= ccounts.size] = -1

    return Labels, Means

# Example of use
if __name__ == '__main__':

    import pickle    
    pts = pickle.load(open('pts.pkl','rb'))                                                                                          
    my_array_T = pts.astype('int').T
    L, M = meanShift(my_array_T, 20)
    print "labels"
    print L
    print "data"
    print M
