from scipy.io import loadmat,savemat
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import numpy as np
import pdb



ptstrj = loadmat('./mat/ptsTrj_part4.mat')
x = csr_matrix(ptstrj['xtracks'], shape=ptstrj['xtracks'].shape).toarray()
y = csr_matrix(ptstrj['ytracks'], shape=ptstrj['ytracks'].shape).toarray()
seidx  = ptstrj['startend'][0]
sample = ptstrj['xtracks'].shape[0]
fnum   = ptstrj['xtracks'].shape[1]

x[x<0]=0
y[y<0]=0


xspeed = np.diff(x)*((x!=0)[:,1:])
yspeed = np.diff(y)*((y!=0)[:,1:])
speed = np.abs(xspeed)+np.abs(yspeed)
# chk if trj is long enough
mask =[]
x_re = []
y_re =[]
xspd =[]
yspd =[]
minspdth = 9 #threshold of min speed
fps = 4
transth  = 60*fps   #transition time (red light time)



print('initialization finished....')
#pdb.set_trace()
for i in range(sample):
    if sum(x[i,:]!=0)>4:  # chk if trj is long enough
        try:
            if max(speed[i,:][x[i,1:]!=0][1:-1])>minspdth: # check if it is not a not move point
                if sum(speed[i,:][x[i,1:]!=0][1:-1] < 3) < transth:  # check if it is a idole point
                    mask.append(i+seidx[0])
                    x_re.append(x[i,:])
                    y_re.append(y[i,:])
                    xspd.append(xspeed[i,:])
                    yspd.append(yspeed[i,:])
        except:
            pass

sample = len(x_re)
adj = np.zeros([sample,sample])
dth = 30
spdth = 5
num = arange(fnum)
x_re = array(x_re)
y_re = array(y_re)
xspd = array(xspd)
yspd = array(yspd)

print(sample)
print('building adj mtx ....')
#pdb.set_trace()

# build adjacent mtx
for i in range(sample):
    print i
    for j in range(sample):
        if i<j:
            tmp1 = x_re[i,:]!=0
            tmp2 = x_re[j,:]!=0
            idx  = num[tmp1&tmp2]
            if len(idx)>0:
                sidx = idx[1:-1]
                sxdiff = np.mean(np.abs(xspd[i,sidx]-xspd[j,sidx]))
                sydiff = np.mean(np.abs(yspd[i,sidx]-yspd[j,sidx]))
                if (sxdiff <spdth ) & (sydiff<spdth ):
                    mdis = mean(np.abs(x_re[i,idx]-x_re[j,idx])+np.abs(y_re[i,idx]-y_re[j,idx]))
                    #mahhattan distance
                    if mdis < dth:
                        adj[i,j] = 1
        else:
            adj[i,j] = adj[j,i]

sparsemtx = csr_matrix(adj)
s,c = connected_components(sparsemtx)

'''
### save data #####
result = {}
result['adj'] = adj
result['c']   = c
result['mask']= mask

savemat('./mat/seg_adj4',result)
'''

### spectral clustering ###

from DPM_spectral_cluster import spec_cluster

DPM_labels = spec_cluster(adj,c,mask)

result = {} 
result['label']=DPM_labels                             
result['mask']=mask                                                
savemat('./mat/label4',result) 
 
