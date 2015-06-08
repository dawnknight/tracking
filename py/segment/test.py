from scipy.io import loadmat,savemat
import pdb
num = 0
atmp = []
flab = []  #final labels
fmask = [] #fimal mask

for idx in range(1,4): 
    #pdb.set_trace() 
    if idx == 1:
        name1 = './mat/label'+str(idx)+'.mat'
        L1 = loadmat(name1)['label'][0]
        M1 = loadmat(name1)['mask'][0]
    else:
        L1 = L2
        M1 = M2

    name2 = './mat/label'+str(idx+1)+'.mat'
    L2 = loadmat(name2)['label'][0]
    M2 = loadmat(name2)['mask'][0]

    labelnum = max(L1)
    commonidx = np.intersect1d(M1,M2)  #idx exist in 2 trucations
    
    print('L1 : {0}, L2 : {1}').format(len(np.unique(L1)),len(np.unique(L2)))

    L2[:] = L2 + labelnum

    '''
    for i in commonidx:
        print i 
        label1 = L1[np.where((M1 == i)!=0)[0][0]]
        print('\nlabel 1 : {0}').format(label1)
        idx1 = np.where(L1==label1)
        print('label 1 group : {0}').format(M1[idx1])
        label2 = L2[np.where((M2 == i)!=0)[0][0]]
        print('label 2 : {0}').format(label2)
        idx2 = np.where(L2==label2)
        print('label 2 group : {0}\n\n').format(M2[idx2])
    '''
    #pdb.set_trace()
    for i in commonidx:
        if i not in atmp:
        
            label1 = L1[np.where((M1 == i)!=0)[0][0]]
            label2 = L2[np.where((M2 == i)!=0)[0][0]]
            idx1 = np.where(L1==label1)
            idx2 = np.where(L2==label2)
            tmp1 = np.intersect1d(M1[idx1],commonidx)
            tmp2 = np.intersect1d(M2[idx2],commonidx)

            L1idx =list(idx1[0])
            L2idx =list(idx2[0])

            diff = np.setdiff1d(tmp1,tmp2)
            atmp = np.unique(np.hstack((tmp1,tmp2)))


            if len(diff)>0:
                for j in diff:
                    if j in tmp1:
                        wlab = L1[np.where((M1 == j)!=0)[0][0]] # wrong labels
                        widx = np.where(L1==wlab) #idx of wrong od wrong labels
                        L1[widx] = label1
                        L1idx.append(np.where(M1 == j)[0][0])
                    else: # j in tmp2:
                        wlab = L2[np.where((M2 == j)!=0)[0][0]]
                        widx = np.where(L2==wlab)
                        L2[widx] = label2
                        L2idx.append(np.where(M2 == j)[0][0])

            L1[L1idx] = label1
            L2[L2idx] = label1

    if idx == 1 :
        flab[:] = flab + list(L1) 
        fmask[:] = fmask + list(M1)

    flab[:] = flab +list(L2)
    fmask[:] = fmask + list(M2)


#pdb.set_trace()
#== eliminate duplicate part == 
   
data = vstack((fmask,flab))
result = data[:,data[0].argsort()]
labels = list(result[1])
mask   = list(result[0])
dpidx = np.where(np.diff(sorted(mask)) == 0)[0]
    
for k in dpidx[::-1]:
    labels.pop(k)
    mask.pop(k)
 

result = {}
result['label']   = labels
result['mask']= mask
savemat('./mat/finalresult',result)













