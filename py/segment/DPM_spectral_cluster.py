import numpy as np
import scipy as sp
import scipy.io as scipy_io
from sklearn.utils import check_random_state
from sklearn import (manifold, datasets, decomposition, ensemble, lda,
                     random_projection,mixture)
from sklearn.manifold import *
from scipy.io import savemat
import pdb


class sparse_subspace_clustering:
    def __init__(self,lambd = 10,dataset = np.random.randn(100),n_dimension = 100,random_state= None):
        """dataset :[sample,feature] """
        self.lambd = lambd
        self.dataset = dataset
        self.random_state = random_state
        self.n_dimension = n_dimension

    def get_adjacency(self,adjacency):
        self.adjacency = adjacency
    def manifold(self):
        random_state = check_random_state(self.random_state)              
        self.embedding_ = spectral_embedding(self.adjacency,n_components=self.n_dimension,eigen_solver='arpack',\
                                             random_state=random_state)*1000

    def clustering(self,n_components,alpha):
        model = mixture.DPGMM(n_components=n_components,alpha=alpha,n_iter = 1000)
        model.fit(self.embedding_)
        self.label = model.predict(self.embedding_)
        return self.label, model


def spec_cluster(adjmtx,subgraphs,mask):

    #file = scipy_io.loadmat('./mat/seg_adj4')
    #feature =(file['adj'] > 0).astype('float')
    #c = file['c']
    #mask = file['mask']
    #labels = np.zeros(c.size)
    feature = (adjmtx > 0).astype('float')
    labels = np.zeros(subgraphs.size) 

    for i in np.unique(subgraphs):
        print i
        sub_index = np.where(subgraphs == i)[0]
        sub_matrix = feature[sub_index][:,sub_index]
        if sub_index.size >3:
            project_dimension = int(np.floor(sub_index.size/20)+1)
            ssc = sparse_subspace_clustering(2000000,feature,n_dimension = project_dimension)
            ssc.get_adjacency(sub_matrix)
            ssc.manifold()
            sub_labels,model = ssc.clustering(n_components=int(np.floor(sub_index.size/2)+1),alpha= 0.1) 
            labels[sub_index] = np.max(labels) + (sub_labels+1)
            print 'number of trajectory %s'%sub_labels.size + '  unique labels %s' % np.unique(sub_labels).size
        else:
            sub_labels = np.ones(sub_index.size)
            labels[sub_index] = np.max(labels) + sub_labels
            print 'number of trajectory %s'%sub_labels.size + '  unique labels %s' % np.unique(sub_labels).size

    j = 0
    labels_new = np.zeros(labels.shape)
    unique_array =np.unique(labels)

    for i in range(unique_array.size):
        labels_new[np.where(labels == unique_array[i])] = j
        j = j+1

    labels = labels_new

    ### save data ###
    #labelsave ={}
    #labelsave['label']=labels
    #labelsave['mask']=mask
    #savemat('./mat/seg_label4',labelsave)
    return labels
