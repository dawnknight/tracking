import numpy as np
import pickle
import scipy.ndimage as nd

mask = nd.imread('/home/andyc/image/000.jpg')[:,:,0]/255

pickle.dump(mask,open('Feb11_mask2.pkl','wb'),True)
