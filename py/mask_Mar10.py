import numpy as np
import pickle
import scipy.ndimage as nd

mask = nd.imread('/home/andyc/image/Feb11_00.jpg')[:,:,0]/255 

pickle.dump(mask,open('mask_Feb11.pkl','wb'),True)

