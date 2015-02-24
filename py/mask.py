import numpy as np
import pickle
import scipy.ndimage as nd

mask = nd.imread('/home/andyc/image/test1.bmp')[:,:,0]/255
imshow(mask,'gray')
pickle.dump(mask,open('./mask/TLC0005_mask.pkl','wb'),True)
