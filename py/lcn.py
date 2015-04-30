'''Local contrast normalization

dividing each pixel by the standard 

deviation of its neighbors
'''

def Lcn(img,blksize):
# f is an M-by-N size image
    m,n = img.shape
    mr  = m%blksize
    nr  = n%blksize
    img = img[:m-mr,:n-nr]
    m,n = img.shape
    imgr = np.zeros(img.shape)
    avg = np.zeros(img.shape)
    sigma = np.zeros(img.shape)
    result = np.zeros(img.shape)

    for i in range(blksize):
        for j in range(blksize):
            imgr[:] = np.roll(img,-i,0)
            imgr[:] = np.roll(imgr,-j,1)
            avg[i::blksize,j::blksize] = imgr.reshape(m/blksize,blksize,n/blksize,blksize).mean(1).mean(-1)
            sigma[i::blksize,j::blksize] = imgr.reshape(m/blksize,blksize,n/blksize,blksize).std(1).std(-1)
    sigma[:] = sigma+1
    result[:] = (img-avg)/sigma
    return result

