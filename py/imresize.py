
define imresize(im,factor): #

    if ((factor<1) or (factor%1>0)):
        print('factor should be positive interger.....')

    nrow,ncol,nd = im.shape

    im_bin - im.reshape(nrow/factor,factor,ncol/factor,factor,nd).mean(1).mean(-2)

    return im_bin 
