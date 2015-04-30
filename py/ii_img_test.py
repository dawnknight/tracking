import cv2

vid = '/home/andyc/Videos/jayst.mp4'
cam = cv2.VideoCapture(video_src)
ret, frame = cam.read()
plt.figure(1,figsize=[10,12])

for idx,alpha in enumerate(arange(38000,50000,50)/100000.):
    print idx
    img = 0.5 +log(frame[:,:,1])-alpha*\
               log(frame[:,:,2])-(1-alpha)*log(frame[:,:,0])

    imshow(img,'gray')
    title(r'$\alpha$ = '+str(alpha))
    name = '/home/andyc/image/AIG/alpha2/'+str(idx).zfill(4)+'.jpg'
    savefig(name)
    clf()
