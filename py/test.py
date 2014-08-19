import lktrack2
import os,glob,cv2,pickle,cv

path = '/home/andyc/image/Feb11/'
imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))
imnames = imlist[0:100]
# create tracker object
lkt = lktrack2.LKTracker(imnames)
# detect in first frame, track in the remaining

lkt.detect_points()
lkt.draw()
for i in range(len(imnames)-1):
    print i
    lkt.track_points()
    lkt.draw()


'''
for im,ft in lkt.track():
    print 'tracking %d features' % len(ft)
# plot the tracks
figure()
imshow(im)
for p in ft:
    plot(p[0],p[1],'bo')
for t in lkt.tracks:
    plot([int(p[0][0]) for p in t],[int(p[0][1]) for p in t])
axis('off')
show()
'''
