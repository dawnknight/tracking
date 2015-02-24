fig, ax = plt.subplots(figsize=(5,5))
plt.imshow(frame[:,:,::-1])

for i in range(len(obj)):
    Color = (randint(0,255)/255.,randint(0,255)/255.,randint(0,255)/255.)
    for ii in range(len(obj[i].ptsTrj)):
        #cv2.circle(vis,(int(obj[i].ptsTrj[ii][-1][0]),int(obj[i].ptsTrj[ii][-1][1])),2,Color)
        x = int(obj[i].ptsTrj[ii][-1][0])
        y = int(obj[i].ptsTrj[ii][-1][1])
        ax.scatter(x,y,color=Color) 

def ckpts(num):
    iidx = []
    for i in range(len(obj)):
        if num in obj[i].pts:
            iidx.append(i)
    return iidx
