import os
import time

nframe   = 900
cmd_wget = "wget http://207.251.86.238/cctv11.jpg"
cmd_mv   = "mv cctv11.jpg /home/andyc/image/dot/baxandcanal/cctv11_{0:03}.jpg"

for ii in range(nframe):
    os.system(cmd_wget)
    os.system(cmd_mv.format(ii))
    time.sleep(1)
