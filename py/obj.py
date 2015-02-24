

class Objects():
    def __init__(self):
        self.ptsTrj= {}
        self.pts = []
        self.Trj = []
        self.xTrj = []
        self.yTrj = []
        self.frame = []
        self.vel = []
        self.pos = []
        self.status = 1   # 1: alive  2: dead

    def Color(self):
        try:
            self.R
        except:
            self.R = randint(0,255)
            self.G = randint(0,255)
            self.B = randint(0,255)
        return (self.R,self.G,self.B)


class Objectpts():
    def __init__(self):
        self.Trj= []
        self.vel = []
        self.pos = []
        self.pa  = 0
        self.color = []
        self.status = 1   # 1: alive  2: dead

