
def errcheck():
    err = []
    for i in range(len(pts)):
        if i not in obj[pts[i].pa].pts:
            err.append(i)
    return err
