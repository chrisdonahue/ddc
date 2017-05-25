import os

def ezname(x):
    x = ''.join(x.strip().split())
    x_clean = []
    for char in x:
        if char.isalnum():
            x_clean.append(char)
        else:
            x_clean.append('_')
    return ''.join(x_clean)

def dot_dot_fp(fp, nup=1):
    for i in xrange(nup):
        fp = os.path.split(fp)[0]
    return fp
