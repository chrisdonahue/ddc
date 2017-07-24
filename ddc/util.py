import os
import pkg_resources

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

def resource_fp(name):
    resource_package = __name__
    resource_path = '/'.join(['resources', name])
    return pkg_resources.resource_filename(resource_package, resource_path)
