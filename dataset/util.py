import os

def ez_name(x):
    x = ''.join(x.strip().split())
    x_clean = []
    for char in x:
        if char.isalnum():
            x_clean.append(char)
        else:
            x_clean.append('_')
    return ''.join(x_clean)

def get_subdirs(root, choose=False):
    subdir_names = sorted(filter(lambda x: os.path.isdir(os.path.join(root, x)), os.listdir(root)))
    if choose:
        for i, subdir_name in enumerate(subdir_names):
            print '{}: {}'.format(i, subdir_name)
        subdir_idxs = [int(x) for x in raw_input('Which subdir(s)? ').split(',')]
        subdir_names = [subdir_names[i] for i in subdir_idxs]
    return subdir_names