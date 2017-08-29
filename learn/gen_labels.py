import itertools
import sys

if __name__ == '__main__':
    narrows, chars = sys.argv[1:3]
    perms = []
    for perm in itertools.product(chars, repeat=int(narrows)):
        perms.append(''.join([str(x) for x in perm]))
    with open('labels_{}_{}.txt'.format(narrows, chars), 'w') as f:
        f.write('\n'.join(perms))
