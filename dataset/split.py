if __name__ == '__main__':
    import argparse
    import glob
    import os
    import random

    from ddc import dot_dot_fp

    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str, help='Input directory (any tree level)')
    parser.add_argument('--depth', type=int, help='Input directory tree depth')
    parser.add_argument('--splits', type=str, help='CSV list of split values for datasets (e.g. 8,1,1')
    parser.add_argument('--split_names', type=str, help='CSV list of split names for datasets (e.g. train,valid,test)')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help='If set, shuffle dataset before split')
    parser.add_argument('--shuffle_seed', type=int, help='If set, use this seed for shuffling')

    parser.set_defaults(
        depth=2,
        splits='8,1,1',
        split_names='train,valid,test',
        shuffle=True,
        shuffle_seed=1337)

    args = parser.parse_args()
    assert args.depth >= 0

    splits = [float(x) for x in args.splits.split(',')]
    split_names = [x.strip() for x in args.split_names.split(',')]
    assert len(splits) == len(split_names)

    glob_wildcard = (['*'] * args.depth) + ['*.sm']
    sm_fps = glob.glob(os.path.join(args.in_dir, *glob_wildcard))
    pack_dirs = set([dot_dot_fp(x) for x in sm_fps])

    for pack_dir in pack_dirs:
        pack_sm_fps = filter(lambda x: os.path.splitext(x)[1] == '.sm', sorted(os.listdir(pack_dir)))
        identifiers = [os.path.splitext(os.path.split(x)[1])[0] for x in pack_sm_fps]

        if args.shuffle:
            random.seed(args.shuffle_seed)
            random.shuffle(identifiers)

        if len(splits) == 0:
            splits = [1.0]
        else:
            splits = [x / sum(splits) for x in splits]

        split_ints = [int(len(identifiers) * split) for split in splits]
        split_ints[0] += len(identifiers) - sum(split_ints)
        assert sum(split_ints) == len(identifiers)

        split_fps = []
        for split_int in split_ints:
            split_fps.append(identifiers[:split_int])
            identifiers = identifiers[split_int:]

        for split, split_name in zip(split_fps, split_names):
            out_name = '{}.txt'.format(split_name)
            out_fp = os.path.join(pack_dir, out_name)
            with open(out_fp, 'w') as f:
                f.write('\n'.join(split))

    depth_dirs = set(pack_dirs)
    for _ in xrange(args.depth):
        depth_dirs = set([dot_dot_fp(x) for x in depth_dirs])
        for depth_dir in depth_dirs:
            for split_name in split_names:
                split_txt_fps = glob.glob(os.path.join(depth_dir, '*', '{}.txt'.format(split_name)))
                split_aggregated = []
                for split_txt_fp in split_txt_fps:
                    split_depth_name = os.path.split(dot_dot_fp(split_txt_fp))[1]
                    with open(split_txt_fp, 'r') as f:
                        split_aggregated += [os.path.join(split_depth_name, x) for x in f.read().splitlines()]
                with open(os.path.join(depth_dir, '{}.txt'.format(split_name)), 'w') as f:
                    f.write('\n'.join(split_aggregated))
