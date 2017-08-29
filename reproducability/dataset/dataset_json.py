if __name__ == '__main__':
    import argparse
    import os
    import random
    from util import get_subdirs

    parser = argparse.ArgumentParser()
    parser.add_argument('json_dir', type=str, help='Input JSON dir')
    parser.add_argument('--dataset_dir', type=str, help='If specified, use different output dir otherwise JSON dir')
    parser.add_argument('--rel', dest='abs', action='store_false', help='If set, output relative paths')
    parser.add_argument('--splits', type=str, help='CSV list of split values for datasets (e.g. 0.8,0.1,0.1)')
    parser.add_argument('--splitnames', type=str, help='CSV list of split names for datasets (e.g. train,test,eval)')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help='If set, shuffle dataset before split')
    parser.add_argument('--shuffle_seed', type=int, help='If set, use this seed for shuffling')
    parser.add_argument('--choose', dest='choose', action='store_true', help='If set, choose from list of packs')

    parser.set_defaults(
        dataset_dir='',
        abs=True,
        splits='1',
        splitnames='',
        shuffle=False,
        shuffle_seed=0,
        choose=False)

    args = parser.parse_args()

    splits = [float(x) for x in args.splits.split(',')]
    split_names = [x.strip() for x in args.splitnames.split(',')]
    assert len(splits) == len(split_names)

    out_dir = args.dataset_dir if args.dataset_dir else args.json_dir
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    pack_names = get_subdirs(args.json_dir, args.choose)

    for pack_name in pack_names:
        pack_dir = os.path.join(args.json_dir, pack_name)
        sub_fps = sorted(os.listdir(pack_dir))

        if args.shuffle:
            random.seed(args.shuffle_seed)
            random.shuffle(sub_fps)

        if args.abs:
            sub_fps = [os.path.abspath(os.path.join(pack_dir, sub_fp)) for sub_fp in sub_fps]

        if len(splits) == 0:
            splits = [1.0]
        else:
            splits = [x / sum(splits) for x in splits]

        split_ints = [int(len(sub_fps) * split) for split in splits]
        split_ints[0] += len(sub_fps) - sum(split_ints)

        split_fps = []
        for split_int in split_ints:
            split_fps.append(sub_fps[:split_int])
            sub_fps = sub_fps[split_int:]

        for split, splitname in zip(split_fps, split_names):
            out_name = '{}{}.txt'.format(pack_name, '_' + splitname if splitname else '')
            out_fp = os.path.join(out_dir, out_name)
            with open(out_fp, 'w') as f:
                f.write('\n'.join(split))
