import numpy as np

if __name__ == '__main__':
    import argparse
    import cPickle as pickle
    import glob
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument('in_dir', type=str, help='')
    parser.add_argument('out_dir', type=str, help='')
    parser.add_argument('--in_channel', type=int, help='')
    parser.add_argument('--n_dt', type=int, help='')

    parser.set_defaults(
        in_channel=0,
        n_dt=1)

    args = parser.parse_args()

    feat_fps = glob.glob(os.path.join(args.in_dir, '*.pkl'))

    for feat_fp in feat_fps:
        song_name = os.path.splitext(os.path.split(feat_fp)[1])[0]
        with open(feat_fp, 'rb') as f:
            song_feats = pickle.load(f)

        song_ch = song_feats[:, :, args.in_channel]
        song_feats_dt = [song_ch]
        for i in xrange(1, args.n_dt + 1):
            song_ch_dt = np.diff(song_ch, n=i, axis=0)
            song_ch_zp = np.zeros_like(song_ch[:i])
            song_ch_dt = np.concatenate([song_ch_zp, song_ch_dt])
            song_feats_dt.append(song_ch_dt)
        song_feats_dt = np.stack(song_feats_dt, axis=2)

        print '{}: {}->{}'.format(song_name, song_feats.shape, song_feats_dt.shape)

        feat_out_fp = os.path.join(args.out_dir, '{}.pkl'.format(song_name))
        with open(feat_out_fp, 'wb') as f:
            pickle.dump(song_feats_dt, f)
