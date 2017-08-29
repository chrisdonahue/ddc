import essentia
from essentia.standard import MonoLoader, FrameGenerator, Windowing, Spectrum, MelBands
import numpy as np
from tqdm import tqdm

def create_analyzers(fs=44100.0,
                     nhop=512,
                     nffts=[1024, 2048, 4096],
                     mel_nband=80,
                     mel_freqlo=27.5,
                     mel_freqhi=16000.0):
    analyzers = []
    for nfft in nffts:
        window = Windowing(size=nfft, type='blackmanharris62')
        spectrum = Spectrum(size=nfft)
        mel = MelBands(inputSize=(nfft // 2) + 1,
                       numberBands=mel_nband,
                       lowFrequencyBound=mel_freqlo,
                       highFrequencyBound=mel_freqhi,
                       sampleRate=fs)
        analyzers.append((window, spectrum, mel))
    return analyzers

def extract_mel_feats(audio_fp, analyzers, fs=44100.0, nhop=512, nffts=[1024, 2048, 4096], log_scale=True):
    # Extract features
    loader = MonoLoader(filename=audio_fp, sampleRate=fs)
    samples = loader()
    feat_channels = []
    for nfft, (window, spectrum, mel) in zip(nffts, analyzers):
        feats = []
        for frame in FrameGenerator(samples, nfft, nhop):
            frame_feats = mel(spectrum(window(frame)))
            feats.append(frame_feats)
        feat_channels.append(feats)

    # Transpose to move channels to axis 2 instead of axis 0
    feat_channels = np.transpose(np.stack(feat_channels), (1, 2, 0))

    # Apply numerically-stable log-scaling
    # Value 1e-16 comes from inspecting histogram of raw values and picking some epsilon >2 std dev left of mean
    if log_scale:
        feat_channels = np.log(feat_channels + 1e-16)

    return feat_channels

if __name__ == '__main__':
    import argparse
    import cPickle as pickle
    import json
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_fps', type=str, nargs='+', help='')
    parser.add_argument('--out_dir', type=str, required=True, help='')
    parser.add_argument('--nhop', type=int, help='')
    parser.add_argument('--nffts', type=str, help='')
    parser.add_argument('--mel_nband', type=int, help='')
    parser.add_argument('--log_scale', dest='log_scale', action='store_true')
    parser.add_argument('--nolog_scale', dest='log_scale', action='store_false')

    parser.set_defaults(
        nhop=512,
        nffts='1024,2048,4096',
        mel_nband=80,
        log_scale=True,
        choose=False)

    args = parser.parse_args()

    nffts = [int(x) for x in args.nffts.split(',')]

    # Create anlyzers
    analyzers = create_analyzers(fs=44100.0, nhop=args.nhop, nffts=nffts, mel_nband=args.mel_nband)

    # Create outdir
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    # Iterate through packs extracting features
    for dataset_fp in args.dataset_fps:
        with open(dataset_fp, 'r') as f:
            json_fps = f.read().splitlines()

        for json_fp in json_fps:
            song_name = os.path.splitext(os.path.split(json_fp)[1])[0]
            print 'Extracting feats from {}'.format(song_name)

            with open(json_fp, 'r') as json_f:
                meta = json.loads(json_f.read())
            song_metadata = {k: meta[k] for k in ['title', 'artist']}

            music_fp = meta['music_fp']
            if not os.path.exists(music_fp):
                raise ValueError('No music for {}'.format(json_fp))

            song_feats = extract_mel_feats(music_fp, analyzers, fs=44100.0, nhop=args.nhop, nffts=nffts, log_scale=args.log_scale)

            feats_fp = os.path.join(args.out_dir, '{}.pkl'.format(song_name))
            with open(feats_fp, 'wb') as f:
                pickle.dump(song_feats, f)
