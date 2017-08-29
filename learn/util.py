import cPickle as pickle
import os
import math
import unicodedata

import numpy as np
from scipy.signal import argrelextrema

def load_id_dict(id_dict_fp):
    with open(id_dict_fp, 'r') as f:
        id_dict = {k:int(i) for k,i in [x.split(',') for x in f.read().splitlines()]}
        if '' in id_dict:
            id_dict[None] = id_dict['']
            del id_dict['']
        return id_dict

def ez_name(x):
    x = x.encode('ascii', 'replace')
    x = ''.join(x.strip().split())
    x_clean = []
    for char in x:
        if char.isalnum():
            x_clean.append(char)
        else:
            x_clean.append('_')
    return ''.join(x_clean)

def stride_csv_arg_list(arg, stride, cast=int):
    assert stride > 0
    l = filter(lambda x: bool(x), [x.strip() for x in arg.split(',')])
    l = [cast(x) for x in l]
    assert len(l) % stride == 0
    result = []
    for i in xrange(0, len(l), stride):
        if stride == 1:
            subl = l[i]
        else:
            subl = tuple(l[i:i + stride])
        result.append(subl)
    return result

def np_pad(x, pad_to, value=0, axis=-1):
    assert x.shape[axis] <= pad_to
    pad = [(0, 0) for i in xrange(x.ndim)]
    pad[axis] = (0, pad_to - x.shape[axis])
    return np.pad(x, pad_width=pad, mode='constant', constant_values=value)

def open_dataset_fps(*args):
    datasets = []
    for data_fp in args:
        if not data_fp:
            datasets.append([])
            continue

        with open(data_fp, 'r') as f:
            song_fps = f.read().split()
        dataset = []
        for song_fp in song_fps:
            with open(song_fp, 'rb') as f:
                dataset.append(pickle.load(f))
        datasets.append(dataset)
    return datasets[0] if len(datasets) == 1 else datasets

def select_channels(dataset, channels):
    for i, (song_metadata, song_features, song_charts) in enumerate(dataset):
        song_features_selected = song_features[:, :, channels]
        dataset[i] = (song_metadata, song_features_selected, song_charts)
        for chart in song_charts:
            chart.song_features = song_features_selected

def apply_z_norm(dataset, mean_per_band, std_per_band):
    for i, (song_metadata, song_features, song_charts) in enumerate(dataset):
        song_features_z = song_features - mean_per_band
        song_features_z /= std_per_band
        dataset[i] = (song_metadata, song_features_z, song_charts)
        for chart in song_charts:
            chart.song_features = song_features_z

def calc_mean_std_per_band(dataset):
    mean_per_band_per_song = [np.mean(song_features, axis=0) for _, song_features, _ in dataset]
    std_per_band_per_song = [np.std(song_features, axis=0) for _, song_features, _ in dataset]
    mean_per_band = np.mean(np.array(mean_per_band_per_song), axis=0)
    std_per_band = np.mean(np.array(std_per_band_per_song), axis=0)

    return mean_per_band, std_per_band

def flatten_dataset_to_charts(dataset):
    return [item for sublist in [x[2] for x in dataset] for item in sublist]

def filter_chart_type(charts, chart_type):
    return filter(lambda x: x.get_type() == chart_type, charts)

def make_onset_feature_context(song_features, frame_idx, radius):
    nframes = song_features.shape[0]
    
    assert nframes > 0

    frame_idxs = xrange(frame_idx - radius, frame_idx + radius + 1)
    context = np.zeros((len(frame_idxs),) + song_features.shape[1:], dtype=song_features.dtype)
    for i, frame_idx in enumerate(frame_idxs):
        if frame_idx >= 0 and frame_idx < nframes:
            context[i] = song_features[frame_idx]
        else:
            context[i] = np.zeros_like(song_features[0])

    return context

def find_pred_onsets(scores, window):
    if window.shape[0] > 0:
        onset_function = np.convolve(scores, window, mode='same')
    else:
        onset_function = scores
    # see page 592 of "Universal onset detection with bidirectional long short-term memory neural networks"
    maxima = argrelextrema(onset_function, np.greater_equal, order=1)[0]
    return set(list(maxima))

def align_onsets_to_sklearn(true_onsets, pred_onsets, scores, tolerance=0):
    # Build one-to-many dicts of candidate matches
    true_to_pred = {}
    pred_to_true = {}
    for true_idx in true_onsets:
        true_to_pred[true_idx] = []
        for pred_idx in xrange(true_idx - tolerance, true_idx + tolerance + 1):
            if pred_idx in pred_onsets:
                true_to_pred[true_idx].append(pred_idx)
                if pred_idx not in pred_to_true:
                    pred_to_true[pred_idx] = []
                pred_to_true[pred_idx].append(true_idx)

    # Create alignments
    true_to_pred_confidence = {}
    pred_idxs_used = set()
    for pred_idx, true_idxs in pred_to_true.items():
        true_idx_use = true_idxs[0]
        if len(true_idxs) > 1:
            for true_idx in true_idxs:
                if len(true_to_pred[true_idx]) == 1:
                    true_idx_use = true_idx
                    break
        true_to_pred_confidence[true_idx_use] = scores[pred_idx]
        assert pred_idx not in pred_idxs_used
        pred_idxs_used.add(pred_idx)

    # Create confidence list
    y_true = np.zeros_like(scores)
    y_true[list(true_onsets)] = 1.0
    y_scores = np.zeros_like(scores)
    for true_idx, confidence in true_to_pred_confidence.items():
        y_scores[true_idx] = confidence

    # Add remaining false positives
    for fp_idx in pred_onsets - pred_idxs_used:
        y_scores[fp_idx] = scores[fp_idx]

    # Truncate predictions to annotated range
    first_onset = min(true_onsets)
    last_onset = max(true_onsets)

    return y_true[first_onset:last_onset + 1], y_scores[first_onset:last_onset + 1]
