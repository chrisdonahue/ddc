import math
import numpy as np

from scipy.signal import fftconvolve
from scipy.io.wavfile import write as wavwrite

def _wav_write(wav_fp, fs, wav_f, normalize=False):
    if normalize:
        wav_f_max = wav_f.max()
        if wav_f_max != 0.0:
            wav_f /= wav_f.max()
    wav_f = np.clip(wav_f, -1.0, 1.0)
    wav = (wav_f * 32767.0).astype(np.int16)
    wavwrite(wav_fp, fs, wav)

# (length, val) pairs
def _linterp(val_start, pts, env_len):
    pt_lens = [pt[0] for pt in pts]
    pt_vals = [pt[1] for pt in pts]
    pt_lens = [int(env_len * (pt_len / sum(pt_lens))) for pt_len in pt_lens]
    pt_lens[-1] -= sum(pt_lens) - env_len
    env = []
    val_curr = val_start
    for pt_len, pt_val in zip(pt_lens, pt_vals):
        env.append(np.linspace(val_curr, pt_val, pt_len, endpoint=False))
        val_curr = pt_val
    return np.concatenate(env)

def write_preview_wav(wav_fp, note_beats_and_abs_times, wav_fs=11025.0):
    wav_len = int(wav_fs * (note_beats_and_abs_times[-1][1] + 0.05))
    dt = 1.0 / wav_fs

    note_type_to_idx = {}
    idx = 0
    for _, beat, time, note_type in note_beats_and_abs_times:
        if note_type == '0' * len(note_type):
            continue
        if note_type not in note_type_to_idx:
            note_type_to_idx[note_type] = idx
            idx += 1
    num_note_types = len(note_type_to_idx)

    pulse_f = np.zeros((num_note_types, wav_len))

    for _, beat, time, note_type in note_beats_and_abs_times:
        sample = int(time * wav_fs)
        if sample > 0 and sample < wav_len and note_type in note_type_to_idx:
            pulse_f[note_type_to_idx[note_type]][sample] = 1.0

    scale = [440.0, 587.33, 659.25, 783.99]
    freqs = [scale[i % 4] * math.pow(2.0, (i // 4) - 1) for i in xrange(num_note_types)]
    metro_f = np.zeros(wav_len)
    for idx in xrange(num_note_types):
        click_len = 0.05
        click_t = np.arange(0.0, click_len, dt)
        click_atk = 0.02
        click_sus = 0.5
        click_rel = 0.2
        click_env = _linterp(0.0, [(click_atk, 1.0), (click_sus, 1.0), (click_rel, 0.0)], len(click_t))
        click_f = click_env * np.sin(2.0 * np.pi * freqs[idx] * click_t)

        metro_f += fftconvolve(pulse_f[idx], click_f, mode='full')[:wav_len]
        #metro_f += pulse_f[idx][:wav_len]

    _wav_write(wav_fp, wav_fs, metro_f, normalize=True)

if __name__ == '__main__':
    import json
    import sys

    json_fp, wav_fp = sys.argv[1:3]
    
    with open(json_fp, 'r') as f:
        meta = json.loads(f.read())

    for i, chart in enumerate(meta['charts']):
        print '{}: {} {} {}'.format(i, chart['type'], chart['difficulty_fine'], chart['desc_or_author'])
    chart_idx = int(raw_input('Which chart? '))
    write_preview_wav(wav_fp, meta['charts'][chart_idx]['notes'])
