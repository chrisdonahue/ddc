"""Class managing a Stepmania "chart"

'Stepfiles' for Stepmania are organized into 'charts': lists of annotations for by an annotator for a song with some difficulty. Many charts can point to one song so we do not want to store song features for every chart. Instead, we have this helper class that will point to song features but store chart features. This way, our training and eval sets can be list of charts, instead of some mess with list of songs attached to multiple charts.
"""

from collections import Counter
import math
import random

import numpy as np

from beatcalc import BeatCalc
from util import make_onset_feature_context, np_pad

class Chart(object):
    def __init__(self, song_metadata, metadata, annotations):
        assert len(annotations) >= 2
        self.song_metadata = song_metadata
        self.metadata = metadata

        self.annotations = annotations
        self.label_counts = Counter()

        self.beat_calc = BeatCalc(song_metadata['offset'], song_metadata['bpms'], song_metadata['stops'])

        self.first_annotation_time = self.annotations[0][2]
        self.last_annotation_time = self.annotations[-1][2]

        assert self.last_annotation_time - self.first_annotation_time > 0.0

        self.time_annotated = self.last_annotation_time - self.first_annotation_time
        self.annotations_per_second = float(len(self.annotations)) / self.time_annotated

    def get_song_metadata(self):
        return self.song_metadata

    def get_coarse_difficulty(self):
        return self.metadata[0]
    
    def get_foot_difficulty(self):
        return self.metadata[1]

    def get_type(self):
        return self.metadata[2]

    def get_freetext(self):
        return self.metadata[3]

    def get_nannotations(self):
        return len(self.annotations)

    def get_time_annotated(self):
        return self.time_annotated

    def get_annotations_per_second(self):
        return self.annotations_per_second

class OnsetChart(Chart):
    def __init__(self, song_metadata, song_features, frame_rate, metadata, annotations):
        super(OnsetChart, self).__init__(song_metadata, metadata, annotations)

        self.song_features = song_features
        self.nframes = song_features.shape[0]

        self.dt = dt = 1.0 / frame_rate
        self.onsets = set(int(round(t / dt)) for _, _, t, _ in self.annotations)
        self.onsets = set(filter(lambda x: x >= 0, self.onsets))
        self.onsets = set(filter(lambda x: x < self.nframes, self.onsets))
        # TODO: filter greater than song_features len?

        self.first_onset = min(self.onsets)
        self.last_onset = max(self.onsets)
        self.nframes_annotated = (self.last_onset - self.first_onset) + 1

        assert self.first_onset >= 0
        assert self.last_onset < self.nframes
        assert self.nframes > 0
        assert self.nframes_annotated > 0
        assert self.nframes >= self.nframes_annotated
        assert len(self.onsets) > 0

        self.blanks = set(range(self.nframes)) - self.onsets
        self._blanks_memoized = {}

    def get_nframes(self):
        return self.nframes

    def get_nframes_annotated(self):
        return self.nframes_annotated

    def get_nonsets(self):
        return len(self.onsets)

    def get_onsets(self):
        return self.onsets

    def get_first_onset(self):
        return self.first_onset

    def get_last_onset(self):
        return self.last_onset

    def get_example(self,
                    frame_idx,
                    dtype,
                    time_context_radius=1,
                    diff_feet_to_id=None,
                    diff_coarse_to_id=None,
                    diff_dipstick=False,
                    freetext_to_id=None,
                    beat_phase=False,
                    beat_phase_cos=False):
        feats_audio = make_onset_feature_context(self.song_features, frame_idx, time_context_radius)
        feats_other = [np.zeros(0, dtype=dtype)]

        if diff_feet_to_id:
            diff_feet = diff_feet_to_id[str(self.get_foot_difficulty())]
            diff_feet_onehot = np.zeros(max(diff_feet_to_id.values()) + 1, dtype=dtype)
            if diff_dipstick:
                diff_feet_onehot[:diff_feet + 1] = 1.0
            else:
                diff_feet_onehot[diff_feet] = 1.0
            feats_other.append(diff_feet_onehot)
        if diff_coarse_to_id:
            diff_coarse = diff_coarse_to_id[str(self.get_coarse_difficulty())]
            diff_coarse_onehot = np.zeros(max(diff_coarse_to_id.values()) + 1, dtype=dtype)
            if diff_dipstick:
                diff_coarse_onehot[:diff_coarse + 1] = 1.0
            else:
                diff_coarse_onehot[diff_coarse] = 1.0
            feats_other.append(diff_coarse_onehot)
        if freetext_to_id:
            freetext = self.get_freetext()
            if freetext not in freetext_to_id:
                freetext = None
            freetext_id = freetext_to_id[freetext]
            freetext_onehot = np.zeros(max(freetext_to_id.values()) + 1, dtype=dtype)
            freetext_onehot[freetext_id] = 1.0
            feats_other.append(freetext_onehot)
        if beat_phase:
            beat = self.beat_calc.time_to_beat(frame_idx * self.dt)
            beat_phase = beat - int(beat)
            feats_other.append(np.array([beat_phase], dtype=dtype))
        if beat_phase_cos:
            beat = self.beat_calc.time_to_beat(frame_idx * self.dt)
            beat_phase = beat - int(beat)
            beat_phase_cos = np.cos(beat * 2.0 * np.pi)
            beat_phase_cos += 1.0
            beat_phase_cos *= 0.5
            feats_other.append(np.array([beat_phase_cos], dtype=dtype))

        y = dtype(frame_idx in self.onsets)
        return np.array(feats_audio, dtype=dtype), np.concatenate(feats_other), y

    def sample(self, n, exclude_onset_neighbors=0, nunroll=0):
        if self._blanks_memoized:
            valid = self._blanks_memoized
        else:
            valid = set(range(self.get_first_onset(), self.get_last_onset() + 1))
            if exclude_onset_neighbors > 0:
                onset_neighbors = [set([x + i for x in self.onsets]) | set([x - i for x in self.onsets]) for i in xrange(1, exclude_onset_neighbors + 1)]
                onset_neighbors = reduce(lambda x, y: x | y, onset_neighbors)
                valid -= onset_neighbors
            if nunroll > 0:
                valid -= set(range(self.get_first_onset(), self.get_first_onset() + nunroll))
            self._blanks_memoized = valid

        assert n <= len(valid)
        return random.sample(valid, n)

    def sample_onsets(self, n):
        assert n <= len(self.onsets)
        return random.sample(self.onsets, n)

    def sample_blanks(self, n, exclude_onset_neighbors=0, exclude_pre_onsets=True, exclude_post_onsets=True, include_onsets=False):
        exclusion_params = (exclude_onset_neighbors, exclude_pre_onsets, exclude_post_onsets, include_onsets)

        if exclusion_params in self._blanks_memoized:
            blanks = self._blanks_memoized[exclusion_params]
        else:
            blanks = self.blanks
            if exclude_onset_neighbors > 0:
                onset_neighbors = [set([x + i for x in self.onsets]) | set([x - i for x in self.onsets]) for i in xrange(1, exclude_onset_neighbors + 1)]
                onset_neighbors = reduce(lambda x, y: x | y, onset_neighbors)
                blanks -= onset_neighbors
            if exclude_pre_onsets:
                blanks -= set(range(self.get_first_onset()))
            if exclude_post_onsets:
                blanks -= set(range(self.get_last_onset(), self.nframes))
            if include_onsets:
                blanks |= self.onsets
            self._blanks_memoized[exclusion_params] = blanks

        assert n <= len(blanks)
        return random.sample(blanks, n)

    def get_subsequence(self,
            subseq_start,
            subseq_len,
            dtype,
            zack_hack_div_2=0,
            **feat_kwargs):
        seq_feats_audio = []
        seq_feats_other = []
        seq_y = []
        for i in xrange(subseq_start - zack_hack_div_2, subseq_start + subseq_len + zack_hack_div_2):
            feats_audio, feats_other, y = self.get_example(i, dtype=dtype, **feat_kwargs)
            seq_feats_audio.append(feats_audio)
            seq_feats_other.append(feats_other)
            seq_y.append(y)
        zhmin = zack_hack_div_2
        zhmax = zack_hack_div_2 + subseq_len

        return np.array(seq_feats_audio, dtype=dtype), np.array(seq_feats_other, dtype=dtype)[zhmin:zhmax], np.array(seq_y, dtype=dtype)[zhmin:zhmax]

class SymbolicChart(Chart):
    def __init__(self, song_metadata, song_features, frame_rate, metadata, annotations, pre=1, post=False):
        super(SymbolicChart, self).__init__(song_metadata, metadata, annotations)

        self.song_features = song_features
        dt = 1.0 / frame_rate

        self.sequence = []

        self.seq_beat_phase = []
        self.seq_beat_diff = []
        self.seq_beat_abs = []
        self.seq_time_diff = []
        self.seq_time_abs = []
        self.seq_prog_diff = []
        self.seq_prog_abs = []
        self.seq_meas_phase = []
        self.seq_meas_wraps = []
        self.seq_beat_wraps = []
        self.seq_frame_idxs = []

        prepend = ['<-{}>'.format(i + 1) for i in range(pre)[::-1]]
        prepend_annotations = [self.annotations[0][:3] + [p] for p in prepend]
        annotations = prepend_annotations + self.annotations
        prog_last = 0.0
        for i in xrange(len(annotations) - 1):
            pulse_last, beat_last, time_last, label_last = annotations[i]
            pulse, beat, time, label = annotations[i + 1]
            prog = (time - self.first_annotation_time) / self.time_annotated

            assert beat >= beat_last
            assert time >= time_last

            self.sequence.append(label_last)

            self.seq_beat_phase.append(beat - math.floor(beat))
            self.seq_beat_diff.append(beat - beat_last)
            self.seq_beat_abs.append(beat)
            self.seq_time_diff.append(time - time_last)
            self.seq_time_abs.append(time)
            self.seq_prog_abs.append(prog)
            self.seq_prog_diff.append(prog - prog_last)
            self.seq_meas_phase.append(float(pulse[2]) / pulse[1])
            self.seq_meas_wraps.append(pulse[0] - pulse_last[0])
            self.seq_beat_wraps.append(int(beat) - int(beat_last))
            self.seq_frame_idxs.append(int(round(time / dt)))

            prog_last = prog

        self.sequence.append(label)

    def get_subsequence(self,
            subseq_start,
            subseq_len,
            meas_phase_cos=False,
            meas_phase_sin=False,
            meas_phase=False,
            beat_phase_cos=False,
            beat_phase_sin=False,
            beat_phase=False,
            beat_diff=False,
            beat_diff_next=False,
            beat_abs=False,
            time_diff=False,
            time_diff_next=False,
            time_abs=False,
            prog_diff=False,
            prog_abs=False,
            beat_phase_nquant=0,
            beat_phase_max_nwraps=0,
            meas_phase_nquant=0,
            meas_phase_max_nwraps=0,
            diff_feet_to_id=None,
            diff_coarse_to_id=None,
            diff_dipstick=False,
            freetext_to_id=None,
            audio_time_context_radius=0,
            audio_deviation_max=0,
            bucket_beat_diff_n=None,
            bucket_beat_diff_max=None,
            bucket_time_diff_n=None,
            bucket_time_diff_max=None,
            dtype=np.float32):
        syms = self.sequence[subseq_start:subseq_start + subseq_len + 1]
        nvalid = len(syms) - 1

        feats = np.zeros((nvalid, 0), dtype=dtype)
        if meas_phase_cos:
            feats = np.append(feats, np.cos(2. * np.pi * np.array(self.seq_meas_phase[subseq_start:subseq_start + nvalid], dtype=dtype))[:, np.newaxis], axis=1)
        if meas_phase_sin:
            feats = np.append(feats, np.sin(2. * np.pi * np.array(self.seq_meas_phase[subseq_start:subseq_start + nvalid], dtype=dtype))[:, np.newaxis], axis=1)
        if meas_phase:
            feats = np.append(feats, np.array(self.seq_meas_phase[subseq_start:subseq_start + nvalid], dtype=dtype)[:, np.newaxis], axis=1)
        if beat_phase:
            feats = np.append(feats, np.array(self.seq_beat_phase[subseq_start:subseq_start + nvalid], dtype=dtype)[:, np.newaxis], axis=1)
        if beat_phase_cos:
            feats = np.append(feats, np.cos(2. * np.pi * np.array(self.seq_beat_phase[subseq_start:subseq_start + nvalid], dtype=dtype))[:, np.newaxis], axis=1)
        if beat_phase_sin:
            feats = np.append(feats, np.sin(2. * np.pi * np.array(self.seq_beat_phase[subseq_start:subseq_start + nvalid], dtype=dtype))[:, np.newaxis], axis=1)
        if beat_diff:
            feats = np.append(feats, np.array(self.seq_beat_diff[subseq_start:subseq_start + nvalid], dtype=dtype)[:, np.newaxis], axis=1)
        if beat_diff_next:
            feat = np.array(self.seq_beat_diff[subseq_start + 1:subseq_start + nvalid + 1], dtype=dtype)
            feat = np_pad(feat, nvalid, axis=0)
            feats = np.append(feats, feat[:, np.newaxis], axis=1)
        if beat_abs:
            feats = np.append(feats, np.array(self.seq_beat_abs[subseq_start:subseq_start + nvalid], dtype=dtype)[:, np.newaxis], axis=1)
        if time_diff:
            feats = np.append(feats, np.array(self.seq_time_diff[subseq_start:subseq_start + nvalid], dtype=dtype)[:, np.newaxis], axis=1)
        if time_diff_next:
            feat = np.array(self.seq_time_diff[subseq_start + 1:subseq_start + nvalid + 1], dtype=dtype)
            feat = np_pad(feat, nvalid, axis=0)
            feats = np.append(feats, feat[:, np.newaxis], axis=1)
        if time_abs:
            feats = np.append(feats, np.array(self.seq_time_abs[subseq_start:subseq_start + nvalid], dtype=dtype)[:, np.newaxis], axis=1)
        if prog_diff:
            feats = np.append(feats, np.array(self.seq_prog_diff[subseq_start:subseq_start + nvalid], dtype=dtype)[:, np.newaxis], axis=1)
        if prog_abs:
            feats = np.append(feats, np.array(self.seq_prog_abs[subseq_start:subseq_start + nvalid], dtype=dtype)[:, np.newaxis], axis=1)
        if beat_phase_nquant > 0:
            beat_phase = np.array(self.seq_beat_phase[subseq_start:subseq_start + nvalid])
            beat_phase_quant = np.rint(beat_phase_nquant * beat_phase).astype(np.int)
            beat_phase_quant = np.mod(beat_phase_quant, beat_phase_nquant)
            beat_phase_onehot = np.zeros((nvalid, beat_phase_nquant), dtype=dtype)
            beat_phase_onehot[np.arange(nvalid), beat_phase_quant] = 1.0
            feats = np.append(feats, beat_phase_onehot, axis=1)
        if beat_phase_max_nwraps > 0:
            beat_wraps = np.array(self.seq_beat_wraps[subseq_start:subseq_start + nvalid], dtype=np.int)
            beat_wraps = np.minimum(beat_wraps, beat_phase_max_nwraps)
            beat_wraps_onehot = np.zeros((nvalid, beat_phase_max_nwraps + 1), dtype=dtype)
            beat_wraps_onehot[np.arange(nvalid), beat_wraps] = 1.0
            feats = np.append(feats, beat_wraps_onehot, axis=1)
        if meas_phase_nquant > 0:
            meas_phase = np.array(self.seq_meas_phase[subseq_start:subseq_start + nvalid])
            meas_phase_quant = np.rint(meas_phase_nquant * meas_phase).astype(np.int)
            meas_phase_quant = np.mod(meas_phase_quant, meas_phase_nquant)
            meas_phase_onehot = np.zeros((nvalid, meas_phase_nquant), dtype=dtype)
            meas_phase_onehot[np.arange(nvalid), meas_phase_quant] = 1.0
            feats = np.append(feats, meas_phase_onehot, axis=1)
        if meas_phase_max_nwraps > 0:
            meas_wraps = np.array(self.seq_meas_wraps[subseq_start:subseq_start + nvalid], dtype=np.int)
            meas_wraps = np.minimum(meas_wraps, meas_phase_max_nwraps)
            meas_wraps_onehot = np.zeros((nvalid, meas_phase_max_nwraps + 1), dtype=dtype)
            meas_wraps_onehot[np.arange(nvalid), meas_wraps] = 1.0
            feats = np.append(feats, meas_wraps_onehot, axis=1)
        if diff_feet_to_id:
            diff_feet = diff_feet_to_id[str(self.get_foot_difficulty())]
            diff_feet_onehot = np.zeros((nvalid, max(diff_feet_to_id.values()) + 1), dtype=dtype)
            if diff_dipstick:
                diff_feet_onehot[np.arange(nvalid), :diff_feet + 1] = 1.0
            else:
                diff_feet_onehot[np.arange(nvalid), diff_feet] = 1.0
            feats = np.append(feats, diff_feet_onehot, axis=1)
        if diff_coarse_to_id:
            diff_coarse = diff_coarse_to_id[self.get_coarse_difficulty()]
            diff_coarse_onehot = np.zeros((nvalid, max(diff_coarse_to_id.values()) + 1), dtype=dtype)
            if diff_dipstick:
                diff_coarse_onehot[np.arange(nvalid), :diff_coarse + 1] = 1.0
            else:
                diff_coarse_onehot[np.arange(nvalid), diff_coarse] = 1.0
            feats = np.append(feats, diff_coarse_onehot, axis=1)
        if freetext_to_id:
            freetext = self.get_freetext()
            if freetext not in freetext_to_id:
                freetext = None
            freetext_id = freetext_to_id[freetext]
            freetext_onehot = np.zeros((nvalid, max(freetext_to_id.values()) + 1), dtype=dtype)
            freetext_onehot[np.arange(nvalid), freetext_id] = 1.0
            feats = np.append(feats, freetext_onehot, axis=1)
        if bucket_beat_diff_n and bucket_beat_diff_max:
            beat_diff = np.array(self.seq_beat_diff[subseq_start:subseq_start + nvalid])

            bucket_inc = bucket_beat_diff_max / bucket_beat_diff_n
            bucket_beat_diff = beat_diff / bucket_inc
            bucket_beat_diff = bucket_beat_diff.astype(np.int32)
            bucket_beat_diff[np.where(bucket_beat_diff > bucket_beat_diff_n)] = bucket_beat_diff_n

            buckets = np.zeros((nvalid, bucket_beat_diff_n + 1), dtype=dtype)
            buckets[np.arange(nvalid), bucket_beat_diff] = 1.0

            feats = np.append(feats, buckets, axis=1)
        if bucket_time_diff_n and bucket_time_diff_max:
            time_diff = np.array(self.seq_time_diff[subseq_start:subseq_start + nvalid])

            bucket_inc = bucket_time_diff_max / bucket_time_diff_n
            bucket_time_diff = time_diff / bucket_inc
            bucket_time_diff = bucket_time_diff.astype(np.int32)
            bucket_time_diff[np.where(bucket_time_diff > bucket_time_diff_n)] = bucket_time_diff_n

            buckets = np.zeros((nvalid, bucket_time_diff_n + 1), dtype=dtype)
            buckets[np.arange(nvalid), bucket_time_diff] = 1.0

            feats = np.append(feats, buckets, axis=1)

        feats_audio = np.zeros((nvalid, 0, 0, 0), dtype=dtype)
        if audio_time_context_radius >= 0 and self.song_features is not None:
            audio_time = audio_time_context_radius * 2 + 1
            feats_audio = np.zeros((nvalid, audio_time) + self.song_features.shape[1:], dtype=dtype)
            frame_idxs = self.seq_frame_idxs[subseq_start:subseq_start + nvalid]
            for i, frame_idx in enumerate(frame_idxs):
                if audio_deviation_max > 0:
                    frame_idx += random.randint(-audio_deviation_max, audio_deviation_max)
                feats_audio[i] = make_onset_feature_context(self.song_features, frame_idx, audio_time_context_radius)

        return syms, feats, feats_audio

    def get_random_subsequence(self, subseq_len, **feat_kwargs):
        assert subseq_len <= self.get_nannotations()

        max_idx = self.get_nannotations() - subseq_len
        i = random.randint(0, max_idx)
        #TODO: first sequence incredibly unlikely to appear, balance this

        return self.get_subsequence(i, subseq_len, **feat_kwargs)

    def get_sequence(self, **feat_kwargs):
        return self.get_subsequence(0, self.get_nannotations(), **feat_kwargs)
