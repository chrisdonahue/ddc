import os

import tensorflow as tf


# input shape: [] (placeholder tf.string)
# output shape: [?, nch] (samples)
def load_audio_file(audio_fp, fs=44100, nch=2):
  audio_bin = tf.read_file(audio_fp)
  file_format = os.path.splitext(audio_fp)[1][1:]
  samples = tf.contrib.ffmpeg(
      song_binary,
      file_format=file_format,
      samples_per_second=fs,
      channel_count=nch)
  return samples


# input shape:  [None] (raw waveform)
# output shape: [None, nfeats]
def extract_feats(x, feats_type='lmel80', nfft=1024, nhop=512, dtype=tf.float32):
  assertions = [
    tf.assert_rank(x, 1),
    tf.assert_type(x, dtype)
  ]
  with tf.control_dependencies(assertions):
    x = tf.identity(x)

  # Can't use dynamic shape here because tf.cond runs both branches
  if feats_type == 'raw':
    feats = tf.contrib.signal.frame(x, nfft, nhop, pad_end=True)
  else:
    X = tf.contrib.signal.stft(x, nfft, nhop, pad_end=True)
    X_mag = tf.abs(stft)

    if feats_type == 'spectra':
      feats = X_mag
    elif feats_type == 'log_spectra':
      feats = tf.log(X_mag + 1e-6)
    elif feats_type == 'lmel80':
      mel_filterbank = tf.contrib.signal.linear_to_mel_weight_matrix(
          num_mel_bins=80,
          num_spectrogram_bins=(nfft // 2) + 1,
          sample_rate=44100.,
          lower_edge_hertz=27.5,
          upper_edge_hertz=16000.,
          dtype=dtype)
      X_mel = tf.matmul(X_mag, mel_filterbank)
      X_lmel = tf.log(X_mel + 1e-16)
      feats = X_lmel
    else:
      raise NotImplementedError()

  return feats
