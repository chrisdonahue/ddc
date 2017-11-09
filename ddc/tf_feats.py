import cPickle as pickle

import tensorflow as tf
import numpy as np

from util import resource_fp

# input shape: [None]
# output shape: [None, nfft]
def short_time(x, window=1024, stride=512):
  assertions = [
    tf.assert_rank(x, 1),
  ]
  with tf.control_dependencies(assertions):
    x = tf.identity(x)

  x_len = tf.cast(tf.shape(x)[0], tf.float32)
  ntotal = tf.cast(tf.ceil(x_len / stride), tf.int32)

  pad_left = tf.constant(window // 2)
  pad_right = tf.constant(window // 2)

  x_padded = tf.pad(x, [[pad_left, pad_right]])

  x_slices = []
  for i in xrange(window):
    x_slices.append(x_padded[i::stride][:ntotal])
  x_slices = tf.stack(x_slices, axis=1)

  return x_slices

# input shape: [None, nfft]
# output shape: [None, nfft]
def hann_window(x, nfft=1024, dtype=tf.float32):
  window = tf.constant(np.hanning(nfft), dtype)
  x_windowed = tf.multiply(x, window)

  return x_windowed

def bh62_window(x, nfft=1024, dtype=tf.float32):
  assert nfft in [1024, 2048, 4096]

  window_fp = resource_fp('window_bh62_nfft{}.pkl'.format(nfft))
  with open(window_fp, 'rb') as f:
    window = pickle.load(f)

  w = tf.constant(window, dtype=dtype)
  x_windowed = tf.multiply(x, w)

  return x_windowed

# input shape: [None, (nfft // 2) + 1]
# output shape: [None, 80]
def mel80(x, nfft=1024, dtype=tf.float32):
  assert nfft in [1024, 2048, 4096]

  xform_fp = resource_fp('triangles_nfft{}_mel80.pkl'.format(nfft))
  with open(xform_fp, 'rb') as f:
    xform = pickle.load(f)

  w = tf.constant(xform, dtype=dtype)
  x_mel80 = tf.matmul(x, w)

  return x_mel80

# input shape is one of:
#   * [None] (raw waveform)
#   * [None, nfft] (pre-strided waveform)
# strided_slice is slow with large nfft; consider striding outside tf
def extract_feats(x, feats_type='mel80', nfft=1024, nhop=512, dtype=tf.float32):
  assertions = [
    tf.assert_rank_at_least(x, 1),
    tf.assert_type(x, dtype)
  ]
  with tf.control_dependencies(assertions):
    x = tf.identity(x)

  # Can't use dynamic shape here because tf.cond runs both branches
  if len(x.get_shape()) == 1:
    x_slices = short_time(x, nfft, nhop)
  else:
    x_slices = x
    x_slices.set_shape([None, nfft])

  if feats_type == 'raw':
    feats = x_slices
  elif feats_type == 'spectra':
    x_windowed = hann_window(x_slices, nfft, dtype=dtype)
    x_fft = tf.spectral.rfft(x_windowed)
    feats = tf.abs(x_fft)
  elif feats_type == 'mel80':
    x_windowed = hann_window(x_slices, nfft, dtype=dtype)
    x_fft = tf.spectral.rfft(x_windowed, [nfft])
    x_mag = tf.abs(x_fft)
    feats = mel80(x_mag, nfft, dtype=dtype)
  elif feats_type == 'lmel80':
    x_windowed = hann_window(x_slices, nfft, dtype=dtype)
    x_fft = tf.spectral.rfft(x_windowed, [nfft])
    x_mag = tf.abs(x_fft)
    x_mel80 = mel80(x_mag, nfft, dtype=dtype)
    feats = tf.log(x_mel80 + 1e-16)
    return feats
  else:
    raise NotImplementedError()

  return feats
