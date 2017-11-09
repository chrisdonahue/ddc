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
