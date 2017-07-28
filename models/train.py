import tensorflow as tf

from ddc.feats import extract_feats

def load_examples(
        charts,
        batch_size,
        rate=100.0,
        num_epochs=None,
        feat_type='lmel80'):
    # Aggregate constants from charts
    audio_fps = []
    feats = []
    step_frames = []
    for chart in charts:
        audio_fps.append(chart.get_audio_fp())
        feats.append(chart.get_feats())
        chart_step_frames = chart.get_step_frames(rate)
        chart_step_frames = ','.join([str(f) for f in chart_step_frames])
        step_frames.append(chart_step_frames)

    # Convert to tensors
    audio_fps = tf.convert_to_tensor(audio_fps, tf.string)
    feats = tf.convert_to_tensor(feats, tf.int32)
    step_frames = tf.convert_to_tensor(step_frames, tf.string)

    # Limit number of epochs
    # https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/training/input.py
    audio_fps = tf.train.limit_epochs(audio_fps, num_epochs)
    feats = tf.train.limit_epochs(feats, num_epochs)
    step_frames = tf.train.limit_epochs(step_frames, num_epochs)

    # Create queue just to shuffle (can't seem to shuffle tuples along an axis)
    chart_queue = tf.RandomShuffleQueue(
            1024,
            40,
            dtypes=[tf.string, tf.int32, tf.string],
            shapes=[[],[],[]],
            seed=None)
    enqueue_op = chart_queue.enqueue_many((audio_fps, feats, step_frames))
    qr = tf.train.QueueRunner(chart_queue, [enqueue_op])
    tf.train.add_queue_runner(qr)

    # Dequeue single tuple example
    audio_fp, feat, chart_step_frames = chart_queue.dequeue()

    # TODO: tf.cond on file extension for decode_audio
    song_binary = tf.read_file(audio_fp)
    song_samples = tf.contrib.ffmpeg.decode_audio(song_binary, file_format='ogg', samples_per_second=44100, channel_count=1)
    song_samples = tf.squeeze(song_samples, axis=1)

    # Convert CSV step frames to integer
    chart_step_frames_split = tf.string_split([chart_step_frames], ',')
    chart_step_frames_split = tf.sparse_tensor_to_dense(chart_step_frames_split, '')[0]
    chart_step_frames = tf.string_to_number(chart_step_frames_split, tf.int32)

    # Make reference
    first_step_frame, last_step_frame = chart_step_frames[0], chart_step_frames[-1]
    with tf.control_dependencies([tf.assert_greater(last_step_frame, first_step_frame)]):
        chart_y = tf.zeros([last_step_frame - first_step_frame + 1], tf.float32)
        chart_y[chart_step_frames - first_step_frame] = 1.

        first_step_sample = first_step_frame * 441
        last_step_sample = last_step_frame * 441
        chart_debug_y = tf.zeros([first_step_sample - last_step_sample + 1], tf.float32)
        chart_debug_y[chart_step_frames * 441 - first_step_sample] = 1.
    chart_debug_y = chart_step_frames

    # Extract feats
    """
    song_feats = []
    for nfft in [1024, 2048, 4096]:
        song_feats.append(extract_feats(song_samples, 'lmel80', nfft=nfft, nhop=441))
    song_feats = tf.stack([feats_1024, feats_2048, feats_4096], axis=2)
    """

    # Trim feats
    chart_x = song_samples#song_samples[first_step_frame * 441:last_step_frame * 441 + 1]

    return chart_x, chart_debug_y


def train(args, charts):
    x, y = load_examples(charts, args.batch_size)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        _x, _y = sess.run([x, y])
        print _x.shape
        print _y

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    from argparse import ArgumentParser
    import json
    import os

    from ddc.chart import Song, PlacementChart

    parser = ArgumentParser()

    io_args = parser.add_argument_group('IO')
    io_args.add_argument('--train_fp', type=str, required=True)
    io_args.add_argument('--json_ext', type=str)

    train_args = parser.add_argument_group('Train')
    train_args.add_argument('--batch_size', type=str)

    parser.set_defaults(
        json_ext='filt',
        batch_size=256
    )

    args = parser.parse_args()

    charts = []
    dataset_dir, _ = os.path.split(args.train_fp)
    with open(args.train_fp, 'r') as f:
        for line in f.read().splitlines():
            json_fp = os.path.join(dataset_dir, line + '.{}.json'.format(args.json_ext))
            with open(json_fp, 'r') as j:
                song_json = json.loads(j.read())
            json_dir, _ = os.path.split(json_fp)
            song = Song(json_dir, song_json)
            for chart_json in song_json['charts']:
                chart = PlacementChart(song, chart_json)
                if chart.get_difficulty() == 'Challenge':
                    charts.append(chart)

    train(args, charts)
