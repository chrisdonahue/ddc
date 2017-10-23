import math
import os
import shutil

import essentia
from essentia.standard import MetadataReader
import numpy as np
import tensorflow as tf
from scipy.signal import argrelextrema
assert tf.__version__ == '0.12.1'

from onset_net import OnsetNet
from sym_net import SymNet
from util import apply_z_norm, make_onset_feature_context
from extract_feats import extract_mel_feats

def load_sp_model(sp_ckpt_fp, batch_size=128):
    with tf.variable_scope('model_sp'):
        model_sp = OnsetNet(
            mode='gen',
            batch_size=batch_size,
            audio_context_radius=7,
            audio_nbands=80,
            audio_nchannels=3,
            nfeats=5,
            cnn_filter_shapes=[(7,3,10),(3,3,20)],
            cnn_init=None,
            cnn_pool=[(1,3),(1,3)],
            cnn_rnn_zack=False,
            rnn_cell_type=None,
            rnn_size=0,
            rnn_nlayers=0,
            rnn_init=None,
            rnn_nunroll=1,
            rnn_keep_prob=1.0,
            dnn_sizes=[256,128],
            dnn_init=None,
            dnn_keep_prob=1.0,
            dnn_nonlin='relu',
            target_weight_strategy='seq',
            grad_clip=None,
            opt=None,
            export_feat_name=None,
            zack_hack=0)
    model_sp_vars = filter(lambda v: 'model_sp' in v.name, tf.all_variables())
    saver = tf.train.Saver(model_sp_vars)
    saver.restore(sess, sp_ckpt_fp)
    return model_sp

def load_ss_model(ss_ckpt_fp):
    with tf.variable_scope('model_ss'):
        model_ss = SymNet(
            mode='gen',
            batch_size=1,
            nunroll=1,
            sym_in_type='bagofarrows',
            sym_embedding_size=0,
            sym_out_type='onehot',
            sym_narrows=4,
            sym_narrowclasses=4,
            other_nfeats=2,
            audio_context_radius=0,
            audio_nbands=0,
            audio_nchannels=0,
            cnn_filter_shapes=[],
            cnn_init=None,
            cnn_pool=[],
            cnn_dim_reduction_size=None,
            cnn_dim_reduction_init=None,
            cnn_dim_reduction_nonlin=None,
            cnn_dim_reduction_keep_prob=None,
            rnn_proj_init=None,
            rnn_cell_type='lstm',
            rnn_size=128,
            rnn_nlayers=2,
            rnn_init=None,
            rnn_keep_prob=1.0,
            dnn_sizes=[],
            dnn_init=None,
            dnn_keep_prob=1.0
        )
    model_ss_vars = filter(lambda v: 'model_ss' in v.name, tf.all_variables())
    saver = tf.train.Saver(model_ss_vars)
    saver.restore(sess, ss_ckpt_fp)
    return model_ss

# These are thresholds producing best perchart Fscore on valid set
_DIFFS = ['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge']
_DIFF_TO_COARSE_FINE_AND_THRESHOLD = {
    'Beginner':     (0, 1, 0.15325437),
    'Easy':         (1, 3, 0.23268291),
    'Medium':       (2, 5, 0.29456162),
    'Hard':         (3, 7, 0.29084727),
    'Challenge':    (4, 9, 0.28875697)
}

_SUBDIV = 192
_DT = 0.01
_HZ = 1.0 / _DT
_BPM = 60 * (1.0 / _DT) * (1.0 / float(_SUBDIV)) * 4.0

_TEMPL = """\
#TITLE:{title};
#ARTIST:{artist};
#MUSIC:{music_fp};
#OFFSET:0.0;
#BPMS:0.0={bpm};
#STOPS:;
{charts}\
"""

_PACK_NAME = 'DanceDanceConvolutionV1'
_CHART_TEMPL = """\
#NOTES:
    dance-single:
    DanceDanceConvolutionV1:
    {ccoarse}:
    {cfine}:
    0.0,0.0,0.0,0.0,0.0:
{measures};\
"""

class CreateChartException(Exception):
    pass

def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1)*s)))

def create_chart_dir(
        artist, title,
        audio_fp,
        norm, analyzers,
        sp_model, sp_batch_size, diffs,
        ss_model, idx_to_label,
        out_dir, delete_audio=False):
    if not artist or not title:
        print 'Extracting metadata from {}'.format(audio_fp)
        meta_reader = MetadataReader(filename=audio_fp)
        metadata = meta_reader()
        if not artist:
            artist = metadata[1]
        if not artist:
            artist = 'Unknown Artist'
        if not title:
            title = metadata[0]
        if not title:
            title = 'Unknown Title'

    print 'Loading {} - {}'.format(artist, title)
    try:
        song_feats = extract_mel_feats(audio_fp, analyzers, nhop=441)
    except:
        raise CreateChartException('Invalid audio file: {}'.format(audio_fp))
    song_feats -= norm[0]
    song_feats /= norm[1]
    song_len_sec = song_feats.shape[0] / _HZ
    print 'Processed {} minutes of features'.format(song_len_sec / 60.0)

    diff_chart_txts = []
    for diff in diffs:
        try:
            coarse, fine, threshold = _DIFF_TO_COARSE_FINE_AND_THRESHOLD[diff]
        except KeyError:
            raise CreateChartException('Invalid difficulty: {}'.format(diff))

        feats_other = np.zeros((sp_batch_size, 1, 5), dtype=np.float32)
        feats_other[:, :, coarse] = 1.0

        print 'Computing step placement scores'
        feats_audio = np.zeros((sp_batch_size, 1, 15, 80, 3), dtype=np.float32)
        predictions = []
        for start in xrange(0, song_feats.shape[0], sp_batch_size):
            for i, frame_idx in enumerate(range(start, start + sp_batch_size)):
                feats_audio[i] = make_onset_feature_context(song_feats, frame_idx, 7)

            feed_dict = {
                sp_model.feats_audio: feats_audio,
                sp_model.feats_other: feats_other
            }

            prediction = sess.run(sp_model.prediction, feed_dict=feed_dict)[:, 0]
            predictions.append(prediction)
        predictions = np.concatenate(predictions)[:song_feats.shape[0]]
        print predictions.shape

        print 'Peak picking'
        predictions_smoothed = np.convolve(predictions, np.hamming(5), 'same')
        maxima = argrelextrema(predictions_smoothed, np.greater_equal, order=1)[0]
        placed_times = []
        for i in maxima:
            t = float(i) * _DT
            if predictions[i] >= threshold:
                placed_times.append(t)
        print 'Found {} peaks, density {} steps per second'.format(len(placed_times), len(placed_times) / song_len_sec)

        print 'Performing step selection'
        state = sess.run(ss_model.initial_state)
        step_prev = '<-1>'
        times_arr = [placed_times[0]] + placed_times + [placed_times[-1]]
        selected_steps = []
        for i in xrange(1, len(times_arr) - 1):
            dt_prev, dt_next = times_arr[i] - times_arr[i-1], times_arr[i+1] - times_arr[i]
            feed_dict = {
                ss_model.syms: np.array([[ss_model.arrow_to_encoding(step_prev, 'bagofarrows')]], dtype=np.float32),
                ss_model.feats_other: np.array([[[dt_prev, dt_next]]], dtype=np.float32),
                ss_model.feats_audio: np.zeros((1, 1, 1, 0, 0), dtype=np.float32),
                ss_model.initial_state: state
            }
            scores, state = sess.run([ss_model.scores, ss_model.final_state], feed_dict=feed_dict)

            step_idx = 0
            while step_idx <= 1:
                step_idx = weighted_pick(scores)
            step = idx_to_label[step_idx]
            selected_steps.append(step)
            step_prev = step
        assert len(placed_times) == len(selected_steps)

        print 'Creating chart text'
        time_to_step = {int(round(t * _HZ)) : step for t, step in zip(placed_times, selected_steps)}
        max_subdiv = max(time_to_step.keys())
        if max_subdiv % _SUBDIV != 0:
            max_subdiv += _SUBDIV - (max_subdiv % _SUBDIV)
        full_steps = [time_to_step.get(i, '0000') for i in xrange(max_subdiv)]
        measures = [full_steps[i:i+_SUBDIV] for i in xrange(0, max_subdiv, _SUBDIV)]
        measures_txt = '\n,\n'.join(['\n'.join(measure) for measure in measures])
        chart_txt = _CHART_TEMPL.format(
            ccoarse=_DIFFS[coarse],
            cfine=fine,
            measures=measures_txt
        )
        diff_chart_txts.append(chart_txt)

    print 'Creating SM'
    out_dir_name = os.path.split(out_dir)[1]
    audio_out_name = out_dir_name + os.path.splitext(audio_fp)[1]
    sm_txt = _TEMPL.format(
        title=title,
        artist=artist,
        music_fp=audio_out_name,
        bpm=_BPM,
        charts='\n'.join(diff_chart_txts))

    print 'Saving to {}'.format(out_dir)
    try:
        os.mkdir(out_dir)
        audio_ext = os.path.splitext(audio_fp)[1]
        shutil.copyfile(audio_fp, os.path.join(out_dir, audio_out_name))
        with open(os.path.join(out_dir, out_dir_name + '.sm'), 'w') as f:
            f.write(sm_txt)
    except:
        raise CreateChartException('Error during output')

    if delete_audio:
        try:
            os.remove(audio_fp)
        except:
            raise CreateChartException('Error deleting audio')

    return True

if __name__ == '__main__':
    import argparse as argparse
    import cPickle as pickle
    import os
    import uuid
    from SimpleXMLRPCServer import SimpleXMLRPCServer
    import zipfile

    from extract_feats import create_analyzers

    parser = argparse.ArgumentParser()
    parser.add_argument('--norm_pkl_fp', type=str)
    parser.add_argument('--sp_ckpt_fp', type=str)
    parser.add_argument('--ss_ckpt_fp', type=str)
    parser.add_argument('--labels_txt_fp', type=str)
    parser.add_argument('--sp_batch_size', type=int)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--port', type=int)

    parser.set_defaults(
        sp_batch_size=256,
        port=13337,
    )

    args = parser.parse_args()

    config = tf.ConfigProto()
    config.log_device_placement = True
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 1.0

    if not os.path.isdir(args.out_dir):
      os.makedirs(args.out_dir)

    with tf.Graph().as_default(), tf.Session(config=config).as_default() as sess:
        print 'Loading band norms'
        with open(args.norm_pkl_fp, 'rb') as f:
            norm = pickle.load(f)

        print 'Creating Mel analyzers'
        analyzers = create_analyzers(nhop=441)

        print 'Loading labels'
        with open(args.labels_txt_fp, 'r') as f:
            idx_to_label = {i + 1:l for i, l in enumerate(f.read().splitlines())}

        print 'Loading step placement model'
        sp_model = load_sp_model(args.sp_ckpt_fp, args.sp_batch_size)
        print 'Loading step selection model'
        ss_model = load_ss_model(args.ss_ckpt_fp)

        def create_chart_closure(artist, title, audio_fp, diffs):
            song_id = uuid.uuid4()

            out_dir = os.path.join(args.out_dir, str(song_id))
            try:
                create_chart_dir(
                    artist, title,
                    audio_fp,
                    norm, analyzers,
                    sp_model, args.sp_batch_size, diffs,
                    ss_model, idx_to_label,
                    out_dir)
            except CreateChartException as e:
                raise e
            except Exception as e:
                raise CreateChartException('Unknown chart creation exception')

            out_zip_fp = os.path.join(args.out_dir, str(song_id) + '.zip')
            print out_zip_fp
            print 'Creating zip {}'.format(out_zip_fp)
            with zipfile.ZipFile(out_zip_fp, 'w', zipfile.ZIP_DEFLATED) as f:
                for fn in os.listdir(out_dir):
                    f.write(os.path.join(out_dir, fn), os.path.join(_PACK_NAME, str(song_id), fn))

            return os.path.abspath(out_zip_fp)

        print 'Waiting for RPCs on port {}'.format(args.port)
        server = SimpleXMLRPCServer(('localhost', args.port))
        server.register_function(create_chart_closure, 'create_chart')
        server.serve_forever()
