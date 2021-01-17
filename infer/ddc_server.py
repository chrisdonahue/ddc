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

def load_sp_model(sp_ckpt_fp, sess, batch_size=128):
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

def load_ss_model(ss_ckpt_fp, sess):
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
    'Medium':         (2, 5, 0.29456162),
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
        sess,
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
    audio_out_name = os.path.split(audio_fp)[1]
    sm_txt = _TEMPL.format(
        title=title,
        artist=artist,
        music_fp=audio_out_name,
        bpm=_BPM,
        charts='\n'.join(diff_chart_txts))

    print 'Saving to {}'.format(out_dir)
    try:
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        audio_out_fp = os.path.join(out_dir, audio_out_name)
        if not os.path.exists(audio_out_fp):
            shutil.copyfile(audio_fp, audio_out_fp)
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


import tempfile

from flask import Flask, jsonify, request, send_from_directory, send_file

_FRONTEND_DIST_DIR = 'frontend'
app = Flask(
    __name__,
    static_url_path='',
    static_folder=_FRONTEND_DIST_DIR)


@app.route('/')
def index():
    return send_from_directory(_FRONTEND_DIST_DIR, 'index.html')


@app.route('/choreograph', methods=['POST'])
def choreograph():
    uploaded_file = request.files.get('audio_file')
    if uploaded_file is None or len(uploaded_file.filename) == 0:
        return 'Audio file required', 400
    try:
        uploaded_file_ext = os.path.splitext(uploaded_file.filename)[1]
    except:
        uploaded_file_ext = ''

    song_artist = request.form.get('song_artist', '')[:1024]
    song_title = request.form.get('song_title', '')[:1024]
    diff_coarse = request.form.get('diff_coarse')
    if diff_coarse is None:
        return 'Need to specify difficulty', 400
    if diff_coarse not in _DIFFS:
        return 'Invalid difficulty specified', 400

    out_dir = tempfile.mkdtemp()
    with tempfile.NamedTemporaryFile(suffix='.zip') as z:
        song_id = uuid.uuid4()

        song_fp = os.path.join(out_dir, '{}{}'.format(str(song_id), uploaded_file_ext))
        uploaded_file.save(song_fp)

        try:
            create_chart_dir(
                song_artist, song_title,
                song_fp,
                NORM, ANALYZERS,
                SESS,
                SP_MODEL, ARGS.sp_batch_size,
                [diff_coarse],
                SS_MODEL, IDX_TO_LABEL,
                out_dir)
        except CreateChartException as e:
            if str(e).startswith('Invalid audio file'):
                return 'Invalid audio file', 400
            shutil.rmtree(out_dir)
            print e
            return 'Unknown error', 500
        except Exception as e:
            shutil.rmtree(out_dir)
            print e
            return 'Unknown error', 500

        print 'Creating zip {}'.format(z.name)
        with zipfile.ZipFile(z.name, 'w', zipfile.ZIP_DEFLATED) as f:
            for fn in os.listdir(out_dir):
                f.write(
                    os.path.join(out_dir, fn),
                    os.path.join(
                        _PACK_NAME,
                        str(song_id),
                        '{}{}'.format(str(song_id), os.path.splitext(fn)[1])))

        shutil.rmtree(out_dir)

        return send_file(
                z.name,
                as_attachment=True,
                attachment_filename='{}.zip'.format(song_id))


@app.after_request
def add_header(r):
    r.headers['Access-Control-Allow-Origin'] = '*'
    r.headers['Access-Control-Allow-Headers'] = '*'
    r.headers['Access-Control-Allow-Methods'] = '*'
    r.headers['Access-Control-Expose-Headers'] = '*'
    return r


if __name__ == '__main__':
    import argparse as argparse
    import cPickle as pickle
    import os
    import uuid
    import zipfile

    from extract_feats import create_analyzers

    parser = argparse.ArgumentParser()
    parser.add_argument('--norm_pkl_fp', type=str)
    parser.add_argument('--sp_ckpt_fp', type=str)
    parser.add_argument('--ss_ckpt_fp', type=str)
    parser.add_argument('--labels_txt_fp', type=str)
    parser.add_argument('--sp_batch_size', type=int)
    parser.add_argument('--max_file_size', type=int)

    parser.set_defaults(
        norm_pkl_fp='server_aux/norm.pkl',
        sp_ckpt_fp='server_aux/model_sp-56000',
        ss_ckpt_fp='server_aux/model_ss-23628',
        labels_txt_fp='server_aux/labels_4_0123.txt',
        sp_batch_size=256,
        max_file_size=None,
    )

    global ARGS
    ARGS = parser.parse_args()

    global SESS
    graph = tf.Graph()
    config = tf.ConfigProto()
    config.log_device_placement = True
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 1.0
    SESS = tf.Session(graph=graph, config=config)

    global NORM
    print 'Loading band norms'
    with open(ARGS.norm_pkl_fp, 'rb') as f:
        NORM = pickle.load(f)

    global ANALYZERS
    print 'Creating Mel analyzers'
    ANALYZERS = create_analyzers(nhop=441)

    global IDX_TO_LABEL
    print 'Loading labels'
    with open(ARGS.labels_txt_fp, 'r') as f:
        IDX_TO_LABEL = {i + 1:l for i, l in enumerate(f.read().splitlines())}

    global SP_MODEL, SS_MODEL
    with graph.as_default():
        print 'Loading step placement model'
        SP_MODEL = load_sp_model(ARGS.sp_ckpt_fp, SESS, ARGS.sp_batch_size)
        print 'Loading step selection model'
        SS_MODEL = load_ss_model(ARGS.ss_ckpt_fp, SESS)

    if ARGS.max_file_size is not None:
        app.config['MAX_CONTENT_LENGTH'] = ARGS.max_file_size
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 80)))
