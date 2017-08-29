from collections import defaultdict
import cPickle as pickle
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score

from onset_net import OnsetNet
from util import *

# Data
tf.app.flags.DEFINE_string('train_txt_fp', '', 'Training dataset txt file with a list of pickled song files')
tf.app.flags.DEFINE_string('valid_txt_fp', '', 'Eval dataset txt file with a list of pickled song files')
tf.app.flags.DEFINE_bool('z_score', False, 'If true, train and test on z-score of training data')
tf.app.flags.DEFINE_string('test_txt_fp', '', 'Test dataset txt file with a list of pickled song files')
tf.app.flags.DEFINE_string('model_ckpt_fp', '', 'File path to model checkpoint if resuming or eval')
tf.app.flags.DEFINE_string('export_feat_name', '', 'If set, export CNN features to this directory')

# Features
tf.app.flags.DEFINE_integer('audio_context_radius', 7, 'Past and future context per training example')
tf.app.flags.DEFINE_integer('audio_nbands', 80, 'Number of bands per frame')
tf.app.flags.DEFINE_integer('audio_nchannels', 3, 'Number of channels per frame')
tf.app.flags.DEFINE_string('audio_select_channels', '', 'List of CSV audio channels. If non-empty, other channels excluded from model.')
tf.app.flags.DEFINE_string('feat_diff_feet_to_id_fp', '', '')
tf.app.flags.DEFINE_string('feat_diff_coarse_to_id_fp', '', '')
tf.app.flags.DEFINE_bool('feat_diff_dipstick', '', '')
tf.app.flags.DEFINE_string('feat_freetext_to_id_fp', '', '')
tf.app.flags.DEFINE_bool('feat_beat_phase', '', '')
tf.app.flags.DEFINE_bool('feat_beat_phase_cos', '', '')

# Network params
tf.app.flags.DEFINE_string('cnn_filter_shapes', '', 'CSV 3-tuples of filter shapes (time, freq, n)')
tf.app.flags.DEFINE_string('cnn_pool', '', 'CSV 2-tuples of pool amounts (time, freq)')
tf.app.flags.DEFINE_bool('cnn_rnn_zack', False, '')
tf.app.flags.DEFINE_integer('zack_hack', 0, '')
tf.app.flags.DEFINE_string('rnn_cell_type', 'lstm', '')
tf.app.flags.DEFINE_integer('rnn_size', 0, '')
tf.app.flags.DEFINE_integer('rnn_nlayers', 0, '')
tf.app.flags.DEFINE_integer('rnn_nunroll', 1, '')
tf.app.flags.DEFINE_float('rnn_keep_prob', 1.0, '')
tf.app.flags.DEFINE_string('dnn_sizes', '', 'CSV sizes for dense layers')
tf.app.flags.DEFINE_float('dnn_keep_prob', 1.0, '')
tf.app.flags.DEFINE_string('dnn_nonlin', 'sigmoid', '')

# Training params
tf.app.flags.DEFINE_integer('batch_size', 256, 'Batch size for training')
tf.app.flags.DEFINE_string('weight_strategy', 'rect', 'One of \'rect\' or \'last\'')
tf.app.flags.DEFINE_bool('randomize_charts', False, '')
tf.app.flags.DEFINE_bool('balanced_class', True, 'If true, balance classes, otherwise use prior.')
tf.app.flags.DEFINE_integer('exclude_onset_neighbors', 0, 'If nonzero, excludes radius around true onsets from dataset as they may be misleading true neg.')
tf.app.flags.DEFINE_bool('exclude_pre_onsets', False, 'If true, exclude all true neg before first onset.')
tf.app.flags.DEFINE_bool('exclude_post_onsets', False, 'If true, exclude all true neg after last onset.')
tf.app.flags.DEFINE_float('grad_clip', 0.0, 'Clip gradients to this value if greater than 0')
tf.app.flags.DEFINE_string('opt', 'sgd', 'One of \'sgd\'')
tf.app.flags.DEFINE_float('lr', 1.0, 'Learning rate')
tf.app.flags.DEFINE_float('lr_decay_rate', 1.0, 'Multiply learning rate by this value every epoch')
tf.app.flags.DEFINE_integer('lr_decay_delay', 0, '')
tf.app.flags.DEFINE_integer('nbatches_per_ckpt', 100, 'Save model weights every N batches')
tf.app.flags.DEFINE_integer('nbatches_per_eval', 10000, 'Evaluate model every N batches')
tf.app.flags.DEFINE_integer('nepochs', 0, 'Number of training epochs, negative means train continuously')
tf.app.flags.DEFINE_string('experiment_dir', '', 'Directory for temporary training files and model weights')

# Eval params
tf.app.flags.DEFINE_string('eval_window_type', '', '')
tf.app.flags.DEFINE_integer('eval_window_width', 0, '')
tf.app.flags.DEFINE_integer('eval_align_tolerance', 2, '')
tf.app.flags.DEFINE_string('eval_charts_export', '', '')
tf.app.flags.DEFINE_string('eval_diff_coarse', '', '')

FLAGS = tf.app.flags.FLAGS
dtype = tf.float32

def main(_):
    assert FLAGS.experiment_dir
    do_train = FLAGS.nepochs != 0 and bool(FLAGS.train_txt_fp)
    do_valid = bool(FLAGS.valid_txt_fp)
    do_train_eval = do_train and do_valid
    do_eval = bool(FLAGS.test_txt_fp)
    do_cnn_export = bool(FLAGS.export_feat_name)

    # Load data
    print 'Loading data'
    train_data, valid_data, test_data = open_dataset_fps(FLAGS.train_txt_fp, FLAGS.valid_txt_fp, FLAGS.test_txt_fp)

    # Select channels
    if FLAGS.audio_select_channels:
        channels = stride_csv_arg_list(FLAGS.audio_select_channels, 1, int)
        print 'Selecting channels {} from data'.format(channels)
        for data in [train_data, valid_data, test_data]:
            select_channels(data, channels)

    # Calculate validation metrics
    if FLAGS.z_score:
        z_score_fp = os.path.join(FLAGS.experiment_dir, 'valid_mean_std.pkl')
        if do_valid and not os.path.exists(z_score_fp):
            print 'Calculating validation metrics'
            mean_per_band, std_per_band = calc_mean_std_per_band(valid_data)
            with open(z_score_fp, 'wb') as f:
                pickle.dump((mean_per_band, std_per_band), f)
        else:
            print 'Loading validation metrics'
            with open(z_score_fp, 'rb') as f:
                mean_per_band, std_per_band = pickle.load(f)

        # Sanitize data
        for data in [train_data, valid_data, test_data]:
            apply_z_norm(data, mean_per_band, std_per_band)

    # Flatten the data into chart references for easier iteration
    print 'Flattening datasets into charts'
    charts_train = flatten_dataset_to_charts(train_data)
    charts_valid = flatten_dataset_to_charts(valid_data)
    charts_test = flatten_dataset_to_charts(test_data)
    print 'Train set: {} charts, valid set: {} charts, test set: {} charts'.format(len(charts_train), len(charts_valid), len(charts_test))

    # Load ID maps
    diff_feet_to_id = None
    if FLAGS.feat_diff_feet_to_id_fp:
        diff_feet_to_id = load_id_dict(FLAGS.feat_diff_feet_to_id_fp)
    diff_coarse_to_id = None
    if FLAGS.feat_diff_coarse_to_id_fp:
        diff_coarse_to_id = load_id_dict(FLAGS.feat_diff_coarse_to_id_fp)
    freetext_to_id = None
    if FLAGS.feat_freetext_to_id_fp:
        freetext_to_id = load_id_dict(FLAGS.feat_freetext_to_id_fp)

    # Create feature config
    feats_config = {
        'time_context_radius': FLAGS.audio_context_radius,
        'diff_feet_to_id': diff_feet_to_id,
        'diff_coarse_to_id': diff_coarse_to_id,
        'diff_dipstick': FLAGS.feat_diff_dipstick,
        'freetext_to_id': freetext_to_id,
        'beat_phase': FLAGS.feat_beat_phase,
        'beat_phase_cos': FLAGS.feat_beat_phase_cos
    }
    nfeats = 0
    nfeats += 0 if diff_feet_to_id is None else max(diff_feet_to_id.values()) + 1
    nfeats += 0 if diff_coarse_to_id is None else max(diff_coarse_to_id.values()) + 1
    nfeats += 0 if freetext_to_id is None else max(freetext_to_id.values()) + 1
    nfeats += 1 if FLAGS.feat_beat_phase else 0
    nfeats += 1 if FLAGS.feat_beat_phase_cos else 0
    print 'Feature configuration (nfeats={}): {}'.format(nfeats, feats_config)

    # Create training data exclusions config
    tn_exclusions = {
        'randomize_charts': FLAGS.randomize_charts,
        'exclude_onset_neighbors': FLAGS.exclude_onset_neighbors,
        'exclude_pre_onsets': FLAGS.exclude_pre_onsets,
        'exclude_post_onsets': FLAGS.exclude_post_onsets,
        'include_onsets': not FLAGS.balanced_class
    }
    train_batch_config = feats_config.copy()
    train_batch_config.update(tn_exclusions)
    print 'Exclusions: {}'.format(tn_exclusions)

    # Create model config
    model_config = {
        'audio_context_radius': FLAGS.audio_context_radius,
        'audio_nbands': FLAGS.audio_nbands,
        'audio_nchannels': FLAGS.audio_nchannels,
        'nfeats': nfeats,
        'cnn_filter_shapes': stride_csv_arg_list(FLAGS.cnn_filter_shapes, 3, int),
        'cnn_init': tf.uniform_unit_scaling_initializer(factor=1.43, dtype=dtype),
        'cnn_pool': stride_csv_arg_list(FLAGS.cnn_pool, 2, int),
        'cnn_rnn_zack': FLAGS.cnn_rnn_zack,
        'zack_hack': FLAGS.zack_hack,
        'rnn_cell_type': FLAGS.rnn_cell_type,
        'rnn_size': FLAGS.rnn_size,
        'rnn_nlayers': FLAGS.rnn_nlayers,
        'rnn_init': tf.random_uniform_initializer(-5e-2, 5e-2, dtype=dtype),
        'rnn_nunroll': FLAGS.rnn_nunroll,
        'rnn_keep_prob': FLAGS.rnn_keep_prob,
        'dnn_sizes': stride_csv_arg_list(FLAGS.dnn_sizes, 1, int),
        'dnn_init': tf.uniform_unit_scaling_initializer(factor=1.15, dtype=dtype),
        'dnn_keep_prob': FLAGS.dnn_keep_prob,
        'dnn_nonlin': FLAGS.dnn_nonlin,
        'grad_clip': FLAGS.grad_clip,
        'opt': FLAGS.opt,
    }
    print 'Model configuration: {}'.format(model_config)

    with tf.Graph().as_default(), tf.Session() as sess:
        if do_train:
            print 'Creating train model'
            with tf.variable_scope('model_sp', reuse=None):
                model_train = OnsetNet(mode='train', target_weight_strategy=FLAGS.weight_strategy, batch_size=FLAGS.batch_size, **model_config)

        if do_train_eval or do_eval or do_cnn_export:
            with tf.variable_scope('model_sp', reuse=do_train):
                eval_batch_size = FLAGS.batch_size
                if FLAGS.rnn_nunroll > 1:
                    eval_batch_size = 1
                model_eval = OnsetNet(mode='eval', target_weight_strategy='seq', batch_size=eval_batch_size, export_feat_name=FLAGS.export_feat_name, **model_config)
                model_early_stop_xentropy_avg = tf.train.Saver(tf.global_variables(), max_to_keep=None)
                model_early_stop_auprc = tf.train.Saver(tf.global_variables(), max_to_keep=None)
                model_early_stop_fscore = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        # Restore or init model
        model_saver = tf.train.Saver(tf.global_variables())
        if FLAGS.model_ckpt_fp:
            print 'Restoring model weights from {}'.format(FLAGS.model_ckpt_fp)
            model_saver.restore(sess, FLAGS.model_ckpt_fp)
        else:
            print 'Initializing model weights from scratch'
            sess.run(tf.global_variables_initializer())

        # Create summaries
        if do_train:
            summary_writer = tf.summary.FileWriter(FLAGS.experiment_dir, sess.graph)

            epoch_mean_xentropy = tf.placeholder(tf.float32, shape=[], name='epoch_mean_xentropy')
            epoch_mean_time = tf.placeholder(tf.float32, shape=[], name='epoch_mean_time')
            epoch_var_xentropy = tf.placeholder(tf.float32, shape=[], name='epoch_var_xentropy')
            epoch_var_time = tf.placeholder(tf.float32, shape=[], name='epoch_var_time')
            epoch_summaries = tf.summary.merge([
                tf.summary.scalar('epoch_mean_xentropy', epoch_mean_xentropy),
                tf.summary.scalar('epoch_mean_time', epoch_mean_time),
                tf.summary.scalar('epoch_var_xentropy', epoch_var_xentropy),
                tf.summary.scalar('epoch_var_time', epoch_var_time)
            ])

            eval_metric_names = ['xentropy_avg', 'pos_xentropy_avg', 'auroc', 'auprc', 'fscore', 'precision', 'recall', 'threshold', 'accuracy', 'perplexity', 'density_rel']
            eval_metrics = {}
            eval_summaries = []
            for eval_metric_name in eval_metric_names:
                name_mean = 'eval_mean_{}'.format(eval_metric_name)
                name_var = 'eval_var_{}'.format(eval_metric_name)
                ph_mean = tf.placeholder(tf.float32, shape=[], name=name_mean)
                ph_var = tf.placeholder(tf.float32, shape=[], name=name_var)
                summary_mean = tf.summary.scalar(name_mean, ph_mean)
                summary_var = tf.summary.scalar(name_var, ph_var)
                eval_summaries.append(tf.summary.merge([summary_mean, summary_var]))
                eval_metrics[eval_metric_name] = (ph_mean, ph_var)
            eval_time = tf.placeholder(tf.float32, shape=[], name='eval_time')
            eval_time_summary = tf.summary.scalar('eval_time', eval_time)
            eval_summaries = tf.summary.merge([eval_time_summary] + eval_summaries)

            # Calculate epoch stuff
            train_nframes = sum([chart.get_nframes_annotated() for chart in charts_train])
            examples_per_batch = FLAGS.batch_size
            examples_per_batch *= FLAGS.rnn_nunroll if FLAGS.weight_strategy == 'rect' else 1
            batches_per_epoch = train_nframes // examples_per_batch
            nbatches = FLAGS.nepochs * batches_per_epoch
            print '{} frames in data, {} batches per epoch, {} batches total'.format(train_nframes, batches_per_epoch, nbatches)

            # Init epoch
            lr_summary = model_train.assign_lr(sess, FLAGS.lr)
            summary_writer.add_summary(lr_summary, 0)
            epoch_xentropies = []
            epoch_times = []

            batch_num = 0
            eval_best_xentropy_avg = float('inf')
            eval_best_auprc = 0.0
            eval_best_fscore = 0.0
            while FLAGS.nepochs < 0 or batch_num < nbatches:
                batch_time_start = time.time()
                feats_audio, feats_other, targets, target_weights = model_train.prepare_train_batch(charts_train, **train_batch_config)
                feed_dict = {
                    model_train.feats_audio: feats_audio,
                    model_train.feats_other: feats_other,
                    model_train.targets: targets,
                    model_train.target_weights: target_weights
                }
                batch_xentropy, _ = sess.run([model_train.avg_neg_log_lhood, model_train.train_op], feed_dict=feed_dict)

                epoch_xentropies.append(batch_xentropy)
                epoch_times.append(time.time() - batch_time_start)

                batch_num += 1

                if batch_num % batches_per_epoch == 0:
                    epoch_num = batch_num // batches_per_epoch
                    print 'Completed epoch {}'.format(epoch_num)

                    lr_decay = FLAGS.lr_decay_rate ** max(epoch_num - FLAGS.lr_decay_delay, 0)
                    lr_summary = model_train.assign_lr(sess, FLAGS.lr * lr_decay)
                    summary_writer.add_summary(lr_summary, batch_num)

                    epoch_xentropy = np.mean(epoch_xentropies)
                    print 'Epoch mean cross-entropy (nats) {}'.format(epoch_xentropy)
                    epoch_summary = sess.run(epoch_summaries, feed_dict={epoch_mean_xentropy: epoch_xentropy, epoch_mean_time: np.mean(epoch_times), epoch_var_xentropy: np.var(epoch_xentropies), epoch_var_time: np.var(epoch_times)})
                    summary_writer.add_summary(epoch_summary, batch_num)

                    epoch_xentropies = []
                    epoch_times = []

                if batch_num % FLAGS.nbatches_per_ckpt == 0:
                    print 'Saving model weights...'
                    ckpt_fp = os.path.join(FLAGS.experiment_dir, 'onset_net_train')
                    model_saver.save(sess, ckpt_fp, global_step=tf.contrib.framework.get_or_create_global_step())
                    print 'Done saving!'

                if do_train_eval and batch_num % FLAGS.nbatches_per_eval == 0:
                    print 'Evaluating...'
                    eval_start_time = time.time()

                    metrics = defaultdict(list)
                    for chart in charts_valid:
                        y_true, y_scores, y_xentropies, y_scores_pkalgn = model_scores_for_chart(sess, chart, model_eval, **feats_config)
                        assert int(np.sum(y_true)) == chart.get_nonsets()

                        chart_metrics = eval_metrics_for_scores(y_true, y_scores, y_xentropies, y_scores_pkalgn)
                        for metrics_key, metric_value in chart_metrics.items():
                            metrics[metrics_key].append(metric_value)

                    metrics = {k: (np.mean(v), np.var(v)) for k, v in metrics.items()}
                    feed_dict = {}
                    results = []
                    for metric_name, (mean, var) in metrics.items():
                        feed_dict[eval_metrics[metric_name][0]] = mean
                        feed_dict[eval_metrics[metric_name][1]] = var
                    feed_dict[eval_time] = time.time() - eval_start_time

                    summary_writer.add_summary(sess.run(eval_summaries, feed_dict=feed_dict), batch_num)

                    xentropy_avg_mean = metrics['xentropy_avg'][0]
                    if xentropy_avg_mean < eval_best_xentropy_avg:
                        print 'Xentropy {} better than previous {}'.format(xentropy_avg_mean, eval_best_xentropy_avg)
                        ckpt_fp = os.path.join(FLAGS.experiment_dir, 'onset_net_early_stop_xentropy_avg')
                        model_early_stop_xentropy_avg.save(sess, ckpt_fp, global_step=tf.contrib.framework.get_or_create_global_step())
                        eval_best_xentropy_avg = xentropy_avg_mean

                    auprc_mean = metrics['auprc'][0]
                    if auprc_mean > eval_best_auprc:
                        print 'AUPRC {} better than previous {}'.format(auprc_mean, eval_best_auprc)
                        ckpt_fp = os.path.join(FLAGS.experiment_dir, 'onset_net_early_stop_auprc')
                        model_early_stop_auprc.save(sess, ckpt_fp, global_step=tf.contrib.framework.get_or_create_global_step())
                        eval_best_auprc = auprc_mean

                    fscore_mean = metrics['fscore'][0]
                    if fscore_mean > eval_best_fscore:
                        print 'Fscore {} better than previous {}'.format(fscore_mean, eval_best_fscore)
                        ckpt_fp = os.path.join(FLAGS.experiment_dir, 'onset_net_early_stop_fscore')
                        model_early_stop_fscore.save(sess, ckpt_fp, global_step=tf.contrib.framework.get_or_create_global_step())
                        eval_best_fscore = fscore_mean

                    print 'Done evaluating'

        if do_cnn_export:
            print 'Exporting CNN features...'
            export_dir = os.path.join(FLAGS.experiment_dir, 'export_{}'.format(FLAGS.export_feat_name))

            if not os.path.exists(export_dir):
                os.makedirs(export_dir)

            # This is ugly... Deal with it
            song_names = []
            for data_fp in [FLAGS.train_txt_fp, FLAGS.valid_txt_fp, FLAGS.test_txt_fp]:
                with open(data_fp, 'r') as f:
                    song_charts_fps = f.read().splitlines()
                song_names += [os.path.splitext(os.path.split(x)[1])[0] for x in song_charts_fps]

            songs_complete = set()
            for charts in [charts_train, charts_valid, charts_test]:
                for chart in charts:
                    song_features_id = id(chart.song_features)
                    if song_features_id in songs_complete:
                        continue

                    song_feats_export = []
                    for feats_audio, feats_other, _, _ in model_eval.iterate_eval_batches(chart, **feats_config):
                        assert feats_other.shape[2] == 0
                        feed_dict = {
                            model_eval.feats_audio: feats_audio,
                            model_eval.feats_other: feats_other,
                        }
                        feats_export = sess.run(model_eval.feats_export, feed_dict=feed_dict)
                        song_feats_export.append(feats_export)

                    song_feats_export = np.concatenate(song_feats_export)
                    song_feats_export = song_feats_export[:chart.song_features.shape[0]]
                    assert song_feats_export.shape[0] == chart.song_features.shape[0]

                    if 'cnn' in FLAGS.export_feat_name:
                        song_feats_export = np.reshape(song_feats_export, (song_feats_export.shape[0], -1, song_feats_export.shape[3]))
                    if 'dnn' in FLAGS.export_feat_name:
                        song_feats_export = song_feats_export[:, :, np.newaxis]

                    assert song_feats_export.ndim == 3

                    out_name = song_names.pop(0)
                    print '{} ({})->{} ({})'.format(chart.get_song_metadata(), chart.song_features.shape, out_name, song_feats_export.shape)

                    with open(os.path.join(export_dir, '{}.pkl'.format(out_name)), 'wb') as f:
                        pickle.dump(song_feats_export, f)

                    songs_complete.add(song_features_id)

            assert len(song_names) == 0

        if do_eval and not do_cnn_export:
            print 'Evaluating...'

            exports = stride_csv_arg_list(FLAGS.eval_charts_export, 1, int)
            diff_concat = lambda d, diff: np.concatenate(d[diff])
            diff_array = lambda d, diff: np.array(d[diff])
            all_concat = lambda d: np.concatenate([diff_concat(d, diff) for diff in diffs])
            all_concat_array = lambda d: np.concatenate([diff_array(d, diff) for diff in diffs])

            # calculate thresholds
            if len(charts_valid) > 0:
                print 'Calculating perchart and micro thresholds for validation data'

                # go through charts calculating scores and thresholds
                diff_to_threshold = defaultdict(list)
                diff_to_y_true_all = defaultdict(list)
                diff_to_y_scores_pkalgn_all = defaultdict(list)
                for chart in charts_valid:
                    y_true, _, _, y_scores_pkalgn = model_scores_for_chart(sess, chart, model_eval, **feats_config)
                    chart_metrics = eval_metrics_for_scores(y_true, None, None, y_scores_pkalgn)

                    diff_to_threshold[chart.get_coarse_difficulty()].append(chart_metrics['threshold'])
                    diff_to_y_true_all[chart.get_coarse_difficulty()].append(y_true)
                    diff_to_y_scores_pkalgn_all[chart.get_coarse_difficulty()].append(y_scores_pkalgn)

                # lambdas for calculating micro thresholds
                diffs = diff_to_y_true_all.keys()

                # calculate diff perchart
                diff_to_threshold_perchart = {diff:np.mean(diff_array(diff_to_threshold, diff)) for diff in diffs}

                # calculate all perchart
                threshold_all_perchart = np.mean(all_concat_array(diff_to_threshold))

                # calculate diff micro
                diff_to_threshold_micro = {}
                for diff in diffs:
                    diff_metrics = eval_metrics_for_scores(diff_concat(diff_to_y_true_all, diff), None, None, diff_concat(diff_to_y_scores_pkalgn_all, diff))
                    diff_to_threshold_micro[diff] = diff_metrics['threshold']

                # calculate all micro
                all_metrics = eval_metrics_for_scores(all_concat(diff_to_y_true_all), None, None, all_concat(diff_to_y_scores_pkalgn_all))
                threshold_all_micro = all_metrics['threshold']

                print 'Diff perchart thresholds: {}'.format(diff_to_threshold_perchart)
                print 'All perchart threshold: {}'.format(threshold_all_perchart)
                print 'Diff_micro thresholds: {}'.format(diff_to_threshold_micro)
                print 'All micro thresholds: {}'.format(threshold_all_micro)

            # run evaluation on test data
            diff_to_y_true = defaultdict(list)
            diff_to_y_scores = defaultdict(list)
            diff_to_y_xentropies = defaultdict(list)
            diff_to_y_scores_pkalgn = defaultdict(list)
            metrics = defaultdict(list)
            for i, chart in enumerate(charts_test):
                chart_coarse = chart.get_coarse_difficulty()
                if FLAGS.eval_diff_coarse:
                    if chart_coarse != FLAGS.eval_diff_coarse:
                        continue

                y_true, y_scores, y_xentropies, y_scores_pkalgn = model_scores_for_chart(sess, chart, model_eval, **feats_config)
                assert int(np.sum(y_true)) == chart.get_nonsets()
                diff_to_y_true[chart_coarse].append(y_true)
                diff_to_y_scores[chart_coarse].append(y_scores)
                diff_to_y_xentropies[chart_coarse].append(y_xentropies)
                diff_to_y_scores_pkalgn[chart_coarse].append(y_scores_pkalgn)

                if i in exports:
                    chart_name = ez_name(chart.get_song_metadata()['title'])
                    chart_export_fp = os.path.join(FLAGS.experiment_dir, '{}_{}.pkl'.format(chart_name, chart.get_foot_difficulty()))
                    chart_eval_save = {
                        'song_metadata': chart.get_song_metadata(),
                        'song_feats': chart.song_features[:, :, 0],
                        'chart_feet': chart.get_foot_difficulty(),
                        'chart_onsets': chart.get_onsets(),
                        'y_true': y_true,
                        'y_scores': y_scores,
                        'y_xentropies': y_xentropies,
                        'y_scores_pkalgn': y_scores_pkalgn
                    }
                    with open(chart_export_fp, 'wb') as f:
                        print 'Saving {} {}'.format(chart.get_song_metadata(), chart.get_foot_difficulty())
                        pickle.dump(chart_eval_save, f)

                if len(charts_valid) > 0:
                    threshold_names = ['diff_perchart', 'all_perchart', 'diff_micro', 'all_micro']
                    thresholds = [diff_to_threshold_perchart[chart_coarse], threshold_all_perchart, diff_to_threshold_micro[chart_coarse], threshold_all_micro]
                else:
                    threshold_names, thresholds = [], []

                chart_metrics = eval_metrics_for_scores(y_true, y_scores, y_xentropies, y_scores_pkalgn, threshold_names, thresholds)
                for metrics_key, metric in chart_metrics.items():
                    metrics[metrics_key].append(metric)

            # calculate micro metrics
            diffs = diff_to_y_true.keys()
            metrics_micro, prc = eval_metrics_for_scores(all_concat(diff_to_y_true), all_concat(diff_to_y_scores), all_concat(diff_to_y_xentropies), all_concat(diff_to_y_scores_pkalgn), return_prc=True)

            # dump PRC for inspection
            with open(os.path.join(FLAGS.experiment_dir, 'prc.pkl'), 'wb') as f:
                pickle.dump(prc, f)

            # calculate perchart metrics
            metrics_perchart = {k: (np.mean(v), np.std(v), np.min(v), np.max(v)) for k, v in metrics.items()}

            # report metrics
            copy_pasta = []

            metrics_report_micro = ['xentropy_avg', 'pos_xentropy_avg', 'auroc', 'auprc', 'fscore', 'precision', 'recall', 'threshold', 'accuracy']
            for metric_name in metrics_report_micro:
                metric_stats = metrics_micro[metric_name]
                copy_pasta += [metric_stats]
                print 'micro_{}: {}'.format(metric_name, metric_stats)

            metrics_report_perchart = ['xentropy_avg', 'pos_xentropy_avg', 'auroc', 'auprc', 'fscore', 'precision', 'recall', 'threshold', 'accuracy', 'perplexity', 'density_rel']
            for threshold_name in threshold_names:
                fmetric_names = ['fscore_{}', 'precision_{}', 'recall_{}', 'threshold_{}']
                metrics_report_perchart += [name.format(threshold_name) for name in fmetric_names]
            for metric_name in metrics_report_perchart:
                metric_stats = metrics_perchart.get(metric_name, (0., 0., 0., 0.))
                copy_pasta += list(metric_stats)
                print 'perchart_{}: {}'.format(metric_name, metric_stats)

            print 'COPY PASTA:'
            print ','.join([str(x) for x in copy_pasta])

def model_scores_for_chart(sess, chart, model, **feat_kwargs):
    if model.do_rnn:
        state = sess.run(model.initial_state)
    targets_all = []
    scores = []
    xentropies = []
    weight_sum = 0.0
    target_sum = 0.0

    chunk_len = FLAGS.rnn_nunroll if model.do_rnn else FLAGS.batch_size
    for feats_audio, feats_other, targets, target_weights in model.iterate_eval_batches(chart, **feat_kwargs):
        feed_dict = {
            model.feats_audio: feats_audio,
            model.feats_other: feats_other,
            model.targets: targets,
            model.target_weights: target_weights
        }
        if model.do_rnn:
            feed_dict[model.initial_state] = state
            state, seq_scores, seq_xentropies = sess.run([model.final_state, model.prediction, model.neg_log_lhoods], feed_dict=feed_dict)
            scores.append(seq_scores[0])
            xentropies.append(seq_xentropies[0])
        else:
            seq_scores, seq_xentropies = sess.run([model.prediction, model.neg_log_lhoods], feed_dict=feed_dict)
            scores.append(seq_scores[:, 0])
            xentropies.append(seq_xentropies[:, 0])

        targets_all.append(targets)
        weight_sum += np.sum(target_weights)
        target_sum += np.sum(targets)

    targets_all = np.concatenate(targets_all)
    assert int(weight_sum) == chart.get_nframes_annotated()
    assert int(target_sum) == chart.get_nonsets()

    # scores may be up to nunroll-1 longer than song feats but will be left-aligned
    scores = np.concatenate(scores)
    xentropies = np.concatenate(xentropies)
    assert scores.shape[0] >= chart.get_nframes()
    assert scores.shape[0] < (chart.get_nframes() + 2 * chunk_len)
    assert xentropies.shape == scores.shape
    if model.do_rnn:
        xentropies = xentropies[chunk_len - 1:]
        scores = scores[chunk_len - 1:]
    scores = scores[:chart.get_nframes()]

    # find predicted onsets (smooth and peak pick)
    if FLAGS.eval_window_type == 'hann':
        window = np.hanning(FLAGS.eval_window_width)
    elif FLAGS.eval_window_type == 'hamming':
        window = np.hamming(FLAGS.eval_window_width)
    else:
        raise NotImplementedError()
    pred_onsets = find_pred_onsets(scores, window)

    # align scores with true to create sklearn-compatible vectors
    true_onsets = set(chart.get_onsets())
    y_true, y_scores_pkalgn = align_onsets_to_sklearn(true_onsets, pred_onsets, scores, tolerance=FLAGS.eval_align_tolerance)
    y_scores = scores[chart.get_first_onset():chart.get_last_onset() + 1]
    y_xentropies = xentropies[chart.get_first_onset():chart.get_last_onset() + 1]

    return y_true, y_scores, y_xentropies, y_scores_pkalgn

def eval_metrics_for_scores(y_true, y_scores, y_xentropies, y_scores_pkalgn, given_threshold_names=[], given_thresholds=[], return_prc=False):
    nonsets = np.sum(y_true)
    if y_xentropies is not None:
        xentropy_avg = np.mean(y_xentropies)
        pos_xentropy_avg = np.sum(np.multiply(y_xentropies, y_true)) / nonsets
    else:
        xentropy_avg = 0.
        pos_xentropy_avg = 0.

    # calculate ROC curve
    fprs, tprs, thresholds = roc_curve(y_true, y_scores_pkalgn)
    auroc = auc(fprs, tprs)

    # calculate PR curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores_pkalgn)
    # https://github.com/scikit-learn/scikit-learn/issues/1423
    auprc = auc(recalls, precisions)

    # find best fscore and associated values
    fscores_denom = precisions + recalls
    fscores_denom[np.where(fscores_denom == 0.0)] = 1.0
    fscores = (2 * (precisions * recalls)) / fscores_denom
    fscore_max_idx = np.argmax(fscores)
    precision, recall, fscore, threshold_ideal = precisions[fscore_max_idx], recalls[fscore_max_idx], fscores[fscore_max_idx], thresholds[fscore_max_idx]

    # calculate density
    predicted_steps = np.where(y_scores_pkalgn >= threshold_ideal)
    density_rel = float(len(predicted_steps[0])) / float(nonsets)

    # calculate accuracy
    y_labels = np.zeros(y_scores_pkalgn.shape[0], dtype=np.int)
    y_labels[predicted_steps] = 1
    accuracy = accuracy_score(y_true.astype(np.int), y_labels)

    # calculate metrics for fixed thresholds
    metrics = {}
    for threshold_name, threshold in zip(given_threshold_names, given_thresholds):
        threshold_closest_idx = np.argmax(thresholds >= threshold)
        precision_closest, recall_closest, fscore_closest, threshold_closest = precisions[threshold_closest_idx], recalls[threshold_closest_idx], fscores[threshold_closest_idx], thresholds[threshold_closest_idx]
        metrics['precision_{}'.format(threshold_name)] = precision_closest
        metrics['recall_{}'.format(threshold_name)] = recall_closest
        metrics['fscore_{}'.format(threshold_name)] = fscore_closest
        metrics['threshold_{}'.format(threshold_name)] = threshold_closest

    # aggregate metrics
    metrics['xentropy_avg'] = xentropy_avg
    metrics['pos_xentropy_avg'] = pos_xentropy_avg
    metrics['auroc'] = auroc
    metrics['auprc'] = auprc
    metrics['fscore'] = fscore
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['threshold'] = threshold_ideal
    metrics['accuracy'] = accuracy
    metrics['perplexity'] = np.exp(xentropy_avg)
    metrics['density_rel'] = density_rel

    if return_prc:
        return metrics, (precisions, recalls)
    else:
        return metrics

if __name__ == '__main__':
    tf.app.run()
