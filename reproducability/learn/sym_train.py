from collections import defaultdict
import cPickle as pickle
import os
import time

import numpy as np
import tensorflow as tf

from sym_net import SymNet
from util import *

# Data
tf.app.flags.DEFINE_string('train_txt_fp', '', 'Training dataset txt file with a list of pickled song files')
tf.app.flags.DEFINE_string('valid_txt_fp', '', 'Eval dataset txt file with a list of pickled song files')
tf.app.flags.DEFINE_string('test_txt_fp', '', 'Test dataset txt file with a list of pickled song files')
tf.app.flags.DEFINE_string('sym_rnn_pretrain_model_ckpt_fp', '', 'File path to model checkpoint with only sym weights')
tf.app.flags.DEFINE_string('model_ckpt_fp', '', 'File path to model checkpoint if resuming or eval')

# Features
tf.app.flags.DEFINE_string('sym_in_type', 'onehot', 'Either \'onehot\' or \'bagofarrows\'')
tf.app.flags.DEFINE_string('sym_out_type', 'onehot', 'Either \'onehot\' or \'bagofarrows\'')
tf.app.flags.DEFINE_integer('sym_narrows', 4, 'Number or arrows in data')
tf.app.flags.DEFINE_integer('sym_narrowclasses', 4, 'Number or arrow classes in data')
tf.app.flags.DEFINE_integer('sym_embedding_size', 32, '')
tf.app.flags.DEFINE_bool('audio_z_score', False, 'If true, train and test on z-score of validation data')
tf.app.flags.DEFINE_integer('audio_deviation_max', 0, '')
tf.app.flags.DEFINE_integer('audio_context_radius', -1, 'Past and future context per training example')
tf.app.flags.DEFINE_integer('audio_nbands', 0, 'Number of bands per frame')
tf.app.flags.DEFINE_integer('audio_nchannels', 0, 'Number of channels per frame')
tf.app.flags.DEFINE_bool('feat_meas_phase', False, '')
tf.app.flags.DEFINE_bool('feat_meas_phase_cos', False, '')
tf.app.flags.DEFINE_bool('feat_meas_phase_sin', False, '')
tf.app.flags.DEFINE_bool('feat_beat_phase', False, '')
tf.app.flags.DEFINE_bool('feat_beat_phase_cos', False, '')
tf.app.flags.DEFINE_bool('feat_beat_phase_sin', False, '')
tf.app.flags.DEFINE_bool('feat_beat_diff', False, '')
tf.app.flags.DEFINE_bool('feat_beat_diff_next', False, '')
tf.app.flags.DEFINE_bool('feat_beat_abs', False, '')
tf.app.flags.DEFINE_bool('feat_time_diff', False, '')
tf.app.flags.DEFINE_bool('feat_time_diff_next', False, '')
tf.app.flags.DEFINE_bool('feat_time_abs', False, '')
tf.app.flags.DEFINE_bool('feat_prog_diff', False, '')
tf.app.flags.DEFINE_bool('feat_prog_abs', False, '')
tf.app.flags.DEFINE_bool('feat_diff_feet', False, '')
tf.app.flags.DEFINE_bool('feat_diff_aps', False, '')
tf.app.flags.DEFINE_integer('feat_beat_phase_nquant', 0, '')
tf.app.flags.DEFINE_integer('feat_beat_phase_max_nwraps', 0, '')
tf.app.flags.DEFINE_integer('feat_meas_phase_nquant', 0, '')
tf.app.flags.DEFINE_integer('feat_meas_phase_max_nwraps', 0, '')
tf.app.flags.DEFINE_string('feat_diff_feet_to_id_fp', '', '')
tf.app.flags.DEFINE_string('feat_diff_coarse_to_id_fp', '', '')
tf.app.flags.DEFINE_bool('feat_diff_dipstick', False, '')
tf.app.flags.DEFINE_string('feat_freetext_to_id_fp', '', '')
tf.app.flags.DEFINE_integer('feat_bucket_beat_diff_n', None, '')
tf.app.flags.DEFINE_float('feat_bucket_beat_diff_max', None, '')
tf.app.flags.DEFINE_integer('feat_bucket_time_diff_n', None, '')
tf.app.flags.DEFINE_float('feat_bucket_time_diff_max', None, '')

# Network params
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size for training')
tf.app.flags.DEFINE_integer('nunroll', 1, '')
tf.app.flags.DEFINE_string('cnn_filter_shapes', '', 'CSV 3-tuples of filter shapes (time, freq, n)')
tf.app.flags.DEFINE_string('cnn_pool', '', 'CSV 2-tuples of pool amounts (time, freq)')
tf.app.flags.DEFINE_integer('cnn_dim_reduction_size', -1, '')
tf.app.flags.DEFINE_float('cnn_dim_reduction_keep_prob', 1.0, '')
tf.app.flags.DEFINE_string('cnn_dim_reduction_nonlin', '', '')
tf.app.flags.DEFINE_string('rnn_cell_type', 'lstm', '')
tf.app.flags.DEFINE_integer('rnn_size', 0, '')
tf.app.flags.DEFINE_integer('rnn_nlayers', 0, '')
tf.app.flags.DEFINE_float('rnn_keep_prob', 1.0, '')
tf.app.flags.DEFINE_string('dnn_sizes', '', 'CSV sizes for dense layers')
tf.app.flags.DEFINE_float('dnn_keep_prob', 1.0, '')

# Training params
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

# Generate params
tf.app.flags.DEFINE_string('generate_fp', '', '')
tf.app.flags.DEFINE_string('generate_vocab_fp', '', '')

FLAGS = tf.app.flags.FLAGS
dtype = tf.float32

def main(_):
    assert FLAGS.experiment_dir
    do_train = FLAGS.nepochs != 0 and bool(FLAGS.train_txt_fp)
    do_valid = bool(FLAGS.valid_txt_fp)
    do_train_eval = do_train and do_valid
    do_eval = bool(FLAGS.test_txt_fp)
    do_generate = bool(FLAGS.generate_fp)

    # Load data
    print 'Loading data'
    train_data, valid_data, test_data = open_dataset_fps(FLAGS.train_txt_fp, FLAGS.valid_txt_fp, FLAGS.test_txt_fp)

    # Calculate validation metrics
    if FLAGS.audio_z_score:
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

    # Filter charts that are too short
    charts_train_len = len(charts_train)
    charts_train = filter(lambda x: x.get_nannotations() >= FLAGS.nunroll, charts_train)
    if len(charts_train) != charts_train_len:
        print '{} charts too small for training'.format(charts_train_len - len(charts_train))
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
        'meas_phase': FLAGS.feat_meas_phase,
        'meas_phase_cos': FLAGS.feat_meas_phase_cos,
        'meas_phase_sin': FLAGS.feat_meas_phase_sin,
        'beat_phase': FLAGS.feat_beat_phase,
        'beat_phase_cos': FLAGS.feat_beat_phase_cos,
        'beat_phase_sin': FLAGS.feat_beat_phase_sin,
        'beat_diff': FLAGS.feat_beat_diff,
        'beat_diff_next': FLAGS.feat_beat_diff_next,
        'beat_abs': FLAGS.feat_beat_abs,
        'time_diff': FLAGS.feat_time_diff,
        'time_diff_next': FLAGS.feat_time_diff_next,
        'time_abs': FLAGS.feat_time_abs,
        'prog_diff': FLAGS.feat_prog_diff,
        'prog_abs': FLAGS.feat_prog_abs,
        'diff_feet': FLAGS.feat_diff_feet,
        'diff_aps': FLAGS.feat_diff_aps,
        'beat_phase_nquant': FLAGS.feat_beat_phase_nquant,
        'beat_phase_max_nwraps': FLAGS.feat_beat_phase_max_nwraps,
        'meas_phase_nquant': FLAGS.feat_meas_phase_nquant,
        'meas_phase_max_nwraps': FLAGS.feat_meas_phase_max_nwraps,
        'diff_feet_to_id': diff_feet_to_id,
        'diff_coarse_to_id': diff_coarse_to_id,
        'freetext_to_id': freetext_to_id,
        'bucket_beat_diff_n': FLAGS.feat_bucket_beat_diff_n,
        'bucket_time_diff_n': FLAGS.feat_bucket_time_diff_n
    }
    nfeats = 0
    for feat in feats_config.values():
        if feat is None:
            continue
        if isinstance(feat, dict):
            nfeats += max(feat.values()) + 1
        else:
            nfeats += int(feat)
    nfeats += 1 if FLAGS.feat_beat_phase_max_nwraps > 0 else 0
    nfeats += 1 if FLAGS.feat_meas_phase_max_nwraps > 0 else 0
    nfeats += 1 if FLAGS.feat_bucket_beat_diff_n > 0 else 0
    nfeats += 1 if FLAGS.feat_bucket_time_diff_n > 0 else 0
    feats_config['diff_dipstick'] = FLAGS.feat_diff_dipstick
    feats_config['audio_time_context_radius'] = FLAGS.audio_context_radius
    feats_config['audio_deviation_max'] = FLAGS.audio_deviation_max
    feats_config['bucket_beat_diff_max'] = FLAGS.feat_bucket_beat_diff_max
    feats_config['bucket_time_diff_max'] = FLAGS.feat_bucket_time_diff_max
    feats_config_eval = dict(feats_config)
    feats_config_eval['audio_deviation_max'] = 0
    print 'Feature configuration (nfeats={}): {}'.format(nfeats, feats_config)

    # Create model config
    rnn_proj_init = tf.constant_initializer(0.0, dtype=dtype) if FLAGS.sym_rnn_pretrain_model_ckpt_fp else tf.uniform_unit_scaling_initializer(factor=1.0, dtype=dtype)
    model_config = {
        'nunroll': FLAGS.nunroll,
        'sym_in_type': FLAGS.sym_in_type,
        'sym_embedding_size': FLAGS.sym_embedding_size,
        'sym_out_type': FLAGS.sym_out_type,
        'sym_narrows': FLAGS.sym_narrows,
        'sym_narrowclasses': FLAGS.sym_narrowclasses,
        'other_nfeats': nfeats,
        'audio_context_radius': FLAGS.audio_context_radius,
        'audio_nbands': FLAGS.audio_nbands,
        'audio_nchannels': FLAGS.audio_nchannels,
        'cnn_filter_shapes': stride_csv_arg_list(FLAGS.cnn_filter_shapes, 3, int),
        'cnn_init': tf.uniform_unit_scaling_initializer(factor=1.43, dtype=dtype),
        'cnn_pool': stride_csv_arg_list(FLAGS.cnn_pool, 2, int),
        'cnn_dim_reduction_size': FLAGS.cnn_dim_reduction_size,
        'cnn_dim_reduction_init': tf.uniform_unit_scaling_initializer(factor=1.0, dtype=dtype),
        'cnn_dim_reduction_nonlin': FLAGS.cnn_dim_reduction_nonlin,
        'cnn_dim_reduction_keep_prob': FLAGS.cnn_dim_reduction_keep_prob,
        'rnn_proj_init': rnn_proj_init,
        'rnn_cell_type': FLAGS.rnn_cell_type,
        'rnn_size': FLAGS.rnn_size,
        'rnn_nlayers': FLAGS.rnn_nlayers,
        'rnn_init': tf.random_uniform_initializer(-5e-2, 5e-2, dtype=dtype),
        'nunroll': FLAGS.nunroll,
        'rnn_keep_prob': FLAGS.rnn_keep_prob,
        'dnn_sizes': stride_csv_arg_list(FLAGS.dnn_sizes, 1, int),
        'dnn_init': tf.uniform_unit_scaling_initializer(factor=1.15, dtype=dtype),
        'dnn_keep_prob': FLAGS.dnn_keep_prob,
        'grad_clip': FLAGS.grad_clip,
        'opt': FLAGS.opt,
    }
    print 'Model configuration: {}'.format(model_config)

    with tf.Graph().as_default(), tf.Session() as sess:
        if do_train:
            print 'Creating train model'
            with tf.variable_scope('model_ss', reuse=None):
                model_train = SymNet(mode='train', batch_size=FLAGS.batch_size, **model_config)

        if do_train_eval or do_eval:
            print 'Creating eval model'
            with tf.variable_scope('model_ss', reuse=do_train):
                eval_batch_size = FLAGS.batch_size
                if FLAGS.rnn_size > 0 and FLAGS.rnn_nlayers > 0:
                    eval_batch_size = 1
                model_eval = SymNet(mode='eval', batch_size=eval_batch_size, **model_config)
                model_early_stop_xentropy_avg = tf.train.Saver(tf.global_variables(), max_to_keep=None)
                model_early_stop_accuracy = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        if do_generate:
            print 'Creating generation model'
            with tf.variable_scope('model_ss', reuse=do_train):
                eval_batch_size = FLAGS.batch_size
                model_gen = SymNet(mode='gen', batch_size=1, **model_config)

        # Restore or init model
        model_saver = tf.train.Saver(tf.global_variables())
        if FLAGS.model_ckpt_fp:
            print 'Restoring model weights from {}'.format(FLAGS.model_ckpt_fp)
            model_saver.restore(sess, FLAGS.model_ckpt_fp)
        else:
            print 'Initializing model weights from scratch'
            sess.run(tf.global_variables_initializer())

            # Restore or init sym weights
            if FLAGS.sym_rnn_pretrain_model_ckpt_fp:
                print 'Restoring pretrained weights from {}'.format(FLAGS.sym_rnn_pretrain_model_ckpt_fp)
                var_list_old = filter(lambda x: 'nosym' not in x.name and 'cnn' not in x.name, tf.global_variables())
                pretrain_saver = tf.train.Saver(var_list_old)
                pretrain_saver.restore(sess, FLAGS.sym_rnn_pretrain_model_ckpt_fp)

        # Create summaries
        if do_train:
            summary_writer = tf.summary.FileWriter(FLAGS.experiment_dir, sess.graph)

            epoch_mean_xentropy = tf.placeholder(tf.float32, shape=[], name='epoch_mean_xentropy')
            epoch_mean_time = tf.placeholder(tf.float32, shape=[], name='epoch_mean_time')
            epoch_var_xentropy = tf.placeholder(tf.float32, shape=[], name='epoch_var_xentropy')
            epoch_var_time = tf.placeholder(tf.float32, shape=[], name='epoch_var_time')
            epoch_time_total = tf.placeholder(tf.float32, shape=[], name='epoch_time_total')
            epoch_summaries = tf.summary.merge([
                tf.summary.scalar('epoch_mean_xentropy', epoch_mean_xentropy),
                tf.summary.scalar('epoch_mean_time', epoch_mean_time),
                tf.summary.scalar('epoch_var_xentropy', epoch_var_xentropy),
                tf.summary.scalar('epoch_var_time', epoch_var_time),
                tf.summary.scalar('epoch_time_total', epoch_time_total)
            ])

            eval_metric_names = ['xentropy_avg', 'accuracy']
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
            train_nexamples = sum([chart.get_nannotations() for chart in charts_train])
            examples_per_batch = FLAGS.batch_size
            examples_per_batch *= model_train.out_nunroll
            batches_per_epoch = train_nexamples // examples_per_batch
            nbatches = FLAGS.nepochs * batches_per_epoch
            print '{} frames in data, {} batches per epoch, {} batches total'.format(train_nexamples, batches_per_epoch, nbatches)

            # Init epoch
            lr_summary = model_train.assign_lr(sess, FLAGS.lr)
            summary_writer.add_summary(lr_summary, 0)
            epoch_xentropies = []
            epoch_times = []

            batch_num = 0
            eval_best_xentropy_avg = float('inf')
            eval_best_accuracy = float('-inf')
            while FLAGS.nepochs < 0 or batch_num < nbatches:
                batch_time_start = time.time()
                syms, feats_other, feats_audio, targets, target_weights = model_train.prepare_train_batch(charts_train, **feats_config)
                feed_dict = {
                    model_train.syms: syms,
                    model_train.feats_other: feats_other,
                    model_train.feats_audio: feats_audio,
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
                    epoch_summary = sess.run(epoch_summaries, feed_dict={epoch_mean_xentropy: epoch_xentropy, epoch_mean_time: np.mean(epoch_times), epoch_var_xentropy: np.var(epoch_xentropies), epoch_var_time: np.var(epoch_times), epoch_time_total: np.sum(epoch_times)})
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

                    for eval_chart in charts_valid:
                        if model_eval.do_rnn:
                            state = sess.run(model_eval.initial_state)

                        neg_log_prob_sum = 0.0
                        correct_predictions_sum = 0.0
                        weight_sum = 0.0
                        for syms, syms_in, feats_other, feats_audio, targets, target_weights in model_eval.eval_iter(eval_chart, **feats_config_eval):
                            feed_dict = {
                                model_eval.syms: syms_in,
                                model_eval.feats_other: feats_other,
                                model_eval.feats_audio: feats_audio,
                                model_eval.targets: targets,
                                model_eval.target_weights: target_weights
                            }
                            if model_eval.do_rnn:
                                feed_dict[model_eval.initial_state] = state
                                xentropies, correct_predictions, state = sess.run([model_eval.neg_log_lhoods, model_eval.correct_predictions, model_eval.final_state], feed_dict=feed_dict)
                            else:
                                xentropies, correct_predictions = sess.run([model_eval.neg_log_lhoods, model_eval.correct_predictions], feed_dict=feed_dict)

                            neg_log_prob_sum += np.sum(xentropies)
                            correct_predictions_sum += np.sum(correct_predictions)
                            weight_sum += np.sum(target_weights)

                        assert int(weight_sum) == eval_chart.get_nannotations()
                        xentropy_avg = neg_log_prob_sum / weight_sum
                        accuracy = correct_predictions_sum / weight_sum

                        metrics['xentropy_avg'].append(xentropy_avg)
                        metrics['accuracy'].append(accuracy)

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

                    accuracy_mean = metrics['accuracy'][0]
                    if accuracy_mean > eval_best_accuracy:
                        print 'Accuracy {} better than previous {}'.format(accuracy_mean, eval_best_accuracy)
                        ckpt_fp = os.path.join(FLAGS.experiment_dir, 'onset_net_early_stop_accuracy')
                        model_early_stop_accuracy.save(sess, ckpt_fp, global_step=tf.contrib.framework.get_or_create_global_step())
                        eval_best_accuracy = accuracy_mean

                    print 'Done evaluating'

        if do_eval:
            print 'Evaluating...'

            metrics = defaultdict(list)

            for test_chart in charts_test:
                if model_eval.do_rnn:
                    state = sess.run(model_eval.initial_state)

                neg_log_prob_sum = 0.0
                correct_predictions_sum = 0.0
                weight_sum = 0.0
                for syms, syms_in, feats_other, feats_audio, targets, target_weights in model_eval.eval_iter(test_chart, **feats_config_eval):
                    feed_dict = {
                        model_eval.syms: syms_in,
                        model_eval.feats_other: feats_other,
                        model_eval.feats_audio: feats_audio,
                        model_eval.targets: targets,
                        model_eval.target_weights: target_weights
                    }
                    if model_eval.do_rnn:
                        feed_dict[model_eval.initial_state] = state
                        xentropies, correct_predictions, state = sess.run([model_eval.neg_log_lhoods, model_eval.correct_predictions, model_eval.final_state], feed_dict=feed_dict)
                    else:
                        xentropies, correct_predictions = sess.run([model_eval.neg_log_lhoods, model_eval.correct_predictions], feed_dict=feed_dict)

                    neg_log_prob_sum += np.sum(xentropies)
                    correct_predictions_sum += np.sum(correct_predictions)
                    weight_sum += np.sum(target_weights)

                assert int(weight_sum) == test_chart.get_nannotations()
                xentropy_avg = neg_log_prob_sum / weight_sum
                accuracy = correct_predictions_sum / weight_sum

                metrics['perplexity'].append(np.exp(xentropy_avg))
                metrics['xentropy_avg'].append(xentropy_avg)
                metrics['accuracy'].append(accuracy)

            metrics = {k: (np.mean(v), np.std(v), np.min(v), np.max(v)) for k, v in metrics.items()}
            copy_pasta = []
            for metric_name in ['xentropy_avg', 'perplexity', 'accuracy']:
                metric_stats = metrics[metric_name]
                copy_pasta += list(metric_stats)
                print '{}: {}'.format(metric_name, metric_stats)
            print 'COPY PASTA:'
            print ','.join([str(x) for x in copy_pasta])

        # TODO: This currently only works for VERY specific model (delta time LSTM)
        if do_generate:
            print 'Generating...'

            with open(FLAGS.generate_fp, 'r') as f:
                step_times = [float(x) for x in f.read().split(',')]

            with open(FLAGS.generate_vocab_fp, 'r') as f:
                idx_to_sym = {i:k for i, k in enumerate(f.read().splitlines())}

            def weighted_pick(weights):
                t = np.cumsum(weights)
                s = np.sum(weights)
                return(int(np.searchsorted(t, np.random.rand(1)*s)))

            state = sess.run(model_gen.initial_state)
            sym_prev = '<-1>'
            step_time_prev = step_times[0]
            seq_scores = []
            seq_sym_idxs = []
            seq_syms = []
            for step_time in step_times:
                delta_step_time = step_time - step_time_prev

                syms_in = np.array([[model_gen.arrow_to_encoding(sym_prev, 'bagofarrows')]], dtype=np.float32)
                feats_other = np.array([[[delta_step_time]]], dtype=np.float32)
                feats_audio = np.zeros((1, 1, 0, 0, 0), dtype=np.float32)
                feed_dict = {
                    model_gen.syms: syms_in,
                    model_gen.feats_other: feats_other,
                    model_gen.feats_audio: feats_audio,
                    model_gen.initial_state: state
                }

                scores, state = sess.run([model_gen.scores, model_gen.final_state], feed_dict=feed_dict)

                sym_idx = 0
                while sym_idx <= 1:
                    sym_idx = weighted_pick(scores)
                    if sym_idx <= 1:
                        print 'rare'
                sym_idx = sym_idx - 1 # remove special
                sym = idx_to_sym[sym_idx]

                seq_scores.append(scores)
                seq_sym_idxs.append(sym_idx)
                seq_syms.append(sym)

                sym_prev = sym
                step_time_prev = step_time

            with open(os.path.join(FLAGS.experiment_dir, 'seq.pkl'), 'wb') as f:
                pickle.dump((seq_scores, seq_sym_idxs, seq_syms), f)

if __name__ == '__main__':
    tf.app.run()
