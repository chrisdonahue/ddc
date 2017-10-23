import math
import random

import tensorflow as tf
import numpy as np

from util import np_pad

dtype = tf.float32
np_dtype = dtype.as_numpy_dtype

# https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/model.py
class SymNet:
    def __init__(self,
                 mode,
                 batch_size,
                 nunroll,
                 sym_in_type,
                 sym_embedding_size,
                 sym_out_type,
                 sym_narrows,
                 sym_narrowclasses,
                 other_nfeats,
                 audio_context_radius,
                 audio_nbands,
                 audio_nchannels,
                 cnn_filter_shapes,
                 cnn_init,
                 cnn_pool,
                 cnn_dim_reduction_size,
                 cnn_dim_reduction_init,
                 cnn_dim_reduction_nonlin,
                 cnn_dim_reduction_keep_prob,
                 rnn_proj_init,
                 rnn_cell_type,
                 rnn_size,
                 rnn_nlayers,
                 rnn_init,
                 rnn_keep_prob,
                 dnn_sizes,
                 dnn_init,
                 dnn_keep_prob,
                 grad_clip=0.0,
                 opt='sgd',
                 dtype=tf.float32):
        if audio_context_radius >= 0:
            audio_context_len = audio_context_radius * 2 + 1
        else:
            audio_context_len = 0

        do_cnn = audio_context_len > 0 and len(cnn_filter_shapes) > 0
        do_rnn = rnn_size > 0 and rnn_nlayers > 0
        do_dnn = len(dnn_sizes) > 0

        in_nunroll = nunroll
        out_nunroll = nunroll if do_rnn else 1

        _IN_SPECIAL = ['<-{}>'.format(i + 1) for i in range(1 if do_rnn else in_nunroll)[::-1]]

        # input sequence
        if sym_in_type == 'onehot':
            in_len = len(_IN_SPECIAL) + int(math.pow(sym_narrowclasses, sym_narrows))
            syms_unrolled = tf.placeholder(tf.int64, shape=[batch_size, in_nunroll], name='syms')
        elif sym_in_type == 'bagofarrows':
            in_len = len(_IN_SPECIAL) + (sym_narrows * sym_narrowclasses)
            syms_unrolled = tf.placeholder(dtype, shape=[batch_size, in_nunroll, in_len], name='syms')
        else:
            raise NotImplementedError()

        # other/audio feats
        feats_other_unrolled = tf.placeholder(dtype, shape=[batch_size, in_nunroll, other_nfeats], name='feats_other')
        feats_audio_unrolled = tf.placeholder(dtype, shape=[batch_size, in_nunroll, audio_context_len, audio_nbands, audio_nchannels], name='feats_audio')

        # targets
        if mode != 'gen':
            if sym_out_type == 'onehot':
                targets_unrolled = tf.placeholder(tf.int64, shape=[batch_size, out_nunroll], name='target_seq')
            else:
                raise NotImplementedError()
        if mode == 'train':
            target_weights_unrolled = tf.ones([batch_size, out_nunroll], dtype)
        elif mode == 'eval':
            target_weights_unrolled = tf.placeholder(dtype, shape=[batch_size, out_nunroll], name='target_weights')

        # reshape inputs to remove nunroll dim; will briefly restore later during RNN if necessary
        if sym_in_type == 'onehot':
            syms = tf.reshape(syms_unrolled, shape=[batch_size * in_nunroll])
        elif sym_in_type == 'bagofarrows':
            syms = tf.reshape(syms_unrolled, shape=[batch_size * in_nunroll, in_len])
        feats_audio = tf.reshape(feats_audio_unrolled, shape=[batch_size * in_nunroll, audio_context_len, audio_nbands, audio_nchannels])
        feats_other = tf.reshape(feats_other_unrolled, shape=[batch_size * in_nunroll, other_nfeats])
        if mode != 'gen':
            targets = tf.reshape(targets_unrolled, shape=[batch_size * out_nunroll])
            target_weights = tf.reshape(target_weights_unrolled, shape=[batch_size * out_nunroll])

        # output represenetation
        if sym_out_type == 'onehot':
            out_len = len(_IN_SPECIAL) + int(math.pow(sym_narrowclasses, sym_narrows))
        elif sym_out_type == 'bagofarrows':
            out_len = sym_narrows * sym_narrowclasses
        else:
            raise NotImplementedError()

        # embed symbolic features
        with tf.device('/cpu:0'):
            # embed
            if sym_embedding_size > 0:
                with tf.variable_scope('sym_embedding'):
                    if sym_in_type == 'onehot':
                        embed_w = tf.get_variable('W', [in_len, sym_embedding_size])
                        feats_sym = tf.nn.embedding_lookup(embed_w, syms)
                    elif sym_in_type == 'bagofarrows':
                        embed_w = tf.get_variable('W', [in_len, sym_embedding_size])
                        embed_b = tf.get_variable('b', [sym_embedding_size])
                        feats_sym = tf.nn.bias_add(tf.matmul(syms, embed_w), embed_b)
                nfeats_sym = sym_embedding_size
            # noembed
            else:
                if sym_in_type == 'onehot':
                    feats_sym = tf.one_hot(syms, in_len)
                elif sym_in_type == 'bagofarrows':
                    feats_sym = syms
                nfeats_sym = in_len

        # CNN audio features
        cnn_output = feats_audio
        if do_cnn:
            layer_last = feats_audio
            nfilt_last = audio_nchannels
            for i, ((ntime, nband, nfilt), (ptime, pband)) in enumerate(zip(cnn_filter_shapes, cnn_pool)):
                layer_name = 'cnn_{}'.format(i)
                with tf.variable_scope(layer_name):
                    filters = tf.get_variable('filters', [ntime, nband, nfilt_last, nfilt], initializer=cnn_init, dtype=dtype)
                    biases = tf.get_variable('biases', [nfilt], initializer=tf.constant_initializer(0.1), dtype=dtype)
                conv = tf.nn.conv2d(layer_last, filters, [1, 1, 1, 1], padding='VALID')
                biased = tf.nn.bias_add(conv, biases)
                convolved = tf.nn.relu(biased)

                pool_shape = [1, ptime, pband, 1]
                pooled = tf.nn.max_pool(convolved, ksize=pool_shape, strides=pool_shape, padding='SAME')
                print '{}: {}'.format(layer_name, pooled.get_shape())

                # TODO: CNN dropout?

                layer_last = pooled
                nfilt_last = nfilt

            cnn_output = layer_last

        # Flatten CNN
        nfeats_conv = reduce(lambda x, y: x * y, [int(x) for x in cnn_output.get_shape()[-3:]])
        feats_conv = tf.reshape(cnn_output, shape=[batch_size * in_nunroll, nfeats_conv])
        print 'feats_sym: {}'.format(feats_sym.get_shape())
        print 'feats_cnn: {}'.format(feats_conv.get_shape())
        print 'feats_other: {}'.format(feats_other.get_shape())

        # Reduce CNN dimensionality
        if cnn_dim_reduction_size >= 0:
            with tf.variable_scope('cnn_dim_reduction'):
                cnn_dim_reduction_W = tf.get_variable('W', [nfeats_conv, cnn_dim_reduction_size], initializer=cnn_dim_reduction_init, dtype=dtype)
                cnn_dim_reduction_b = tf.get_variable('b', [cnn_dim_reduction_size], initializer=tf.constant_initializer(0.0), dtype=dtype)

                nfeats_conv = cnn_dim_reduction_size
                feats_conv = tf.nn.bias_add(tf.matmul(feats_conv, cnn_dim_reduction_W), cnn_dim_reduction_b)
                if mode == 'train' and cnn_dim_reduction_keep_prob < 1.0:
                    feats_conv = tf.nn.dropout(feats_conv, cnn_dim_reduction_keep_prob)

                if cnn_dim_reduction_nonlin == 'sigmoid':
                    feats_conv = tf.nn.sigmoid(feats_conv)
                elif cnn_dim_reduction_nonlin == 'tanh':
                    feats_conv = tf.nn.tanh(feats_conv)
                elif cnn_dim_reduction_nonlin == 'relu':
                    feats_conv = tf.nn.relu(feats_conv)

                print 'feats_cnn_reduced: {}'.format(feats_conv.get_shape())

        # Project to RNN size
        rnn_output_inspect = None
        if do_rnn:
            nfeats_nosym = nfeats_conv + other_nfeats
            # TODO: should this be on cpu? (batch_size, nunroll, sym_embedding_size + nfeats)
            feats_nosym = tf.concat(1, [feats_conv, feats_other])

            with tf.variable_scope('rnn_proj'):
                rnn_proj_sym_w = tf.get_variable('W', [nfeats_sym, rnn_size], initializer=rnn_proj_init, dtype=dtype)
                rnn_proj_nosym_w = tf.get_variable('nosym_W', [nfeats_nosym, rnn_size], initializer=rnn_proj_init, dtype=dtype)
                rnn_proj_b = tf.get_variable('b', [rnn_size], initializer=tf.constant_initializer(0.0), dtype=dtype)

            rnn_inputs_sym = tf.matmul(feats_sym, rnn_proj_sym_w)
            rnn_inputs_nosym = tf.matmul(feats_nosym, rnn_proj_nosym_w)
            rnn_inputs_prebias = tf.add(rnn_inputs_sym, rnn_inputs_nosym)
            rnn_inputs = tf.nn.bias_add(rnn_inputs_prebias, rnn_proj_b)
            rnn_inputs = tf.reshape(rnn_inputs, shape=[batch_size, nunroll, rnn_size])
            rnn_inputs = tf.split(1, nunroll, rnn_inputs)
            rnn_inputs = [tf.squeeze(input_, [1]) for input_ in rnn_inputs]

            if rnn_cell_type == 'rnn':
                cell_fn = tf.nn.rnn_cell.BasicRNNCell
            elif rnn_cell_type == 'gru':
                cell_fn = tf.nn.rnn_cell.GRUCell
            elif rnn_cell_type == 'lstm':
                cell_fn = tf.nn.rnn_cell.BasicLSTMCell
            else:
                raise NotImplementedError()
            cell = cell_fn(rnn_size)

            if mode == 'train' and rnn_keep_prob < 1.0:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=rnn_keep_prob)

            if rnn_nlayers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * rnn_nlayers)

            initial_state = cell.zero_state(batch_size, dtype)

            # RNN
            # TODO: weight init
            with tf.variable_scope('rnn_unroll'):
                state = initial_state
                outputs = []
                for i in xrange(nunroll):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(rnn_inputs[i], state)
                    outputs.append(cell_output)
                final_state = state

            rnn_output_inspect = tf.concat(1, outputs)

            rnn_output = tf.reshape(rnn_output_inspect, [batch_size * nunroll, rnn_size])
            rnn_output_size = rnn_size
        else:
            nfeats_tot = nfeats_sym + nfeats_conv + other_nfeats
            feats_all = tf.concat(1, [feats_sym, feats_conv, feats_other])
            rnn_output = tf.reshape(feats_all, shape=[batch_size, in_nunroll * nfeats_tot])
            rnn_output_size = in_nunroll * nfeats_tot
        print 'rnn_output: {}'.format(rnn_output.get_shape())

        # Dense NN
        dnn_output = rnn_output
        dnn_output_size = rnn_output_size
        dnn_output_inspect = None
        if do_dnn:
            last_layer = rnn_output
            last_layer_size = rnn_output_size
            for i, layer_size in enumerate(dnn_sizes):
                layer_name = 'dnn_{}'.format(i)
                with tf.variable_scope(layer_name):
                    dnn_w = tf.get_variable('W', shape=[last_layer_size, layer_size], initializer=dnn_init, dtype=dtype)
                    dnn_b = tf.get_variable('b', shape=[layer_size], initializer=tf.constant_initializer(0.0), dtype=dtype)
                projected = tf.nn.bias_add(tf.matmul(last_layer, dnn_w), dnn_b)
                # TODO: argument nonlinearity, change bias to 0.1 if relu
                last_layer = tf.nn.sigmoid(projected)
                if mode == 'train' and dnn_keep_prob < 1.0:
                    last_layer = tf.nn.dropout(last_layer, dnn_keep_prob)
                last_layer_size = layer_size
                print '{}: {}'.format(layer_name, last_layer.get_shape())

            dnn_output = last_layer
            dnn_output_size = last_layer_size

            dnn_output_inspect = dnn_output

        with tf.variable_scope('sym_rnn_output'):
            if sym_out_type == 'onehot':
                # Output projection
                softmax_w = tf.get_variable('softmax_w', [dnn_output_size, out_len])
                softmax_b = tf.get_variable('softmax_b', [out_len])

                # Calculate logits, shape is [batch_size * nunroll, iolen]
                logits = tf.nn.bias_add(tf.matmul(dnn_output, softmax_w), softmax_b)
                scores = tf.nn.softmax(logits)
                predictions = tf.argmax(scores, 1)

                # Reshape only for inspection
                scores_inspect = tf.reshape(scores, [batch_size, out_nunroll, out_len])
                predictions_inspect = tf.reshape(predictions, [batch_size, out_nunroll])
            elif sym_out_type == 'bagofarrows':
                raise NotImplementedError()
                softmax_w = tf.get_variable('softmax_w', [rnn_size, out_len])
                softmax_b = tf.get_variable('softmax_b', [out_len])

                # Concat outputs to (batch_size, nunroll * rnn_size)
                output = tf.concat(1, outputs)
                # TODO: remove this once verify that it's unnecessary
                output = tf.reshape(output, [batch_size, out_nunroll, rnn_size])
                # Reshape outputs to (batch_size * nunroll, rnn_size) for matmul
                output = tf.reshape(output, [batch_size * out_nunroll, rnn_size])

                # Calculate logits, shape is [batch_size * nunroll, sym_narrows * sym_narrowclasses]
                logits = tf.nn.bias_add(tf.matmul(output, softmax_w), softmax_b)
                logits = tf.reshape(logits, [batch_size * nunroll, sym_narrows, sym_narrowclasses])
                arrows = tf.nn.softmax(logits)
                score = tf.reshape(arrows, [batch_size * nunroll, out_len])

        if mode != 'gen':
            neg_log_lhoods = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
            print neg_log_lhoods.get_shape()
            """
            # calculate cross-entropy, result is [batch_size * nunroll] where each entry is an unscaled neg ln prob
            neg_log_lhoods = tf.nn.seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(targets, [-1])],
                [tf.reshape(target_weights, [-1])])
            print neg_log_lhoods.get_shape()
            # weights may not be all 1s so we have to sum them
            #avg_neg_log_lhood = tf.reduce_sum(neg_log_lhoods) / (batch_size * tf.reduce_sum(target_weights))
            """
            if mode == 'train':
                # weights are all 1s so we can take mean
                avg_neg_log_lhood = tf.reduce_mean(neg_log_lhoods)
            elif mode == 'eval':
                neg_log_lhoods = tf.multiply(neg_log_lhoods, target_weights)
                avg_neg_log_lhood = tf.reduce_sum(neg_log_lhoods) / tf.reduce_sum(target_weights)
            neg_log_lhoods_inspect = tf.reshape(neg_log_lhoods, shape=[batch_size, out_nunroll])

        if mode != 'gen':
            correct_predictions = tf.cast(tf.equal(predictions, targets), dtype)
            if mode == 'train':
                accuracy = tf.reduce_mean(correct_predictions)
            if mode == 'eval':
                correct_predictions = tf.multiply(correct_predictions, target_weights)
                accuracy = tf.reduce_sum(correct_predictions) / tf.reduce_sum(target_weights)
            correct_predictions_inspect = tf.reshape(correct_predictions, shape=[batch_size, out_nunroll])

        if mode == 'train':
            lr = tf.Variable(0.0, trainable=False)
            self._lr = lr
            self._lr_summary = tf.summary.scalar('learning_rate', self._lr)

            tvars = tf.trainable_variables()
            grads = tf.gradients(avg_neg_log_lhood, tvars)
            if grad_clip > 0.0:
                grads, _ = tf.clip_by_global_norm(grads, grad_clip)

            if opt == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif opt == 'adam':
                optimizer = tf.train.AdamOptimizer(lr)
            else:
                raise NotImplementedError()

            train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())

        self.syms = syms_unrolled
        self.feats_other = feats_other_unrolled
        self.feats_audio = feats_audio_unrolled
        self.rnn_output = rnn_output_inspect
        self.dnn_output = dnn_output_inspect
        self.scores = scores_inspect
        self.predictions = predictions_inspect
        if mode != 'gen':
            self.neg_log_lhoods = neg_log_lhoods_inspect
            self.avg_neg_log_lhood = avg_neg_log_lhood
            self.correct_predictions = correct_predictions_inspect
            self.accuracy = accuracy
            self.targets = targets_unrolled
            self.target_weights = target_weights_unrolled
        if mode == 'train':
            self.train_op = train_op
        if mode != 'train' and do_rnn:
            self.initial_state = initial_state
            self.final_state = final_state

        self.mode = mode
        self.batch_size = batch_size
        self.in_nunroll = in_nunroll
        self.out_nunroll = out_nunroll
        self.sym_in_type = sym_in_type
        self.sym_out_type = sym_out_type
        self.sym_narrows = sym_narrows
        self.sym_narrowclasses = sym_narrowclasses
        self.do_rnn = do_rnn
        self._IN_SPECIAL = _IN_SPECIAL
        self._arrow_to_encoding = {}

    def assign_lr(self, sess, lr_new):
        assert self.mode == 'train'
        sess.run(tf.assign(self._lr, lr_new))
        return sess.run(self._lr_summary)

    def arrow_to_encoding(self, arrow, encoding):
        if encoding in self._arrow_to_encoding and arrow in self._arrow_to_encoding[encoding]:
            return self._arrow_to_encoding[encoding][arrow]

        if encoding == 'onehot':
            if arrow in self._IN_SPECIAL:
                result = self._IN_SPECIAL.index(arrow)
            else:
                multipliers = [int(math.pow(self.sym_narrowclasses, self.sym_narrows - i - 1)) for i in xrange(self.sym_narrows)]
                result = len(self._IN_SPECIAL) + sum([multipliers[i] * int(arrowclass) for i, arrowclass in enumerate(arrow)])
        elif encoding == 'bagofarrows':
            in_len = len(self._IN_SPECIAL) + (self.sym_narrows * self.sym_narrowclasses)
            result = np.zeros(in_len, dtype=np_dtype)
            if arrow in self._IN_SPECIAL:
                result[self._IN_SPECIAL.index(arrow)] = 1.0
            else:
                for i, arrowclass in enumerate(arrow):
                    result[len(self._IN_SPECIAL) + (i * self.sym_narrowclasses) + int(arrowclass)] = 1.0

        if encoding not in self._arrow_to_encoding:
            self._arrow_to_encoding[encoding] = {}
        self._arrow_to_encoding[encoding][arrow] = result
        return result

    def prepare_train_batch(self, charts, diff_feet, diff_aps, **seq_feat_kwargs):
        batch_syms_input = []
        batch_syms_target = []
        batch_feats_other = []
        batch_feats_audio = []
        for _ in xrange(self.batch_size):
            chart = charts[random.randint(0, len(charts) - 1)]

            syms, feats_other, feats_audio = chart.get_random_subsequence(self.in_nunroll, **seq_feat_kwargs)
            assert len(syms) == self.in_nunroll + 1
            input_syms = [self.arrow_to_encoding(sym, self.sym_in_type) for sym in syms[:-1]]
            target_syms = [self.arrow_to_encoding(sym, self.sym_out_type) for sym in syms[(self.in_nunroll - self.out_nunroll) + 1:]]

            if diff_feet:
                feats_other = np.append(feats_other, chart.get_foot_difficulty() * np.ones((self.in_nunroll, 1), dtype=np_dtype), axis=1)
            if diff_aps:
                feats_other = np.append(feats_other, chart.get_annotations_per_second() * np.ones((self.in_nunroll, 1), dtype=np_dtype), axis=1)

            batch_syms_input.append(input_syms)
            batch_syms_target.append(target_syms)
            batch_feats_other.append(feats_other)
            batch_feats_audio.append(feats_audio)

        if self.sym_in_type == 'onehot':
            batch_syms_input = np.array(batch_syms_input, dtype=np.int64)
        elif self.sym_in_type == 'bagofarrows':
            batch_syms_input = np.array(batch_syms_input, dtype=np_dtype)
        batch_feats_other = np.array(batch_feats_other, dtype=np_dtype)
        batch_feats_audio = np.array(batch_feats_audio, dtype=np_dtype)
        if self.sym_out_type == 'onehot':
            batch_syms_target = np.array(batch_syms_target, dtype=np.int64)
        batch_target_weights = np.ones((self.batch_size, self.out_nunroll), dtype=np_dtype)

        return batch_syms_input, batch_feats_other, batch_feats_audio, batch_syms_target, batch_target_weights

    def eval_iter(self, eval_chart, diff_feet, diff_aps, **seq_feat_kwargs):
        pad_ax_to_n = lambda x, a, n: np_pad(x, n, axis=a)

        subseq_len = self.in_nunroll
        subseq_start = 0
        if self.do_rnn:
            subseq_stride = subseq_len
            subseq_end = eval_chart.get_nannotations()
        else:
            subseq_stride = self.batch_size
            subseq_end = eval_chart.get_nannotations() - subseq_len

        for i in xrange(subseq_start, subseq_end, subseq_stride):
            batch_syms = []
            batch_syms_inputs = []
            batch_feats_other = []
            batch_feats_audio = []
            batch_syms_targets = []
            for j in xrange(self.batch_size):
                if i + j >= eval_chart.get_nannotations():
                    break
                syms, feats_other, feats_audio = eval_chart.get_subsequence(i + j, subseq_len, **seq_feat_kwargs)

                validlen = feats_other.shape[0]

                input_syms = [self.arrow_to_encoding(sym, self.sym_in_type) for sym in syms[:-1]]
                if self.sym_in_type == 'onehot':
                    input_syms = np.array(input_syms, dtype=np.int64)
                elif self.sym_in_type == 'bagofarrows':
                    input_syms = np.array(input_syms, dtype=np_dtype)

                target_syms = [self.arrow_to_encoding(sym, self.sym_out_type) for sym in syms[(self.in_nunroll - self.out_nunroll) + 1:]]
                if self.sym_out_type == 'onehot':
                    target_syms = np.array(target_syms, dtype=np.int64)

                if diff_feet:
                    feats_other = np.append(feats_other, eval_chart.get_foot_difficulty() * np.ones((validlen, 1), dtype=np_dtype), axis=1)
                if diff_aps:
                    feats_other = np.append(feats_other, eval_chart.get_annotations_per_second() * np.ones((validlen, 1), dtype=np_dtype), axis=1)

                batch_syms.append(syms)
                batch_syms_inputs.append(input_syms)
                batch_feats_other.append(feats_other)
                batch_feats_audio.append(feats_audio)
                batch_syms_targets.append(target_syms)

            batch_syms_inputs = np.array(batch_syms_inputs)
            batch_feats_other = np.array(batch_feats_other)
            batch_feats_audio = np.array(batch_feats_audio)
            batch_syms_targets = np.array(batch_syms_targets)
            batch_target_weights = np.ones_like(batch_syms_targets, dtype=np_dtype)

            if self.do_rnn:
                batch_syms_inputs = pad_ax_to_n(batch_syms_inputs, 1, self.in_nunroll)
                batch_feats_other = pad_ax_to_n(batch_feats_other, 1, self.in_nunroll)
                batch_feats_audio = pad_ax_to_n(batch_feats_audio, 1, self.in_nunroll)
                batch_syms_targets = pad_ax_to_n(batch_syms_targets, 1, self.in_nunroll)
                batch_target_weights = pad_ax_to_n(batch_target_weights, 1, self.in_nunroll)
            else:
                batch_syms_inputs = pad_ax_to_n(batch_syms_inputs, 0, self.batch_size)
                batch_feats_other = pad_ax_to_n(batch_feats_other, 0, self.batch_size)
                batch_feats_audio = pad_ax_to_n(batch_feats_audio, 0, self.batch_size)
                batch_syms_targets = pad_ax_to_n(batch_syms_targets, 0, self.batch_size)
                batch_target_weights = pad_ax_to_n(batch_target_weights, 0, self.batch_size)

            yield batch_syms, batch_syms_inputs, batch_feats_other, batch_feats_audio, batch_syms_targets, batch_target_weights
