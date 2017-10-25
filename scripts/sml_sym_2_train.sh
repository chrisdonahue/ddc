source sml_0_push.sh

EXP_DIR=/tmp/train_sym
rm -rf ${EXP_DIR}
mkdir -p ${EXP_DIR}

python sym_train.py \
        --train_txt_fp=${SM_DIR}/data/chart_sym/${1}/symbolic/${1}_train.txt \
        --valid_txt_fp=${SM_DIR}/data/chart_sym/${1}/symbolic/${1}_valid.txt \
        --sym_in_type=bagofarrows \
        --sym_out_type=onehot \
        --sym_narrows=4 \
        --sym_narrowclasses=4 \
        --sym_embedding_size=0 \
        --feat_time_diff \
        --feat_time_diff_next \
        --batch_size=64 \
        --nunroll=64 \
        --cnn_filter_shapes= \
        --cnn_pool= \
        --rnn_cell_type=lstm \
        --rnn_size=128 \
        --rnn_nlayers=2 \
        --rnn_keep_prob=0.5 \
        --dnn_sizes= \
        --dnn_keep_prob=0.5 \
        --grad_clip=5.0 \
        --opt=sgd \
        --lr=1.0 \
        --lr_decay_rate=1.0 \
        --lr_decay_delay=10 \
        --nbatches_per_ckpt=200 \
        --nbatches_per_eval=200 \
        --nepochs=1000 \
        --experiment_dir=${EXP_DIR}
