source sml_0_push.sh

EXP_DIR=/tmp/train
rm -rf ${EXP_DIR}
mkdir -p ${EXP_DIR}

python onset_train.py \
        --train_txt_fp=${SM_DIR}/data/chart_onset/${1}/mel80hop441/${1}_train.txt \
        --valid_txt_fp=${SM_DIR}/data/chart_onset/${1}/mel80hop441/${1}_valid.txt \
        --z_score \
        --audio_context_radius=7 \
        --audio_nbands=80 \
        --audio_nchannels=3 \
        --audio_select_channels=0,1,2 \
        --feat_diff_coarse_to_id_fp=${SM_DIR}/labels/${1}/diff_coarse_to_id.txt \
        --cnn_filter_shapes=7,3,10,3,3,20 \
        --cnn_pool=1,3,1,3 \
        --rnn_cell_type=lstm \
        --rnn_size=200 \
        --rnn_nlayers=0 \
        --rnn_nunroll=1 \
        --rnn_keep_prob=0.5 \
        --dnn_nonlin=sigmoid \
        --dnn_sizes=256,128 \
        --dnn_keep_prob=0.5 \
        --batch_size=256 \
        --weight_strategy=rect \
        --nobalanced_class \
        --exclude_onset_neighbors=2 \
        --exclude_pre_onsets \
        --exclude_post_onsets \
        --grad_clip=5.0 \
        --opt=sgd \
        --lr=0.1 \
        --lr_decay_rate=1.0 \
        --lr_decay_delay=0 \
        --nbatches_per_ckpt=4000 \
        --nbatches_per_eval=4000 \
        --nepochs=128 \
        --experiment_dir=${EXP_DIR} \
        --eval_window_type=hamming \
        --eval_window_width=5 \
        --eval_align_tolerance=2
