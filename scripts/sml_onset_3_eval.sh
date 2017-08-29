SM_DIR=${WORK}/sm
EXP_DIR=${SM_DIR}/trained/onset/17_01_23_00_cnn_diff_eval

pushd ../smlearn

python onset_train.py \
	--test_txt_fp=${SM_DIR}/data/chart_onset/fraxtil_test.txt \
	--model_ckpt_fp=${EXP_DIR}/onset_net_early_stop-88800 \
	--audio_context_radius=7 \
	--audio_nbands=80 \
	--audio_nchannels=3 \
	--feat_diff_feet_to_id_fp=${SM_DIR}/data/labels/fraxtil/diff_feet_to_id.txt \
	--cnn_filter_shapes=7,3,10,3,3,20 \
	--cnn_pool=1,3,1,3 \
	--rnn_cell_type=lstm \
	--rnn_size=200 \
	--rnn_nlayers=0 \
	--rnn_nunroll=1 \
	--rnn_keep_prob=0.5 \
	--dnn_sizes=256,128 \
	--dnn_keep_prob=0.5 \
	--batch_size=256 \
	--weight_strategy=rect \
	--exclude_onset_neighbors=2 \
	--exclude_pre_onsets \
	--exclude_post_onsets \
	--experiment_dir=${EXP_DIR} \
	--eval_hann_width=5 \
	--eval_align_tolerance=2
