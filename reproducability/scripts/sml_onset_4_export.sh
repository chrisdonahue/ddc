source sml_0_push.sh

SM_DIR=/work/03860/cdonahue/maverick/sm

EXP_DIR=${SM_DIR}/trained/onset/17_02_05_00_fraxnew_cnn_export

python onset_train.py \
	--train_txt_fp=${SM_DIR}/data/chart_onset/fraxtil/mel80nfft3/fraxtil_train.txt \
	--valid_txt_fp=${SM_DIR}/data/chart_onset/fraxtil/mel80nfft3/fraxtil_valid.txt \
	--z_score \
	--test_txt_fp=${SM_DIR}/data/chart_onset/fraxtil/mel80nfft3/fraxtil_test.txt \
	--model_ckpt=${EXP_DIR}/onset_net_early_stop_auprc-312000 \
	--export_feat_name=cnn_1 \
	--audio_context_radius=7 \
	--audio_nbands=80 \
	--audio_nchannels=3 \
	--audio_select_channels=0,1,2 \
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
	--experiment_dir=${EXP_DIR}
