python ddc_server.py \
	--norm_pkl_fp server_aux/norm.pkl \
	--sp_ckpt_fp server_aux/model_sp-56000 \
	--ss_ckpt_fp server_aux/model_ss-23628 \
	--labels_txt_fp server_aux/labels_4_0123.txt \
	--sp_batch_size 256 \
	--out_dir ~/ddc_infer \
	--port=1337
