DDC=ddc-1.0/infer
sudo docker run --runtime=nvidia \
	-p 1337:1337 \
	-e NVIDIA_VISIBLE_DEVICES=0 \
	--env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
	-v ~/audio:/audio \
	me/ddc \
  python ${DDC}/ddc_server.py \
  	--norm_pkl_fp ${DDC}/server_aux/norm.pkl \
	--sp_ckpt_fp ${DDC}/server_aux/model_sp-56000 \
	--ss_ckpt_fp ${DDC}/server_aux/model_ss-23628 \
	--labels_txt_fp ${DDC}/server_aux/labels_4_0123.txt \
	--sp_batch_size 256 \
	--out_dir ${DDC}/ddc_infer \
	--port=1337
