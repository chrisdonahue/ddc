source sml_0_push.sh

python create_charts.py \
	${SMDATA_DIR}/json_filt${3}/${1}_train.txt \
	${SMDATA_DIR}/json_filt${3}/${1}_valid.txt \
	${SMDATA_DIR}/json_filt${3}/${1}_test.txt \
	--out_dir=${SMDATA_DIR}/chart_sym/${1}/${2}${3} \
	--chart_type=sym \
	--frame_rate=44100,441 \
	--feats_dir=${SMDATA_DIR}/feats/${1}/${2}
