source sml_0_push.sh

python create_charts.py \
	${SMDATA_DIR}/json_filt/${1}_train.txt \
	${SMDATA_DIR}/json_filt/${1}_valid.txt \
	${SMDATA_DIR}/json_filt/${1}_test.txt \
	--out_dir=${SMDATA_DIR}/chart_onset/${1}/mel80hop441 \
	--chart_type=onset \
	--frame_rate=44100,441 \
	--feats_dir=${SMDATA_DIR}/feats/${1}/mel80hop441
