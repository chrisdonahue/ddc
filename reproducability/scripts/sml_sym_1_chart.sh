source sml_0_push.sh

python create_charts.py \
	${SMDATA_DIR}/json_filt/${1}_train.txt \
	${SMDATA_DIR}/json_filt/${1}_valid.txt \
	${SMDATA_DIR}/json_filt/${1}_test.txt \
	--out_dir=${SMDATA_DIR}/chart_sym/${1}/symbolic \
	--chart_type=sym
