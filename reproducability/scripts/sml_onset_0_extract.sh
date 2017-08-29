source sml_0_push.sh

python extract_feats.py \
	${SMDATA_DIR}/json_filt/${1}.txt \
	--out_dir=${SMDATA_DIR}/feats/${1}/mel80hop441 \
	--nhop=441 \
	--nffts=1024,2048,4096 \
	--log_scale
