source smd_0_push.sh

python filter_json.py \
	${SMDATA_DIR}/json_raw/${1} \
	${SMDATA_DIR}/json_filt${2}/${1} \
	--chart_types=dance-single \
	--chart_difficulties=Beginner,Easy,Medium,Hard,Challenge \
	--min_chart_feet=1 \
	--max_chart_feet=-1 \
	--substitutions=M,0,4,2 \
	--arrow_types=1,2,3 \
	--max_jump_size=-1 \
	--remove_zeros \
	--permutations=0123,3120,0213,3210
