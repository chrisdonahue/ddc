python -m ddc.datasets.sm.filter \
	${SM_DATA_DIR}/json/*.txt \
	--chart_types=dance-single \
	--chart_difficulties=Beginner,Easy,Medium,Hard,Challenge \
	--min_chart_feet=1 \
	--max_chart_feet=-1 \
	--substitutions=M,0,4,2 \
	--arrow_types=1,2,3 \
	--max_jump_size=-1 \
	--remove_zeros \
	--permutations=0123,3120,0213,3210
