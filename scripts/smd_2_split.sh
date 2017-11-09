python -m ddc.datasets.sm.split \
	${SM_DATA_DIR}/json \
	--splits=8,1,1 \
	--split_names=train,valid,test \
	--shuffle \
	--shuffle_seed=1337
