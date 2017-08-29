source smd_0_push.sh

python dataset_json.py \
	${SMDATA_DIR}/json_filt/${1} \
	--splits=8,1,1 \
	--splitnames=train,valid,test \
	--shuffle \
	--shuffle_seed=1337

rm ${SMDATA_DIR}/json_filt/*${1}*.txt
for f in ${SMDATA_DIR}/json_filt/${1}/*train*.txt; do (cat "${f}"; echo) >> ${SMDATA_DIR}/json_filt/${1}_train.txt; done
for f in ${SMDATA_DIR}/json_filt/${1}/*valid*.txt; do (cat "${f}"; echo) >> ${SMDATA_DIR}/json_filt/${1}_valid.txt; done
for f in ${SMDATA_DIR}/json_filt/${1}/*test*.txt; do (cat "${f}"; echo) >> ${SMDATA_DIR}/json_filt/${1}_test.txt; done
for f in ${SMDATA_DIR}/json_filt/${1}/*.txt; do (cat "${f}"; echo) >> ${SMDATA_DIR}/json_filt/${1}.txt; done
