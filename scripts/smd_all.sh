source var.sh

rm -rf ${SMDATA_DIR}/json_*
mkdir ${SMDATA_DIR}/json_raw
mkdir ${SMDATA_DIR}/json_filt

for COLL in fraxtil itg
do
	./smd_1_extract.sh ${COLL} --itg
done

for COLL in speirs sudzi
do
	./smd_1_extract.sh ${COLL} --itg
done

for COLL in fraxtil itg speirs sudzi
do
	./smd_2_filter.sh ${COLL}
	./smd_3_dataset.sh ${COLL} filt
done
