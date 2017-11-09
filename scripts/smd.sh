rm -rf ${SM_DATA_DIR}/json

./smd_1_extract.sh
./smd_2_split.sh
./smd_3_filter.sh
./smd_4_analyze.sh
