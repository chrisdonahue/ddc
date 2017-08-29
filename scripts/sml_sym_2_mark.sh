source sml_0_push.sh

TRAIN_DIR=/tmp/ngram
rm -rf ${TRAIN_DIR}
mkdir -p ${TRAIN_DIR}

python ngram.py \
	${SM_DIR}/data/json_filt/${1}_train.txt \
	${TRAIN_DIR}/model_${2}.pkl \
	--k=${2} \
	--task=train

python ngram.py \
	${SM_DIR}/data/json_filt/${1}_test.txt \
	${TRAIN_DIR}/model_${2}.pkl \
	--k=${2} \
	--task=eval
