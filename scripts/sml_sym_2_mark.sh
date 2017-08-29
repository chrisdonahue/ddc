source sml_0_push.sh

TRAIN_DIR=${SM_DIR}/trained/ngram/17_02_10_trained
rm -rf ${TRAIN_DIR}
mkdir -p ${TRAIN_DIR}
git rev-parse HEAD > ${TRAIN_DIR}/gitsha.txt

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
