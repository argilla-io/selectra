DATA_DIR=gs://selectra-1
MODEL_NAME=selectra_small
HPARAMS=/home/recognai/selectra/hparams/hparams_small.json


python electra/run_pretraining.py --data-dir $DATA_DIR --model-name $MODEL_NAME --hparams $HPARAMS
