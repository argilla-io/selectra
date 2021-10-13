TF_CHECKPOINT_PATH=../models/medium
CONFIG_FILE=./config_medium.json
PYTORCH_DUMP_PATH=../models/medium/pytorch_model
VOCAB_FILE=../vocab/vocab_50000_0.5/vocab.txt

python convert_electra_original_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=$TF_CHECKPOINT_PATH --config_file=$CONFIG_FILE --pytorch_dump_path=$PYTORCH_DUMP_PATH/pytorch_model.bin --discriminator_or_generator=discriminator
cp $VOCAB_FILE $PYTORCH_DUMP_PATH/vocab.txt
cp $CONFIG_FILE $PYTORCH_DUMP_PATH/config.json
