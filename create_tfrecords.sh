DISK=/home/recognai/disk
CORPUS=$DISK/oscar_files
OUTPUT=$DISK/tfrecords
VOCAB=$DISK/vocab_50000_0.5/vocab.txt

python /home/recognai/selectra/electra/build_pretraining_dataset.py \
	--corpus-dir $CORPUS \
	--vocab-file $VOCAB \
	--output-dir $OUTPUT \
	--blanks-separate-docs True \
	--max-seq-length 512 \
	--num-processes 8 \
	--no-lower-case
