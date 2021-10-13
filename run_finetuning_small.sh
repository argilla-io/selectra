python electra/run_finetuning.py --data-dir gs://selectra-1 --model-name selectra_small --hparams hparams/finetune_conll2002pos_small.json
python electra/run_finetuning.py --data-dir gs://selectra-1 --model-name selectra_small --hparams hparams/finetune_conll2002ner_small.json
python electra/run_finetuning.py --data-dir gs://selectra-1 --model-name selectra_small --hparams hparams/finetune_pawsx_small.json
python electra/run_finetuning.py --data-dir gs://selectra-1 --model-name selectra_small --hparams hparams/finetune_xnli_small.json
