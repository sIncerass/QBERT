# How to do fine-controlled quantization

Notice the extra argument is config.json

quantization of CoLA:
```bash
export CUDA_VISIBLE_DEVICES=6,7; python run_classifier.py --task_name CoLA --do_train --do_eval --do_lower_case --data_dir ../data/glue_data/CoLA --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 1 --output_dir ../results/ --config_dir=new_config.json
```
quantization of SQUAD:
```bash
export CUDA_VISIBLE_DEVICES=6,7; python run_squad.py  --bert_model bert-base-uncased  --do_train  --do_predict  --do_lower_case  --train_file $SQUAD_DIR/train-v1.1.json  --predict_file $SQUAD_DIR/dev-v1.1.json  --train_batch_size 12  --learning_rate 3e-5  --num_train_epochs 2.0  --max_seq_length 384  --doc_stride 128  --output_dir ../results/ --config_dir=new_config.json
```
 
Please take a look at example_config.json for details of how to control QuantLinear layers.

For fine-controlled per QuantLinear control, directly manipulate the bits in json is good enough.

For coarse-grained per Block control, a script is provided to generate block-wise sync. Please see the detail of the script for further usage.
```bash
  python block_editor.py
```

For scheduling run_classifier.py, change the run_classifier.py by discarding output_dir exisiting check. 
```bash
export CUDA_VISIBLE_DEVICES=6,7; python schedule_run_classifier.py --output_dir ../results/ --block_wise_bits_mask 8 8 8 8 8 8 6 6 6 2 2 2 --block_wise_order 5 6 7 4 3 2 1 0 8 9 10 11
```

For setting the fine-tuning schedule based on the HAWQ paper, run the following file:
```bash
python tuning_order.py --bert_model bert-base-uncased --config_dir new_config.json --task_name CoLA --lambdas 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
```
The `tuning_order` function can also be directly used in other scripts to generate the fine-tuning sequence.


To run HAWQ of CoLA, run the following command:
```bash
export CUDA_VISIBLE_DEVICES=0; python hawq_classifier.py  --task_name CoLA --bert_model bert-base-uncased --lambdas 6.67 9.38 14.55 18.26 18.81 21.75 16.43 15.37 25.82 1.71 0.80 1.01   --bits_order 8 5 4 3 6 7 2 1 0 9 11 10
export CUDA_VISIBLE_DEVICES=0; python hawq_classifier.py  --task_name ner --bert_model bert-base-cased --lambdas 71.77 66.07 57.8  54.43 50.38 42.11 41.06 36.46 18.22 9.8  9.37  4.29   --bits_order 6  7  5  4  8  3  1  2  0 10 11  9
```
where `lambdas` and `bits_order` are based on the Eigenvalues information.
