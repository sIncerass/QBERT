## June 11th


export CUDA_VISIBLE_DEVICES=1,2,6,7; python hessian/squad_eigens.py  --bert_model bert-base-uncased  --do_train  --do_lower_case   --train_file ../data/squad_data/train-v1.1.json   --learning_rate 3e-5   --max_seq_length 384   --doc_stride 128      --train_batch_size 6 --task_name squad --seed 123 --data_percentage 0.1 --method poweriter --output_dir out_hessian/positive_a8_cw_e2/

python run_squad.py \
  --bert_model out_hessian/negative_a8_cw_e2/ \
  --do_predict \
  --do_lower_case \
  --train_file ../data/squad_data/train-v1.1.json \
  --predict_file ../data/squad_data/dev-v1.1.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir out_hessian

export CUDA_VISIBLE_DEVICES=0,1,2,3; 

python run_squad_test.py  --bert_model bert-base-uncased  --do_train  --do_predict  --do_lower_case  --train_file ../data/squad_data/train-v1.1.json  --predict_file ../data/squad_data/dev-v1.1.json  --train_batch_size 12  --learning_rate 3e-5  --num_train_epochs 2.0  --max_seq_length 384  --doc_stride 128  --output_dir out_squad


export PYTHONPATH=$PYTHONPATH:$PWD
export CUDA_VISIBLE_DEVICES=0,1,2,3; 
python ner_quant.py  --bert_model bert-base-cased  --do_train  --do_eval --data_dir ../data/ner_data  --train_batch_size 32  --learning_rate 5e-5  --num_train_epochs 5.0  --max_seq_length 128  --output_dir out --warmup_proportion 0.4 --task_name ner

ner
{"0": {"loss": 0.006103242978018029, "f1": 0.9339480301760268, "block_wise_bits": [2, 6, 8, 6, 8, 4, 4, 2, 2, 2, 2, 2]}

sst-t
{"0": {"normal": {"acc": 0.8577981651376146, "eval_loss": 0.3577010922065569, "global_step": 2105, "loss": 0.2667777091848737}, "block_wise_bits": [5, 2, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2]}}

squad
ori_negative hessian full precision: 0.885, 
4 bits: 0.8790 - 0.8778 (2 epoch 3e-5) - (2 epoch 2e-5)

ori_positive hessian full precision: 0.883, 
4 bits: 0.8747 - 0.8786 (2 epoch 3e-5) - (2 epoch 2e-5)