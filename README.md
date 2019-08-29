# NLP-quantization

This repo is based on [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) and [pytorch-hessian-eigenthings](https://github.com/noahgolmant/pytorch-hessian-eigenthings).


## Installation

This repo was tested on Python 3.5+ and PyTorch 1.1.0

PyTorch pretrained bert and requirements need to be installed by pip as follows:
```bash
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$PWD
```

## Commands that are tested
For training of CoLA dataset:
```bash
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py

python download_glue_data.py

export GLUE_DIR=/path/to/glue

export OUTPUT-DIR=/path/to/output_dir

export CUDA_VISIBLE_DEVICES=0,1,2,3; python run_classifier.py   --task_name CoLA   --do_train   --do_eval   --do_lower_case   --data_dir $GLUE_DIR/CoLA    --bert_model bert-base-uncased   --max_seq_length 128   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3.0   --output_dir $OUTPUT-DIR

```
For training of SQUAD dataset:

Get SQUAD dataset:
```bash
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

wget https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py
```
Training:
```bash
export SQUAD_DIR=/path/to/squad

export OUTPUT-DIR=/path/to/output_dir

export CUDA_VISIBLE_DEVICES=0,1,2,3; python run_squad.py  --bert_model bert-base-uncased  --do_train  --do_predict  --do_lower_case  --train_file $SQUAD_DIR/train-v1.1.json  --predict_file $SQUAD_DIR/dev-v1.1.json  --train_batch_size 12  --learning_rate 3e-5  --num_train_epochs 2.0  --max_seq_length 384  --doc_stride 128  --output_dir $OUTPUT-DIR
```
For training of NER (CoNLL-03) dataset:

Get NER (CoNLL-03):
```bash
wget https://raw.githubusercontent.com/kamalkraj/BERT-NER/master/data/train.txt

wget https://raw.githubusercontent.com/kamalkraj/BERT-NER/master/data/test.txt

wget https://raw.githubusercontent.com/kamalkraj/BERT-NER/master/data/valid.txt
```
Training:
```bash
export NER_DIR=/path/to/ner

export OUTPUT-DIR=/path/to/output_dir

export CUDA_VISIBLE_DEVICES=0,1,2,3; python run_ner.py  --bert_model bert-base-cased  --do_train  --do_eval --data_dir $NER_DIR  --train_batch_size 32  --learning_rate 5e-5  --num_train_epochs 5.0  --max_seq_length 128  --output_dir $OUTPUT-DIR --warmup_proportion 0.4 --task_name ner
```

## Hessian analysis
For CoLA dataset:
```
export CUDA_VISIBLE_DEVICES=0,1; python  hessian/classifier_eigens.py   --task_name CoLA   --do_train   --do_lower_case   --data_dir $GLUE_DIR/CoLA    --bert_model bert-base-uncased   --max_seq_length 128   --train_batch_size 32   --learning_rate 2e-5    --output_dir /scratch/linjian2/tmp/CoLA  --task_name cola --seed 123 --data_percentage 0.01
```
For SQUAD dataset:
```
export CUDA_VISIBLE_DEVICES=0,1; python hessian/squad_eigens.py  --bert_model bert-base-uncased  --do_train  --do_lower_case   --train_file $SQUAD_DIR/train-v1.1.json   --learning_rate 3e-5   --max_seq_length 384   --doc_stride 128   --output_dir /scratch/linjian2/tmp/debug_squad/   --train_batch_size 12 --task_name squad --seed 123 --data_percentage 0.001
```
For NER (CoNLL-03) dataset:
```
export CUDA_VISIBLE_DEVICES=0,1; python hessian/ner_eigens.py  --bert_model bert-base-cased  --do_train    --data_dir $NER_DIR/   --learning_rate 5e-5   --max_seq_length 128   --output_dir /scratch/linjian2/tmp/debug_squad/   --train_batch_size 12 --task_name ner --seed 123 --data_percentage 0.01
```
___
___

### Fine-tuning with BERT: examples

We showcase several fine-tuning examples based on (and extended from) [the original implementation](https://github.com/google-research/bert/):

- a *sequence-level classifier* on nine different GLUE tasks,
- a *token-level classifier* on the question answering dataset SQuAD, and
- a *sequence-level multiple-choice classifier* on the SWAG classification corpus.
- a *BERT language model* on another target corpus

#### GLUE results on dev set

We get the following results on the dev set of GLUE benchmark with an uncased BERT base 
model. All experiments were run on a P100 GPU with a batch size of 32.

| Task | Metric | Result |
|-|-|-|
| CoLA | Matthew's corr. | 57.29 |
| SST-2 | accuracy | 93.00 |
| MRPC | F1/accuracy | 88.85/83.82 |
| STS-B | Pearson/Spearman corr. | 89.70/89.37 |
| QQP | accuracy/F1 | 90.72/87.41 |
| MNLI | matched acc./mismatched acc.| 83.95/84.39 |
| QNLI | accuracy | 89.04 |
| RTE | accuracy | 61.01 |
| WNLI | accuracy | 53.52 |

Some of these results are significantly different from the ones reported on the test set
of GLUE benchmark on the website. For QQP and WNLI, please refer to [FAQ #12](https://gluebenchmark.com/faq) on the webite.

Before running anyone of these GLUE tasks you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

```shell
export GLUE_DIR=/path/to/glue
export TASK_NAME=MRPC

python run_classifier.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/$TASK_NAME/
```

where task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI.

The dev set results will be present within the text file 'eval_results.txt' in the specified output_dir. In case of MNLI, since there are two separate dev sets, matched and mismatched, there will be a separate output folder called '/tmp/MNLI-MM/' in addition to '/tmp/MNLI/'.

The code has not been tested with half-precision training with apex on any GLUE task apart from MRPC, MNLI, CoLA, SST-2. The following section provides details on how to run half-precision training with MRPC. With that being said, there shouldn't be any issues in running half-precision training with the remaining GLUE tasks as well, since the data processor for each task inherits from the base class DataProcessor.

#### MRPC

This example code fine-tunes BERT on the Microsoft Research Paraphrase
Corpus (MRPC) corpus and runs in less than 10 minutes on a single K-80 and in 27 seconds (!) on single tesla V100 16GB with apex installed.

Before running this example you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

```shell
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/
```

Our test ran on a few seeds with [the original implementation hyper-parameters](https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks) gave evaluation results between 84% and 88%.

**Fast run with apex and 16 bit precision: fine-tuning on MRPC in 27 seconds!**
First install apex as indicated [here](https://github.com/NVIDIA/apex).
Then run
```shell
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/ \
  --fp16
```

#### SQuAD

This example code fine-tunes BERT on the SQuAD dataset. It runs in 24 min (with BERT-base) or 68 min (with BERT-large) on a single tesla V100 16GB.

The data for SQuAD can be downloaded with the following links and should be saved in a `$SQUAD_DIR` directory.

*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
*   [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

```shell
export SQUAD_DIR=/path/to/SQUAD

python run_squad.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/
```

Training with the previous hyper-parameters gave us the following results:
```bash
{"f1": 88.52381567990474, "exact_match": 81.22043519394512}
```

#### SWAG

The data for SWAG can be downloaded by cloning the following [repository](https://github.com/rowanz/swagaf)

```shell
export SWAG_DIR=/path/to/SWAG

python run_swag.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_lower_case \
  --do_eval \
  --data_dir $SWAG_DIR/data \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 80 \
  --output_dir /tmp/swag_output/ \
  --gradient_accumulation_steps 4
```

Training with the previous hyper-parameters on a single GPU gave us the following results:
```
eval_accuracy = 0.8062081375587323
eval_loss = 0.5966546792367169
global_step = 13788
loss = 0.06423990014260186
```
