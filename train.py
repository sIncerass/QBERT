# schdule specific order for training different layer iteratively
import os
import json
import re
import argparse
from run_classifier import run_classifier_w_args, get_default_args
from run_squad import run_squad_w_args
from run_ner import run_ner_w_args
from util import BertConfig_generic
from tuning_order import processors_classification
from schedule_run_classifier import execute_command

GLUE_DATA_PATH = '../data/glue_data/'
NER_DATA_PATH = '../data/ner_data/'
SQUAD_DATA_PATH = '../data/squad_data/'


def schedule_run(output_dir,
                 block_wise_bits,
                 emb_bits,
                 do_fullprecision=False,
                 freeze_embedding=False,
                 task_name='CoLA',
                 quantize_activation=False,
                 embedding_epoch=0,
                 tuning_epoch=1):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    full_precision_bits = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]

    result_info = dict()

    if do_fullprecision:
        # Train to full acc then quantize.
        run_config = BertConfig_generic(
            freeze_embedding=False, block_wise_bits_mask=full_precision_bits,
            emb_bits=emb_bits)
        result_info[0] = execute_command("bert-base-uncased", output_dir, True,
                                         run_config, task_name)
        result_info[0]['block_wise_bits'] = block_wise_bits.copy()
    else:
        task_nameid = task_name.lower()
        os.system(f'cp saved-models/{task_nameid}/* {output_dir}')

    quantized_config = None

    quantized_config = BertConfig_generic(
        quantize_activation, freeze_embedding, 
        block_wise_bits, emb_bits=emb_bits)

    for i in range(tuning_epoch):
        # run and save the model
        result_info[i + embedding_epoch] = execute_command(
            output_dir,
            output_dir,
            False,
            config_obj=quantized_config,
            task_name=task_name)
        result_info[i + embedding_epoch]['block_wise_bits'] = block_wise_bits.copy()

    open(output_dir + "result_logs.json",
         "a").write(json.dumps(result_info) + '\n')
    return json.dumps(result_info) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="", type=str, help="output for stored model")
    parser.add_argument(
        "--do_fullprecision",
        action='store_true',
        help="if true, will train full precision with 3 epoches; otherwise will use the saved results of full precision"
    )
    parser.add_argument(
        "--block_wise_bits",
        default=[32] * 12,
        nargs='+',
        type=int,
        help="block_wise_bits_mask")
    parser.add_argument(
        "--embedding_epoch",
        default=0,
        type=int,
        help="number of training epochs for embedding")
    parser.add_argument(
        "--tuning_epoch",
        default=1,
        type=int,
        help="number of training epochs for encoders")
    parser.add_argument(
        '--freeze_embedding',
        action='store_true',
        help="Whether to perform fine-tune with embedding frozen")
    parser.add_argument(
        '--quantize_activation',
        action='store_true',
        help="Whether to quantize the activation layers")
    parser.add_argument(
        "--task_name",
        default="ner",
        type=str,
        required=True,
        help="The name of the task to train.")
    parser.add_argument(
        '--emb_bits',
        default=[32, 32, 32],
        nargs='+',
        type=int,
        help="bits for embedding layer, following word, pos, type; 15261:256:1 as #param.")

    args = parser.parse_args()
    block_wise_bits = args.block_wise_bits
    schedule_run(args.output_dir,
                 block_wise_bits, args.emb_bits,
                 args.do_fullprecision, args.freeze_embedding, 
                 args.task_name, args.quantize_activation,
                 args.embedding_epoch, args.tuning_epoch)


if __name__ == "__main__":
    main()
