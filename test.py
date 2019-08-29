# schdule specific order for training different layer iteratively
import os
import json
import re
import argparse
import ray

from tuning_order import tuning_order
from schedule_run_classifier import schedule_run
from util import BertConfig_generic

quantize_assigned_bits = [2, 4, 6, 8]
block_wise_bits_mask = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]

TOTAL_GPUS = 4
PER_JOB_GPUS = 2


def hawq(bert_model, lambdas, task_name, bits_order, do_fullprecision,
         freeze_tune, freeze_embedding, quantize_activation, quantize_embedding):
    '''
    Perform Hessian aware quantization.
    The 12 layers are grouped into 4 blocks, each one containing 3 layers:
        layer[bits_order[0]:bits_order[3]] belong to block1,
        layer[bits_order[3]:bits_order[6]] belong to block2,
        layer[bits_order[6]:bits_order[9]] belong to block3,
        layer[bits_order[9]:bits_order[12]] belong to block4.

    Each block is assigned 1 bit, with the constraint:
        bits_block1 >= bits_block2 >= bits_block3 >= bits_block4
    '''

    def get_valid_bit_assignments():
        quantize_bits_list = []
        for bits_block1 in [4, 6, 8]:
            for bits_block2 in [4, 6, 8]:
                for bits_block3 in [4, 6, 8]:
                    for bits_block4 in [2, 4, 6, 8]:
                        # get bits of each block
                        if (bits_block1 < bits_block2) or (
                                bits_block2 < bits_block3) or (bits_block3 <
                                                               bits_block4):
                            continue
                        quantize_bits = [bits_block1] * 3 + [bits_block2] * \
                            3 + [bits_block3] * 3 + [bits_block4] * 3
                        quantize_bits_list.append(quantize_bits)
        return quantize_bits_list

    quantize_bits_list = get_valid_bit_assignments()
    #ray.init(num_gpus=TOTAL_GPUS)

    #@ray.remote(num_gpus=PER_JOB_GPUS)
    def run(quantize_bits, index):
        for i in range(12):
            block_wise_bits_mask[bits_order[i]] = quantize_bits[i]

        quantized_config = BertConfig_generic(
            block_wise_bits_mask=block_wise_bits_mask)
        block_wise_order = tuning_order(bert_model, quantized_config, lambdas,
                                        task_name)
        # fine tune the quantized model
        return schedule_run(
            f'results/experiment-{index}/',
            block_wise_order,
            block_wise_bits_mask,
            do_fullprecision=do_fullprecision,
            freeze_tune=freeze_tune,
            freeze_embedding=freeze_embedding,
            task_name=task_name,
            quantize_activation=quantize_activation,
            quantize_embedding=quantize_embedding)
    results = []
    for q_b in quantize_bits_list:
        if sum(q_b) == 48:
            results.append( run(q_b, len(results)) )
    #results = ray.get(
    #    [run.remote(quantize_bits, index) for index, quantize_bits in enumerate(quantize_bits_list)])
    #results = run(quantize_bits_list[0], 0)
    with open("results/HAWQ-results.json", "w") as writer:
        writer.write(str(results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train.")
    parser.add_argument(
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument(
        "--lambdas",
        default=[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        nargs='+',
        type=float,
        help="lambda (eigenvalue in the HAWQ paper)")
    parser.add_argument(
        "--bits_order",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        nargs='+',
        type=int,
        help="the ordering of bit assignment")
    parser.add_argument(
        "--do_fullprecision",
        action='store_true',
        help=
        "if true, will train full precision with 3 epoches; otherwise will use the saved results of full precision results"
    )
    parser.add_argument(
        '--freeze_tune',
        action='store_true',
        help="Whether to perform fine-tune with other layers frozen")
    parser.add_argument(
        '--freeze_embedding',
        action='store_true',
        help="Whether to perform fine-tune with embedding frozen")
    parser.add_argument(
        '--quantize_activation',
        action='store_true',
        help="Whether to quantize the activation layers")
    parser.add_argument(
        '--quantize_embedding',
        action='store_true',
        help="Whether to quantize the activation layers")

    args = parser.parse_args()
    hawq(args.bert_model, args.lambdas, args.task_name, args.bits_order,
         args.do_fullprecision, args.freeze_tune, args.freeze_embedding,
         args.quantize_activation, args.quantize_embedding)


if __name__ == "__main__":
    main()
