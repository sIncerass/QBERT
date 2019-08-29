# schdule specific order for training different layer iteratively
import os
import json
import re
import argparse
import ray

from train import schedule_run
from util import BertConfig_generic

quantize_assigned_bits = [3, 4, 5, 6, 7, 8]
block_wise_bits = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]

TOTAL_GPUS = 8
PER_JOB_GPUS = 1


def hawq_direct_search(task_name, bits_order, quantize_activation):
    '''
    Perform Hessian aware quantization.
    The 12 layers are grouped into 6 blocks, each one containing 2 layers:
        layer[bits_order[0]:bits_order[2]] belong to block1,
        ...
        layer[bits_order[10]:bits_order[12]] belong to block6.
    Each block is assigned 1 bit, with the constraint:
        bits_block1 >= bits_block2 >= ... >= bits_block6
    '''
    def get_valid_bit_assignments():
        quantize_bits_list = []
        for bits_block1 in [6,7,8]:
            for bits_block2 in [5,6,7,8]:
                for bits_block3 in [4,5,6,7]:
                    for bits_block4 in [2,3,4,5,6]:
                        for bits_block5 in [2,3,4,5,6]:
                            for bits_block6 in [2,3,4,5,6]:
                                # get bits of each block
                                if (bits_block1 < bits_block2) or (
                                        bits_block2 < bits_block3) or (
                                        bits_block3 < bits_block4) or (
                                        bits_block4 < bits_block5) or (
                                        bits_block5 < bits_block6):
                                    continue
                                quantize_bits = [bits_block1] * 2 + [bits_block2] * 2 +\
                                                [bits_block3] * 2 + [bits_block4] * 2 +\
                                                [bits_block5] * 2 + [bits_block6] * 2
                                if sum(quantize_bits) / 12.0 < 5.5:
                                    quantize_bits_list.append(quantize_bits)
        print(len(quantize_bits_list))
        return quantize_bits_list

    quantize_bits_list = get_valid_bit_assignments()
    ray.init(num_gpus=TOTAL_GPUS)

    @ray.remote(num_gpus=PER_JOB_GPUS)
    def run(quantize_bits):
        for i in range(12):
            block_wise_bits[bits_order[i]] = quantize_bits[i]

        # fine tune the quantized model
        return schedule_run(
            f'results-direct-search/',
            block_wise_bits,
            emb_bits=[8,8,32],
            task_name=task_name,
            quantize_activation=quantize_activation)

    results = ray.get(
        [run.remote(quantize_bits) for quantize_bits in quantize_bits_list])

    with open("results-direct-search/HAWQ-directsearch-results.json", "w") as writer:
        for i in range(len(results)):
            writer.write(results[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train.")
    parser.add_argument(
        "--bits_order",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        nargs='+',
        type=int,
        help="the ordering of bit assignment")
    parser.add_argument(
        '--quantize_activation',
        action='store_true',
        help="Whether to quantize the activation layers")

    args = parser.parse_args()
    hawq_direct_search(
        args.task_name,
        args.bits_order,
        args.quantize_activation)


if __name__ == "__main__":
    main()