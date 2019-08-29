# schdule specific order for training different layer iteratively
import os
import json
import re
import argparse

from schedule_run_classifier import BlendingGlueRunner
from util import BertConfig_generic


def main():

    result_list = []
    for i in range(100):
        block_wise_bits = [8] * 12
        quantized_config = BertConfig_generic(
            block_wise_bits_mask=block_wise_bits)
        quantized_config.set('eval-blending-alpha', 0.01 * i)
        runner = BlendingGlueRunner()
        result = runner.run(
            cache_model='saved-models/cola/',
            output_dir='blending-results/',
            config_obj=quantized_config)
        result_list.append(result)

    open("result_logs.json", "w").write(json.dumps(result_list) + '\n')


if __name__ == "__main__":
    main()
