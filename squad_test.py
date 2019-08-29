import os
import json
import re
import argparse
from run_classifier import run_classifier_w_args, get_default_args
from run_ner import run_ner_w_args
from run_squad import run_squad_w_args
from util import BertConfig_generic
from tuning_order import processors_classification

GLUE_DATA_PATH = '../data/glue_data/'
NER_DATA_PATH = '../data/ner_data/'
SQUAD_DATA_PATH = '../data/squad_data/'


class Runner():
    def __init__(self, full=True, task_name='CoLA'):
        self.args = get_default_args()
        self.args.do_train = True
        self.args.do_eval = True
        self.args.do_lower_case = True
        self.args.task_name = task_name
        self.full = full

    def run_w_args(self):
        raise NotImplmentedError()

    def get_args(self):
        return self.args


class GlueRunner(Runner):
    def __init__(self, full=True, task_name='CoLA'):
        super(GlueRunner, self).__init__(full, task_name)
        self.args.max_seq_length = 128
        self.args.train_batch_size = 32
        self.args.data_dir = os.path.join(GLUE_DATA_PATH, task_name)
        self.args.learning_rate = 2e-5
        self.args.num_train_epochs = 3 if self.full else 1

    def run_w_args(self):
        return run_classifier_w_args(self.args)


class BlendingGlueRunner(GlueRunner):
    def __init__(self, task_name='CoLA'):
        super(BlendingGlueRunner, self).__init__(False, task_name)
        self.args.no_update = True

    def run(self, cache_model, output_dir, config_obj=None):
        self.args.bert_model = f'{cache_model}'
        self.args.output_dir = f'{output_dir}'
        self.args.config = config_obj
        return self.run_w_args()


class NerRunner(Runner):
    def __init__(self, full=True, task_name='ner'):
        super(NerRunner, self).__init__(full, task_name)
        self.args.max_seq_length = 128
        self.args.train_batch_size = 32
        self.args.data_dir = NER_DATA_PATH
        self.args.learning_rate = 5e-5 if self.full else 2e-5
        self.args.warmup_proportion = 0.4 if self.full else 0.1
        self.args.num_train_epochs = 5 if self.full else 3
        self.args.do_lower_case = False

    def run_w_args(self):
        return run_ner_w_args(self.args)


class SquadRunner(Runner):
    def __init__(self, full=True, task_name='squad'):
        super(SquadRunner, self).__init__(full, task_name)
        self.args.max_seq_length = 384
        self.args.train_batch_size = 12
        self.args.doc_stride = 128
        self.args.train_file = os.path.join(SQUAD_DATA_PATH, "train-v1.1.json")
        self.args.predict_file = os.path.join(SQUAD_DATA_PATH, "dev-v1.1.json")
        self.args.learning_rate = 3e-5
        self.args.num_train_epochs = 2 if self.full else 1
        self.args.do_predict = True
        self.args.version_2_with_negative = False
        self.args.max_query_length = 64
        self.args.predict_batch_size = 8
        self.args.warmup_proportion = 0.1
        self.args.n_best_size = 20
        self.args.max_answer_length = 30
        self.args.gradient_accumulation_steps = 1
        self.args.null_score_diff_threshold = 0.0
        self.args.verbose_logging = False

    def run_w_args(self):
        return run_squad_w_args(self.args)


def execute_command(cache_model,
                    output_dir,
                    full=False,
                    config_obj=None,
                    task_name='CoLA'):
    task_name_id = task_name.lower()
    if task_name_id in processors_classification:
        if task_name_id == 'ner':
            cache_model = "bert-base-cased" if cache_model == "bert-base-uncased" else cache_model
            runner = NerRunner(full=full, task_name=task_name)
        else:
            runner = GlueRunner(full=full, task_name=task_name)
    elif task_name_id == 'squad':
        runner = SquadRunner(full=full, task_name=task_name)
    elif task_name_id == 'swag':
        raise NotImplementedError("Do not support swag yet.")
    else:
        raise ValueError("Taks name should be GLUE task/swag/squad/ner.")
    runner.args.bert_model = f'{cache_model}'
    runner.args.output_dir = f'{output_dir}'
    runner.args.config = config_obj
    return runner.run_w_args()


def schedule_run(output_dir,
                 block_wise_order,
                 block_wise_bits_mask,
                 do_fullprecision=False,
                 freeze_tune=False,
                 freeze_embedding=False,
                 task_name='CoLA',
                 quantize_activation=False):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Uniform Quantization Settings
    # block_wise_bits = [32] * 12
    # block_wise_bits = [4] * 12

    # NER
    # block_wise_bits =  [2, 6, 6, 6, 6, 4, 4, 4, 4, 2, 2, 2]
    # channel-wise best setting
    # block_wise_bits =  [2, 6, 6, 6, 6, 4, 4, 4, 4, 2, 2, 2]
    ## layer-wise best setting
    # block_wise_bits =  [2, 6, 8, 6, 8, 4, 4, 2, 2, 2, 2, 2]

    # SST-2
    # block_wise_bits = [4, 3, 4, 5, 5, 5, 5, 5, 5, 3, 2, 2]
    # channel-wise best setting
    # block_wise_bits = [4, 3, 4, 5, 5, 5, 5, 5, 5, 3, 2, 2]
    ## layer-wise best setting
    # block_wise_bits = [5, 2, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2]

    # SQUAD
    block_wise_bits = [4, 4, 4, 5, 5, 5, 5, 5, 5, 2, 2, 2]
    # channel-wise best setting
    # block_wise_bits = [4, 4, 4, 5, 5, 5, 5, 5, 5, 2, 2, 2]
    ## layer-wise best setting
    # block_wise_bits = [2, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2]

    # MNLI
    # block_wise_bits = [3, 4, 4, 5, 5, 5, 5, 5, 5, 3, 2, 2]
    # channel-wise best setting
    # block_wise_bits = [3, 4, 4, 5, 5, 5, 5, 5, 5, 3, 2, 2]
    ## layer-wise best setting
    # block_wise_bits = [2, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2]

    result_info = dict()

    """
    if do_fullprecision:
        # Train to full acc then quantize.
        run_config = BertConfig_generic(
            freeze_embedding=False, block_wise_bits_mask=block_wise_bits)
        result_info[0] = execute_command("bert-base-uncased", output_dir, True,
                                         run_config, task_name)
        result_info[0]['block_wise_bits'] = block_wise_bits.copy()
    else:
    """

    task_nameid = task_name.lower()
    #os.system(f'cp saved-models/{task_nameid}/* {output_dir}')
    os.system(f'cp ../epoch2/* {output_dir}')
    """
    for order_iter, order_block in enumerate(block_wise_order):
        # write the configuration file
        block_wise_bits[order_block] = block_wise_bits_mask[order_block]

        quantized_config = None

        if freeze_tune:
            quantized_config = BertConfig_generic(
                quantize_activation,
                freeze_embedding,
                block_wise_bits,
                block_to_tune=order_block)
        else:
    """

    quantized_config = BertConfig_generic(
                quantize_activation, freeze_embedding, block_wise_bits, emb_bits=[8, 8, 32])

        # run and save the model
    result_info = execute_command(
             output_dir,
             output_dir,
             False,
             config_obj=quantized_config,
             task_name=task_name)

        #result_info[order_iter + 1] = execute_command(
        #    output_dir, output_dir, config_obj=quantized_config, task_name=task_name)
        #result_info[order_iter + 1]['block_wise_bits'] = block_wise_bits.copy()

    open(output_dir + "result_logs.json",
         "w").write(json.dumps(result_info) + '\n')
    return json.dumps(result_info) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="", type=str, help="output for stored model")
    parser.add_argument(
        "--do_fullprecision",
        action='store_true',
        help=
        "if true, will train full precision with 3 epoches; otherwise will use the saved results of full precision"
    )
    parser.add_argument(
        "--block_wise_bits_mask",
        default=[32] * 12,
        nargs='+',
        type=int,
        help="block_wise_bits_mask")
    parser.add_argument(
        "--block_wise_order",
        default=[5, 6, 7, 4, 3, 2, 1, 0, 8, 9, 10, 11],
        nargs='+',
        type=int,
        help="block_wise_order")
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
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train.")

    args = parser.parse_args()
    block_wise_order = args.block_wise_order
    block_wise_bits_mask = args.block_wise_bits_mask
    schedule_run(args.output_dir, block_wise_order, block_wise_bits_mask,
                 args.do_fullprecision, args.freeze_tune,
                 args.freeze_embedding, args.task_name, args.quantize_activation)


if __name__ == "__main__":
    main()
