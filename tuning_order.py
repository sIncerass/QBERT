# schdule specific order for training different layer iteratively
import os
import json
import re
import argparse
from modeling import BertForSequenceClassification_Quant as BertForSequenceClassification
from modeling import BertForTokenClassification_Quant as BertForTokenClassification 
from modeling import BertForQuestionAnswering_Quant as BertForQuestionAnswering                                          
from run_classifier import *
from run_ner import *

processors_classification = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "ner": NerProcessor
}

def prepare_model(bert_model, config_obj, task_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_name = task_name.lower()
    # prepare model for classification or ner
    if task_name in processors_classification:
        processor = processors_classification[task_name]()
        label_list = processor.get_labels()
        num_labels = len( label_list ) + 1 if task_name == "ner" else len( label_list )
        model_obj = BertForNer if task_name == "ner" else BertForSequenceClassification
        model = model_obj.from_pretrained(bert_model, 
                                        cache_dir=None, num_labels=num_labels, config=config_obj)
    # prepare model for squad
    elif task_name == 'squad':
        model = BertForQuestionAnswering.from_pretrained(bert_model, 
                                        cache_dir=None, config=config_obj)
    # prepare model for swag
    elif task_name == 'swag':
        raise NotImplementedError("Do not support swag yet.")
    else:
        raise ValueError("Taks name should be GLUE task/swag/squad/ner.")
    model.to(device)
    model = torch.nn.DataParallel(model)
    return model

def get_delta_weights_norm2(model, layer):
    encoder_layer = model.module.bert.encoder.layer[layer]
    sum_delta_weights_norm2 = 0.
    for linear_layer in [
            encoder_layer.attention.self.query,
            encoder_layer.attention.self.key,
            encoder_layer.attention.self.value,
            encoder_layer.attention.output.dense,
            encoder_layer.intermediate.dense, encoder_layer.output.dense
    ]:
        weights = linear_layer.weight
        weights_quant = linear_layer.weight_function(linear_layer.weight,
                                                     linear_layer.weight_bit)
        delta_weights = weights - weights_quant
        delta_weights_norm = delta_weights.norm()
        sum_delta_weights_norm2 += delta_weights_norm * delta_weights_norm
    return sum_delta_weights_norm2


def tuning_order(cache_model, config_obj, lambdas, task_name):
    """Get the fine-tuning sequence based on the HAWQ paper.
    Args:
        cache_model: bert pre-trained model or directory to fetch the pretrained model
        config_obj: configuration object
        task_name: task name (cola)
        lambdas: list of lambda for each block (in HAWQ paper, lambda is the largest eigenvalue)
    Returns:
        numpy array: fine tuning sequence
    """
    Sigmas = np.zeros(12)

    model = prepare_model(cache_model, config_obj, task_name)

    for i in range(12):
        delta_weights_norm2 = get_delta_weights_norm2(model, i)
        Sigmas[i] = lambdas[i] * delta_weights_norm2

    return Sigmas.argsort()[::-1].tolist()


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
        '--config_dir', type=str, default='', help="Config file for Bert")
    parser.add_argument(
        "--lambdas",
        default=[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        nargs='+',
        type=float,
        help="lambda (eigenvalue in the HAWQ paper)")

    args = parser.parse_args()
    print(
        tuning_order(args.bert_model, args.config_dir, args.lambdas,
                     args.task_name))


if __name__ == "__main__":
    main()
