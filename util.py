import json
import re

from modeling import BertConfig_Quant


def get_all_module_names():
    # Ideally this should take a model.
    # Currently it fetchs from a config file.
    #
    # Return: A list of names.
    with open('example_config.json') as json_file:
        data = json.load(json_file)
        # freeze embedding or not
        # modify layer first bits mask, then gradient flag
        return data['layer_bits'].keys()


def generate_per_block_config(per_block_value):
    config_obj = {}
    for k in get_all_module_names():
        matches = re.search(r'layer.(?P<layer_num>\d+).*', k)
        layer_num = matches.group('layer_num')
        config_obj[k] = per_block_value[int(layer_num)]
    return config_obj

def generate_emb_config(per_emb_value):
    config_obj = {} 
    emb_keys = ['word_embeddings', 'position_embeddings', 'token_type_embeddings']
    for v, k in zip(per_emb_value, emb_keys):
        config_obj[k] = v
    return config_obj

class BertConfig_generic(BertConfig_Quant):
    def __init__(self,
                 quantize_activation=False,
                 freeze_embedding=False,
                 block_wise_bits_mask=None,
                 block_to_tune=None,
                 emb_bits=None,
                 vocab_size_or_config_json_file=-1):

        super(BertConfig_generic,
              self).__init__(vocab_size_or_config_json_file)

        self.set('quantize_activation', quantize_activation)
        # if do quantize for embedding, no freeze_embedding then
        assert not (sum(emb_bits) != 96 and freeze_embedding)
        self.set('freeze_embedding', freeze_embedding)
        self.set('emb_bits',
                    generate_emb_config(emb_bits))

        if block_wise_bits_mask is not None:
            self.set('layer_bits',
                     generate_per_block_config(block_wise_bits_mask))

        if block_to_tune is not None:
            assert isinstance(block_to_tune, int)
            # Setup the requie grad for the block.
            block_wise_gradient_flag = [False] * len(block_wise_bits_mask)
            # for block_gradient_flag in args.block_wise_gradient_flag:
            block_wise_gradient_flag[block_to_tune] = True

            self.set('layer_requires_grad',
                     generate_per_block_config(block_wise_gradient_flag))

    def set(self, k, v):
        self.__dict__[k] = v
