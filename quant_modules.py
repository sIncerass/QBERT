from quantization_utils.quant_utils import *
import numpy as np
from torch.nn.modules.conv import Conv2d as _Conv2d
from torch.nn import Linear as _linear
from torch.nn import Embedding as _Embedding
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
import time

# Activation Look-Up Table
# ACT_QUANT_BITS_MAP = {2:5, 3:6, 4:6, 5:8, 6:8, 7:8, 8:8, 32:32}

# Activation 8-bit
ACT_QUANT_BITS_MAP = {1:8, 2:8, 3:8, 4:8, 5:8, 6:8, 7:8, 8:8, 32:8}


class QuantEmbedding(_Embedding):
    """docstring for QuantEmbedding"""
    def __init__(self, 
                 num_embeddings, 
                 embedding_dim,
                 padding_idx=None,
                 max_norm=None, 
                 norm_type=2., 
                 scale_grad_by_freq=False,
                 mode='mean', 
                 sparse=False, 
                 _weight=None, 
                 weight_bit=8,
                 full_precision_flag=True,
                 quant_mode="asymmetric",
                 alpha=None,
                 per_channel=True,
                 group_quantization=True,
                 group_number=1,
                 weight_percentile=False):

        super(QuantEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx,
                 max_norm, norm_type, scale_grad_by_freq,
                 sparse, _weight)

        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.alpha = alpha
        self.quant_mode = quant_mode
        self.input_size = num_embeddings
        self.output_size = embedding_dim
        self.momentum = 0.99

        # self.x_min = torch.zeros(1)
        # self.x_max = torch.zeros(1)
        # if not per_channel:
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        # else:
        #    self.register_buffer('x_min', torch.zeros(input_size))
        #    `self.register_buffer('x_max', torch.zeros(input_size))

        self.per_channel = per_channel

        self.weight_percentile = weight_percentile

        self.group_quantization = group_quantization
        self.group_number = group_number

        if quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

    def __repr__(self):
        s = super(QuantEmbedding, self).__repr__()
        s.replace(
            ")", ", weight_bit={}, "
            "full_precision_flag={})".format(self.weight_bit,
                                             self.full_precision_flag))
        return s

    def reset_bits(self, weight_bit=8):
        self.full_precision_flag = False
        self.weight_bit = weight_bit

    def forward(self, x):
        # print(x.shape)
        w = self.weight
        self.channel_num = w.shape[1]
        # print("w shape:", w.shape)

        if self.per_channel:
            if not self.group_quantization:
                # print(1, time.time())
                # x_transform = w.data.transpose(0, 1).contiguous().view(self.channel_num, -1) # output_dim as channel
                x_transform = w.data.transpose(0, 1).contiguous()
                # print("x_transform, shape:", x_transform.shape)
                w_min = x_transform.min(dim=1)[0]
                w_max = x_transform.max(dim=1)[0]
                # print("w_min shape:", w_min.shape)
                # print("w_max shape:", w_max.shape)
                # w_min = torch.zeros(self.channel_num).cuda()
                # w_max = torch.zeros(self.channel_num).cuda()
                # for i in range(self.channel_num):
                #   w_min[i] = w.data[:, i, :, :].min()
                #   w_max[i] = w.data[:, i, :, :].max()
                # print(w_min)
                # print(w_max)

                if not self.weight_percentile:
                    pass

                elif self.weight_percentile:
                    # print("percentile = ", self.percentile)
                    lower_percentile = 0.1
                    upper_percentile = 99.9
                    input_length = x_transform[0].view(-1).shape[0]

                    # print("self.channel_num = ", self.channel_num)
                    # print("input_length = ", input_length)

                    lower_index = round(input_length * lower_percentile * 0.01)
                    upper_index = round(input_length * upper_percentile * 0.01)

                    lower_bound, _ = torch.topk(x_transform, lower_index, largest=False, sorted=False)
                    upper_bound, _ = torch.topk(x_transform, input_length - upper_index, largest=True, sorted=False)

                    # print("lower_bound.shape = ", lower_bound.shape)
                    # print("w_min shape:", w_min.shape)

                    w_min = lower_bound.max(dim=1)[0]
                    w_max = upper_bound.min(dim=1)[0]

                    # print("w_min_new shape:", w_min.shape)

                    # for i in range(self.channel_num):
                    #     w_min[i], w_max[i] = get_percentile_min_max(
                    #          x_transform[i].view(-1), 0.1, 99.9, output_tensor=True)

            elif self.group_quantization:
                x_transform = w.data.transpose(0, 1).contiguous()
                # w_min = torch.zeros(x_transform.size()[0])
                # w_max = torch.zeros(x_transform.size()[0])
                w_min = x_transform.min(dim=1)[0]
                w_max = x_transform.max(dim=1)[0]

                # please make sure group_length is an integer
                group_length = w_max.size()[0] // self.group_number

                if not self.weight_percentile:
                    temp_w_min = w_min.clone()
                    temp_w_max = w_max.clone()
                    # temp_w_min = x_transform.min(dim=1)[0]
                    # temp_w_max = x_transform.max(dim=1)[0]

                    for i in range(self.group_number):
                        w_min[i * group_length: (i + 1) * group_length] = \
                            temp_w_min[i * group_length: (i + 1) * group_length].min().repeat(group_length)
                        w_max[i * group_length: (i + 1) * group_length] = \
                            temp_w_max[i * group_length: (i + 1) * group_length].max().repeat(group_length)
                        # print("shape = ", temp_w_max[i * group_length: (i + 1) * group_length].max().shape)
                        # print("enlarged shape = ", temp_w_max[i * group_length: (i + 1) * group_length] \
                        #       .max().repeat(group_length).shape)
                        # if i == 1:
                        #     print("w_max_1_2 = ", w_max[i * group_length: (i + 1) * group_length])

                elif self.weight_percentile:
                    # print("percentile = ", self.percentile)
                    for i in range(self.group_number):
                        temp_w_min, temp_w_max = get_percentile_min_max(x_transform
                                [i * group_length: (i + 1) * group_length].view(-1), 0.1, 99.9, output_tensor=True)
                        w_min[i * group_length: (i + 1) * group_length] = temp_w_min.repeat(group_length)
                        w_max[i * group_length: (i + 1) * group_length] = temp_w_max.repeat(group_length)

        elif not self.per_channel:
            if not self.weight_percentile:
                w_min = w.data.min().expand(1)
                w_max = w.data.max().expand(1)
            elif self.weight_percentile:
                w_min, w_max = get_percentile_min_max(w.clone().view(-1), 0.1, 99.9)

        # print("w_min: ", w_min)
        # print("w_min size: ", w_min.size())

        # Initialization
        # if self.x_min.size()[0] == 3072:
        #     print("self.x_max = ", self.x_max[7:11])

        # print("self.x_min: ", self.x_min)
        # print("self.x_min size: ", self.x_min.size())

        if self.x_min.size()[0] == 1:
            if self.x_min == self.x_max:
                self.x_min = w_min
                self.x_max = w_max

            # print("True x_min = ", self.x_min[0:8])
            # if self.per_channel:
            #     self.x_min = self.x_min.expand(self.channel_num).cuda()
            #     self.x_max = self.x_max.expand(self.channel_num).cuda()
            # print(self.x_max)

        # print("self.x_min 2: ", self.x_min)

            # exponential moving average (EMA)
            # use momentum to prevent the quantized values change greatly every
            # iteration
        self.x_min = self.momentum * self.x_min + (1. - self.momentum) * w_min
        self.x_max = self.momentum * self.x_max + (1. - self.momentum) * w_max

        # print("self.x_min 3: ", self.x_min)

        # if self.x_min.size()[0] == 3072:
        #     print("True self.x_max = ", self.x_max[7:11])
        # print("True self.x_min size:", self.x_min.size())
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, self.x_min,
                                     self.x_max, self.per_channel, self.weight_percentile)
        else:
            w = self.weight

        # print("self.x_min 4: ", self.x_min)

        if self.alpha is None:
            return F.embedding(
                x, w, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            assert self.full_precision_flag == False
            # quantized = self.alpha * F.linear(x, w, bias=self.bias)
            # non_quantized = (1 - self.alpha) * F.linear(
            #     x, self.weight, bias=self.bias)
            quantized = self.alpha * w
            non_quantized = (1 - self.alpha) * self.weight

            return F.embedding(
                x, non_quantized + quantized, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)

    # Original forward() of QuantEmbedding()
    # def forward(self, input):
    #     if not self.full_precision_flag:
    #         w = self.weight_function(self.weight, self.weight_bit)
    #     else:
    #         w = self.weight
    #
    #     return F.embedding(
    #         input, w, self.padding_idx, self.max_norm,
    #         self.norm_type, self.scale_grad_by_freq, self.sparse)
        

class QuantLinear(_linear):
    def __init__(self,
                 input_size,
                 output_size,
                 weight_bit=8,
                 full_precision_flag=True,
                 quant_mode="asymmetric",
                 alpha=None,
                 per_channel=True,
                 group_quantization=True,
                 group_number=1,
                 weight_percentile=False):
        # WARNING on alpha blending:
        # This is an feature that should only be used during evaluation.
        # And it must be used with the quantized config.
        super(QuantLinear, self).__init__(input_size, output_size)
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.alpha = alpha
        self.quant_mode = quant_mode
        self.input_size = input_size
        self.output_size = output_size
        self.momentum = 0.99

        # self.x_min = torch.zeros(1)
        # self.x_max = torch.zeros(1)
        #if not per_channel:
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        #else:
        #    self.register_buffer('x_min', torch.zeros(input_size))
        #    self.register_buffer('x_max', torch.zeros(input_size))

        self.per_channel = per_channel

        self.weight_percentile = weight_percentile

        self.group_quantization = group_quantization
        self.group_number = group_number

        if quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

    def reset_bits(self, weight_bit=8):
        self.full_precision_flag = False
        self.weight_bit = weight_bit

    def reset_alpha(self, alpha):
        assert alpha >= 0.0
        assert alpha <= 1.0
        self.alpha = alpha

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s.replace(
            ")", ", weight_bit={}, "
            "full_precision_flag={})".format(self.weight_bit,
                                             self.full_precision_flag))
        return s

    def forward(self, x):
        # print(x.shape)
        w = self.weight
        self.channel_num = w.shape[1]
        # print("w shape:", w.shape)

        if self.per_channel:
            if not self.group_quantization:
                # print(1, time.time())
                # x_transform = w.data.transpose(0, 1).contiguous().view(self.channel_num, -1) # output_dim as channel
                x_transform = w.data.transpose(0, 1).contiguous()
                # print("x_transform, shape:", x_transform.shape)
                w_min = x_transform.min(dim=1)[0]
                w_max = x_transform.max(dim=1)[0]
                # print("w_min shape:", w_min.shape)
                # print("w_max shape:", w_max.shape)
                # w_min = torch.zeros(self.channel_num).cuda()
                # w_max = torch.zeros(self.channel_num).cuda()
                # for i in range(self.channel_num):
                #   w_min[i] = w.data[:, i, :, :].min()
                #   w_max[i] = w.data[:, i, :, :].max()
                # print(w_min)
                # print(w_max)

                if not self.weight_percentile:
                    pass

                elif self.weight_percentile:
                    # print("percentile = ", self.percentile)
                    lower_percentile = 0.1
                    upper_percentile = 99.9
                    input_length = x_transform[0].view(-1).shape[0]

                    # print("self.channel_num = ", self.channel_num)
                    # print("input_length = ", input_length)

                    lower_index = round(input_length * lower_percentile * 0.01)
                    upper_index = round(input_length * upper_percentile * 0.01)

                    lower_bound, _ = torch.topk(x_transform, lower_index, largest=False, sorted=False)
                    upper_bound, _ = torch.topk(x_transform, input_length - upper_index, largest=True, sorted=False)

                    # print("lower_bound.shape = ", lower_bound.shape)
                    # print("w_min shape:", w_min.shape)

                    w_min = lower_bound.max(dim=1)[0]
                    w_max = upper_bound.min(dim=1)[0]

                    # print("w_min_new shape:", w_min.shape)

                    # for i in range(self.channel_num):
                    #     w_min[i], w_max[i] = get_percentile_min_max(
                    #          x_transform[i].view(-1), 0.1, 99.9, output_tensor=True)

            elif self.group_quantization:
                x_transform = w.data.transpose(0, 1).contiguous()
                # w_min = torch.zeros(x_transform.size()[0])
                # w_max = torch.zeros(x_transform.size()[0])
                w_min = x_transform.min(dim=1)[0]
                w_max = x_transform.max(dim=1)[0]

                # please make sure group_length is an integer
                group_length = w_max.size()[0] // self.group_number

                if not self.weight_percentile:
                    temp_w_min = w_min.clone()
                    temp_w_max = w_max.clone()
                    # temp_w_min = x_transform.min(dim=1)[0]
                    # temp_w_max = x_transform.max(dim=1)[0]

                    for i in range(self.group_number):
                        w_min[i * group_length: (i + 1) * group_length] = \
                            temp_w_min[i * group_length: (i + 1) * group_length].min().repeat(group_length)
                        w_max[i * group_length: (i + 1) * group_length] = \
                            temp_w_max[i * group_length: (i + 1) * group_length].max().repeat(group_length)
                        # print("shape = ", temp_w_max[i * group_length: (i + 1) * group_length].max().shape)
                        # print("enlarged shape = ", temp_w_max[i * group_length: (i + 1) * group_length] \
                        #       .max().repeat(group_length).shape)
                        # if i == 1:
                        #     print("w_max_1_2 = ", w_max[i * group_length: (i + 1) * group_length])

                elif self.weight_percentile:
                    # print("percentile = ", self.percentile)
                    for i in range(self.group_number):
                        temp_w_min, temp_w_max = get_percentile_min_max(x_transform
                                [i * group_length: (i + 1) * group_length].view(-1), 0.1, 99.9, output_tensor=True)
                        w_min[i * group_length: (i + 1) * group_length] = temp_w_min.repeat(group_length)
                        w_max[i * group_length: (i + 1) * group_length] = temp_w_max.repeat(group_length)

        elif not self.per_channel:
            if not self.weight_percentile:
                w_min = w.data.min().expand(1)
                w_max = w.data.max().expand(1)
            elif self.weight_percentile:
                w_min, w_max = get_percentile_min_max(w.clone().view(-1), 0.1, 99.9)

        # print("w_min: ", w_min)
        # print("w_min size: ", w_min.size())

        # Initialization
        # if self.x_min.size()[0] == 3072:
        #     print("self.x_max = ", self.x_max[7:11])
        # print("self.x_min: ", self.x_min)
        # print("self.x_min size: ", self.x_min.size())

        if self.x_min.size()[0] == 1:
            # print("1")
            if self.x_min == self.x_max:
                # print("2")
                self.x_min = w_min
                self.x_max = w_max
                # print("w_min size: ", w_min.size())

            # print("True x_min = ", self.x_min[0:8])
            # if self.per_channel:
            #     self.x_min = self.x_min.expand(self.channel_num).cuda()
            #     self.x_max = self.x_max.expand(self.channel_num).cuda()
            # print(self.x_max)

        # print("self.x_min 2: ", self.x_min)

            # exponential moving average (EMA)
            # use momentum to prevent the quantized values change greatly every
            # iteration
        self.x_min = self.momentum * self.x_min + (1. - self.momentum) * w_min
        self.x_max = self.momentum * self.x_max + (1. - self.momentum) * w_max

        # print("self.x_min 3: ", self.x_min)

        # if self.x_min.size()[0] == 3072:
        #     print("True self.x_max = ", self.x_max[7:11])
        # print("True self.x_min size:", self.x_min.size())
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, self.x_min,
                                     self.x_max, self.per_channel, self.weight_percentile)
        else:
            w = self.weight

        # print("self.x_min 4: ", self.x_min)

        if self.alpha is None:
            return F.linear(x, w, bias=self.bias)
        else:
            assert self.full_precision_flag == False
            # quantized = self.alpha * F.linear(x, w, bias=self.bias)
            # non_quantized = (1 - self.alpha) * F.linear(
            #     x, self.weight, bias=self.bias)
            quantized = self.alpha * w
            non_quantized = (1 - self.alpha) * self.weight

            return F.linear(x, quantized + non_quantized, bias=self.bias)


class QuantLinear_Act(QuantLinear):
    def __init__(self, quant_linear):
        super(QuantLinear_Act, self).__init__(
            quant_linear.input_size, quant_linear.output_size,
            quant_linear.weight_bit, quant_linear.full_precision_flag,
            quant_linear.quant_mode)
        self.weight_bit_act = ACT_QUANT_BITS_MAP[self.weight_bit]

        self.percentile = False

        self.quant_act = QuantAct_bert(
            activation_bit=self.weight_bit_act,
            full_precision_flag=self.full_precision_flag,
            percentile=self.percentile)

    def reset_bits(self, weight_bit=8):
        super(QuantLinear_Act, self).reset_bits(weight_bit)
        self.weight_bit_act = ACT_QUANT_BITS_MAP[weight_bit]

        self.quant_act.reset_bits(self.weight_bit_act)

    def forward(self, x):
        if self.full_precision_flag:
            return super(QuantLinear_Act, self).forward(x)
        else:
            # x = self.quant_act(x)
            x = super(QuantLinear_Act, self).forward(x)
            return self.quant_act(x)


class QuantAct_bert(Module):
    def __init__(self,
                 activation_bit=32,
                 momentum=0.99,
                 full_precision_flag=True,
                 running_stat=True,
                 quant_mode="asymmetric",
                 show_flag=False,
                 percentile=False):
        super(QuantAct_bert, self).__init__()

        self.activation_bit = activation_bit
        self.momentum = momentum
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.show_flag = show_flag

        self.percentile = percentile

        self.x_min = 0.
        self.x_max = 0.

        if quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            self.act_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

    def __repr__(self):
        return "{}(activation_bit={}, " \
               "full_precision_flag={}, Act_min: {}, " \
               "Act_max: {})".format(self.__class__.__name__, self.activation_bit,
                                     self.full_precision_flag, self.x_min, self.x_max)

    def reset_bits(self, weight_bit=8):
        self.full_precision_flag = False
        self.activation_bit = weight_bit

    def forward(self, x):

        if self.running_stat:
            if not self.percentile:
                x_min = x.data.min()
                x_max = x.data.max()
            else:
                x_min, x_max = get_percentile_min_max(x.clone().view(-1), 0.1, 99.9)

            # print("self.x_max = ", self.x_max)
            # Initialization
            if self.x_min == self.x_max:
                self.x_min = x_min
                self.x_max = x_max

            self.x_min = self.momentum * self.x_min + \
                (1. - self.momentum) * x_min
            self.x_max = self.momentum * self.x_max + \
                (1. - self.momentum) * x_max

        if not self.full_precision_flag:
            if self.quant_mode == "asymmetric":
                quant_act = self.act_function(x, self.activation_bit,
                                              self.x_min, self.x_max)
            elif self.quant_mode == "symmetric":
                magnitude = max(abs(self.x_min), abs(self.x_max))
                quant_act = self.act_function(x, self.activation_bit, magnitude)
            return quant_act
        else:
            return x


class QuantAct(Module):
    def __init__(self,
                 activation_bit,
                 out_channels=1,
                 momentum=0.99,
                 full_precision_flag=True,
                 running_stat=True,
                 quant_mode="asymmetric",
                 per_channel=False,
                 show_flag=False):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.momentum = momentum
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.show_flag = show_flag

        self.register_buffer("x_min", torch.Tensor(out_channels).zero_())
        self.register_buffer("x_max", torch.Tensor(out_channels).zero_())

        if quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            self.act_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

    def __repr__(self):
        return "{}(activation_bit={}, " \
               "full_precision_flag={}, Act_min: {}, " \
               "Act_max: {})".format(self.__class__.__name__, self.activation_bit,
                                     self.full_precision_flag, self.x_min[0], self.x_max[0])

    def forward(self, x):

        self.channel_num = x.data.size()[1]

        if self.running_stat:
            if self.per_channel:
                # print(1, time.time())
                x_transform = x.data.transpose(0, 1).contiguous().view(
                    self.channel_num, -1)
                x_min = x_transform.min(dim=1)[0]
                x_max = x_transform.max(dim=1)[0]
            # x_min = torch.zeros(self.channel_num).cuda()
            # x_max = torch.zeros(self.channel_num).cuda()
            # for i in range(self.channel_num):
            # 	x_min[i] = x.data[:, i, :, :].min()
            # 	x_max[i] = x.data[:, i, :, :].max()
            # print(x_min)
            # print(x_max)
            else:
                x_min = x.data.min()
                x_max = x.data.max()

            # print(self.x_min)

            # Initialization
            if self.x_min.size()[0] == 1:
                if self.x_min == self.x_max:
                    self.x_min = x_min
                    self.x_max = x_max

                if self.per_channel:
                    self.x_min = self.x_min.expand(self.channel_num).cuda()
                    self.x_max = self.x_max.expand(self.channel_num).cuda()
            # print(self.x_max)

            # exponential moving average (EMA)
            # use momentum to prevent the quantized values change greatly every
            # iteration

            self.x_min = self.momentum * self.x_min + \
                (1. - self.momentum) * x_min
            self.x_max = self.momentum * self.x_max + \
                (1. - self.momentum) * x_max

        # print(self.x_max)
        # print(2, time.time())

        if not self.full_precision_flag:
            if self.quant_mode == "asymmetric":
                if self.per_channel:
                    quant_act = self.act_function(x, self.activation_bit,
                                                  self.x_min, self.x_max, True)
                else:
                    quant_act = self.act_function(x, self.activation_bit,
                                                  self.x_min.item(),
                                                  self.x_max.item())
                if self.show_flag:
                    print(self.x_min, self.x_max)
                    print(x.data.min(), x.data.max())
                    print(quant_act)
                return quant_act
            elif self.quant_mode == "symmetric":
                magnitude = max(abs(self.x_min[0]), abs(self.x_max[0]))
                return self.act_function(x, self.activation_bit, magnitude)
        else:
            return x


class Quant_Conv2d(_Conv2d):
    def __init__(self,
                 weight_bit,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 full_precision_flag=True,
                 quant_mode="asymmetric",
                 per_channel=False):
        super(Quant_Conv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias)

        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit

        if quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

    def __repr__(self):
        s = super(Quant_Conv2d, self).__repr__()
        s.replace(
            ")", ", weight_bit={}, "
            "full_precision_flag={})".format(self.weight_bit,
                                             self.full_precision_flag))
        return s

    def forward(self, x):
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit)
        else:
            w = self.weight

        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class QuantBnConv2d(_Conv2d):
    def __init__(self,
                 weight_bit,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 eps=1e-5,
                 momentum=0.9,
                 affine=True,
                 full_precision_flag=True,
                 quant_mode="asymmetric"):
        super(QuantBnConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias)

        self.weight_bit = weight_bit
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.full_precision_flag = full_precision_flag

        if quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

        self.register_buffer("running_mean", torch.Tensor(out_channels))
        self.register_buffer("running_var", torch.Tensor(out_channels))

        if self.affine:
            self.alpha = Parameter(torch.Tensor(out_channels))
            self.beta = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("alpha", None)
            self.register_parameter("beta", None)
        self._reset_parameters()

    def _reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1.)
        if self.affine:
            self.alpha.data.uniform_()
            self.beta.data.zero_()

    def __repr__(self):
        conv_s = super(QuantBnConv2d, self).__repr__()
        bn_s = "BN:(eps={}, affine={})".format(self.eps, self.affine)
        s = "{}:(Conv:{}\n{}\n" "weight_bit={}, full_precision={})".format(
            self.__class__.__name__, conv_s, bn_s, self.weight_bit,
            self.full_precision)
        return s

    def forward(self, x):
        out_channels = self.weight.size(0)
        w_view = (out_channels, 1, 1, 1)

        if self.training:
            y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                         self.dilation, self.groups)
            y_data = y.transpose(0, 1).contiguous().view(out_channels, -1)
            y_mean = y_data.mean(1)
            y_var = y_data.var(1)

            self.running_mean = self.momentum * self.running_mean + \
                (1. - self.momentum) * y_mean.data
            self.running_var = self.momentum * self.running_var + \
                (1. - self.momentum) * y_var.data

            safe_std = torch.sqrt(y_var + self.eps)
            weight, bias = self._fold_bn(y_mean, safe_std, w_view)

            if not self.full_precision_flag:
                weight = self.weight_function(weight, self.weight_bit)

            return F.conv2d(x, weight, bias, self.stride, self.padding,
                            self.dilation, self.groups)
        else:
            y_mean = Variable(self.running_mean)
            safe_std = Variable(torch.sqrt(self.running_var + self.eps))

            weight, bias = self._fold_bn(y_mean, safe_std, w_view)

            if not self.full_precision_flag:
                weight = self.weight_function(weight, self.weight_bit)

            return F.conv2d(x, weight, bias, self.stride, self.padding,
                            self.dilation, self.groups)

    def _fold_bn(self, y_mean, safe_std, w_view):
        if self.affine:
            weight = self.weight * (self.alpha / safe_std).view(w_view)
            beta = self.beta - self.alpha * y_mean / safe_std
            if self.bias is not None:
                bias = self.alpha * self.bias / safe_std + beta
            else:
                bias = beta
        else:
            weight = self.weight / safe_std.view(w_view)
            beta = -y_mean / safe_std
            if self.bias is not None:
                bias = self.bias / safe_std + beta
            else:
                bias = beta
        return weight, bias
