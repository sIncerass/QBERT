## Helper functions
import torch
import numpy as np


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def de_variable(v):
    '''
    normalize the vector and detach it from variable
    '''

    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item() + 1e-6
    v = [vi / s for vi in v]
    return v


def run_method(method, *argv):
    """
        Pure syntax sugar.
        method: string 
    """
    # Pure syntax sugar.
    from power_iteration import get_hessian_trace, power_iteration, get_trace_family, power_iteration_eigenvecs

    if method == 'trace':
        get_hessian_trace(*argv)
    elif method == 'tracefamily':
        get_trace_family(*argv)
    elif method == 'poweriter':
        power_iteration(*argv)
    elif method == 'poweriter-eigvecs':
        power_iteration_eigenvecs(*argv)
    else:
        raise NotImplementedError("Only support trace and poweriteration.")


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    # v = [vi / s for vi in v]
    return v


def orthonormal(w, v_list):
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)


def total_number_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])