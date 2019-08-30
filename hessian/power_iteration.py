"""BERT finetuning runner."""
import torch
from tqdm import tqdm, trange
import csv
import os
import logging
from os.path import dirname, join

from utils import group_product, de_variable, orthonormal, total_number_parameters, group_add

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def add_poweriter_arguments(parser):
    parser.add_argument(
        '--relative_tolerance',
        type=float,
        default=1e-2,
        metavar='float',
        help='Relative tolerance for stopping power iteration (default: 1e-4)')
    parser.add_argument(
        '--data_percentage',
        type=float,
        default=0.1,
        metavar='float',
        help='Data percentage for stopping power iteration (default: 0.1)')
    parser.add_argument(
        '--num_iter',
        type=int,
        default=100,
        metavar='int',
        help='Number of iterations for trace calculation (default: 20)')


def get_hessian_trace(train_dataloader, model, get_loss, args, device):
    """
    compute the trace/num_params of model parameters Hessian with a full dataset.
    """
    percentage_index = len(train_dataloader.dataset) * \
        args.data_percentage / args.train_batch_size
    print(f'percentage_index: {percentage_index}')

    # change the model to evaluation mode, otherwise the batch Normalization Layer will change.
    # If you call this functino during training, remember to change the mode
    # back to training mode.
    model.eval()
    model.zero_grad()

    parent_dir = join(dirname(__file__), os.pardir)

    results_dir = join(parent_dir, 'results')

    csv_path = join(
        results_dir,
        f'{args.task_name}-{args.data_percentage}-seed-{args.seed}-trace.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(
        csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['block', 'iters', 'eigenvalue'])

    # make sure requires_grad are true
    for module in model.modules():
        for param in module.parameters():
            param.requires_grad = True

    for block_id in range(12):
        model_block = model.module.bert.encoder.layer[block_id]

        eigenvalue_sum = 0
        for i in range(args.num_iter):
            v = [
                torch.randn(p.size()).to(device)
                for p in model_block.parameters()
            ]
            v = de_variable(v)

            acc_Hv = [
                torch.zeros(p.size()).to(device)
                for p in model_block.parameters()
            ]
            for step, batch in enumerate(
                    tqdm(train_dataloader, desc="Iteration")):
                if step < percentage_index:

                    loss = get_loss(batch)

                    loss.backward(create_graph=True)
                    grads = [param.grad for param in model_block.parameters()]
                    params = model_block.parameters()

                    Hv = torch.autograd.grad(
                        grads,
                        params,
                        grad_outputs=v,
                        only_inputs=True,
                        retain_graph=True)
                    acc_Hv = [
                        acc_Hv_p + Hv_p for acc_Hv_p, Hv_p in zip(acc_Hv, Hv)
                    ]
                    model.zero_grad()
            # calculate raylay quotients
            eigenvalue = group_product(acc_Hv, v).item() / percentage_index
            eigenvalue_sum += eigenvalue

            writer.writerow([f'{block_id}', f'{i}', f'{eigenvalue}'])
            csv_file.flush()

        print(
            f'result for iter {args.num_iter} is : {eigenvalue_sum/args.num_iter}'
        )


def get_trace_family(train_dataloader, model, get_loss, args, device, n_v=1):
    """
    compute the trace/num_params of model parameters Hessian with a full dataset.
    """
    percentage_index = len(train_dataloader.dataset) * \
        args.data_percentage / args.train_batch_size
    print(f'percentage_index: {percentage_index}')

    # change the model to evaluation mode, otherwise the batch Normalization Layer will change.
    # If you call this functino during training, remember to change the mode
    # back to training mode.
    model.eval()
    model.zero_grad()

    parent_dir = join(dirname(__file__), os.pardir)

    results_dir = join(parent_dir, 'results')

    csv_path = join(
        results_dir,
        f'{args.task_name}-{args.data_percentage}-seed-{args.seed}-{args.method}.csv'
    )
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(
        csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(
        ['block', 'iters', 'slq-trace', 'slq-abstrace', 'slq-quadtrace'])

    # make sure requires_grad are true
    for module in model.modules():
        for param in module.parameters():
            param.requires_grad = True

    for block_id in range(12):
        slq_trace = 0
        slq_quadtrace = 0
        slq_abstrace = 0
        for _ in range(n_v):
            model_block = model.module.bert.encoder.layer[block_id]

            alpha_list = []
            beta_list = []
            v_list = []
            beta = 0
            w = None

            v = [
                torch.randn(p.size()).to(device)
                for p in model_block.parameters()
            ]
            v = de_variable(v)
            for i in range(args.num_iter):

                acc_Hv = [
                    torch.zeros(p.size()).to(device)
                    for p in model_block.parameters()
                ]
                for step, batch in enumerate(
                        tqdm(train_dataloader, desc="Iteration")):
                    if step < percentage_index:

                        loss = get_loss(batch)

                        loss.backward(create_graph=True)
                        grads = [
                            param.grad for param in model_block.parameters()
                        ]
                        params = model_block.parameters()

                        Hv = torch.autograd.grad(
                            grads,
                            params,
                            grad_outputs=v,
                            only_inputs=True,
                            retain_graph=True)
                        acc_Hv = [
                            acc_Hv_p + Hv_p
                            for acc_Hv_p, Hv_p in zip(acc_Hv, Hv)
                        ]
                        model.zero_grad()
                acc_Hv = [
                    w * total_number_parameters(model_block) for w in acc_Hv
                ]
                if i == 0:
                    # Do specific manipluations.
                    # Lanczos algo on wiki, step 2, first iteration.
                    alpha = group_product(acc_Hv, v).item() / percentage_index
                    # Reviwer Double-check: here I choose to average the hessian vector.
                    w = [w / percentage_index for w in acc_Hv]

                    w = [w_i - alpha * v_i for w_i, v_i in zip(w, v)]
                    v_list.append(v)

                    alpha_list.append(alpha)

                    # Prepare for next step
                    beta = (group_product(w, w)**0.5).detach()
                    beta_list.append(beta)
                    # v = [w_i / beta for w_i in w]
                    v = orthonormal(w, v_list)
                    v_list.append(v)

                else:

                    # calculate raylay quotients
                    eigenvalue = group_product(acc_Hv,
                                               v).item() / percentage_index
                    w = [w / percentage_index for w in acc_Hv]
                    alpha = eigenvalue
                    alpha_list.append(alpha)

                    w = [
                        w_i - alpha * v_i - beta * v_old_i
                        for w_i, v_i, v_old_i in zip(w, *v_list[-2:])
                    ]

                    beta = (group_product(w, w)**0.5).detach()
                    beta_list.append(beta)
                    logger.info(f'num_iter: {i}, beta: {beta}')
                    assert beta != 0
                    # v = [w_i / beta for w_i in w]
                    v = orthonormal(w, v_list)
                    v_list.append(v)

            assert len(alpha_list) == args.num_iter
            beta_list.pop()  # The last one is unesscarry
            assert len(beta_list) == (args.num_iter - 1)

            m = args.num_iter
            T = torch.zeros(m, m).to(device)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]
            a_, b_ = torch.eig(T, eigenvectors=True)

            eigen_list = a_[:, 0]
            weight_list = b_[0, :]**2
            slq_trace += eigen_list @ weight_list
            slq_quadtrace += (eigen_list**2) @ weight_list
            slq_abstrace += torch.abs(eigen_list) @ weight_list

        slq_trace /= n_v
        slq_quadtrace /= n_v
        slq_abstrace /= n_v
        writer.writerow([
            f'{block_id}', f'{i}', f'{slq_trace}', f'{slq_abstrace}',
            f'{slq_quadtrace}'
        ])
        csv_file.flush()


def power_iteration(train_dataloader, model, get_loss, args, device):

    percentage_index = len(train_dataloader.dataset) * \
        args.data_percentage / args.train_batch_size
    print(f'percentage_index: {percentage_index}')

    model.eval()

    parent_dir = join(dirname(__file__), os.pardir)

    results_dir = join(parent_dir, 'results')

    csv_path = join(
        results_dir,
        f'{args.task_name}-{args.data_percentage}-seed-{args.seed}-eigens.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(
        csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['block', 'iters', 'max_eigenvalue'])

    # make sure requires_grad are true
    for module in model.modules():
        for param in module.parameters():
            param.requires_grad = True

    block_id = 0
    for block_id in range(12):
        model_block = model.module.bert.encoder.layer[block_id]

        v = [
            torch.randn(p.size()).to(device) for p in model_block.parameters()
        ]
        v = de_variable(v)

        lambda_old, lambdas = 0., 1.
        i = 0
        while (abs((lambdas - lambda_old) / lambdas) >= 0.01):

            lambda_old = lambdas

            acc_Hv = [
                torch.zeros(p.size()).cuda() for p in model_block.parameters()
            ]
            for step, batch in enumerate(
                    tqdm(train_dataloader, desc="Iteration")):
                if step < percentage_index:

                    loss = get_loss(batch)

                    loss.backward(create_graph=True)
                    grads = [param.grad for param in model_block.parameters()]
                    params = model_block.parameters()

                    Hv = torch.autograd.grad(
                        grads,
                        params,
                        grad_outputs=v,
                        only_inputs=True,
                        retain_graph=True)
                    acc_Hv = [
                        acc_Hv_p + Hv_p for acc_Hv_p, Hv_p in zip(acc_Hv, Hv)
                    ]
                    model.zero_grad()
            # calculate raylay quotients
            lambdas = group_product(acc_Hv, v).item() / percentage_index

            v = de_variable(acc_Hv)
            logger.info(f'lambda: {lambdas}')
            writer.writerow([f'{block_id}', f'{i}', f'{lambdas}'])
            csv_file.flush()

            i += 1


def power_iteration_eigenvecs(train_dataloader, model, get_loss, args, device):
    # Also eig vectors

    percentage_index = len(train_dataloader.dataset) * \
        args.data_percentage / args.train_batch_size
    print(f'percentage_index: {percentage_index}')

    model.eval()

    parent_dir = join(dirname(__file__), os.pardir)

    results_dir = join(parent_dir, 'results')

    csv_path = join(
        results_dir,
        f'{args.task_name}-{args.data_percentage}-seed-{args.seed}-eigens.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(
        csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['block', 'iters', 'max_eigenvalue'])

    # make sure requires_grad are true
    for module in model.modules():
        for param in module.parameters():
            param.requires_grad = True

    block_id = 0

    def get_eigens():
        vs = []
        lambdass = []
        for block_id in range(12):
            model_block = model.module.bert.encoder.layer[block_id]

            v = [
                torch.randn(p.size()).to(device)
                for p in model_block.parameters()
            ]
            v = de_variable(v)

            lambda_old, lambdas = 0., 1.
            i = 0
            while (abs((lambdas - lambda_old) / lambdas) >= 0.01):

                lambda_old = lambdas

                acc_Hv = [
                    torch.zeros(p.size()).cuda()
                    for p in model_block.parameters()
                ]
                for step, batch in enumerate(
                        tqdm(train_dataloader, desc="Iteration")):
                    if step < percentage_index:

                        loss = get_loss(batch)

                        loss.backward(create_graph=True)
                        grads = [
                            param.grad for param in model_block.parameters()
                        ]
                        params = model_block.parameters()

                        Hv = torch.autograd.grad(
                            grads,
                            params,
                            grad_outputs=v,
                            only_inputs=True,
                            retain_graph=True)
                        acc_Hv = [
                            acc_Hv_p + Hv_p
                            for acc_Hv_p, Hv_p in zip(acc_Hv, Hv)
                        ]
                        model.zero_grad()
                # calculate raylay quotients
                lambdas = group_product(acc_Hv, v).item() / percentage_index

                v = de_variable(acc_Hv)
                logger.info(f'lambda: {lambdas}')
                writer.writerow([f'{block_id}', f'{i}', f'{lambdas}'])
                csv_file.flush()

                i += 1
            vs.append(v)
            lambdass.append(lambdas)
            break
        return vs, lambdass

    v1, lambdas1 = get_eigens()
    # calculate second eig vec
    logger.info("Calculate the second eig vec")
    block_id = 0
    vs = []
    lambdass = []
    for block_id in range(12):
        model_block = model.module.bert.encoder.layer[block_id]

        v = [
            torch.randn(p.size()).to(device) for p in model_block.parameters()
        ]
        v = de_variable(v)

        lambda_old, lambdas = 0., 1.
        i = 0
        while (abs((lambdas - lambda_old) / lambdas) >= 0.01):

            lambda_old = lambdas

            acc_Hv = [
                torch.zeros(p.size()).cuda() for p in model_block.parameters()
            ]
            for step, batch in enumerate(
                    tqdm(train_dataloader, desc="Iteration")):
                if step < percentage_index:

                    loss = get_loss(batch)

                    loss.backward(create_graph=True)
                    grads = [param.grad for param in model_block.parameters()]
                    params = model_block.parameters()

                    Hv = torch.autograd.grad(
                        grads,
                        params,
                        grad_outputs=v,
                        only_inputs=True,
                        retain_graph=True)
                    acc_Hv = [
                        acc_Hv_p + Hv_p for acc_Hv_p, Hv_p in zip(acc_Hv, Hv)
                    ]
                    model.zero_grad()
            acc_Hv = [acc_hv / percentage_index for acc_hv in acc_Hv]
            # make the product be (H - lambda * v1 * v1^T) v
            tmp = lambdas1[block_id] * group_product(v1[block_id], v).item()

            acc_Hv = group_add(acc_Hv, v1[block_id], alpha=-tmp)
            # calculate raylay quotients
            lambdas = group_product(acc_Hv, v).item() / percentage_index

            v = de_variable(acc_Hv)
            logger.info(f'lambda: {lambdas}')
            writer.writerow([f'{block_id}', f'{i}', f'{lambdas}'])
            csv_file.flush()

            i += 1

        vs.append(v)
        lambdass.append(lambdas)
        break
    v2, lambdas2 = vs, lambdass
    landscape_data = {
        'v1': v1,
        'v2': v2,
        'lambdas1': lambdas1,
        'lambdas2': lambdas2
    }
    import pickle
    outfile = open(f'results/{args.task_name}_landscape_data', 'wb')
    pickle.dump(landscape_data, outfile)
    outfile.close()