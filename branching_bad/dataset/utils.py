import torch as th
import numpy as np
from collections import defaultdict


class Collator:
    def __init__(self, compiler):
        self.compiler = compiler

    def __call__(self, batch):
        return format_train_data_with_compiler(batch, self.compiler)


class PLADCollator(Collator):
    def __call__(self, batch):
        # pop out the target:
        return plad_collate_with_compiler(batch, self.compiler)


def val_collate_fn(batch):
    batch = th.stack(batch, dim=0).to("cuda")
    return batch, None, None, None


def wrap_format_with_compiler(format_func, compiler):

    def inner_func(batch):
        batch = format_func(batch, compiler)
        return batch
    return inner_func


def plad_collate_with_compiler(batch, compiler):

    targets = []
    for val in batch:
        targets.append(val[4])
    canvas, actions, actions_validity, n_actions = format_train_data_with_compiler(
        batch, compiler)
    targets = np.stack(targets, 0)
    targets = th.from_numpy(targets).to(canvas.device)
    canvas = th.cat([canvas, targets], 0)
    actions = th.cat([actions, actions], 0)
    actions_validity = th.cat([actions_validity, actions_validity], 0)
    n_actions = th.cat([n_actions, n_actions], 0)

    return canvas, actions, actions_validity, n_actions


def format_train_data_with_compiler(batch, compiler):

    collapsed_draws = defaultdict(list)
    collapsed_inversions = defaultdict(list)
    all_draws = []
    all_graphs = []

    actions = []
    actions_validity = []
    n_actions = []
    for val in batch:

        draw_transforms, draw_inversions, graph = val[0]

        all_draws.append(draw_transforms)
        all_graphs.append(graph)
        for draw_type, transforms in draw_transforms.items():
            collapsed_draws[draw_type].extend(transforms)
            collapsed_inversions[draw_type].extend(draw_inversions[draw_type])

        actions.append(val[1])
        actions_validity.append(val[2])
        n_actions.append(val[3])

    for draw_type in draw_transforms.keys():
        if len(collapsed_draws[draw_type]) == 0:
            continue
        collapsed_inversions[draw_type] = th.from_numpy(
            np.array(collapsed_inversions[draw_type])).to(compiler.device)
        collapsed_inversions[draw_type] = collapsed_inversions[draw_type].unsqueeze(
            1)
        collapsed_draws[draw_type] = th.stack(
            collapsed_draws[draw_type], 0).to(compiler.device)
    canvas = compiler.batch_evaluate_with_graph(collapsed_draws, all_draws,
                                                collapsed_inversions,
                                                all_graphs)

    canvas = canvas.reshape(-1, 64, 64)
    canvas = (canvas <= 0).float()

    actions = np.stack(actions, 0)
    actions = th.from_numpy(actions).to(canvas.device)

    actions_validity = np.concatenate(actions_validity, 0)
    actions_validity = th.from_numpy(actions_validity).to(canvas.device)

    n_actions = np.array(n_actions)
    n_actions = th.from_numpy(n_actions).to(canvas.device)

    return canvas, actions, actions_validity, n_actions


def format_train_data(batch):

    canvas = []
    actions = []
    actions_validity = []
    n_actions = []
    for val in batch:
        canvas.append(val[0])
        actions.append(val[1])
        actions_validity.append(val[2])
        n_actions.append(val[3])

    canvas = th.stack(canvas, 0)
    canvas = canvas.reshape(-1, 64, 64)

    actions = np.stack(actions, 0)
    actions = th.from_numpy(actions).to(canvas.device)

    actions_validity = np.concatenate(actions_validity, 0)
    actions_validity = th.from_numpy(actions_validity).to(canvas.device)

    n_actions = np.array(n_actions)
    n_actions = th.from_numpy(n_actions).to(canvas.device)

    return canvas, actions, actions_validity, n_actions
