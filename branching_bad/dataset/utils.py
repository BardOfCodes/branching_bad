import torch as th
import numpy as np


def format_train_data(batch):

    canvas = [x[0] for x in batch]
    canvas = th.stack(canvas, 0)
    canvas = canvas.reshape(-1, 64, 64)

    n_actions = [x[2] for x in batch]
    actions = [x[1] for x in batch]

    targets = []
    for ind, cur_n_actions in enumerate(n_actions):
        targets.append(actions[ind][:cur_n_actions])

    n_actions = np.array(n_actions)
    n_actions = th.from_numpy(n_actions).to(canvas.device)

    targets = np.concatenate(targets, 0)
    targets = th.from_numpy(targets).to(canvas.device)

    actions = np.stack(actions, 0)
    actions = th.from_numpy(actions).to(canvas.device)

    return canvas, actions, targets, n_actions
