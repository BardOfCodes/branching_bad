import torch as th
import os
import random
import numpy as np
from .dataset_registry import DatasetRegistry
from branching_bad.domain import CSG2DExecutor, GenNNInterpreter

DATA_PATHS = {
    1: "synthetic/one_ops/expressions.txt",
    2: "synthetic/two_ops/expressions.txt",
    3: "synthetic/three_ops/expressions.txt",
    4: "synthetic/four_ops/expressions.txt",
    5: "synthetic/five_ops/expressions.txt",
    6: "synthetic/six_ops/expressions.txt",
    7: "synthetic/seven_ops/expressions.txt",
    8: "synthetic/eight_ops/expressions.txt",
    9: "synthetic/nine_ops/expressions.txt",
    10: "synthetic/ten_ops/expressions.txt",
    11: "synthetic/eleven_ops/expressions.txt",
    12: "synthetic/twelve_ops/expressions.txt",
    13: "synthetic/thirteen_ops/expressions.txt",
    14: "synthetic/fourteen_ops/expressions.txt",
    15: "synthetic/fifteen_ops/expressions.txt",
}
TRAIN_PROPORTION = 0.8


@DatasetRegistry.register("CSG2D")
class CSG2DDataset(th.utils.data.IterableDataset):

    def __init__(self, config, subset, device, *args, **kwargs):
        super(CSG2DDataset, self).__init__(*args, **kwargs)

        # load all the files
        expressions = []
        for n_ops, data_file in DATA_PATHS.items():
            if n_ops in config.EXPR_N_OPS:
                data_file = os.path.join(config.DATA_PATH, data_file)
                new_expressions = open(data_file, "r").readlines()
                new_expressions = [x.strip().split("__")
                                   for x in new_expressions]
                train_limit = int(len(new_expressions) * TRAIN_PROPORTION)
                if subset == "train":
                    selected_expressions = new_expressions[:train_limit]
                elif subset == "val":
                    selected_expressions = new_expressions[train_limit:]
                elif subset == "test":
                    selected_expressions = new_expressions
                expressions.extend(selected_expressions)

        if subset == "train":
            random.shuffle(expressions)

        self.expressions = expressions
        self.cache = dict()
        self.executor = CSG2DExecutor(config.EXECUTOR, device=device)
        self.model_translator = GenNNInterpreter(config.NN_INTERPRETER)
        self.epoch_size = config.EPOCH_SIZE
        self.max_actions = config.MAX_ACTIONS

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):

        # if index in self.cache:
        #     draw_transforms, inversion_array, intersection_matrix, actions, n_actions = self.cache[
        #         index]
        #     draw_transforms = {k: v.cuda() for k, v in draw_transforms.items()}
        #     intersection_matrix = intersection_matrix.cuda()
        #     inversion_array = inversion_array.cuda()
            
        # else:
        expression = self.expressions[index]
        draw_transforms, inversion_array, intersection_matrix = self.executor.compile(
            expression)
        actions = self.model_translator.expression_to_action(expression)
        n_actions = actions.shape[0]
        actions = np.pad(actions, ((0, self.max_actions - n_actions), (0, 0)),  mode="constant", constant_values=0)
        
        # store_transform = {k: v.cpu() for k, v in draw_transforms.items()}
        # self.cache[index] = (
        #     store_transform, inversion_array.cpu(), intersection_matrix.cpu(), actions, n_actions)

        execution = self.executor.execute(
            draw_transforms, inversion_array, intersection_matrix)
        execution = (execution < 0).float()

        return execution, actions, n_actions

    def __iter__(self):
        for i in range(self.epoch_size):
            yield self[i]
