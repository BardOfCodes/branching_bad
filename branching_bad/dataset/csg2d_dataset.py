import torch as th
import os
import random
import numpy as np
import h5py
import _pickle as cPickle
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
# TRAIN_PROPORTION = 0.0002

CAD_FILE = 'cad/cad.h5' 

@DatasetRegistry.register("SynthCSG2DDataset")
class SynthCSG2DDataset(th.utils.data.IterableDataset):

    def __init__(self, config, subset, device, *args, **kwargs):
        super(SynthCSG2DDataset, self).__init__(*args, **kwargs)

        self.device = device
        self.executor = CSG2DExecutor(config.EXECUTOR, device="cpu")
        self.model_translator = GenNNInterpreter(config.NN_INTERPRETER)
        self.epoch_size = config.EPOCH_SIZE
        self.max_actions = config.MAX_ACTIONS
        self.action_validity = np.zeros((self.max_actions, 7), dtype=bool)
        
        self.bake_file = config.BAKE_FILE
        self.cache = {}
        if not os.path.exists(self.bake_file):
            # load all the files
            expressions = self.get_expressions(config, subset)
            self.expressions = expressions
            cache = self.bake_dataset()
            self.cache = {i:cache[i] for i in range(len(cache))}
        else:
            cache = cPickle.load(open(self.bake_file, "rb"))
            if subset == "train":
                random.shuffle(cache)
            self.cache = {i:cache[i] for i in range(len(cache))}
            self.expressions = [x[-1] for x in cache]

    def get_expressions(self, config, subset):
        
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
            
        return expressions
    def bake_dataset(self):
        
        cache_obj = []
        print("Baking dataset...")
        for i in range(self.epoch_size):
            if i % 1000 == 0:
                print(f"{i}/{self.epoch_size}")
            expr_obj, actions, action_validity, n_actions = self.make_item(i)
            expression = self.expressions[i]
            cache_entry = (expr_obj, actions, action_validity, n_actions, expression)
            cache_obj.append(cache_entry)
        print("Done baking dataset.")
        
        cPickle.dump(cache_obj, open(self.bake_file, "wb"))
        
    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        
        if index in self.cache.keys():
            expr_obj, actions, action_validity, n_actions, _ = self.cache[index]
        else:
            expr_obj, actions, action_validity, n_actions = self.make_item(index)
            self.cache[index] = (expr_obj, actions, action_validity, n_actions)
        draw_obj = {k:th.from_numpy(v) for k,v in expr_obj[0].items()}
        expr_obj = (draw_obj, expr_obj[1], expr_obj[2])
        return expr_obj, actions, action_validity, n_actions

    def __iter__(self):
        self.executor.set_device(self.device)
        for i in range(self.epoch_size):
            yield self[i]

    def make_item(self, index):
        
        expression = self.expressions[index]
        draw_transforms, inversion_array, intersection_matrix = self.executor.compile(
            expression)
        
        actions = self.model_translator.expression_to_action(expression)
        n_actions = actions.shape[0]
        actions = np.pad(actions, ((0, self.max_actions - n_actions),
                        (0, 0)),  mode="constant", constant_values=0)
        action_validity = self.action_validity.copy()
        action_validity[:n_actions, :] = True
        # map zeros when param is not to be measured.
        action_validity[actions[:,-1]==0, 1:] = False
        
        draw_transforms = {k:v.cpu().numpy() for k,v in draw_transforms.items()}
        
        expr_obj = (draw_transforms, inversion_array, intersection_matrix)
        return expr_obj, actions, action_validity, n_actions
    
class CADCSG2DDataset(SynthCSG2DDataset):
    
    
    def __init__(self, config, subset, device, *args, **kwargs):
        super(SynthCSG2DDataset, self).__init__(*args, **kwargs)

        # load all the files
        data_file = os.path.join(config.DATA_PATH, CAD_FILE)
        
        hf = h5py.File(data_file, "r")
        if self.mode == "train":
            data = np.array(hf.get(name="%s_images" % "train"))
        elif self.mode == "val":
            data = np.array(hf.get(name="%s_images" % "val"))
        elif self.mode == "test":
            data = np.array(hf.get(name="%s_images" % "test"))
        hf.close()
        self.targets = data

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):

        output = self.targets[index].copy()
        output = th.tensor(output)
        
        return output
