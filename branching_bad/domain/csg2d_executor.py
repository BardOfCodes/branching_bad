from .parser import CSG2DParser
from .parser import MacroParser
from .compiler import CSG2DCompiler
from collections import defaultdict
import numpy as np
import torch as th


class CSG2DExecutor:

    def __init__(self, config, device):

        self.resolution = config.RESOLUTION
        self.parser = CSG2DParser(device)
        self.compiler = CSG2DCompiler(self.resolution, device)

    def compile(self, expression):
        parsed_graphs, draw_count = self.parser.parse(expression)
        draw_transforms, inversion_array, intersection_matrix = self.compiler.fast_sub_compile(
            parsed_graphs, draw_count)
        return draw_transforms, inversion_array, intersection_matrix

    def execute(self, draw_transforms, inversion_array, intersection_matrix):

        canvas = self.compiler.evaluate(
            draw_transforms, inversion_array, intersection_matrix)
        return canvas

    def get_cmd_list(self,):
        return self.parser.get_cmd_list()

    def set_device(self, device):
        self.parser.set_device(device)
        self.compiler.set_device(device)

    def eval_batch_execute(self, pred_expressions_batch):

        storage_count = []
        cache = []
        counter = 0

        for ind, expressions in enumerate(pred_expressions_batch):
            start_counter = counter
            for expr in expressions:
                expr_obj = self.compile(expr)
                cache.append(expr_obj)
            counter += len(expressions)
            end_counter = counter
            storage_count.append((start_counter, end_counter))

        collapsed_draws = defaultdict(list)
        collapsed_inversions = defaultdict(list)
        all_draws = []
        all_graphs = []

        for val in cache:

            draw_transforms, draw_inversions, graph = val

            all_draws.append(draw_transforms)
            all_graphs.append(graph)
            for draw_type, transforms in draw_transforms.items():
                collapsed_draws[draw_type].extend(transforms)
                collapsed_inversions[draw_type].extend(
                    draw_inversions[draw_type])

        for draw_type in draw_transforms.keys():
            if len(collapsed_draws[draw_type]) == 0:
                continue
            collapsed_inversions[draw_type] = th.from_numpy(
                np.array(collapsed_inversions[draw_type])).to(self.compiler.device)
            collapsed_inversions[draw_type] = collapsed_inversions[draw_type].unsqueeze(
                1)
            collapsed_draws[draw_type] = th.stack(
                collapsed_draws[draw_type], 0).to(self.compiler.device)
        canvas = self.compiler.batch_evaluate_with_graph(collapsed_draws, all_draws,
                                                         collapsed_inversions,
                                                         all_graphs)

        canvas = canvas.reshape(-1, 64, 64)
        canvas = (canvas <= 0).float()
        pred_canvas = []
        for ind, (start, end) in enumerate(storage_count):
            pred_canvas.append(canvas[start:end])

        return pred_canvas


class MacroExecutor(CSG2DExecutor):

    def __init__(self, config, device):

        self.resolution = config.RESOLUTION
        self.parser = MacroParser(device)
        self.compiler = CSG2DCompiler(self.resolution, device)

    def update_macros(self, macro_dict):
        self.parser.update_macros(macro_dict)

    def get_dsl_size(self):
        return self.parser.get_dsl_size()
