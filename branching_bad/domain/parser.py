import math
from collections import defaultdict
import numpy as np
import torch as th
from .utils import euler2quat

class CSG2DParser():

    def __init__(self,device):

        self.command_n_param = {
            "sphere": 0,
            "cuboid": 0,
            "translate": 2,
            "rotate": 1,
            "scale": 2,
            "union": 0,
            "intersection": 0,
            "difference": 0,
        }
        self.command_symbol_to_type = {
            "sphere": "D",
            "cuboid": "D",
            "translate": "T",
            "rotate": "T",
            "scale": "T",
            "union": "B",
            "intersection": "B",
            "difference": "B",
        }

        self.device = device

    def parse(self, expression_list, use_torch=False):
        command_list = []
        draw_count = defaultdict(int)
        for ind, expr in enumerate(expression_list):
            command_symbol = expr.split("(")[0]
            if command_symbol == "$":
                # END OF EXPRESSION
                break
            else:
                command_type = self.command_symbol_to_type[command_symbol]
                command_dict = {'type': command_type, "symbol": command_symbol}
                n_param = self.command_n_param[command_symbol]
                if n_param > 0:
                    param_str = expr.split("(")[1][:-1]
                    param = np.array([float(x.strip())
                                     for x in param_str.split(",")])
                    if use_torch:
                        param = th.tensor(param, device=self.device)
                    if command_symbol == "rotate":
                        param = -param
                    elif command_symbol == "translate":
                        param = -param
                    elif command_symbol == "scale":
                        param = 1/param
                    command_dict['param'] = param
                if command_type == "D":
                    command_dict["ID"] = draw_count[command_symbol]
                    draw_count[command_symbol] += 1

                command_list.append(command_dict)
        return command_list, draw_count
