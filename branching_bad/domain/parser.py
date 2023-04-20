import math
from collections import defaultdict
import numpy as np
import torch as th
from .utils import euler2quat

class CSG2DParser():

    def __init__(self,device):

        self.command_n_param = {
            "sphere": 5,
            "cuboid": 5,
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
        self.transform_sequence = ["translate", "scale", "rotate"]

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
                command_dict["ID"] = draw_count[command_symbol]
                draw_count[command_symbol] += 1
                n_param = self.command_n_param[command_symbol]
                if n_param > 0:
                    param_str = expr.split("(")[1][:-1]
                    param = np.array([float(x.strip())
                                     for x in param_str.split(",")])
                    
                    param[:2] *= -1
                    param[2:4] **= -1
                    param[4] *= math.pi / 180.
                    # Now convert into MCSG3D
                    for ind, command_symbol in enumerate(self.transform_sequence):
                        transform_dict = {
                            'type': "T", "symbol": command_symbol, "param": param[ind*2: (ind+1)*2]}
                        command_list.append(transform_dict)
                command_list.append(command_dict)

        return command_list, draw_count
