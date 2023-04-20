import numpy as np
import torch as th

from .constants import (ROTATE_MULTIPLIER, SCALE_ADDITION, TRANSLATE_MIN, TRANSLATE_MAX,
                        SCALE_MIN, SCALE_MAX, ROTATE_MIN, ROTATE_MAX, DRAW_MIN, DRAW_MAX, CONVERSION_DELTA)
empty_param_actions = ["union", "intersection", "difference", "$"]
draw_commands = ["sphere", "cuboid"]
transform_commands = ["translate", "rotate", "scale"]


class GenNNInterpreter:

    def __init__(self, config):
        self.quantization = config.QUANTIZATION
        self.empty_params = np.zeros((5), dtype=np.int32)
        self.empty_param_action = np.array([0])
        self.active_param_action = np.array([1])

        self.conversion_delta = CONVERSION_DELTA

        self.two_scale_delta = (
            2 - 2 * self.conversion_delta)/(self.quantization - 1)

        self.command_index = {
            'sphere': 0,
            'cuboid': 1,
            'union': 2,
            'intersection': 3,
            'difference': 4,
            '$': 5,
        }

        self.index_to_expr = {}
        for key, value in self.command_index.items():
            self.index_to_expr[value] = key

        for key, value in self.command_index.items():
            self.command_index[key] = np.array([value], dtype=np.int32)
        self.expr_to_n_params = {
            'sphere': 5,
            'cuboid': 5,
            'union': 0,
            'intersection': 0,
            'difference': 0,
            '$': 0,

        }

    def expression_to_action(self, expression_list):
        action_array = []
        for expr in expression_list:
            cmd_type_params_action_type = self.single_expression_to_action(
                expr)
            action_array.append(cmd_type_params_action_type)
        last_expr = expression_list[-1]
        if last_expr != "$":
            cmd_type_params_action_type = self.single_expression_to_action("$")
            action_array.append(cmd_type_params_action_type)
        action_array = np.stack(action_array, 0)

        return action_array

    def single_expression_to_action(self, expr):
        command_symbol = expr.split("(")[0]
        if command_symbol in empty_param_actions:
            cmd_type = self.command_index[command_symbol]
            params = self.empty_params.copy()
            action_type = self.empty_param_action.copy()
            action_array = np.concatenate([cmd_type, params, action_type], 0)

        elif command_symbol in draw_commands:
            cmd_type = self.command_index[command_symbol]
            action_type = self.active_param_action.copy()
            # Then there are 3 transforms:
            param_str = expr.split("(")[1][:-1]
            param = np.array([float(x.strip()) for x in param_str.split(",")])
            translate_param = np.clip(
                param[:2], TRANSLATE_MIN + self.conversion_delta, TRANSLATE_MAX - self.conversion_delta)
            scale_param = np.clip(
                param[2:4], SCALE_MIN + self.conversion_delta, SCALE_MAX - self.conversion_delta)
            scale_param -= SCALE_ADDITION
            rotate_param = np.clip(
                param[4:5], ROTATE_MIN + self.conversion_delta, ROTATE_MAX - self.conversion_delta)
            rotate_param = rotate_param / ROTATE_MULTIPLIER
            # param = [self.mid_point + np.round( (x-1.25)/self.two_scale_delta) for x in param]
            param = np.concatenate(
                [translate_param, scale_param, rotate_param], 0)
            param = (param - (-1 + self.conversion_delta)) / \
                self.two_scale_delta
            param = np.round(param).astype(np.uint32)
            action_array = np.concatenate([cmd_type, param, action_type], 0)
        return action_array

    def action_to_expression(self, actions):
        _size = actions.shape[0]
        pointer = 0
        expression_list = []
        for pointer in range(_size):
            cur_actions = actions[pointer]
            cur_command = cur_actions[0]
            cur_expr = self.index_to_expr[cur_command]
            n_param = self.expr_to_n_params[cur_expr]
            if n_param > 0:
                # Has to be for transform or mirror
                param = np.array(cur_actions[1:])
                translate_param = -1 + self.conversion_delta + \
                    param[:2] * self.two_scale_delta
                scale_param = -1 + self.conversion_delta + \
                    param[2:4] * self.two_scale_delta + SCALE_ADDITION
                rotate_param = (-1 + self.conversion_delta +
                                param[4:5] * self.two_scale_delta) * ROTATE_MULTIPLIER

                param = np.concatenate(
                    [translate_param, scale_param, rotate_param], 0)
                param_str = ", ".join([f"{x}" for x in param])
                cur_expr = f"{cur_expr}({param_str})"
            expression_list.append(cur_expr)

        return expression_list

    # compute state space as well.
