import copy
import math
from collections import defaultdict
import numpy as np
import torch as th
from .utils import euler2quat
from .constants import TRANSLATE_MIN, TRANSLATE_MAX, ROTATE_MIN, ROTATE_MAX, SCALE_MIN, SCALE_MAX, ROTATE_MULTIPLIER, SCALE_ADDITION, CONVERSION_DELTA


class CSG2DParser():

    def __init__(self, device):

        self.command_n_param = {
            "sphere": 5,
            "cuboid": 5,
            "union": 0,
            "intersection": 0,
            "difference": 0,
            # "translate": 2,
            # "scale": 2,
            # "rotate": 1,
        }
        self.command_symbol_to_type = {
            "sphere": "D",
            "cuboid": "D",
            # "translate": "T",
            # "scale": "T",
            # "rotate": "T",
            "union": "B",
            "intersection": "B",
            "difference": "B",
        }
        self.transform_sequence = ["translate", "scale", "rotate"]

        self.device = device
        self.novel_cmds = []

    def get_novel_cmds(self):
        return self.novel_cmds

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
                    command_dict["ID"] = draw_count[command_symbol]
                    draw_count[command_symbol] += 1
                    all_transforms = self.get_transforms(expr)
                    command_list.extend(all_transforms)
                command_list.append(command_dict)

        return command_list, draw_count

    def get_cmd_list(self):
        cmds = [x for x, y in self.command_n_param.items()]
        cmds += ["$"]
        return cmds

    def get_transforms(self, expr, adjust_param=True):
        param_str = expr.split("(")[1][:-1]
        param = np.array([float(x.strip())
                          for x in param_str.split(",")])
        if adjust_param:
            param[:2] *= -1
            param[2:4] = 1/(param[2:4] + CONVERSION_DELTA)
            param[4] *= math.pi / 180.
        # Now convert into MCSG3D
        all_transforms = []
        for ind, command_symbol in enumerate(self.transform_sequence):
            transform_dict = {
                'type': "T", "symbol": command_symbol, "param": param[ind*2: (ind+1)*2]}

            all_transforms.append(transform_dict)
        return all_transforms

    def set_device(self, device):
        self.device = device


class MacroParser(CSG2DParser):
    """Convert Macros when found"""

    def __init__(self, device):
        super(MacroParser, self).__init__(device)
        self.macro_extension_dict = {}
        self.fcsg_mode = True
        self.named_expression = {}
        self.deprecated_cmds = {}

    def update_macros(self, add_macros, remove_macros):
        # remove the existing "novel cmds"
        self.novel_cmds = []
        # remove:
        for macro in remove_macros:
            name = macro.name
            assert name in self.named_expression.keys()
            self.deprecated_cmds[name] = copy.copy(self.named_expression[name])
            del self.named_expression[name]
            del self.command_n_param[name]
            del self.command_symbol_to_type[name]
            del self.macro_extension_dict[name]
            # self.novel_cmds.remove(name)
        for macro in add_macros:
            # expression = self.get_expression(macro.commands, clip=True,
            #                                  quantize=True, resolution=33)
            expression = macro.subexpression
            name = macro.name
            self.named_expression[name] = macro.subexpression
            self.command_n_param[name] = 5
            self.command_symbol_to_type[name] = "M"
            self.macro_extension_dict[name] = expression
            self.novel_cmds.append(name)
            
        

    def parse(self, expression_list, adjust_param=True, use_torch=False):
        command_list = []
        draw_count = defaultdict(int)
        pointer = 0
        max_pointer = len(expression_list)
        while (pointer < max_pointer):
            expr = expression_list[pointer]
            command_symbol = expr.split("(")[0]
            if command_symbol == "$":
                # END OF EXPRESSION
                break
            else:
                command_type = self.command_symbol_to_type[command_symbol]
                if command_type == "M":
                    all_transforms = self.get_transforms(expr, adjust_param)
                    command_list.extend(all_transforms)
                    # Add the remaining to the list:
                    extension = self.macro_extension_dict[command_symbol]
                    expression_list = expression_list[:pointer] + \
                        extension + expression_list[pointer+1:]
                    max_pointer = len(expression_list)
                    pointer -= 1

                else:
                    command_dict = {'type': command_type,
                                    "symbol": command_symbol}
                    n_param = self.command_n_param[command_symbol]
                    if n_param > 0:
                        command_dict["ID"] = draw_count[command_symbol]
                        draw_count[command_symbol] += 1
                        all_transforms = self.get_transforms(
                            expr, adjust_param)
                        command_list.extend(all_transforms)
                    command_list.append(command_dict)
            pointer += 1

        return command_list, draw_count

    def parse_for_graph(self, expression_list, adjust_param=True, use_torch=False):
        command_list = []
        draw_count = defaultdict(int)
        pointer = 0
        max_pointer = len(expression_list)
        while (pointer < max_pointer):
            expr = expression_list[pointer]
            command_symbol = expr.split("(")[0]
            if command_symbol == "$":
                # END OF EXPRESSION
                break
            else:
                command_type = self.command_symbol_to_type[command_symbol]
                if command_type == "M":
                    all_transforms = self.get_transforms(expr, adjust_param)
                    all_transforms[-1]['true_expression'] = expr
                    all_transforms[-1]['MACRO'] = command_symbol
                    command_list.extend(all_transforms)
                    # Add the remaining to the list:
                    # macro_marker = command_symbol
                    # mark_macro = True
                    extension = self.macro_extension_dict[command_symbol]
                    expression_list = expression_list[:pointer] + \
                        extension + expression_list[pointer+1:]
                    max_pointer = len(expression_list)
                    pointer -= 1

                else:
                    command_dict = {'type': command_type,
                                    "symbol": command_symbol,
                                    "true_expression": expr}
                    n_param = self.command_n_param[command_symbol]
                    if n_param > 0:
                        command_dict["ID"] = draw_count[command_symbol]
                        draw_count[command_symbol] += 1
                        all_transforms = self.get_transforms(
                            expr, adjust_param)
                        command_list.extend(all_transforms)
                    command_list.append(command_dict)
            pointer += 1

        return command_list, draw_count

    def get_dsl_size(self):

        size = len(self.command_n_param.keys()) + 1
        return size

    def describe_all_macros(self):
        for macro, expression in self.named_expression.items():
            print("Macro: ", macro)
            print("Expression: ", expression)
            print("")

    def get_expression(self, command_list, clip=True, quantize=True, resolution=33):
        # we need to first clean up the command list
        # if a command has a true expression with a macro we need to remove all its children
        canvas_counter = 0
        bool_counter = 0
        new_command_list = []
        inside_macro = False
        pointer = 0
        max_pointer = len(command_list)
        while (pointer < max_pointer):
            cmd = command_list[pointer]
            if inside_macro:
                if cmd['type'] == "B":
                    bool_counter += 1
                if cmd['type'] in ["D", "M"]:
                    canvas_counter += 1
                # when stop?
                n_new_canvas = canvas_counter - reject_start_canvas_size
                n_new_ops = bool_counter - reject_start_bool_size
                if n_new_canvas == n_new_ops + 1:
                    inside_macro = False
            else:
                new_command_list.append(cmd)
                if cmd['type'] == "B":
                    bool_counter += 1
                if cmd['type'] in ["D", "M"]:
                    canvas_counter += 1
                if "MACRO" in cmd.keys():
                    # This is an extension of a macro.
                    macro_symbol = cmd['MACRO']
                    # skip all cmd till loop ends:
                    new_cmd = {'type': "M", "symbol": macro_symbol}
                    new_command_list.append(new_cmd)
                    reject_start_canvas_size = canvas_counter
                    reject_start_bool_size = bool_counter
                    inside_macro = True
            pointer += 1

        # command_list = self.resolve_command_list(command_list)
        fcsg_expression = mcsg_commands_to_lower_expr(new_command_list, fcsg_mode=self.fcsg_mode,
                                                      clip=clip, quantize=quantize, resolution=resolution)
        return fcsg_expression


def mcsg_commands_to_lower_expr(command_list, fcsg_mode=True, clip=True, quantize=False, resolution=33):

    scale_stack = [np.array([1, 1])]
    translate_stack = [np.array([0, 0])]
    rotate_stack = [np.array([0])]
    lower_expr = []

    conversion_delta = CONVERSION_DELTA
    if quantize:
        two_scale_delta = (2 - 2 * conversion_delta)/(resolution - 1)
    for command in command_list:
        if 'macro_mode' in command.keys():
            expr = command['macro_mode']
            if not expr is None:
                lower_expr.append(expr)
        else:
            c_type = command['type']
            c_symbol = command['symbol']

            if c_type == "B":
                lower_expr.append(c_symbol)
                cloned_scale = np.copy(scale_stack[-1])
                scale_stack.append(cloned_scale)
                cloned_translate = np.copy(translate_stack[-1])
                translate_stack.append(cloned_translate)
                cloned_rotate = np.copy(rotate_stack[-1])
                rotate_stack.append(cloned_rotate)

            elif c_type == "T":
                param = command['param']
                if isinstance(param, th.Tensor):
                    param = param.detach().cpu().data.numpy()
                if c_symbol == "scale":
                    cur_scale = scale_stack.pop()
                    new_param = cur_scale * param
                    scale_stack.append(new_param)
                elif c_symbol == "translate":
                    cur_scale = scale_stack[-1]
                    cur_translate = translate_stack.pop()
                    new_translate = cur_scale * param + cur_translate
                    translate_stack.append(new_translate)
                elif c_symbol == "rotate":
                    cur_rotate = rotate_stack.pop()
                    new_rotate = cur_rotate + param
                    rotate_stack.append(new_rotate)
            elif c_type in ["D", "M"]:
                t_p = translate_stack.pop()
                s_p = scale_stack.pop()
                # gotta invert?
                # t_p = -t_p
                # s_p = 1/(s_p + CONVERSION_DELTA)
                if clip:
                    t_p = np.clip(t_p, TRANSLATE_MIN + conversion_delta,
                                  TRANSLATE_MAX - conversion_delta)
                    s_p = np.clip(s_p, SCALE_MIN + conversion_delta,
                                  SCALE_MAX - conversion_delta)
                if fcsg_mode:
                    r_p = rotate_stack.pop()
                    if clip:
                        r_p = np.clip(
                            r_p, ROTATE_MIN + conversion_delta, ROTATE_MAX - conversion_delta)
                    param = np.concatenate([t_p, s_p, r_p], 0)
                else:
                    param = np.concatenate([t_p, s_p], 0)

                if quantize:
                    param[2:4] -= SCALE_ADDITION
                    if fcsg_mode:
                        param[4:5] /= ROTATE_MULTIPLIER
                    param = (param - (-1 + conversion_delta)) / two_scale_delta
                    param = np.round(param)
                    param = (param * two_scale_delta) + (-1 + conversion_delta)
                    param[2:4] += SCALE_ADDITION
                    if fcsg_mode:
                        param[4:5] *= ROTATE_MULTIPLIER

                param_str = ", ".join(["%f" % x for x in param])
                draw_expr = "%s(%s)" % (c_symbol, param_str)
                lower_expr.append(draw_expr)
    # lower_expr.append("$")
    return lower_expr
