import numpy as np
import torch as th
import copy

class Macro:

    def __init__(self, expr_to_splice, era, id, subexpression):

        self.commands = [copy.deepcopy(x)
                                      for x in expr_to_splice["commands"]] 
        self.canonical_cmds = [copy.deepcopy(x)
                                      for x in expr_to_splice["canonical_commands"]] 
        self.name = f"MACRO_{era}_{id}"
        self.subexpression = subexpression
        
    
    def get_cmd(self):
        cmd = {"type": "M", "symbol": self.name}
        return cmd

    def resolve_param_chain(self, prev_canonical_commands, device=th.device("cuda"), dtype=th.float32):

        # [x.copy() for x in node['subexpr_info']['canonical_commands']]
        current_canonical_commands = [copy.deepcopy(x)
                                      for x in prev_canonical_commands]
        current_canonical_commands[1]['param'] *= -1
        current_canonical_commands[0]['param'] = 1 / \
            (current_canonical_commands[0]['param'] + 1e-9)
        # get param from these:
        transform_chain = current_canonical_commands[::-
                                                     1] + self.canonical_cmds + [self.get_cmd()]
        transform_chain = distill_transform_chains(
            transform_chain, device, dtype)
        
        return transform_chain


def distill_transform_chains(command_list, device, dtype):
    new_command_list = []
    # scale_base = th.tensor([1, 1, 1], device=device, dtype=dtype)
    # translate_base = th.tensor([0, 0, 0], device=device, dtype=dtype)
    scale_base = np.array([1, 1], dtype=np.float32)
    translate_base = np.array([0, 0], dtype=np.float32)

    transform_chain = []
    for command in command_list:
        if 'macro_mode' in command.keys():
            resolved_transform_list = resolve_transform_chain(
                transform_chain, scale_base, translate_base)
            new_command_list.extend(resolved_transform_list)
            new_command_list.append(command)
        else:
            c_type = command['type']
            c_symbol = command['symbol']
            if c_type == "B":
                resolved_transform_list = resolve_transform_chain(
                    transform_chain, scale_base, translate_base)
                new_command_list.extend(resolved_transform_list)
                new_command_list.append(command)
                transform_chain = []
            elif c_type == "T":
                if c_symbol == "scale":
                    transform_chain.append(command)
                elif c_symbol == "translate":
                    transform_chain.append(command)
                elif c_symbol == "rotate":
                    resolved_transform_list = resolve_transform_chain(
                        transform_chain, scale_base, translate_base)
                    new_command_list.extend(resolved_transform_list)
                    new_command_list.append(command)
                    transform_chain = []
            elif c_type in ["D", "M"]:
                resolved_transform_list = resolve_transform_chain(
                    transform_chain, scale_base, translate_base)
                new_command_list.extend(resolved_transform_list)
                new_command_list.append(command)
                transform_chain = []
            else:
                resolved_transform_list = resolve_transform_chain(
                    transform_chain, scale_base, translate_base)
                new_command_list.extend(resolved_transform_list)
                new_command_list.append(command)
                transform_chain = []

    return new_command_list


def resolve_transform_chain(transform_chain, scale_base, translate_base):
    # scale_value = scale_base.clone()
    # translate_value = translate_base.clone()
    scale_value = scale_base.copy()
    translate_value = translate_base.copy()

    for command in transform_chain:
        c_symbol = command['symbol']
        if c_symbol == "translate":
            param = command['param']
            translate_value += scale_value * param
        elif c_symbol == "scale":
            param = command['param']
            scale_value *= param
    resolved_commands = []
    # Now we have the chain.
    # check which should go first?

    if ((scale_value - scale_base)**2).mean() > 1e-4:
        command = {"type": "T", "symbol": "scale", "param": scale_value}
        resolved_commands.append(command)
    if (translate_value**2).mean() > 1e-4:
        first_trans = translate_value / scale_value
        if np.sum(np.abs(first_trans) <= 1) < np.sum(np.abs(translate_value) <= 1):
            command = {"type": "T", "symbol": "translate",
                       "param": first_trans}
            resolved_commands.insert(0, command)
        else:
            command = {"type": "T", "symbol": "translate",
                       "param": translate_value}
            resolved_commands.append(command)

    return resolved_commands
