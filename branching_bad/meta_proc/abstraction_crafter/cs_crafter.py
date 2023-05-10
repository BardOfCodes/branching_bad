from collections import defaultdict
import sys
import torch as th
# import CSG.bc_trainers as bc_trainers
from .code_splicer import BootADSplicer
from stable_baselines3.common import utils
from stable_baselines3.common.vec_env import DummyVecEnv
import _pickle as cPickle
from branching_bad.utils.metrics import StatEstimator
import os
from .macro import Macro

from CSG.utils.train_utils import arg_parser, load_config
import CSG.env as csg_env


class CSCrafter():

    def __init__(self, bbad_config):

        # create config
        self.config_file = bbad_config.CONFIG_FILE
        self.length_tax_rate = bbad_config.LENGTH_TAX_RATE
        self.save_dir = bbad_config.SAVE_DIR
        self.language_name = bbad_config.LANGUAGE_NAME  # FCSG2D
        self.delete_threshold = bbad_config.DELETE_THRESHOLD

        device = th.device("cuda")
        arg_list = ["--config", self.config_file,]
        # create the rewriter
        if bbad_config.MACHINE == "CCV":
            arg_list.extend(["--machine", "CCV"])

        args = arg_parser.parse_args(arg_list)
        config = load_config(args)
        config.BC.CS.USE_CANONICAL = True
        self.config = config
        self.bc_config = config.BC
        self.seed = 0

        self.subexpr_cache = BootADSplicer(self.save_dir, config.BC.CS.CACHE_CONFIG,
                                           config.BC.CS.MERGE_SPLICE, eval_mode=False, language_name=self.language_name)
        self.subexpr_cache.bbad_setup(bbad_config.CS_SPLICER)
        self.logger = utils.configure_logger(
            1, config.LOG_DIR, "CS_%s" % config.EXP_NAME, False)

    def craft_abstractions(self, expression_bank, era, executor):

        temp_env = self.get_temp_env()
        best_program_dict = self.convert_expression_bank(
            expression_bank, executor, temp_env)

        new_expression_bank, new_macros = self.subexpr_cache.generate_cache_and_index(
            best_program_dict, temp_env, executor, era)
        # cPickle.dump(self.subexpr_cache, open(os.path.join(self.save_dir, "subexpr_cache.pkl"), "wb"))
        # self.subexpr_cache = cPickle.load(open(os.path.join(self.save_dir, "subexpr_cache.pkl"), "rb"))
        return new_expression_bank, new_macros

        # convert high matches into abstractions.
    def remove_abstractions(self, expression_bank, executor, add_macros):

        count_dict = defaultdict(int)
        cmd_pointer = defaultdict(list)
        all_cmds = executor.get_cmd_list()

        for ind, expression in enumerate(expression_bank):
            cur_cmds = [x.split("(")[0] for x in expression['expression']]
            for cmd in all_cmds:
                if cmd in cur_cmds:
                    count_dict[cmd] += 1
                    cmd_pointer[cmd].append(ind)
        # sort the expression dictionary:
        remove_indices = []
        new_macro_names = [x.name for x in add_macros]
        remove_macros = []
        thresold = self.delete_threshold * len(expression_bank)
        print("Removing macros with less than %d occurences" % thresold)
        for cmd, count in count_dict.items():
            if cmd in new_macro_names:
                continue
            if count < thresold:
                remove_indices.extend(cmd_pointer[cmd])
                _, era, candidate_ind = cmd.split("_")
                era = int(era)
                candidate_ind = int(candidate_ind)
                expr = executor.parser.named_expression[cmd]
                dummy_dict = {'commands': [], "canonical_commands": []}
                rem_macro = Macro(dummy_dict, era, candidate_ind, expr)
                remove_macros.append(rem_macro)

        print("Removing %d macros" % len(remove_macros))
        for macro in remove_macros:
            print("macro", macro.name, macro.subexpression)

        new_expression_bank = []
        remove_indices = set(remove_indices)
        for ind, expr in enumerate(expression_bank):
            if ind not in remove_indices:
                new_expression_bank.append(expr)

        return new_expression_bank, remove_macros

    def convert_expression_bank(self, expression_bank, executor, temp_env):
        origin_type = "BS"
        slot_id = "CAD"
        best_program_dict = defaultdict(list)
        base_parser = temp_env.program_generator.parser
        base_compiler = temp_env.program_generator.compiler
        for ind, expression in enumerate(expression_bank):
            target_id = expression['target_index']
            key = (slot_id, target_id, origin_type)
            expression['target_id'] = target_id
            expression['slot_id'] = slot_id
            expression['reward'] = expression['score']
            best_program_dict[key].append(expression)
        return best_program_dict

    def get_temp_env(self):

        bc_env_class = getattr(csg_env, self.bc_config.ENV.TYPE)
        # bc_env = bc_env_class(config, config.BC, seed=seed)
        bc_env = bc_env_class(config=self.config, phase_config=self.bc_config,
                              seed=self.seed, n_proc=self.bc_config.N_ENVS, proc_id=0)

        bc_env.reset()

        return bc_env

    def craft_branching_abstractions(self, expression_bank, era, executor, n_branches):
        
        temp_env = self.get_temp_env()
        best_program_dict = self.convert_expression_bank(
            expression_bank, executor, temp_env)

        new_expression_bank, new_macros = self.subexpr_cache.branching_abstraction(
            best_program_dict, temp_env, executor, era, n_branches)
        # cPickle.dump(self.subexpr_cache, open(os.path.join(self.save_dir, "subexpr_cache.pkl"), "wb"))
        # self.subexpr_cache = cPickle.load(open(os.path.join(self.save_dir, "subexpr_cache.pkl"), "rb"))
        return new_expression_bank, new_macros