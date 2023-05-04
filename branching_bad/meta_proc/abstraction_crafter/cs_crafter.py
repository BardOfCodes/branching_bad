from collections import defaultdict
import sys
import torch as th
sys.path.insert(0, "/home/aditya/projects/rl/rl_csg")
from CSG.utils.train_utils import arg_parser, load_config, prepare_model_config_and_env
# import CSG.bc_trainers as bc_trainers
from CSG.bc_trainers.rewrite_engines.rewriters import CodeSplicer
from CSG.bc_trainers.rewrite_engines.subexpr_cache import MergeSplicerCache
from .code_splicer import BootADSplicer
from stable_baselines3.common import utils
import CSG.env as csg_env
from stable_baselines3.common.vec_env import DummyVecEnv
import _pickle as cPickle
import os


class CSCrafter():
    
    def __init__(self, bbad_config):
        
        # create config
        self.config_file = bbad_config.CONFIG_FILE
        self.length_tax_rate = bbad_config.LENGTH_TAX_RATE
        self.save_dir = bbad_config.SAVE_DIR
        self.language_name = bbad_config.LANGUAGE_NAME # FCSG2D
        
        device = th.device("cuda")
        arg_list = ["--config", self.config_file]
        # create the rewriter
        args = arg_parser.parse_args(arg_list)
        config = load_config(args)
        config.BC.CS.USE_CANONICAL = True
        self.config = config
        self.bc_config = config.BC
        self.seed = 0
        
        
        self.subexpr_cache = BootADSplicer(self.save_dir, config.BC.CS.CACHE_CONFIG, 
                                               config.BC.CS.MERGE_SPLICE, eval_mode=False, language_name=self.language_name)
        self.subexpr_cache.bbad_setup(bbad_config.CS_SPLICER)
        self.logger = utils.configure_logger(1, config.LOG_DIR , "CS_%s" % config.EXP_NAME, False)
        
    
    def craft_abstractions(self, expression_bank, era, executor):
        
        temp_env = self.get_temp_env()
        best_program_dict = self.convert_expression_bank(expression_bank, executor, temp_env)
        
        new_expression_bank, new_macros = self.subexpr_cache.generate_cache_and_index(
            best_program_dict, temp_env, executor, era)
        # cPickle.dump(self.subexpr_cache, open(os.path.join(self.save_dir, "subexpr_cache.pkl"), "wb"))
        # self.subexpr_cache = cPickle.load(open(os.path.join(self.save_dir, "subexpr_cache.pkl"), "rb"))
        return new_expression_bank, new_macros
        
        # convert high matches into abstractions.
    
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
            expression['reward']  = expression['score']            
            best_program_dict[key].append(expression)
        return best_program_dict
            
            
    
    def get_temp_env(self):
        
        
        bc_env_class = getattr(csg_env, self.bc_config.ENV.TYPE)
        # bc_env = bc_env_class(config, config.BC, seed=seed)
        bc_env = bc_env_class(config=self.config, phase_config=self.bc_config,
                              seed=self.seed, n_proc=self.bc_config.N_ENVS, proc_id=0)
         
        bc_env.reset()
        
        return bc_env
        