import os
from wacky import CfgNode as CN,  CfgNodeFactory as CNF
# from configs.subconfigs.model import NESTED_MODEL as MODEL
# from configs.ablations.pretrain import PretrainConfFactory
from configs.ablations.plad import PLADConfFactory


class NaiveBootConfFactory(CNF):

    def __init__(self, name="NaiveBOOT_2", machine="local"):

        # pretrain config
        config = PLADConfFactory(name, machine).config
        config.EXPERIMENT_MODE = "NaiveBOOTAD"
        config.MODEL.LOAD_WEIGHTS = os.path.join(config.MACHINE_DIR, "checkpoints/PretrainTEST/best_model.pt")
        # config.MODEL.LOAD_WEIGHTS = os.path.join(config.MACHINE_DIR, "checkpoints/NaiveBOOT_1/best_model.pt")
        
        config.ABSTRACTION = CN()
        config.ABSTRACTION.NAME = "CS"
        config.ABSTRACTION.CONFIG_FILE = "/home/aditya/projects/rl/rl_csg/configs/ablations/finals/final_srt.py"
        config.ABSTRACTION.LENGTH_TAX_RATE = config.OBJECTIVE.LENGTH_TAX
        config.ABSTRACTION.SAVE_DIR = os.path.join(config.MACHINE_DIR, f"CS/{config.NAME}")
        config.ABSTRACTION.LANGUAGE_NAME = "FCSG2D"
        config.ABSTRACTION.NAME = "CS"
        config.ABSTRACTION.RELOAD_LATEST = False
        
        config.ABSTRACTION.DSL_WEIGHT = - 1e-4
        config.ABSTRACTION.TASK_WEIGHT = 1
        config.ABSTRACTION.DSL_SCORE_TOLERANCE = - 1e-3
        config.ABSTRACTION.DSL_PATIENCE = 10
        
        config.ABSTRACTION.CS_SPLICER = CN()
        config.ABSTRACTION.CS_SPLICER.LENGTH_TAX = config.OBJECTIVE.LENGTH_TAX
        config.ABSTRACTION.CS_SPLICER.MACRO_PER_ERA = 3
        config.ABSTRACTION.CS_SPLICER.NOVELTY_REWARD = 0# 0.15
        config.ABSTRACTION.CS_SPLICER.MERGE_BIT_DISTANCE = 5
        config.ABSTRACTION.CS_SPLICER.MAX_NEW_PERCENTAGE = 0.5
        
        config.OBJECTIVE.NOVEL_CMD_WEIGHT = 5
        
        config.MODEL.RELOAD_ON_DSL_UPDATE = False
        
        config.PLAD.DATASET.NAME = "MacroDataset"
        
        
        self.config = config
