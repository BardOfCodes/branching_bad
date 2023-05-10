import os
from wacky import CfgNode as CN,  CfgNodeFactory as CNF
# from configs.subconfigs.model import NESTED_MODEL as MODEL
# from configs.ablations.pretrain import PretrainConfFactory
from configs.ablations.naive_boot import NaiveBootConfFactory


class BranchingBootConfFactory(CNF):

    def __init__(self, name="BranchingBad", machine="local"):
        
        config = NaiveBootConfFactory(name, machine).config
        config.EXPERIMENT_MODE = "BranchingBAD"
        config.N_BRANCHES = 3
        self.config = config
