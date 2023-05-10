import os
from wacky import CfgNode as CN,  CfgNodeFactory as CNF
# from configs.subconfigs.model import NESTED_MODEL as MODEL
from configs.ablations.pretrain import PretrainConfFactory


class PLADConfFactory(CNF):

    def __init__(self, name="PLADRE", machine="local"):

        # pretrain config
        config = PretrainConfFactory(name, machine).config
        config.EXPERIMENT_MODE = "PLAD"
        config.PLAD = CN()
        config.PLAD.INNER_PATIENCE = 5
        config.PLAD.OUTER_PATIENCE = 1
        config.PLAD.MAX_INNER_ITER = 7
        config.PLAD.MAX_OUTER_ITER = 3
        config.PLAD.N_EXPR_PER_ENTRY = 3

        config.TRAIN.DATASET.NAME = "CADCSG2DDataset"
        config.TRAIN.DATASET.LOAD_EXTRA = True
        config.TRAIN.DATASET.EPOCH_SIZE = int(256 * 1000/4)

        config.PLAD.DATASET = config.TRAIN.DATASET.clone()
        config.PLAD.DATASET.NAME = "PLADCSG2DDataset"
        config.PLAD.DATASET.BAKE_FILE = os.path.join(
            config.DATA_PATH, "plad_bakery.pt")
        # config.MODEL.LOAD_WEIGHTS = os.path.join(config.MACHINE_DIR, "checkpoints/PretrainTEST/pretrain_model.pt")
        config.MODEL.LOAD_WEIGHTS = ""# os.path.join(config.MACHINE_DIR, "checkpoints/PretrainTEST/pretrain_model.pt")
        config.DATA_LOADER.SEARCH_BATCH_SIZE = 512
        config.DATA_LOADER.VAL_BATCH_SIZE = 512
        # change train dataset as well:

        config.DATA_LOADER.TRAIN_WORKERS = 4
        config.DATA_LOADER.VAL_WORKERS = 0

        self.config = config
