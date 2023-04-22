import os
from wacky import CfgNode as CN,  CfgNodeFactory as CNF
from configs.subconfigs.domain import CSG2D
from configs.subconfigs.model import MODEL



class PretrainConfFactory(CNF):

    def __init__(self, name="PretrainNew", machine="local"):
        super(PretrainConfFactory, self).__init__()

        if machine == "local": 
            MACHINE_DIR = "/media/aditya/DATA/projects/branching_bad"
            DATA_PATH = "/home/aditya/data/synthetic_data/FCSG2D_data"
        else:
            MACHINE_DIR = "/users/aganesh8/data/aganesh8/projects/branching_bad"
            DATA_PATH = "/users/aganesh8/data/aganesh8/data/synthetic_data/FCSG2D_data"
        canvas_res = 64
        input_quant = 33
        # Create Config object:
        config = CN()
        config.NAME = name
        config.EXPERIMENT_MODE = "Pretrain"

        config.DEVICE = "cuda"

        config.DOMAIN = CSG2D.clone()
        config.MODEL = MODEL.clone()
        
        config.TRAIN = CN()
        config.TRAIN.DATASET = CN()
        config.TRAIN.DATASET.NAME = "SynthCSG2DDataset"
        config.TRAIN.DATASET.EPOCH_SIZE = (9 - 3) * int(8e4)
        config.TRAIN.DATASET.MAX_ACTIONS = MODEL.OUTPUT_SEQ_LENGTH
        config.TRAIN.DATASET.EXPR_N_OPS = list(range(3, 9))
        config.TRAIN.DATASET.DATA_PATH = DATA_PATH
        config.TRAIN.DATASET.EXECUTOR = CN()
        config.TRAIN.DATASET.EXECUTOR.RESOLUTION = canvas_res
        config.TRAIN.DATASET.BAKE_FILE = os.path.join(DATA_PATH, "synth_fcsg2d_baked.pt")

        config.LOSS = CN()
        config.LOSS.CMD_NEG_COEF = 0.1

        NN_INTERPRETER = CN()
        NN_INTERPRETER.QUANTIZATION = input_quant
        config.TRAIN.DATASET.NN_INTERPRETER = NN_INTERPRETER.clone()

        config.DATA_LOADER = CN()
        config.DATA_LOADER.BATCH_SIZE = 512
        config.DATA_LOADER.TRAIN_WORKERS = 4
        config.DATA_LOADER.VAL_WORKERS = 0
        

        config.TRAIN_SPECS = CN()
        config.TRAIN_SPECS.LR = 0.0001
        config.TRAIN_SPECS.NUM_EPOCHS = 5000
        config.TRAIN_SPECS.LOG_INTERVAL = 50
        config.MODEL = MODEL.clone()

        config.VAL = config.TRAIN.clone()
        
        # non essentials:
        
        config.NOTIFICATION = CN()
        config.NOTIFICATION.ENABLE = False
        config.NOTIFICATION.CHANNEL = "aditya"
        config.NOTIFICATION.WEBHOOK = "https://hooks.slack.com/services/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"

        config.LOGGER = CN()
        config.LOGGER.LOG_DIR = os.path.join(MACHINE_DIR, "logs", config.NAME)

        config.SAVER = CN()
        config.SAVER.DIR = os.path.join(
            MACHINE_DIR, "checkpoints", config.NAME)
        config.SAVER.EPOCH = 1

        self.config = config
