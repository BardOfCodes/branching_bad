import os
from wacky import CfgNode as CN,  CfgNodeFactory as CNF
from configs.subconfigs.domain import CSG2D
from configs.subconfigs.model import MODEL

MACHINE_DIR = "/home/aditya/projects/branching_bad"
DATA_PATH = "/home/aditya/data/synthetic_data/FCSG2D_data"
class PretrainConfFactory(CNF):
    
    def __init__(self, name="Pretrain"):
        super(PretrainConfFactory, self).__init__()
        
        canvas_res = 64
        input_quant = 33
        # Create Config object:
        config = CN()
        config.NAME = name
        config.EXPERIMENT_MODE = "Pretrain"
        
        config.DEVICE = "cuda"
        
        config.DOMAIN = CSG2D.clone()
        config.MODEL = MODEL.clone()
        
        config.NOTIFICATION = CN()
        config.NOTIFICATION.ENABLE = False
        config.NOTIFICATION.CHANNEL = "aditya"
        config.NOTIFICATION.WEBHOOK = "https://hooks.slack.com/services/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"
        
        config.LOGGER = CN()
        config.LOGGER.LOG_DIR = os.path.join(MACHINE_DIR, "logs", config.NAME)
        
        config.TRAIN = CN()
        config.TRAIN.DATASET = CN()
        config.TRAIN.DATASET.NAME = "CSG2D"
        config.TRAIN.DATASET.EPOCH_SIZE = int(1e6)
        config.TRAIN.DATASET.MAX_ACTIONS = 21
        config.TRAIN.DATASET.EXPR_N_OPS = list(range(3, 6))
        config.TRAIN.DATASET.DATA_PATH = DATA_PATH
        config.TRAIN.DATASET.EXECUTOR = CSG2D.clone()
        config.TRAIN.DATASET.EXECUTOR.RESOLUTION = canvas_res
        
        NN_INTERPRETER = CN()
        NN_INTERPRETER.QUANTIZATION = input_quant
        config.TRAIN.DATASET.NN_INTERPRETER = NN_INTERPRETER.clone()
        
        config.DATA_LOADER = CN()
        config.DATA_LOADER.BATCH_SIZE = 400
        config.DATA_LOADER.NUM_WORKERS = 14
        
        config.TRAIN_SPECS = CN()
        config.TRAIN_SPECS.LR = 0.003
        config.TRAIN_SPECS.NUM_EPOCHS = 500
        config.TRAIN_SPECS.LOG_INTERVAL = 100
        config.TRAIN_SPECS.BATCH_SIZE = 200
        
        
        
        
        
        config.MODEL = MODEL.clone()
        
        config.VAL = config.TRAIN.clone()
        
        self.config = config
        
    