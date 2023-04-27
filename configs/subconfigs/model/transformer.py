from wacky import CfgNode as CN

MODEL = CN()
MODEL.NAME = "BaseTransformer"
MODEL.LOAD_WEIGHTS = None
MODEL.DROPOUT = 0.01
MODEL.VISUAL_SEQ_LENGTH = 16
MODEL.OUTPUT_SEQ_LENGTH = 22
MODEL.POST_ATTN_SIZE = 512
MODEL.NUM_HEADS = 16
MODEL.ATTN_SIZE = 256
MODEL.INIT_DEVICE = "cuda"
MODEL.RETURN_ALL = True
MODEL.REAL_PARAM_SCALE = 33
MODEL.COMMAND_TOKEN_COUNT = 6  # [3 bool, 2 draw, 1 end]
MODEL.CMD_LOGSF_SCALER = 5.
MODEL.NUM_ENC_LAYERS = 8
MODEL.NUM_DEC_LAYERS = 8
MODEL.CMD_PARAM_MAPPER_MLP = [256 * 6, 256 * 3, 256 * 1]
MODEL.POST_ATTN_MLP = [256, 256, 512]
MODEL.CMD_MLP = [512, 256, 256]
MODEL.PARAM_MLP = [512, 256, 128, MODEL.REAL_PARAM_SCALE * 5]
# V2
# MODEL.NUM_ENC_LAYERS = 16
# MODEL.NUM_DEC_LAYERS = 16
# MODEL.CMD_PARAM_MAPPER_MLP = [256 * 6, 256 * 1]
# MODEL.POST_ATTN_MLP = [256, 512]
# MODEL.CMD_MLP = [512, 256]
# MODEL.PARAM_MLP = [512, MODEL.REAL_PARAM_SCALE * 5]

NESTED_MODEL = MODEL.clone()
NESTED_MODEL.NAME = "NestedTransformer"
NESTED_MODEL.N_PARAMS = 5
NESTED_MODEL.PARAM_MLP = [256, 128, MODEL.REAL_PARAM_SCALE]