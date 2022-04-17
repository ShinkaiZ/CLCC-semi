from yacs.config import CfgNode as CN

_C = CN()

_C.EXP = "exp1"
_C.COMMENT = "no comment"
_C.DEBUG = False

_C.INFER = CN()

_C.MODEL = CN()
_C.MODEL.NAME = "Name"  # Model name
_C.MODEL.METHOD = "Base"
_C.MODEL.BCEDiceLoss = False
_C.MODEL.FinalConLoss = False
_C.MODEL.FinalConsistLoss = False
_C.MODEL.WEIGHT = ""
_C.MODEL.FEATURE_SCALE = 1
_C.MODEL.PROJECT_NUM = 5

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 42
_C.SYSTEM.OPT_L = "O0"
_C.SYSTEM.CUDA = True
_C.SYSTEM.MULTI_GPU = False
_C.SYSTEM.NUM_WORKERS = 2

_C.DIRS = CN()
_C.DIRS.DATA = "./data/"
_C.DIRS.WEIGHTS = "./weights/"
_C.DIRS.OUTPUTS = "./outputs/"
_C.DIRS.LOGS = "./logs/"
_C.DIRS.TEST = "./output/"

_C.DATA = CN()
_C.DATA.INP_CHANNELS = 3
_C.DATA.SEG_CLASSES = 1
_C.DATA.NAME = ["COCO"]
_C.DATA.SIZE = 320
_C.DATA.CROP_SIZE = 64
_C.DATA.LABEL = 0.1

_C.OPT = CN()
_C.OPT.OPTIMIZER = "adamw"
_C.OPT.GD_STEPS = 1
_C.OPT.WARMUP_EPOCHS = 0
_C.OPT.BASE_LR = 1e-3
_C.OPT.WEIGHT_DECAY = 1e-2
_C.OPT.WARM_UP = 0
_C.OPT.MAX_C = 1.0

_C.METRIC = CN()
_C.METRIC.SIGMOID = True
_C.METRIC.THRESHOLD = 0.5

_C.TRAIN = CN()
_C.TRAIN.METHOD = "full"
_C.TRAIN.FOLD = 0
_C.TRAIN.EPOCHS = 50
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.LB_BATCH_SIZE = 1

_C.VAL = CN()
_C.VAL.FOLD = 0
_C.VAL.EPOCH = 1
_C.VAL.BATCH_SIZE = 1

_C.CONST = CN()

_C.TEST = CN()
_C.TEST.FOLD = 0
_C.TEST.BATCH_SIZE = 1


def get_cfg_defaults():
    return _C.clone()

