IMG_PATH = '/data/home/wsr/Workspace/dl/dr10_dataset/19_r_21.5/dataset/fullsize/images_19_r_21.5.npy'
LABEL_PATH = '/data/home/wsr/Workspace/dl/dr10_dataset/19_r_21.5/dataset/fullsize/labels_19_r_21.5.npy'
EBV_PATH = '/data/home/wsr/Workspace/dl/dr10_dataset/19_r_21.5/dataset/fullsize/ebv_19_r_21.5.npy'

Z_USE_DEFAULT = False
Z_MIN = 0.
Z_MAX = 1.5

ZBINS_GAP_DEFAULT = 0.0022

NBINS_MAX = 180

BATCH_SIZE = 512
EPOCH = 30
LEARNING_RATE=1e-3

VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2

MODEL_NAME = '19_r_21.5_Pasquet19_0819_gputest'
CONTINUE_TRAIN = False
CONTINUE_EPOCH = 0

# 模型pro设置
DATA_AUGMENTATION = False
DATA_EBV = True
MODEL_MODE = 'regression'