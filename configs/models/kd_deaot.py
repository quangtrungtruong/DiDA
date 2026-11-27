from .default_deaot import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_ENGINE = 'kd_deaotengine'
        # self.TRAIN_LAMBDA_KD = 0.0001
        self.TRAIN_LAMBDA_KD = 2
        self.TRAIN_BETA_KD = 3
        self.ENABLE_PSEUDOLABEL = True

        self.S_MODEL_LSTT_NUM = 1
        self.S_MODEL_ENCODER = 'mobilenetv2'
        self.S_MODEL_ENCODER_DIM = [24, 32, 96, 1280]
        self.TRAIN_LONG_TERM_MEM_GAP = 2

        # self.S_MODEL_ENCODER = 'resnet50'
        # self.S_MODEL_ENCODER_DIM = [256, 512, 1024, 1024]
        # self.S_MODEL_LSTT_NUM = 1

        # DeAOTL
        # self.T_MODEL_ENCODER = 'mobilenetv2'
        # self.T_MODEL_LSTT_NUM = 3
        # self.T_TRAIN_LONG_TERM_MEM_GAP = 2
        # self.T_TEST_LONG_TERM_MEM_GAP = 5
        # self.T_MODEL_ENCODER_DIM = [24, 32, 96, 1280]

        # R50-DeAOT-L
        self.T_MODEL_ENCODER = 'resnet50'
        self.T_MODEL_LSTT_NUM = 3
        self.T_TRAIN_LONG_TERM_MEM_GAP = 2
        self.T_TEST_LONG_TERM_MEM_GAP = 5
        self.T_MODEL_ENCODER_DIM = [256, 512, 1024, 1024]  # 4x, 8x, 16x, 16x

        # SwinB-DeAOT-L
        # self.T_MODEL_ENCODER = 'swin_base'
        # self.T_MODEL_ALIGN_CORNERS = False
        # self.T_MODEL_ENCODER_DIM = [128, 256, 512, 512]  # 4x, 8x, 16x, 16x
        # self.T_MODEL_LSTT_NUM = 3
        # self.T_TRAIN_LONG_TERM_MEM_GAP = 2
        # self.T_TEST_LONG_TERM_MEM_GAP = 5


        self.KD_SCHEME = 'offline'
        self.FEAT_KD_DISTILL = 'cka'
        # self.FEAT_KD_DISTILL = 'it'
        self.DISTILL_FEAT = True
        self.DISTILL_LOGIT = True
        # self.T_CKPT = './pretrain_models/SwinB_DeAOTL_PRE_YTB_DAV.pth'
        self.T_CKPT = './pretrain_models/R50_DeAOTL_PRE_YTB_DAV.pth'
        # self.T_CKPT = './pretrain_models/mobilenet_v2-b0353104.pth'
        self.T_ENGINE = 'deaotengine'

        self.MODEL_NAME = ('DISTILLLOGIT_' + str(self.DISTILL_LOGIT) + '_DISTILLFEAT_' + str(self.DISTILL_FEAT) + '_LAMBDABETA_' + str(self.TRAIN_LAMBDA_KD) + '_' + str(self.TRAIN_BETA_KD) + '_LSTT_' + str(self.S_MODEL_LSTT_NUM) + '_' +
                           str(self.T_MODEL_LSTT_NUM) + '_' + self.S_MODEL_ENCODER + '_' + self.T_MODEL_ENCODER + '_KD_DeAOT')