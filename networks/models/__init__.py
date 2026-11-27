from networks.models.aot import AOT
from networks.models.deaot import DeAOT
from networks.models.kd_deaot import KD_DeAOT


def build_vos_model(name, cfg, type=None, **kwargs):
    if name == 'aot':
        return AOT(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    elif name == 'deaot':
        return DeAOT(cfg, encoder=cfg.MODEL_ENCODER, type=type, **kwargs)
    elif name == 'kd_deaot':
        return KD_DeAOT(cfg, encoder=cfg.S_MODEL_ENCODER, MODEL_LSTT_NUM=cfg.S_MODEL_LSTT_NUM, MODEL_ENCODER_DIM=cfg.S_MODEL_ENCODER_DIM, type='student', **kwargs), \
            KD_DeAOT(cfg, encoder=cfg.T_MODEL_ENCODER, MODEL_LSTT_NUM=cfg.T_MODEL_LSTT_NUM, MODEL_ENCODER_DIM=cfg.T_MODEL_ENCODER_DIM, type='teacher', **kwargs)
    else:
        raise NotImplementedError
