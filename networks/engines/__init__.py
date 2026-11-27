from networks.engines.aot_engine import AOTEngine, AOTInferEngine
from networks.engines.deaot_engine import DeAOTEngine, DeAOTInferEngine
from networks.engines.kd_deaot_engine import KD_DeAOTEngine, KD_DeAOTInferEngine


def build_engine(name, phase='train', **kwargs):
    if name == 'aotengine':
        if phase == 'train':
            return AOTEngine(**kwargs)
        elif phase == 'eval':
            return AOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    elif name == 'deaotengine':
        if phase == 'train':
            return DeAOTEngine(**kwargs)
        elif phase == 'eval':
            return DeAOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    elif name == 'kd_deaotengine':
        if phase == 'train':
            return KD_DeAOTEngine(**kwargs)
        elif phase == 'eval':
            return KD_DeAOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
