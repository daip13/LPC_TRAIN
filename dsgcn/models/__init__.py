from .dsgcn import dsgcn
from .conv3d import Conv3d
from .histgram_std import histgramstd
__factory__ = {
     'dsgcn': dsgcn,
     'Conv3d': Conv3d,
     'histgram_std': histgramstd,
}


def build_model(name, *args, **kwargs):
    if name not in __factory__:
        raise KeyError("Unknown model:", name)
    return __factory__[name](*args, **kwargs)
