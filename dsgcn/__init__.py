from .test_cluster_mall import test_cluster_mall
from .train_cluster_mall import train_cluster_mall


__factory__ = {
     'test_mall': test_cluster_mall,
     'train_mall': train_cluster_mall
}

def build_handler(phase, stage):
    key_handler = '{}_{}'.format(phase, stage)
    if key_handler not in __factory__:
        raise KeyError("Unknown op:", key_handler)
    return __factory__[key_handler]
