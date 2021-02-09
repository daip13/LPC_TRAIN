from .cluster_dataset import ClusterDataset_mall
from .cluster_det_processor import ClusterDetProcessor_mall
from .build_dataloader import build_dataloader


__factory__ = {
     'mall': ClusterDetProcessor_mall,
}


def build_dataset_mall(cfg):
    return ClusterDataset_mall(cfg)

def build_processor(name):
    if name not in __factory__:
        raise KeyError("Unknown processor:", name)
    return __factory__[name]
