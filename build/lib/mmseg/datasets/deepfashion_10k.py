# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DeepFashion10KDataset(BaseSegDataset):
    """
        Deep Fashion 10k dataset.
    """
    METAINFO = dict(
        classes=(
            'background',
            'top',
            'outer',
            'skirt',
            'dress',
            'pants',
            'leggings',
            'headwear',
            'eyeglass',
            'neckwear',
            'belt',
            'footwear',
            'bag',
            'hair',
            'face',
            'skin',
            'ring',
            'wrist wearing',
            'socks',
            'gloves',
            'necklace',
            'rompers',
            'earrings',
            'tie'),

        palette=[
            [0, 0, 0], [255, 0, 0], [0, 128, 0], [0, 0, 255],
            [0, 128, 128], [238, 130, 238], [128, 128, 128], [255, 255, 0],
            [255, 153, 18], [255, 125, 64], [127, 255, 0], [175, 238, 238],
            [138, 43, 226], [210, 105, 30], [0, 0, 139], [72, 61, 139],
            [255, 20, 147], [255, 192, 203], [205, 92, 92], [32, 178, 170],
            [132, 112, 255], [160, 82, 45], [255, 222, 173], [240, 230, 140],
            ],)
           
        # palette=[
        #     [0, 0, 0], [0, 192, 64], [0, 64, 96], [128, 192, 192],
        #     [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
        #     [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
        #     [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
        #     [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
        #     [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
        #     ],)

    def __init__(self,
                #  ann_file,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
