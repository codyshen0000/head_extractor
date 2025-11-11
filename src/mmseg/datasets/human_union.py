# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class HumanUnionDataset(BaseSegDataset):
    """
        Human Union dataset.
    """
    METAINFO = dict(
        classes=(
            'background',
            'hat',
            'hair',
            'sunglasses',
            'upper-clothes',
            'skirt',
            'pants',
            'dress',
            'belt',
            'shoes',
            'face',
            'legs',
            'arms',
            'bag',
            'scarf',
            'glove',
            'socks',
            'jumpsuits',),
        
        palette=[
            [0, 0, 0], 
            [128, 0, 0], 
            [0, 128, 0], 
            [128, 128, 0], 
            [0, 0, 128], 
            [128, 0, 128], 
            [0, 128, 128], 
            [128, 128, 128],
            [64, 0, 0], 
            [192, 0, 0], 
            [64, 128, 0], 
            [192, 128, 0], 
            [64, 0, 128], 
            [66, 66, 66], 
            [123, 66, 123], 
            [22, 33, 44], 
            [77, 88, 99], 
            [23, 24, 77]],)

    def __init__(self,
                #  ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
