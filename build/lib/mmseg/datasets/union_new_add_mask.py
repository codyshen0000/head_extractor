# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class UnionNewAddMaskDataset(BaseSegDataset):
    """
        union dataset.
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
            'accessories',
            'belt',
            'footwear',
            'bag',
            'hair',
            'skin',
            'rompers',
            'face_mask'
            ),

        palette=[
            [0, 0, 0], # background
            [0, 0, 255], # top
            [120, 63, 3], # outer
            [127, 127, 127],  # skirt
            [207, 205, 80], # dress
            [0, 255, 0], # pants
            [223, 83, 230], # leggings
            [41, 135, 207], # accessories, including wrist wear,ring,tie,etc...
            [51, 51, 0], # belt
            [255, 153, 0], # footwear
            [103, 78, 167], # bag
            [255, 0, 0], # hair
            [195, 124, 142], # skin
            [28, 28, 74], # rompers
            [147, 196, 125], # face mask
            ],)
           

    def __init__(self,
                #  ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
