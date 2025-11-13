# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class UnionNewKDataset(BaseSegDataset):
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
            ),

        palette=[
            [0, 0, 0], # background
            [255, 0, 0], # top
            [4, 63, 120], # outer
            [127, 127, 127],  # skirt
            [80, 205, 207], # dress
            [0, 255, 0], # pants
            [230, 83, 223], # leggings
            [207, 135, 41], # accessories, including wrist wear,ring,tie,etc...
            [0, 51, 51], # belt
            [0, 153, 255], # footwear
            [167,78,103], # bag
            [0, 0, 255], # hair
            [142, 124, 195], # skin
            [74, 28, 28], # rompers
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
