# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class iMaterialist_5Cat_Dataset(BaseSegDataset):
    """
        iMaterialist 2019 dataset.
    """
    METAINFO = dict(
        classes=(
            'background',
            'upperbody',
            'lowerbody',
            'head_related',
            'others'
            ),

        palette=[
            [0, 0, 0], 
            [0, 0, 255], [255, 0, 0], [0, 255, 0], [128, 0, 196], 
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
