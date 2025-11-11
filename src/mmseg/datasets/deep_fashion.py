# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DeepFashionDataset(BaseSegDataset):
    """
        Deep Fashion dataset.
    """
    METAINFO = dict(
        classes=('background',
                'sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear',
                'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress',
                'long sleeve dress', 'vest dress', 'sling dress'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [66, 66, 66]])

    def __init__(self,
                #  ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            # ann_file=ann_file,
            **kwargs)
        # assert fileio.exists(self.data_prefix['img_path'],
        #                      self.backend_args) and osp.isfile(self.ann_file)
