# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class fashion3categoryDataset(BaseSegDataset):
    """imaterialist_fashion dataset. Simplified to 3 catgories

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.jpg' for Fashion dataset.
    """
    METAINFO = dict(
        classes=('background', 'upper_body', 'lower_body', 'whole_body'),
        palette=[[0, 0, 0], [255,0,0], [0, 255, 0], [0, 0, 255]])
        # palette=[[0, 0, 0], [1,1,1], [2, 2, 2], [3, 3, 3]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
