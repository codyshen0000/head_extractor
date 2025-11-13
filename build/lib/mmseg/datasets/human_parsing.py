# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class HumanParsingDataset(BaseSegDataset):
    """
        Human Parsing dataset.
    """
    METAINFO = dict(
        classes=(
            'Background', 'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel', 'Hair', 'Sunglasses', 'Upper-clothes', 'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm'),

        palette=[
                [0, 0, 0], [255, 0, 0], [0, 128, 0], [0, 0, 255],
                [0, 128, 128], [238, 130, 238], [128, 128, 128], [255, 255, 0],
                [255, 153, 18], [255, 125, 64], [127, 255, 0], [175, 238, 238],
                [138, 43, 226], [210, 105, 30], [0, 0, 139], [72, 61, 139],
                [255, 20, 147], [255, 192, 203], [205, 92, 92], [32, 178, 170],
                [132, 112, 255], [160, 82, 45], [255, 222, 173],
                [240, 230, 140], [173, 216, 230], [250, 128, 114], [107, 142, 35],
                [72, 209, 204], [199, 21, 133], [25, 25, 112], [123, 104, 238],
                [0, 250, 154], [34, 139, 34], [219, 112, 147], [240, 128, 128],
                [143, 188, 143], [47, 79, 79], [188, 143, 143], [100, 149, 237],
                [102, 205, 170], [255, 160, 122], [147, 112, 219], [60, 179, 113],
                [139, 0, 139], [255, 215, 0], [233, 150, 122], [0, 206, 209],
                [148, 0, 211], [144, 238, 144], [255, 105, 180], [30, 144, 255],
                [255, 140, 0], [153, 50, 204], [220, 20, 60], [46, 139, 87],
                [240, 230, 155], [255, 99, 71]
            ])
           
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
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
