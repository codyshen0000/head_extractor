# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class iMaterialistDataset(BaseSegDataset):
    """
        iMaterialist 2019 dataset.
    """
    METAINFO = dict(
        classes=(
            'background',
            'shirt, blouse',
            'top, t-shirt, sweatshirt',
            'sweater',
            'cardigan',
            "jacket",
            "vest",
            "pants",
            "shorts",
            "skirt",
            "coat",
            "dress",
            "jumpsuit",
            "cape",
            "glasses",
            "hat",
            "headband, head covering, hair accessory",
            "tie",
            "glove",
            "watch",
            "belt",
            "leg warmer",
            "tights, stockings",
            "sock",
            "shoe",
            "bag, wallet",
            "scarf",
            "umbrella",
            "hood",
            "collar",
            "lapel",
            "epaulette",
            "sleeve",
            "pocket",
            "neckline",
            "buckle",
            "zipper",
            "applique",
            "bead",
            "bow",
            "flower",
            "fringe",
            "ribbon",
            "rivet",
            "ruffle",
            "sequin",
            "tassel",
            ),

        palette=[
            [0, 0, 0], 
            [234, 191, 155], [186, 99, 123], [46, 100, 157], [154, 71, 196], 
            [15, 185, 171], [13, 89, 100], [67, 216, 41], [212, 139, 166], 
            [10, 101, 73], [198, 51, 168], [38, 174, 154], [150, 192, 158], 
            [194, 243, 120], [10, 224, 173], [214, 94, 149], [211, 126, 18], 
            [96, 7, 165], [255, 35, 14], [83, 127, 78], [106, 23, 51], 
            [41, 244, 224], [38, 86, 244], [244, 234, 150], [233, 247, 180], 
            [222, 117, 26], [2, 90, 51], [27, 176, 90], [178, 160, 25], 
            [75, 52, 236], [119, 65, 186], [163, 254, 113], [39, 140, 118], 
            [235, 112, 193], [134, 107, 77], [57, 169, 93], [251, 104, 47], 
            [224, 14, 49], [20, 123, 134], [178, 32, 212], [116, 194, 248], 
            [211, 196, 233], [93, 36, 29], [113, 99, 55], [5, 7, 250], 
            [172, 174, 41], [101, 98, 209],
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
