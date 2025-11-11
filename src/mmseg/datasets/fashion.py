# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FashionDataset(BaseSegDataset):
    """imaterialist_fashion dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=("shirt, blouse", "top, t-shirt, sweatshirt", "sweater",
                 "cardigan", "jacket", "vest", "pants", "shorts", "skirt",
                 "coat", "dress", "jumpsuit", "cape", "glasses", "hat", 
                 "headband, head covering, hair accessory", "tie", "glove",
                 "watch", "belt", "leg warmer", "tights, stockings", "sock",
                 "shoe", "bag, wallet", "scarf", "umbrella", "hood", "collar",
                 "lapel", "epaulette", "sleeve", "pocket", "neckline", "buckle",
                 "zipper", "applique", "bead", "bow", "flower", "fringe",
                 "ribbon", "rivet", "ruffle", "sequin", "tassel",),
        palette=[[205, 124, 84], [117, 64, 134], [66, 75, 191], [152, 102, 149], 
                 [68, 20, 174], [104, 106, 176], [26, 123, 233], [65, 148, 108], 
                 [90, 227, 255], [53, 74, 138], [174, 5, 217], [36, 9, 3], [175, 93, 71], 
                 [86, 96, 239], [221, 101, 166], [156, 227, 224], [186, 223, 138], 
                 [151, 121, 189], [118, 43, 207], [137, 157, 76], [224, 160, 18], 
                 [100, 109, 226], [88, 31, 162], [101, 153, 76], [140, 252, 51], 
                 [121, 107, 19], [228, 250, 222], [251, 148, 245], [155, 29, 0], 
                 [99, 246, 138], [182, 66, 5], [103, 232, 180], [50, 75, 12], 
                 [79, 181, 229], [172, 98, 94], [19, 137, 226], [191, 182, 104], 
                 [141, 97, 101], [216, 134, 90], [31, 33, 23], [255, 224, 125], 
                 [199, 82, 200], [196, 10, 110], [244, 144, 145], [232, 145, 29], [51, 185, 206]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_label.jpg',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
