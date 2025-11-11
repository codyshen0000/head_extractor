auto_scale_lr = dict(base_batch_size=16, enable=False)
backbone_embed_multi = dict(decay_mult=0.0, lr_mult=0.1)
backbone_norm_multi = dict(decay_mult=0.0, lr_mult=0.1)
crop_size = (
    896,
    896,
)
custom_keys = dict({
    'backbone.dinov2':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.dinov2.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.10.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.11.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.12.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.13.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.14.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.15.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.16.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.17.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.18.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.19.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.2.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.20.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.21.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.22.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.23.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.3.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.4.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.5.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.6.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.7.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.8.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.blocks.9.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.dinov2.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'level_embed':
    dict(decay_mult=0.0, lr_mult=1.0),
    'pos_embed':
    dict(decay_mult=0.0, lr_mult=0.1),
    'query_embed':
    dict(decay_mult=0.0, lr_mult=1.0),
    'query_feat':
    dict(decay_mult=0.0, lr_mult=1.0)
})
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        896,
        896,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '/mnt/data_ssd/limaopeng/limaopeng/segmentation/mmsegmentation/configs/person_parsing/human_parsing_fashion_dataset.py'
dataset_type = 'HumanParsingFashionDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=1,
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
fp16 = True
model = dict(
    backbone=dict(
        freeze=False,
        load_from='./checkpoints/depth_anything_vitl14.pth',
        type='DINOv2',
        version='large'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            896,
            896,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        enforce_decoder_input_project=False,
        feat_channels=1024,
        in_channels=[
            1024,
            1024,
            1024,
            1024,
        ],
        loss_cls=dict(
            class_weight=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.1,
            ],
            loss_weight=2.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction='mean',
            type='mmdet.DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        num_classes=57,
        num_queries=200,
        num_transformer_feat_level=3,
        out_channels=1024,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                init_cfg=None,
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        embed_dims=1024,
                        feedforward_channels=4096,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dropout=0.0,
                        embed_dims=1024,
                        im2col_step=64,
                        init_cfg=None,
                        norm_cfg=None,
                        num_heads=32,
                        num_levels=3,
                        num_points=4)),
                num_layers=6),
            init_cfg=None,
            norm_cfg=dict(num_groups=32, type='GN'),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=512),
            type='mmdet.MSDeformAttnPixelDecoder'),
        positional_encoding=dict(normalize=True, num_feats=512),
        train_cfg=dict(
            assigner=dict(
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        use_sigmoid=True,
                        weight=5.0),
                    dict(
                        eps=1.0,
                        pred_act=True,
                        type='mmdet.DiceCost',
                        weight=5.0),
                ],
                type='mmdet.HungarianAssigner'),
            importance_sample_ratio=0.75,
            num_points=12544,
            oversample_ratio=3.0,
            sampler=dict(type='mmdet.MaskPseudoSampler')),
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    attn_drop=0.0,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=1024,
                    num_heads=32,
                    proj_drop=0.0),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    add_identity=True,
                    dropout_layer=None,
                    embed_dims=1024,
                    feedforward_channels=4096,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(
                    attn_drop=0.0,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=1024,
                    num_heads=32,
                    proj_drop=0.0)),
            num_layers=9,
            return_intermediate=True),
        type='Mask2FormerHead'),
    neck=dict(
        embed_dim=1024, rescales=[
            4,
            2,
            1,
            0.5,
        ], type='Feature2Pyramid'),
    test_cfg=dict(crop_size=(
        896,
        896,
    ), mode='slide', stride=(
        426,
        426,
    )),
    train_cfg=dict(),
    type='EncoderDecoder')
num_classes = 57
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=3e-05,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone.dinov2':
            dict(decay_mult=1.0, lr_mult=0.1),
            'backbone.dinov2.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.10.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.11.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.12.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.13.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.14.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.15.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.16.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.17.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.18.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.19.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.2.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.20.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.21.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.22.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.23.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.3.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.4.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.5.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.6.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.7.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.8.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.blocks.9.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.dinov2.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'level_embed':
            dict(decay_mult=0.0, lr_mult=1.0),
            'pos_embed':
            dict(decay_mult=0.0, lr_mult=0.1),
            'query_embed':
            dict(decay_mult=0.0, lr_mult=1.0),
            'query_feat':
            dict(decay_mult=0.0, lr_mult=1.0)
        }),
        norm_decay_mult=0.0),
    type='OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ), eps=1e-08, lr=3e-05, type='AdamW', weight_decay=0.05)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=160000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='val/images', seg_map_path='val/labels'),
        data_root=
        '/mnt/data_ssd/limaopeng/limaopeng/segmentation/dataset/human_parsing_fashion_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                896,
                896,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='DeepFashion10KDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        896,
        896,
    ), type='Resize'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=160000, type='IterBasedTrainLoop', val_interval=5000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(img_path='train/images', seg_map_path='train/labels'),
        data_root=
        '/mnt/data_ssd/limaopeng/limaopeng/segmentation/dataset/human_parsing_fashion_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(
                max_size=896,
                resize_type='ResizeShortestEdge',
                scales=[
                    448,
                    537,
                    627,
                    716,
                    806,
                    896,
                    985,
                    1075,
                    1164,
                    1254,
                    1344,
                    1433,
                    1523,
                    1612,
                    1702,
                    1792,
                ],
                type='RandomChoiceResize'),
            # dict(
            #     cat_max_ratio=0.75, crop_size=(
            #         896,
            #         896,
            #     ), type='RandomCrop'),
            # dict(prob=0.5, type='RandomFlip'),
            # dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='DeepFashion10KDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(
        max_size=896,
        resize_type='ResizeShortestEdge',
        scales=[
            448,
            537,
            627,
            716,
            806,
            896,
            985,
            1075,
            1164,
            1254,
            1344,
            1433,
            1523,
            1612,
            1702,
            1792,
        ],
        type='RandomChoiceResize'),
    # dict(cat_max_ratio=0.75, crop_size=(
    #     896,
    #     896,
    # ), type='RandomCrop'),
    # dict(prob=0.5, type='RandomFlip'),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='val/images', seg_map_path='val/labels'),
        data_root=
        '/mnt/data_ssd/limaopeng/limaopeng/segmentation/dataset/human_parsing_fashion_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                896,
                896,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='DeepFashion10KDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/human_parsing_fashion_dataset_20250429'
