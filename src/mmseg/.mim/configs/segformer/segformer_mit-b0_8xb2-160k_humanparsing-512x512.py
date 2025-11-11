_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/human_parsing.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
# train: 15914      val: 1792
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(num_classes=18))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        # type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
        type='AdamW', lr=0.0006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=38000, # batch size 64
        by_epoch=False,
    )
]

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=5),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=5,
#         end=150,
#         by_epoch=True,
#     )
# ]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=38000, val_interval=1250)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')
# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=5),
#     sampler_seed=dict(type='DistSamplerSeedHook'),
#     visualization=dict(type='SegVisualizationHook'))

train_dataloader = dict(batch_size=64, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1250),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))