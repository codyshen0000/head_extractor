_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/imaterialist_5cat.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
# train: 12013      val: 688
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
checkpoint = 'checkpoints/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(num_classes=24))

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

total_train_img = 42988
total_epoch = 120
train_batch_size = 64
total_iter = (int(total_train_img / train_batch_size) + 1) * total_epoch
linear_lr_end = (int(total_train_img / train_batch_size) + 1) * total_epoch / 20
val_save_interval = int(total_train_img / train_batch_size) + 1
base_lr = 0.0009 * (train_batch_size / 64)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=linear_lr_end),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=linear_lr_end,
        end=total_iter,
        by_epoch=False,
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=total_iter, val_interval=val_save_interval)

train_dataloader = dict(batch_size=train_batch_size, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, save_best='mIoU', interval=val_save_interval),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

work_dir = './work_dirs/imaterialist_5cat/segformer-b0'
