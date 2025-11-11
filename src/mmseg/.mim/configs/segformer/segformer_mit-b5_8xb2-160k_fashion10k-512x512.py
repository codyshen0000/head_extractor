_base_ = ['./segformer_mit-b0_8xb2-160k_fashion10k-512x512.py']

# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
checkpoint = 'checkpoints/mit_b5_20220624-658746d9.pth'  # noqa

# model settings
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

total_train_img = 12013
total_epoch = 120
train_batch_size = 1
total_iter = (int(total_train_img / train_batch_size) + 1) * total_epoch
linear_lr_end = (int(total_train_img / train_batch_size) + 1) * total_epoch / 10
val_save_interval = int(total_train_img / train_batch_size) + 1
base_lr = 0.0009 * (train_batch_size / 64)
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

work_dir = './work_dirs/fashion10k/segformer-b5'