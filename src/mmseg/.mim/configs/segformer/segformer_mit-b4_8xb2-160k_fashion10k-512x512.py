_base_ = ['./segformer_mit-b0_8xb2-160k_fashion10k-512x512.py']

# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b4_20220624-d588d980.pth'  # noqa
checkpoint = 'checkpoints/mit_b4_20220624-d588d980.pth'  # noqa

# model settings
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 8, 27, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

total_train_img = 12013
total_epoch = 120
train_batch_size = 1
total_iter = (int(total_train_img / train_batch_size) + 1) * total_epoch
linear_lr_end = (int(total_train_img / train_batch_size) + 1) * total_epoch / 10
val_save_interval = int(total_train_img / train_batch_size) + 1
base_lr = 0.0009 * (train_batch_size / 64)

work_dir = './work_dirs/fashion10k/segformer-b4'

# total_train_img = 12013 * 2
# total_epoch = 120
# train_batch_size = 32
# total_iter = (int(total_train_img / train_batch_size) + 1) * 120
# linear_lr_end = (int(total_train_img / train_batch_size) + 1) * 5
# val_save_interval = int(total_iter / 10)
# base_lr = 0.0006 * (train_batch_size / 64)

# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         # type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
#         type='AdamW', lr=0.0006, betas=(0.9, 0.999), weight_decay=0.01),
#     paramwise_cfg=dict(
#         custom_keys={
#             'pos_block': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.),
#             'head': dict(lr_mult=10.)
#         }))

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=linear_lr_end),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=linear_lr_end,
#         end=total_iter, # batch size 64
#         by_epoch=False,
#     )
# ]

# train_cfg = dict(
#     type='IterBasedTrainLoop', max_iters=total_iter, val_interval=val_save_interval)

# train_dataloader = dict(batch_size=train_batch_size, num_workers=4)
# val_dataloader = dict(batch_size=1, num_workers=4)
# test_dataloader = val_dataloader
