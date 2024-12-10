_base_ = ['../../../_base_/default_runtime.py']

# runtime
max_epochs = 200
base_lr = 5e-4  # 与HRNet一致的学习率
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=base_lr),  # 与HRNet一致的Adam优化器
)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,  # warm-up阶段
        by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],  # 学习率在170和200时衰减
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth')),
    head=dict(
        type='HeatmapHead',
        in_channels=768,
        out_channels=17,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

# base dataset settings
dataset_type = 'SpineDataset'
data_mode = 'topdown'
data_root = '/root/task2/dataset'  # 记得改数据根路径

backend_args = dict(backend='local')

# pipelines (统一与HRNet相同的数据增强)
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),  # 使用与HRNet相同的变换
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/sample/spine_keypoints_rgb_1_v2_train.json',  # 修改后的路径
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/sample/spine_keypoints_rgb_1_v2_val.json',  # 修改后的路径
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='SpineAccuracy',
    thr_list=[0.01, 0.03, 0.05, 0.1, 0.15],
)
test_evaluator = val_evaluator

# hooks
default_hooks = dict(checkpoint=dict(save_best='PCK@0.01', rule='greater'))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]

visualizer = dict(
    draw_bbox=False,
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
    radius=4
)
