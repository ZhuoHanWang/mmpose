# 自己根据模型编写/root/task2/mmpose/mmpose/models/backbones/csp_darknet.py
_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=200, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

#key 'PCK' return from mmpose.evaluation.metrics.keypoint_2d_metrics.py
default_hooks = dict(checkpoint=dict(save_best='PCK@0.01', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(     # 给定数据集上的平均值和标准差
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPDarknet',  # 修改为 CSPDarknet
        arch='P5',  # 可以根据需要选择 P5 或 P6
        deepen_factor=1.0,
        widen_factor=1.0,
        out_indices=(2, 3, 4),  # 输出层的索引
        frozen_stages=-1,
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        conv_cfg=None,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        norm_eval=False,
        init_cfg=dict(type='Kaiming', layer='Conv2d')),
    head=dict(
        type='HeatmapHead',
        in_channels=1024,
        out_channels=17,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'SpineDataset'
data_mode = 'topdown'
data_root = '/root/task2/dataset'   # 记得改数据根路径

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
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
        ann_file='annotations/sample/spine_keypoints_rgb_1_v2_train.json',
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
        ann_file='annotations/sample/spine_keypoints_rgb_1_v2_val.json',
        bbox_file=None,
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
#mmpose.evaluation.metrics.keypoint_2d_metrics.py
val_evaluator = dict(
    # type='PCKAccuracy',
    # thr=0.05,
    type='SpineAccuracy',
    thr_list=[0.01, 0.03, 0.05, 0.1, 0.15],
)
test_evaluator = val_evaluator

visualizer = dict(
    draw_bbox=False,
    vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    ],
    radius=4
)
