_base_ = ['../../../_base_/default_runtime.py']

# /root/task2/mmpose/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-243frm_8xb32-240e_h36m.py
# runtime
train_cfg = dict(max_epochs=200, val_interval=1)

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01))

# learning policy
param_scheduler = [
    dict(type='ExponentialLR', gamma=0.99, end=100, by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

#key 'PCK' return from mmpose.evaluation.metrics.keypoint_2d_metrics.py
default_hooks = dict(checkpoint=dict(save_best='PCK@0.01', rule='greater'))
custom_hooks = [
    dict(type='ForwardHook', module='backbone'),
    dict(type='ForwardHook', module='neck.reshape'),
    dict(type='ForwardHook', module='neck.channel_mapper'),
]

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='DSTFormer',
        in_channels=3,
        feat_size=1536,
        depth=5,
        num_heads=8,
        mlp_ratio=2,
        seq_len=16 * 16,  # 确保 seq_len = height * width
        att_fuse=True,
    ),
    neck=dict(
        reshape=dict(
            type='ReshapeNeck',
            input_shape=(-1, 256, 1536),
            output_shape=(-1, 1536, 16, 16),
        ),
        channel_mapper=dict(
            type='ChannelMapper',
            in_channels=[1536],
            out_channels=512,
            num_outs=1,
            kernel_size=1,
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN'),
        ),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=512,  # 与 ChannelMapper 输出一致
        out_channels=17,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    )
)


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
