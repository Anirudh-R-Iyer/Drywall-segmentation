_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/drywall_qa_original.py',
    '../_base_/default_runtime.py'
]


# 3 classes
model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        size=(432, 432),
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255
    ),
    backbone=dict(
    type='MixVisionTransformer',
    in_channels=3,
    embed_dims=64,
    num_stages=4,
    num_layers=[2, 2, 2, 2],
    num_heads=[1, 2, 5, 8],
    patch_sizes=[7, 3, 3, 3],
    sr_ratios=[8, 4, 2, 1],
    out_indices=(0, 1, 2, 3),
    mlp_ratio=4,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],   # B1 channels
        num_classes=3,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=[1.0, 4.0, 2.0],
                loss_weight=1.0
            ),
            dict(type='DiceLoss', loss_weight=1.2),
        ],
    )
)




img_scale = (432, 432)
crop_size = (432, 432)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=img_scale, ratio_range=(0.75, 1.25), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=3e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
)

#iter-based training schedule
max_iters = 20000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=max_iters,
        by_epoch=False
    )
]


default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2000, save_best='mIoU'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1)#2000)
)
load_from = '/home/anirudh/mmsegmentation/work_dirs/segformer_b1_drywall_640/best_mIoU_epoch_0.pth'


seed = 42
deterministic = True

fp16 = dict(loss_scale='dynamic')
