_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/drywall_qa.py',
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
    decode_head=dict(
        num_classes=3,
        loss_decode=[
        dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1.0, 6.0, 2.0]  # bg, crack, tape
            ),
        dict(type='DiceLoss', loss_weight=1.0),
        ],
    )
)



img_scale = (432, 432)
crop_size = (432, 432)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=img_scale, ratio_range=(0.75, 1.25), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.85),
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
        lr=6e-4,
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
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
)
load_from = '/home/anirudh/mmsegmentation/work_dirs/segformer_b0_drywall_432/best_mIoU_epoch_0.pth'



seed = 42
deterministic = True


fp16 = dict(loss_scale='dynamic')
