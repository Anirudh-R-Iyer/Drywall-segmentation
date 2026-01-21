dataset_type = 'BaseSegDataset'
data_root = '/home/anirudh/Desktop/ASU/Wall/data/drywall_qa'

metainfo = dict(
    classes=('background', 'crack', 'taping_area'),
    palette=[(0, 0, 0), (255, 0, 0), (0, 255, 0)]
)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        seg_map_suffix='.png',
        pipeline=[]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img_path='img_dir/valid', seg_map_path='ann_dir/valid'),
        seg_map_suffix='.png',
        pipeline=[]
    )
)

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator
