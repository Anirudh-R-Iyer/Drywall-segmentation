Download combined drywall dataset from:
 - [640x640](https://app.roboflow.com/trial-kggdn/walls-hkbgt/1)
 - [432x432](https://app.roboflow.com/trial-kggdn/merged-pkzfk/1)

And run tools/coco_mask.py (change the DIR variables inside the file) to convert COCO segmentation polygon format to segment masking format.

output_data/
├── ann_dir/                # annotation masks
│   ├── train/
│   │   └── imgs.png
│   ├── valid/
│   │   └── imgs.png
│   └── test/
│       └── imgs.png
│
└── img_dir/                # original images
    ├── train/
    │   └── imgs.png
    ├── valid/
    │   └── imgs.png
    └── test/
        └── imgs.png
