Download combined drywall dataset from:
 - 640x640 https://app.roboflow.com/trial-kggdn/walls-hkbgt/1
 - 432x432 https://app.roboflow.com/trial-kggdn/merged-pkzfk/1

And run tools/coco_mask.py (change the DIR variables inside the file) to convert COCO segmentation polygon format to segment masking format.

output data
|
|
  ann_dir //stores the annotation masks
  |
    test
      |
        imgs.png
  |
    train
  |
    valid
|
|
  img_dir //stores the original images
  |
    test
      |
        imgs.png
  |
    train
  |
    valid