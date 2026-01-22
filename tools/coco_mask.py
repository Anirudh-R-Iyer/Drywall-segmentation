import json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import shutil
import os

COCO_JSON = "/home/anirudh/Downloads/Walls.v1i.coco-mmdetection/test/_annotations.coco.json"
IMAGES_DIR = "/home/anirudh/Downloads/Walls.v1i.coco-mmdetection/test"  # folder containing jpg images
OUT_ROOT = "original_res_data/drywall_qa"   # output dataset root
SPLIT = "test" # dataset split name


# class mapping: COCO category_id to semantic class id
CAT_MAP = {
    1: 1,  # crack
    2: 2   # drywall join / taping area
}

# output dirs
img_out = Path(OUT_ROOT) / "img_dir" / SPLIT
ann_out = Path(OUT_ROOT) / "ann_dir" / SPLIT
img_out.mkdir(parents=True, exist_ok=True)
ann_out.mkdir(parents=True, exist_ok=True)

# load COCO
with open(COCO_JSON, "r") as f:
    coco = json.load(f)

images = {im["id"]: im for im in coco["images"]}
annotations = coco["annotations"]

# group annotations by image
ann_by_img = {}
for ann in annotations:
    ann_by_img.setdefault(ann["image_id"], []).append(ann)

print("Categories:", coco["categories"])
print("Total images:", len(images))

# process each image
for img_id, img_info in images.items():
    W, H = img_info["width"], img_info["height"]
    fname = img_info["file_name"]

    # init semantic mask
    semantic = np.zeros((H, W), dtype=np.uint8)

    for ann in ann_by_img.get(img_id, []): 
        cat_id = ann["category_id"]
        if cat_id not in CAT_MAP:
            continue

        sem_id = CAT_MAP[cat_id] # get semantic class id
        seg = ann["segmentation"] # list of polygons

        # Polygon to RLE mask
        rles = mask_utils.frPyObjects(seg, H, W)
        rle = mask_utils.merge(rles) # merge multiple polygons
        m = mask_utils.decode(rle)  # HxW

        semantic[m == 1] = sem_id 

    # save mask
    mask_path = ann_out / Path(fname).with_suffix(".png").name
    Image.fromarray(semantic, mode="L").save(mask_path) 

    # copy image
    shutil.copyfile(
        Path(IMAGES_DIR) / fname,
        img_out / fname
    )

print(f"Images: {img_out}")
print(f"Masks: {ann_out}")
