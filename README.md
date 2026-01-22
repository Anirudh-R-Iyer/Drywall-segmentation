## Goal
The goal of this project is to segment drywall defects (cracks and taped joints/seams) from images, and to support user-driven querying (e.g., “segment crack” or “segment drywall seam”) at inference time.

## Installation

Please refer to [get_started.md](docs/en/get_started.md#installation) for installation and [dataset_prepare.md](docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for dataset preparation.

## Train
To train the models simply use the command:
  - python tools/train.py configs/drywall/* --workdir path/to/work_dir

## Test
Ensure that in config, the default hooks has visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
To train the models simply use the command:
  - python tools/test.py path/to/config path/to/work_dir --show_dir /path/to/save/images

## Inference

# Single image
python tools/prompt.py --checkpoint=path/to/checkpoint --config=configs/drywall/*.py (segformer_b0_drywall_640.py) --image=/path/to/img_folder --prompt="highlight cracks" or "show tape"  --out-dir="work_dirs/test" 

# Multiple images (with timed runs)
python prompt.py --checkpoint=path/to/checkpoint --config=configs/drywall/*.py (segformer_b0_drywall_640.py) --image-folder=/path/to/img_folder --prompt="highlight cracks" or "show tape" --out-dir=/path/to/dir --time-runs 1 (remove time-runs args if don't want to time inference)

