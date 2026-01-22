import os, glob, time, argparse
from typing import Optional
import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmseg.apis import init_model, inference_model


# 0 = background, 1 = crack, 2 = tape/joint/seam
def prompt_to_class_id(prompt):
    # converts prompt string to class id
    p = prompt.lower()
    if "crack" in p:
        return 1
    if "tape" in p or "joint" in p or "seam" in p:
        return 2
    return None


def parse_argmax_mask(result):
    # try common mmseg return: SegDataSample.pred_sem_seg.data tensor
        sem = result.pred_sem_seg
        data = getattr(sem, "data", None)
        if data is not None:
            arr = data.squeeze(0)  # (H,W) tensor
            arr_np = arr.detach().cpu().numpy()
            return arr_np.astype(np.uint8)
        return None

    


def measure_time_ms(model, image_path, runs= 20, warmup= 3):
    for _ in range(warmup): # warming up model to get inference once initialized
        _ = inference_model(model, image_path)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = [] # array to hold time for each run
    for _ in range(runs):
        t0 = time.time()
        _ = inference_model(model, image_path)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - t0) # append time taken
    return (sum(times)/len(times)) * 1000.0 # return avg time in ms

def overlay_and_save(img_path, mask, out_mask, out_overlay):
    cv2.imwrite(out_mask, mask) # save mask
    img = cv2.imread(img_path) # read original image
    if img is None:
        return
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) # resize mask if needed
    overlay = img.copy() 
    inds = mask > 0 # where mask is present
    overlay[inds] = (overlay[inds] * 0.5 + np.array([0,0,255]) * 0.5).astype(np.uint8) # overlay in red
    cv2.imwrite(out_overlay, overlay) # save overlay

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--image-folder", default=None) # to test multiple images
    p.add_argument("--prompt", required=True) # e.g., "segment crack"
    p.add_argument("--out-dir", default="prompt_outs") 
    p.add_argument("--time-runs", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "overlays"), exist_ok=True)

    ckpt = args.checkpoint

    if ckpt is None:
        raise FileNotFoundError("No checkpoint found; pass --checkpoint or put .pth in --work-dir")
    
    cfg_path = args.config
    if cfg_path is None:
        raise FileNotFoundError("No config found; pass --config or put config in --work-dir")

    model = init_model(Config.fromfile(cfg_path), ckpt, device=args.device) # initialize model

    # if args.time_runs and args.image:
    #     ms = measure_time_ms(model, args.image, runs=args.time_runs, warmup=3)
    #     print(f"Avg inference: {ms:.1f} ms")

    imgs = []
    if args.image_folder:
        for ext in ("*.jpg","*.png","*.jpeg","*.bmp"):
            imgs += sorted(glob.glob(os.path.join(args.image_folder, ext)))
    elif args.image:
        imgs = [args.image]
    else:
        raise ValueError("Provide --image or --image-folder")

    cid = prompt_to_class_id(args.prompt)
    if cid is None:
        raise ValueError("Prompt not recognized. Use words containing 'crack' or 'tape/joint/seam'")

    times = []
    if args.time_runs > 0:
        for _ in range(3): # warming up model to get inference once initialized
            _ = inference_model(model, imgs[0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
    for img in imgs:
        t0 = time.time()
        result = inference_model(model, img)
        if args.time_runs > 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        if args.time_runs > 0:
            times.append(t1 - t0)
        pred = parse_argmax_mask(result)
        if pred is None:
            raise RuntimeError("Could not parse model output. Inspect inference_model return type.")
        mask = (pred == cid).astype('uint8') * 255
        base = os.path.splitext(os.path.basename(img))[0]
        out_mask = os.path.join(args.out_dir, f"{base}_{str(args.prompt).replace(' ', '_')}.png")
        out_overlay = os.path.join(args.out_dir, f"overlays/{base}_{str(args.prompt).replace(' ', '_')}_overlay.png")
        overlay_and_save(img, mask, out_mask, out_overlay)
        print(f"Saved: {out_mask}, {out_overlay}")
    
    if args.time_runs > 0 and len(times) > 0:
        avg_ms = (sum(times)/len(times)) * 1000.0
        print(f"Avg inference over {len(times)} runs: {avg_ms:.1f} ms")

if __name__ == "__main__":
    main()
