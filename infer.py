import argparse
import os
import sys
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


sys.path.append('/public_bme/data/jianght/mmcv-1.5.0/')
sys.path.append('/public_bme/data/jianght/Co-DETR')

# 引入 mmdet API
from mmdet.apis import init_detector, inference_detector
from mmcv import Config

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet Inference to CSV')
    parser.add_argument('--config', help='Config file path')
    parser.add_argument('--weights', help='weights file path')
    parser.add_argument('--img-dir', type=str, default='/public_bme2/bme-wangqian2/jianght/Datas/ISBI-riva-cervical-cytology-challenge-track-a/test_final', help='Directory containing test images')
    parser.add_argument('--score-thr', type=float, default=0.0001, help='Bbox score threshold')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading from: {args.config} ...")
    model = init_detector(args.config, args.weights, device=args.device)

    if not os.path.exists(args.img_dir):
        raise FileNotFoundError(f"Img dir do not exist: {args.img_dir}")
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    target_images = [f for f in os.listdir(args.img_dir) if f.lower().endswith(valid_extensions)]
    target_images.sort()

    results = []
    
    print(f"Total {len(target_images)} images...")
    
    TARGET_SIZE = 100
    IMG_BOUNDARY = 1024
    STRICT_EDGE_THR = 5


    for img_name in tqdm(target_images):
        img_path = os.path.join(args.img_dir, img_name)
        
        result = inference_detector(model, img_path)


        if isinstance(result, tuple):
            bbox_result = result[0]
        else:
            bbox_result = result


        for class_id, bboxes in enumerate(bbox_result):
            if bboxes.shape[0] > 0:
                for box in bboxes:
                    x1, y1, x2, y2, score = box
                    
                    if score < args.score_thr:
                        continue
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    half_size = TARGET_SIZE / 2.0  # 50.0
                    new_x1 = cx - half_size
                    new_x2 = cx + half_size
                    new_y1 = cy - half_size
                    new_y2 = cy + half_size
                    

                    if x1 < STRICT_EDGE_THR:
                        new_x1 = x2 - TARGET_SIZE
                        new_x2 = x2
                    elif x2 > IMG_BOUNDARY - STRICT_EDGE_THR:
                        new_x1 = x1
                        new_x2 = x1 + TARGET_SIZE
        
                    if y1 < STRICT_EDGE_THR:
                        new_y1 = y2 - TARGET_SIZE
                        new_y2 = y2
                    elif y2 > IMG_BOUNDARY - STRICT_EDGE_THR:
                        new_y1 = y1
                        new_y2 = y1 + TARGET_SIZE
                    
                    cx = (new_x1 + new_x2) / 2 
                    cy = (new_y1 + new_y2) / 2 

                    results.append({
                    "image_filename": img_name,
                    "x": cx, 
                    "y": cy,
                    "width": float(TARGET_SIZE),    # fixed 100
                    "height": float(TARGET_SIZE),   # fixed 100
                    "conf": float(score),
                    "class": int(class_id)
                })


    base_cols = ['image_filename', 'x', 'y', 'width', 'height', 'conf', 'class']


    df_result = pd.DataFrame(results)

    if not df_result.empty:
        df_result['class'] = df_result['class'].astype(int)

    df_result.insert(0, 'id', range(len(df_result)))
    
    final_cols = ['id'] + base_cols
    df_result = df_result[final_cols]

    out_csv = args.weights.replace('.pth', f'_thresh{args.score_thr}.csv')
    df_result.to_csv(out_csv, index=False)
    
    print(f"Restut have saved to : {out_csv}")
    print(f"Total {len(df_result)} bounding boxes。")
    if not df_result.empty:
        print(df_result.head())

if __name__ == '__main__':
    main()