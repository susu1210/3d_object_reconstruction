import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import glob
import os
import cv2

def evaluate(mask_path, pred_mask_path, smooth=1e-10):
    """
    :param mask_path:
    :param pred_mask_path:
    :param smooth:
    :return: mean_iou, mean_f1, mean_fpr
    """
    preds = sorted(glob.glob(os.path.join(pred_mask_path, "*.png")))

    n_frame = len(preds)
    iou_per_frame = []
    f1_per_frame = []
    fpr_per_frame = []
    for frame in range(0, n_frame):
        gt_path = os.path.join(mask_path, preds[frame].split('/')[-1])
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        pred = cv2.imread(preds[frame], cv2.IMREAD_GRAYSCALE)

        if np.sum(gt)==0: #no exist label in this loop
            iou_per_frame.append(np.nan)
            f1_per_frame.append(np.nan)
        else:
            intersect = np.sum(np.logical_and(pred, gt))
            union = np.sum(np.logical_or(pred, gt))

            iou = (intersect + smooth) / (union +smooth)
            iou_per_frame.append(iou)

            f1 = (2 * intersect + smooth) / (np.sum(pred == 255) + np.sum(gt == 255) + smooth)
            f1_per_frame.append(f1)

            if np.sum(pred == 255) == 0:
                fpr_per_frame.append(np.nan)
            else:
                fpr = 1 - (intersect+smooth)/(np.sum(pred == 255))
                fpr_per_frame.append(fpr)
    return np.nanmean(iou_per_frame), np.nanmean(f1_per_frame), np.nanmean(fpr_per_frame)


if __name__ == "__main__":
    YcbObjects = ["YcbMustardBottle",
                  "YcbGelatinBox",
                  "YcbPottedMeatCan",
                  "YcbTomatoSoupCan",
                  "YcbCrackerBox",
                  "YcbSugarBox",
                  "YcbBanana",
                  "YcbTennisBall"]
    DataPath = ["Data", "Data_stuck"]
    SegMethods = ["BackFlow", "OSVOS" ]


    iou = []
    f1= []
    fpr = []
    seg_method = []
    data_type = []
    objs = []
    for s in SegMethods:
        for d in DataPath:
            for obj in YcbObjects:
                annotation_path = os.path.join(os.getcwd(), d, obj, "annotations")
                mask_path = os.path.join(os.getcwd(), d, obj, s+"_results")
                cur_iou, cur_f1, cur_fpr = evaluate(annotation_path, mask_path)
                iou.append(cur_iou)
                f1.append(cur_f1)
                fpr.append(cur_fpr)
                seg_method.append(s)
                data_type.append(d)
                objs.append(obj)
                print(s, d, obj, iou[-1], f1[-1], fpr[-1])
    results_df = pd.DataFrame({
        'seg method': seg_method,
        'data type': data_type,
        'object name': objs,
        'IoU': iou,
        'F1': f1,
        'FP rate': fpr,
    })
    results_df.to_csv(os.path.join(os.getcwd(), 'seg_comparison.csv'),index=False)