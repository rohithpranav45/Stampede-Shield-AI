import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0
    return interArea / unionArea

def load_predictions(parquet_file):
    df = pd.read_parquet(parquet_file)
    # Expect columns: frame, id, x1, y1, x2, y2, confidence (optional)
    if 'confidence' not in df.columns:
        df['confidence'] = 1.0
    return df

def load_ground_truth(gt_file):
    # Expect columns: frame, id, x1, y1, x2, y2
    df = pd.read_csv(gt_file, sep=None, engine='python')
    return df

def match_predictions_to_gt(preds, gts, iou_thr):
    matched_gt = set()
    tp, fp = 0, 0
    pred_scores = []
    pred_labels = []
    for idx, pred in preds.iterrows():
        pred_box = [pred['x1'], pred['y1'], pred['x2'], pred['y2']]
        best_iou = 0
        best_gt_idx = None
        for gt_idx, gt in gts.iterrows():
            if gt_idx in matched_gt:
                continue
            gt_box = [gt['x1'], gt['y1'], gt['x2'], gt['y2']]
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= iou_thr:
            tp += 1
            matched_gt.add(best_gt_idx)
            pred_labels.append(1)
        else:
            fp += 1
            pred_labels.append(0)
        pred_scores.append(pred['confidence'])
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn, pred_scores, pred_labels

def compute_precision_recall_ap(pred_df, gt_df, iou_thr):
    all_scores = []
    all_labels = []
    total_tp, total_fp, total_fn = 0, 0, 0
    for frame in sorted(set(pred_df['frame']).union(gt_df['frame'])):
        preds = pred_df[pred_df['frame'] == frame]
        gts = gt_df[gt_df['frame'] == frame]
        tp, fp, fn, scores, labels = match_predictions_to_gt(preds, gts, iou_thr)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_scores.extend(scores)
        all_labels.extend(labels)
    # Sort by confidence
    sorted_indices = np.argsort(-np.array(all_scores))
    all_labels = np.array(all_labels)[sorted_indices]
    all_scores = np.array(all_scores)[sorted_indices]
    tp_cum = np.cumsum(all_labels)
    fp_cum = np.cumsum(1 - all_labels)
    recalls = tp_cum / (total_tp + total_fn + 1e-8)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
    # AP: area under precision-recall curve
    ap = np.trapz(precisions, recalls)
    return precisions, recalls, ap

def plot_pr_curve(precisions, recalls, ap, iou_thr, out_path):
    plt.figure()
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (IoU={iou_thr}, AP={ap:.3f})')
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate mAP for pedestrian detection/tracking.')
    parser.add_argument('--pred_parquet', type=str, required=True, help='Predictions parquet file')
    parser.add_argument('--gt_txt', type=str, required=True, help='Ground truth .txt file (frame,id,x1,y1,x2,y2)')
    parser.add_argument('--iou_thr', type=float, default=0.5, help='IoU threshold for mAP (default 0.5)')
    parser.add_argument('--out_prefix', type=str, default='Evaluation/map_eval', help='Prefix for output files')
    args = parser.parse_args()

    pred_df = load_predictions(args.pred_parquet)
    gt_df = load_ground_truth(args.gt_txt)
    precisions, recalls, ap = compute_precision_recall_ap(pred_df, gt_df, args.iou_thr)
    print(f'AP@IoU={args.iou_thr}: {ap:.3f}')
    plot_pr_curve(precisions, recalls, ap, args.iou_thr, args.out_prefix + f'_pr_curve.png')
    with open(args.out_prefix + '_results.txt', 'w') as f:
        f.write(f'AP@IoU={args.iou_thr}: {ap:.3f}\n')
    print(f'Results saved to {args.out_prefix}_results.txt and PR curve to {args.out_prefix}_pr_curve.png')

if __name__ == '__main__':
    main() 