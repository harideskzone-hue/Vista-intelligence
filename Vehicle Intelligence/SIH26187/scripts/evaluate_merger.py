import json
from pathlib import Path

OUTPUT_DIR = Path("output/phase_4i")

with open(OUTPUT_DIR / "manual_annotations.json") as f:
    annotations = json.load(f)

with open(OUTPUT_DIR / "merger_audit.json") as f:
    merger_data = json.load(f)

# Count ground truth
total_pairs = len(annotations)
gt_same = sum(1 for v in annotations.values() if v["label"] == "SAME")
gt_diff = sum(1 for v in annotations.values() if v["label"] == "DIFFERENT")
gt_unc = sum(1 for v in annotations.values() if v["label"] == "UNCERTAIN")

print("=== MANUAL VERIFICATION ===")
print(f"Total candidate pairs: {total_pairs}")
print(f"SAME: {gt_same}")
print(f"DIFFERENT: {gt_diff}")
print(f"UNCERTAIN: {gt_unc}")
print()

# Exclude UNCERTAIN from evaluation
valid_pairs = {k: v for k, v in annotations.items() if v["label"] in ["SAME", "DIFFERENT"]}

# Evaluate at different thresholds
thresholds = [0.001, 0.50, 0.60, 0.70, 0.80]

print("=== THRESHOLD ANALYSIS ===")
print(f"{'Score threshold':>15} | {'Predicted merges':>16} | {'Correct (TP)':>12} | {'Incorrect (FP)':>14} | {'Precision':>10} | {'Recall':>10}")
print("-" * 88)

best_threshold = 0.50
best_tp, best_fp, best_fn = 0, 0, 0

for t in thresholds:
    predicted_merges = 0
    tp = 0
    fp = 0
    fn = 0
    
    for res in merger_data:
        pair_id = f"{res['trackA']}->{res['trackB']}"
        if pair_id not in valid_pairs:
            continue
            
        gt_label = valid_pairs[pair_id]["label"]
        pred_merge = res["overall_score"] >= t
        
        if pred_merge:
            predicted_merges += 1
            if gt_label == "SAME":
                tp += 1
            else:
                fp += 1
        else:
            if gt_label == "SAME":
                fn += 1
                
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if t == 0.50:
        best_tp, best_fp, best_fn = tp, fp, fn
        
    print(f"{t:15.2f} | {predicted_merges:16d} | {tp:12d} | {fp:14d} | {precision:10.2f} | {recall:10.2f}")

print()
print("=== MERGER PERFORMANCE (at threshold 0.50) ===")
print(f"True positives : {best_tp}")
print(f"False positives: {best_fp}")
print(f"False negatives: {best_fn}")
precision_50 = best_tp / (best_tp + best_fp) if (best_tp + best_fp) > 0 else 1.0
recall_50 = best_tp / (best_tp + best_fn) if (best_tp + best_fn) > 0 else 0.0
print(f"Precision      : {precision_50:.2f}")
print(f"Recall         : {recall_50:.2f}")

# Write to report file
with open(OUTPUT_DIR / "manual_review_report.txt", "w") as f:
    f.write("=== MANUAL VERIFICATION ===\n")
    f.write(f"Total candidate pairs: {total_pairs}\n")
    f.write(f"SAME: {gt_same}\n")
    f.write(f"DIFFERENT: {gt_diff}\n")
    f.write(f"UNCERTAIN: {gt_unc}\n\n")
    
    f.write("=== MERGER PERFORMANCE (at threshold 0.50) ===\n")
    f.write(f"True positives : {best_tp}\n")
    f.write(f"False positives: {best_fp}\n")
    f.write(f"False negatives: {best_fn}\n")
    f.write(f"Precision      : {precision_50:.2f}\n")
    f.write(f"Recall         : {recall_50:.2f}\n\n")
    
    f.write("=== THRESHOLD ANALYSIS ===\n")
    f.write(f"{'Score threshold':>15} | {'Predicted merges':>16} | {'Correct (TP)':>12} | {'Incorrect (FP)':>14} | {'Precision':>10} | {'Recall':>10}\n")
    f.write("-" * 88 + "\n")
    
    for t in thresholds:
        predicted_merges = 0
        tp, fp, fn = 0, 0, 0
        for res in merger_data:
            pair_id = f"{res['trackA']}->{res['trackB']}"
            if pair_id not in valid_pairs: continue
            gt_label = valid_pairs[pair_id]["label"]
            pred_merge = res["overall_score"] >= t
            if pred_merge:
                predicted_merges += 1
                if gt_label == "SAME": tp += 1
                else: fp += 1
            else:
                if gt_label == "SAME": fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f.write(f"{t:15.2f} | {predicted_merges:16d} | {tp:12d} | {fp:14d} | {precision:10.2f} | {recall:10.2f}\n")
