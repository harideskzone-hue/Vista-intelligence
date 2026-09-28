import os
import cv2
import json
import xml.etree.ElementTree as ET
import re
import sys
from pathlib import Path
import numpy as np

# Ensure SIH26187 imports work
project_root = "/Users/hariharans/Documents/SIH26187/Vehicle Intelligence/SIH26187"
if project_root not in sys.path:
    sys.path.append(project_root)

from anpr.detector import PlateDetector
from anpr.ocr import PlateOCR
from integration.vehicle_anpr_pipeline import VehicleANPRPipeline
from core.config import config
from core.schemas import PlateStatus

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.upper()
    text = text.replace(" ", "")
    text = text.replace("-", "")
    # Strip any other non-alphanumeric punctuation
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def parse_dataset(dataset_dir: str):
    annotations_dir = os.path.join(dataset_dir, "Annotations")
    images_dir = os.path.join(dataset_dir, "images")
    
    samples = []
    
    if not os.path.exists(annotations_dir):
        print(f"Annotations dir not found: {annotations_dir}")
        return samples
        
    for xml_file in sorted(os.listdir(annotations_dir)):
        if not xml_file.endswith(".xml"):
            continue
            
        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()
        
        filename = root.find("filename").text
        img_path = os.path.join(images_dir, filename)
        
        if not os.path.exists(img_path):
            continue
            
        # Extract plate objects
        plates = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name == "number_plate":
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                
                gt_text = ""
                attributes = obj.find("attributes")
                if attributes is not None:
                    for attr in attributes.findall("attribute"):
                        if attr.find("name").text == "number_plate_text":
                            gt_text = attr.find("value").text
                            
                plates.append({
                    "bbox": [xmin, ymin, xmax, ymax],
                    "text": gt_text
                })
        
        if plates:
            samples.append({
                "image_path": img_path,
                "plates": plates
            })
            
    return samples

def run_evaluation():
    dataset_dir = "/Users/hariharans/Documents/SIH26187/data/anpr_dataset_10e"
    samples = parse_dataset(dataset_dir)
    print(f"Loaded {len(samples)} annotated images.")
    
    # Initialize un-modified production components
    print("Initializing components...")
    plate_detector = PlateDetector(model_path=config.plate_model_path, conf_threshold=config.plate_det_conf_threshold)
    ocr_engine = PlateOCR(onnx_model_path=config.ocr_model_path, plate_config_path=config.ocr_config_path)
    pipeline = VehicleANPRPipeline(camera_id="BENCHMARK", source_type="image")
    
    results = {
        "dataset_size": len(samples),
        "path_A_detection": {
            "total_gt_boxes": 0,
            "total_predictions": 0,
            "TP": 0, 
            "FP": 0, 
            "FN": 0, 
            "IoUs": []
        },
        "path_B_ocr": {"total": 0, "exact_matches": 0, "total_chars": 0, "errors": 0},
        "path_C_e2e": {
            "total_gt_plates": 0,
            "e2e_exact_matches": 0,
            "waterfall": {
                "gt_plate_contained_in_vehicle": 0,
                "plate_localized_and_attributed": 0,
                "crop_accepted": 0,
                "ocr_produced_output": 0,
                "format_valid": 0,
                "exact_match": 0
            }
        }
    }
    
    print("Running evaluation...")
    for i, sample in enumerate(samples):
        img_path = sample["image_path"]
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        gt_plates = sample["plates"]
        
        # --- PATH A: Plate Detection ---
        results["path_A_detection"]["total_gt_boxes"] += len(gt_plates)
        
        detections = plate_detector.detect(img)
        results["path_A_detection"]["total_predictions"] += len(detections)
        # Sort by confidence for greedy one-to-one matching
        detections.sort(key=lambda x: x[1], reverse=True)
        
        matched_gt_indices = set()
        
        for d_idx, (det_box, conf) in enumerate(detections):
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, gt in enumerate(gt_plates):
                if j in matched_gt_indices:
                    continue
                iou = bb_intersection_over_union(det_box, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
                    
            if best_iou >= 0.5:
                results["path_A_detection"]["TP"] += 1
                matched_gt_indices.add(best_gt_idx)
                results["path_A_detection"]["IoUs"].append(best_iou)
            else:
                results["path_A_detection"]["FP"] += 1
                
        results["path_A_detection"]["FN"] += len(gt_plates) - len(matched_gt_indices)
        
        # --- PATH B: OCR Conditional ---
        for gt in gt_plates:
            gt_text_norm = normalize_text(gt["text"])
            if not gt_text_norm:
                continue
                
            xmin, ymin, xmax, ymax = map(int, gt["bbox"])
            # Clamp box
            xmin = max(0, min(img.shape[1], xmin))
            ymin = max(0, min(img.shape[0], ymin))
            xmax = max(0, min(img.shape[1], xmax))
            ymax = max(0, min(img.shape[0], ymax))
            
            crop = img[ymin:ymax, xmin:xmax]
            if crop.size > 0:
                results["path_B_ocr"]["total"] += 1
                pred_text, conf, _ = ocr_engine.recognize(crop)
                pred_text_norm = normalize_text(pred_text)
                
                if pred_text_norm == gt_text_norm:
                    results["path_B_ocr"]["exact_matches"] += 1
                    
                dist = levenshtein_distance(pred_text_norm, gt_text_norm)
                results["path_B_ocr"]["total_chars"] += len(gt_text_norm)
                results["path_B_ocr"]["errors"] += dist
                
        # --- PATH C: E2E Pipeline ---
        pipeline.reset()
        vrecords = pipeline.process_frame(img, frame_idx=1)
        
        for gt in gt_plates:
            gt_text_norm = normalize_text(gt["text"])
            if not gt_text_norm:
                continue
                
            results["path_C_e2e"]["total_gt_plates"] += 1
            
            # Find corresponding vehicle record (plate must be inside vehicle)
            # YOLO vehicle bbox should encapsulate the plate
            matched_vrecord = None
            best_overlap = 0.0
            
            for vr in vrecords:
                # Calculate overlap between vehicle box and plate box
                vx1, vy1, vx2, vy2 = vr.bbox
                interArea = max(0, min(gt["bbox"][2], vx2) - max(gt["bbox"][0], vx1)) * max(0, min(gt["bbox"][3], vy2) - max(gt["bbox"][1], vy1))
                plateArea = (gt["bbox"][2] - gt["bbox"][0]) * (gt["bbox"][3] - gt["bbox"][1])
                overlap = interArea / float(plateArea) if plateArea > 0 else 0
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    matched_vrecord = vr
                    
            if best_overlap > 0.5 and matched_vrecord is not None:
                results["path_C_e2e"]["waterfall"]["gt_plate_contained_in_vehicle"] += 1
                
                # Establish attribution: verify the pipeline actually localized this GT plate
                # We replicate the pipeline's plate detection on the vehicle crop to get absolute coordinates
                vx1, vy1, vx2, vy2 = map(int, matched_vrecord.bbox)
                vx1_c = max(0, min(img.shape[1], vx1))
                vy1_c = max(0, min(img.shape[0], vy1))
                vx2_c = max(0, min(img.shape[1], vx2))
                vy2_c = max(0, min(img.shape[0], vy2))
                
                attributed = False
                veh_crop = img[vy1_c:vy2_c, vx1_c:vx2_c]
                if veh_crop.size > 0:
                    plate_detections = plate_detector.detect(veh_crop)
                    for (px1, py1, px2, py2), _ in plate_detections:
                        abs_box = [px1 + vx1_c, py1 + vy1_c, px2 + vx1_c, py2 + vy1_c]
                        if bb_intersection_over_union(abs_box, gt["bbox"]) >= 0.5:
                            attributed = True
                            break
                
                anpr_res = matched_vrecord.anpr
                if attributed and anpr_res.status != PlateStatus.NO_PLATE_DETECTED:
                    results["path_C_e2e"]["waterfall"]["plate_localized_and_attributed"] += 1
                    
                    if anpr_res.status not in (PlateStatus.INSUFFICIENT_RESOLUTION, PlateStatus.LOW_QUALITY):
                        results["path_C_e2e"]["waterfall"]["crop_accepted"] += 1
                        
                        if anpr_res.observations > 0: 
                            results["path_C_e2e"]["waterfall"]["ocr_produced_output"] += 1
                            
                            if anpr_res.format_valid:
                                results["path_C_e2e"]["waterfall"]["format_valid"] += 1
                                
                            if anpr_res.plate:
                                pred_text_norm = normalize_text(anpr_res.plate)
                                if pred_text_norm == gt_text_norm:
                                    results["path_C_e2e"]["waterfall"]["exact_match"] += 1
                                    results["path_C_e2e"]["e2e_exact_matches"] += 1

    # Calculate final metrics
    A = results["path_A_detection"]
    TP = A["TP"]
    FP = A["FP"]
    FN = A["FN"]
    
    assert TP + FN == A["total_gt_boxes"], f"Math error: TP({TP}) + FN({FN}) != Total GT Boxes ({A['total_gt_boxes']})"
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = sum(A["IoUs"]) / len(A["IoUs"]) if A["IoUs"] else 0.0
    
    results["path_A_detection"]["metrics"] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou
    }
    
    B = results["path_B_ocr"]
    exact_match_rate = B["exact_matches"] / B["total"] if B["total"] > 0 else 0
    cer = B["errors"] / B["total_chars"] if B["total_chars"] > 0 else 0
    
    results["path_B_ocr"]["metrics"] = {
        "exact_match_rate": exact_match_rate,
        "cer": cer
    }
    
    C = results["path_C_e2e"]
    e2e_acc = C["e2e_exact_matches"] / C["total_gt_plates"] if C["total_gt_plates"] > 0 else 0
    
    vehicle_gate_survival_rate = C["waterfall"]["gt_plate_contained_in_vehicle"] / C["total_gt_plates"] if C["total_gt_plates"] > 0 else 0
    ocr_after_e2e_conditional_match = C["waterfall"]["exact_match"] / C["waterfall"]["ocr_produced_output"] if C["waterfall"]["ocr_produced_output"] > 0 else 0
    
    results["path_C_e2e"]["metrics"] = {
        "e2e_exact_accuracy": e2e_acc,
        "vehicle_gate_survival_rate": vehicle_gate_survival_rate,
        "ocr_after_e2e_conditional_match": ocr_after_e2e_conditional_match
    }
    
    out_path = "/Users/hariharans/Documents/SIH26187/reports/phase10/phase10e_anpr_baseline.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Evaluation complete. Results saved to {out_path}")

if __name__ == "__main__":
    run_evaluation()
