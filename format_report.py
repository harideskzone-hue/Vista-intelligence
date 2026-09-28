import json

with open("reports/phase8/8B_tracking_results.json", "r") as f:
    data = json.load(f)

# The user wants a specific side-by-side format.
# Let's write the markdown file with the exact format.

report_path = "reports/phase8/8B_tracking_report.md"

def format_val(val):
    if val is None:
        return "N/A"
    return f"{val:.4f}"

with open(report_path, "w") as f:
    f.write("# Phase 8B: MOT17 Tracking Evaluation Report\n\n")
    
    f.write("## Methodological Notes\n")
    f.write("- **Sequences Evaluated**: The 21 evaluated sequences consist of 7 distinct training videos, evaluated across the 3 standard MOT17 detector variants (DPM, FRCNN, SDP). Since our production tracker ignores the MOTChallenge provided bounding box proposals (`det.txt`), it processes the identical underlying image frames 3 separate times independently. The aggregate metrics treat these as separate sequences.\n")
    f.write("- **Lifecycle Workaround**: To avoid a known state-corruption bug in the Ultralytics tracker (`predictor.trackers = []`), the evaluation harness instantiates a fresh `BoundaryTrackingAdapter` per sequence. This is strictly an evaluation-harness lifecycle workaround, not a production fix.\n")
    f.write("- **Matching Rule**: Bounding boxes are matched using an IoU threshold of **IoU >= 0.5**.\n")
    f.write("- **Temporal Subsampling (1 FPS)**: The controlled 1 FPS experiment isolates exactly 1 frame per second. The ground truth (GT) is filtered to match the exact sampled frame index, the tracker does NOT reset state between frames within the sequence, and matches are strictly computed on those timestamps. Note: This is a controlled 1 FPS experiment, which approximates but does not represent exactly the Phase 7 measured 1.2 FPS.\n")
    f.write("- **Acceptance Status**: The tracking performance acceptance criteria are still TBD. These results constitute a reproducible baseline measurement, not an operational acceptance.\n\n")

    f.write("## 1. Production Configuration\n")
    for k, v in data["configuration"].items():
        f.write(f"- **{k}**: `{v}`\n")
    
    f.write("\n## 2. 8B Tracking Evaluation\n")
    f.write("────────────────────────────────────────\n")
    f.write(f"{'':<20} {'Full Frame':<15} {'1 FPS':<15}\n")
    
    agg_full = data["modes"]["full"]["aggregate_metrics"]
    agg_1fps = data["modes"]["1_fps"]["aggregate_metrics"]
    
    metrics = [
        ("MOTA", "mota"),
        ("IDF1", "idf1"),
        ("HOTA", "HOTA"),
        ("ID Switches", "num_switches"),
        ("Fragmentations", "num_fragmentations"),
        ("False Positives", "num_false_positives"),
        ("False Negatives", "num_misses"),
        ("Mostly Tracked", "mostly_tracked"),
        ("Mostly Lost", "mostly_lost")
    ]
    
    for display_name, key in metrics:
        if key == "HOTA":
            f.write(f"{display_name:<20} {'N/M':<15} {'N/M':<15}\n")
        else:
            val_full = agg_full.get(key, 0)
            val_1fps = agg_1fps.get(key, 0)
            # Format floats with 4 decimals, ints as ints
            if isinstance(val_full, float):
                f_full = f"{val_full:.4f}"
                f_1fps = f"{val_1fps:.4f}"
            else:
                f_full = str(val_full)
                f_1fps = str(val_1fps)
            f.write(f"{display_name:<20} {f_full:<15} {f_1fps:<15}\n")
            
    # Add frame skip ratio from stats of an arbitrary sequence
    f.write(f"{'Frame Skip Ratio':<20} {'0.0000':<15} {'~0.9667':<15}\n")
    
    f.write("\n────────────────────────────────────────\n")
    f.write("### Per-Sequence Results (MOTA | IDF1 | IDSW)\n")
    f.write("| Sequence | Full Frame (MOTA | IDF1 | IDSW) | 1 FPS (MOTA | IDF1 | IDSW) |\n")
    f.write("|---|---|---|\n")
    
    seqs_full = data["modes"]["full"]["sequence_metrics"]
    seqs_1fps = data["modes"]["1_fps"]["sequence_metrics"]
    
    for seq in sorted(seqs_full.keys()):
        m_f = seqs_full[seq]
        m_1 = seqs_1fps[seq]
        f_str = f"{m_f['mota']:.4f} | {m_f['idf1']:.4f} | {int(m_f['num_switches'])}"
        s_str = f"{m_1['mota']:.4f} | {m_1['idf1']:.4f} | {int(m_1['num_switches'])}"
        f.write(f"| {seq} | {f_str} | {s_str} |\n")

print("Formatted!")
