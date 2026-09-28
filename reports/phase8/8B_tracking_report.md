# Phase 8B: MOT17 Tracking Evaluation Report

## Methodological Notes
- **Sequences Evaluated**: The 21 evaluated sequences consist of 7 distinct training videos, evaluated across the 3 standard MOT17 detector variants (DPM, FRCNN, SDP). Since our production tracker ignores the MOTChallenge provided bounding box proposals (`det.txt`), it processes the identical underlying image frames 3 separate times independently. The aggregate metrics treat these as separate sequences.
- **Lifecycle Workaround**: To avoid a known state-corruption bug in the Ultralytics tracker (`predictor.trackers = []`), the evaluation harness instantiates a fresh `BoundaryTrackingAdapter` per sequence. This is strictly an evaluation-harness lifecycle workaround, not a production fix.
- **Matching Rule**: Bounding boxes are matched using an IoU threshold of **IoU >= 0.5**.
- **Temporal Subsampling (1 FPS)**: The controlled 1 FPS experiment isolates exactly 1 frame per second. The ground truth (GT) is filtered to match the exact sampled frame index, the tracker does NOT reset state between frames within the sequence, and matches are strictly computed on those timestamps. Note: This is a controlled 1 FPS experiment, which approximates but does not represent exactly the Phase 7 measured 1.2 FPS.
- **Acceptance Status**: The tracking performance acceptance criteria are still TBD. These results constitute a reproducible baseline measurement, not an operational acceptance.

## 1. Production Configuration
- **git_commit**: `734b5195435ee876970f26eb391922c70607278a`
- **model_weights**: `/Users/hariharans/Documents/SIH26187/models/yolo11n.pt`
- **model_sha256**: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`
- **tracker**: `bytetrack.yaml`
- **confidence_threshold**: `0.35`
- **iou_threshold**: `0.7`
- **image_size**: `480`
- **device**: `mps:0`
- **python_version**: `3.10.11`
- **ultralytics_version**: `8.4.105`
- **pytorch_version**: `2.13.0`

## 2. 8B Tracking Evaluation
────────────────────────────────────────
                     Full Frame      1 FPS          
MOTA                 0.2382          0.0985         
IDF1                 0.3335          0.1674         
HOTA                 N/M             N/M            
ID Switches          1206.0000       444.0000       
Fragmentations       3396.0000       243.0000       
False Positives      8763.0000       243.0000       
False Negatives      246660.0000     10374.0000     
Mostly Tracked       195.0000        75.0000        
Mostly Lost          921.0000        1152.0000      
Frame Skip Ratio     0.0000          ~0.9667        

────────────────────────────────────────
### Per-Sequence Results (MOTA | IDF1 | IDSW)
| Sequence | Full Frame (MOTA | IDF1 | IDSW) | 1 FPS (MOTA | IDF1 | IDSW) |
|---|---|---|
| MOT17-02-DPM | 0.1451 | 0.2288 | 38 | 0.0485 | 0.0843 | 27 |
| MOT17-02-FRCNN | 0.1451 | 0.2288 | 38 | 0.0485 | 0.0843 | 27 |
| MOT17-02-SDP | 0.1451 | 0.2288 | 38 | 0.0485 | 0.0843 | 27 |
| MOT17-04-DPM | 0.1771 | 0.2786 | 47 | 0.0627 | 0.1377 | 17 |
| MOT17-04-FRCNN | 0.1771 | 0.2786 | 47 | 0.0627 | 0.1377 | 17 |
| MOT17-04-SDP | 0.1771 | 0.2786 | 47 | 0.0627 | 0.1377 | 17 |
| MOT17-05-DPM | 0.4623 | 0.5693 | 89 | 0.2541 | 0.3319 | 30 |
| MOT17-05-FRCNN | 0.4623 | 0.5693 | 89 | 0.2541 | 0.3319 | 30 |
| MOT17-05-SDP | 0.4623 | 0.5693 | 89 | 0.2541 | 0.3319 | 30 |
| MOT17-09-DPM | 0.5343 | 0.5638 | 61 | 0.1173 | 0.2058 | 31 |
| MOT17-09-FRCNN | 0.5343 | 0.5638 | 61 | 0.1173 | 0.2058 | 31 |
| MOT17-09-SDP | 0.5343 | 0.5638 | 61 | 0.1173 | 0.2058 | 31 |
| MOT17-10-DPM | 0.2452 | 0.2986 | 84 | 0.0714 | 0.1292 | 13 |
| MOT17-10-FRCNN | 0.2452 | 0.2986 | 84 | 0.0714 | 0.1292 | 13 |
| MOT17-10-SDP | 0.2452 | 0.2986 | 84 | 0.0714 | 0.1292 | 13 |
| MOT17-11-DPM | 0.5152 | 0.5231 | 34 | 0.2759 | 0.3274 | 29 |
| MOT17-11-FRCNN | 0.5152 | 0.5231 | 34 | 0.2759 | 0.3274 | 29 |
| MOT17-11-SDP | 0.5152 | 0.5231 | 34 | 0.2759 | 0.3274 | 29 |
| MOT17-13-DPM | 0.1361 | 0.2245 | 49 | 0.0193 | 0.0210 | 1 |
| MOT17-13-FRCNN | 0.1361 | 0.2245 | 49 | 0.0193 | 0.0210 | 1 |
| MOT17-13-SDP | 0.1361 | 0.2245 | 49 | 0.0193 | 0.0210 | 1 |
