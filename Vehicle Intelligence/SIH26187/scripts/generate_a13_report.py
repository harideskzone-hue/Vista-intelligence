import json
import pandas as pd
from pathlib import Path

def generate_report():
    json_path = Path("output/a13_boundary_records.json")
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    if df.empty:
        print("No data found.")
        return
        
    report = []
    report.append("# A.13 Camera/Optical Boundary Analysis Report\n")
    report.append("> [!IMPORTANT]")
    report.append("> **Ground Truth Available**: NO (Empirical observational distributions only)")
    report.append("> **Accuracy Boundary**: NOT ESTABLISHED (Cannot prove impossibility without manual plate annotations)")
    report.append("> **Operational Evidence Boundary**: PRELIMINARY (Multi-dimensional evidence rather than hard single-variable thresholds)\n")
    
    report.append("## 1. Benchmark Condition Coverage Status\n")
    report.append("| Condition | Video File | Status | Observations | Notes |")
    report.append("| :--- | :--- | :---: | ---: | :--- |")
    report.append("| 1. Day / Urban 4K | `pexels-george-morina-6719160 (2160p)` | **COVERED** | 580 | High-resolution daylight urban traffic |")
    report.append("| 2. Night / Low-Light | `night_driving` | **UNAVAILABLE / EXCLUDED** | 0 | No decodable low-light traffic video present on system |")
    report.append("| 3. Highway / Fast Motion | `highway_fast` | **COVERED** | 207 | Motion blur, high-speed vehicle approaches |")
    report.append("| 4. Far / CCTV / Small | `far_vehicles` | **COVERED** | 1,938 | Distant traffic, wide-angle CCTV surveillance |")
    
    report.append("\n## 2. Per-Condition Observational Summary\n")
    report.append("| Condition | Crop Count | Median WxH | Median Area | Median Sharpness | Valid Format % |")
    report.append("| :--- | ---: | :--- | ---: | ---: | ---: |")
    
    for video in df['video'].unique():
        vdf = df[df['video'] == video]
        calls = len(vdf)
        med_w = vdf['width'].median()
        med_h = vdf['height'].median()
        med_area = vdf['area'].median()
        med_sharp = vdf['sharpness'].median()
        valid_pct = (vdf['is_valid'].sum() / calls * 100) if calls > 0 else 0
        
        report.append(f"| {video} | {calls} | {med_w:.0f}x{med_h:.0f} | {med_area:.0f} px | {med_sharp:.0f} | {valid_pct:.1f}% |")
        
    report.append("\n## 3. Pixel Area Distribution & Readability\n")
    report.append("Empirical analysis of crop bounding box area vs. valid Indian registration format detection:\n")
    
    bins = [0, 200, 400, 800, 1500, float('inf')]
    labels = ['< 200 px', '200-400 px', '400-800 px', '800-1500 px', '> 1500 px']
    df['area_bin'] = pd.cut(df['area'], bins=bins, labels=labels, right=False)
    
    report.append("| Pixel Area | Crop Count | Median Conf | Median Sharpness | Valid Indian Format % |")
    report.append("| :--- | ---: | ---: | ---: | ---: |")
    
    for label in labels:
        bdf = df[df['area_bin'] == label]
        count = len(bdf)
        if count == 0:
            report.append(f"| {label} | 0 | - | - | - |")
            continue
            
        med_conf = bdf['det_conf'].median()
        med_sharp = bdf['sharpness'].median()
        valid_pct = (bdf['is_valid'].sum() / count * 100)
        
        report.append(f"| {label} | {count} | {med_conf:.2f} | {med_sharp:.0f} | {valid_pct:.1f}% |")
        
    report.append("\n> [!NOTE]")
    report.append("> **Non-Monotonicity Finding**: Valid format percentage does NOT increase monotonically with area (200-400 px is 9.5%, 400-800 px is 9.0%, 800-1500 px is 6.4%, >1500 px is 10.3%).")
    report.append("> Therefore, a simple cutoff like `min_area = 1000 px` is **scientifically indefensible** as a guarantee of readability. Only `< 200 px` shows 0.0% validity, indicating extreme small crops yield very weak optical information.")

    report.append("\n## 4. Sharpness (Laplacian Variance) Distribution\n")
    report.append("Empirical analysis of image sharpness (Laplacian variance) vs. valid format detection:\n")
    
    sbins = [0, 100, 300, 600, 1000, float('inf')]
    slabels = ['< 100 (Blurry)', '100-300', '300-600', '600-1000', '> 1000 (Sharp)']
    df['sharp_bin'] = pd.cut(df['sharpness'], bins=sbins, labels=slabels, right=False)
    
    report.append("| Sharpness (Var) | Crop Count | Median Area | Valid Indian Format % |")
    report.append("| :--- | ---: | ---: | ---: |")
    
    for label in slabels:
        bdf = df[df['sharp_bin'] == label]
        count = len(bdf)
        if count == 0:
            report.append(f"| {label} | 0 | - | - |")
            continue
            
        med_area = bdf['area'].median()
        valid_pct = (bdf['is_valid'].sum() / count * 100)
        
        report.append(f"| {label} | {count} | {med_area:.0f} px | {valid_pct:.1f}% |")
        
    report.append("\n> [!WARNING]")
    report.append("> **Sharpness Pitfall**: The highest sharpness bucket (`> 1000`) exhibits the *lowest* valid format percentage (7.5%), while blurry crops (`< 100`) have 12.8%.")
    report.append("> Laplacian variance is heavily affected by high-frequency background noise, bumper textures, and sensor artifacts. Sharpness must NOT be used as a hard single-variable filter.")

    report.append("\n## 5. Architectural Recommendations for A.14 Production Quality Gate\n")
    report.append("Based on the empirical evidence, the A.14 production pipeline must adhere to the following principles:\n")
    report.append("1. **No Artificial Single-Variable Cutoffs**: `plate_min_area`, `plate_min_sharpness`, and `plate_min_contrast` are marked as **NOT ESTABLISHED**.")
    report.append("2. **Fail-Closed Multi-Dimensional Evidence Assessment**: The quality gate collects geometry (`width`, `height`, `area`, `aspect_ratio`), sharpness (`Laplacian`), and contrast (`RMS std`), applying physical sanity checks (e.g. rejection of microscopic `< 100 px` crops or degenerate aspect ratios) without fabricating certainty.")
    report.append("3. **Separation of Concerns**: Valid Indian regex format is NOT synonymous with plate correctness. The system explicitly separates:")
    report.append("   - `plate_detected`: presence of bounding box")
    report.append("   - `optical_quality`: raw metrics (area, sharpness, contrast)")
    report.append("   - `ocr_text`: raw string prediction from FastPlateOCR")
    report.append("   - `ocr_confidence`: model probability distribution")
    report.append("   - `format_valid`: regex conformance")
    report.append("   - `temporal_support`: corroborating observation count across track history")
    report.append("   - `identity_status`: `NO_PLATE_DETECTED`, `INSUFFICIENT_RESOLUTION`, `LOW_QUALITY`, `OCR_UNCERTAIN`, `PROBABLE`, `CONFIRMED`")
    report.append("4. **Frozen A.10 Scheduling**: Keep `plate_stride = 6`, `min_vehicle_area = 10000`, and `verification_stride = 60` frozen as vehicle-crop scheduling optimizations (not plate optical limits).")

    report_path = Path('a13_boundary_report.md')
    report_path.write_text('\n'.join(report))
    print(f"Report successfully generated at {report_path.absolute()}")

if __name__ == '__main__':
    generate_report()
