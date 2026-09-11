import pandas as pd
from pathlib import Path
import json

def normalize_name(name):
    """Normalize file stem for robust mapping: lower, strip spaces, parens, hyphens, etc."""
    n = str(name).lower()
    for char in [' ', '(', ')', '-', '_']:
        n = n.replace(char, '')
    return n

def main():
    INPUT_DIR = Path("input/indian_anpr_zenodo")
    excel_path = INPUT_DIR / "number_plate.xlsx"
    images_dir = INPUT_DIR / "images"
    
    if not excel_path.exists() or not images_dir.exists():
        print("Error: Dataset not found.")
        return

    df = pd.read_excel(excel_path)
    
    col_img = df.columns[0]
    col_gt = df.columns[1]
    
    # 1. Map physical files
    actual_files = list(images_dir.rglob("*"))
    # Only consider files that look like images
    actual_files = [f for f in actual_files if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    norm_to_file = {}
    collisions = []
    
    for f in actual_files:
        norm = normalize_name(f.stem)
        if norm in norm_to_file:
            collisions.append(norm)
        else:
            norm_to_file[norm] = f

    print(f"Total Excel rows                  = {len(df)}")
    print(f"Unique normalized physical files  = {len(norm_to_file)}")
    
    if collisions:
        print(f"WARNING: Physical file collisions found: {collisions[:5]}")
        
    # 2. Map Ground Truth
    mapped = {}
    ambiguous = []
    unmatched_gt = []
    
    unique_norm_gt = set()
    
    for _, row in df.iterrows():
        img_val = str(row[col_img]).strip()
        gt_val = str(row[col_gt]).strip().upper().replace(" ", "")
        
        # Original stem from excel (handles .png extension)
        stem = Path(img_val).stem
        norm = normalize_name(stem)
        unique_norm_gt.add(norm)
        
        if norm in norm_to_file:
            # Match
            physical_path = str(norm_to_file[norm].relative_to(INPUT_DIR))
            mapped[physical_path] = gt_val
        else:
            unmatched_gt.append(img_val)
            
    unmatched_images = set(norm_to_file.keys()) - unique_norm_gt
    
    print(f"Unique normalized GT names        = {len(unique_norm_gt)}")
    print(f"One-to-one matches                = {len(mapped)}")
    print(f"Ambiguous matches                 = {len(ambiguous)}")
    print(f"Unmatched GT                      = {len(unmatched_gt)}")
    print(f"Unmatched images                  = {len(unmatched_images)}")
    
    if len(mapped) == len(df):
        out_path = INPUT_DIR / "mapped_dataset.json"
        with open(out_path, "w") as f:
            json.dump(mapped, f, indent=2)
        print(f"\nSUCCESS: 100% one-to-one mapping verified. Saved to {out_path}")
    else:
        print("\nFAILURE: Did not achieve 100% mapping.")
        if unmatched_gt: print(f"Sample unmatched GT: {unmatched_gt[:5]}")
        if unmatched_images: print(f"Sample unmatched images: {list(unmatched_images)[:5]}")

if __name__ == "__main__":
    main()
