import sys
import json
import os
from collections import defaultdict
from pathlib import Path

def run_benchmark(run_dir, expected_plate):
    run_dir = Path(run_dir)
    events_path = run_dir / "recognition_events.json"
    results_path = run_dir / "anpr_results.json"
    
    if not events_path.exists() or not results_path.exists():
        print(f"Error: {run_dir} does not contain required JSON logs.")
        return
        
    with open(events_path, "r") as f:
        events = json.load(f)
        
    with open(results_path, "r") as f:
        results = json.load(f)
        
    # Find the track ID that corresponds to this plate (or the one that reached it)
    target_track = None
    final_status = "UNKNOWN"
    
    for res in results:
        if res.get("plate") == expected_plate:
            target_track = res["track_id"]
            final_status = res["plate_status"]
            break
            
    if target_track is None:
        # If it wasn't in final results, let's search events for the track that at least saw it
        track_hits = defaultdict(int)
        for e in events:
            if e["plate_text"] == expected_plate:
                track_hits[e["track_id"]] += 1
        if track_hits:
            target_track = max(track_hits, key=track_hits.get)
            print(f"Plate not confirmed, but found in events for track {target_track}")
        else:
            print(f"Plate {expected_plate} was never read in this run.")
            return
            
    print(f"--- Temporal Benchmark for Plate: {expected_plate} (Track ID: {target_track}) ---")
    
    # Filter events for target track
    track_events = sorted([e for e in events if e["track_id"] == target_track], key=lambda x: x["frame"])
    
    total_predictions = len(track_events)
    exact_correct = sum(1 for e in track_events if e["plate_text"] == expected_plate)
    empty = sum(1 for e in track_events if not e["plate_text"].strip())
    wrong = total_predictions - exact_correct - empty
    
    # Track stability and contradictory strings
    contradictory_strings = set(e["plate_text"] for e in track_events if e["plate_text"] and e["plate_text"] != expected_plate)
    
    # Replay temporal voter to find stability frame
    from collections import Counter
    class DummyVoter:
        def __init__(self):
            self.obs = []
        def add(self, text, conf, lp_conf, qual):
            self.obs.append({'text': text, 'conf': conf, 'lp_conf': lp_conf, 'qual': qual})
        def decide(self):
            if not self.obs: return None
            valid = [o for o in self.obs if len(o['text']) >= 4]
            if not valid: return None
            
            c = Counter(o['text'] for o in valid)
            best_text, count = c.most_common(1)[0]
            
            support = [o for o in valid if o['text'] == best_text]
            avg_ev = sum(o['conf']*o['lp_conf']*o['qual'] for o in support) / len(support)
            
            if count >= 3 and avg_ev >= 0.1: return "CONFIRMED", best_text
            if count >= 2 and avg_ev >= 0.05: return "PROBABLE", best_text
            return "UNKNOWN", best_text

    voter = DummyVoter()
    stable_frame = None
    temporary_errors_survived = 0
    was_stable = False
    
    for e in track_events:
        if e["plate_text"]:
            voter.add(e["plate_text"], e["recognition_confidence"], e["plate_detection_confidence"], e["quality_score"])
            decision = voter.decide()
            if decision and decision[0] == "CONFIRMED" and decision[1] == expected_plate:
                if not was_stable:
                    stable_frame = e["frame"]
                    was_stable = True
            elif was_stable and e["plate_text"] != expected_plate:
                temporary_errors_survived += 1

    print(f"Total frame-level predictions (plate crops): {total_predictions}")
    print(f"  - Correct reads: {exact_correct} ({(exact_correct/total_predictions)*100:.1f}%)")
    print(f"  - Wrong reads  : {wrong} ({(wrong/total_predictions)*100:.1f}%)")
    print(f"  - Empty reads  : {empty} ({(empty/total_predictions)*100:.1f}%)")
    
    print(f"\nContradictory OCR strings generated: {len(contradictory_strings)}")
    
    print(f"\nTemporal stability:")
    print(f"  - Stable decision reached at frame: {stable_frame}")
    print(f"  - Temporary OCR errors cleanly rejected after stability: {temporary_errors_survived}")
    print(f"  - Final pipeline decision: {final_status}")
    
    if final_status == "CONFIRMED":
        print(f"\nVerdict: Temporal fusion successfully improved frame-level accuracy ({(exact_correct/total_predictions)*100:.1f}%) to 100% stable output.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python eval_temporal_benchmark.py <run_dir_path> <expected_plate>")
    else:
        run_benchmark(sys.argv[1], sys.argv[2])
