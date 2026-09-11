class TrajectoryAssociator:
    def __init__(self, gap_threshold=40, geometry_threshold=0.50, reid_threshold=0.85):
        self.gap_threshold = gap_threshold
        self.geometry_threshold = geometry_threshold
        self.reid_threshold = reid_threshold
        
        self.trajectory_map = {}
        self.next_id = 1
        self.audit_log = []
        
    def get_trajectory_id(self, track_id):
        if track_id not in self.trajectory_map:
            self.trajectory_map[track_id] = f"T{self.next_id:03d}"
            self.next_id += 1
        return self.trajectory_map[track_id]
        
    def evaluate_candidate(self, trackA, trackB, gap, geom_score, reid_score, class_match):
        # Default initialization
        self.get_trajectory_id(trackA)
        self.get_trajectory_id(trackB)
        
        if not class_match:
            decision = "DIFFERENT"
            reason = "Class mismatch"
        elif gap >= self.gap_threshold:
            if geom_score >= 0.30 and reid_score >= 0.70:
                decision = "REVIEW"
                reason = "Plausible but temporal gap exceeds threshold"
            else:
                decision = "DIFFERENT"
                reason = "Large gap and implausible appearance/geometry"
        elif geom_score < self.geometry_threshold:
            if geom_score >= 0.20 and reid_score >= 0.80:
                decision = "REVIEW"
                reason = "Poor geometry but strong appearance"
            else:
                decision = "DIFFERENT"
                reason = "Geometry clearly incompatible"
        elif reid_score < self.reid_threshold:
            if geom_score >= 0.50 and reid_score >= 0.75:
                decision = "REVIEW"
                reason = "Good geometry but Re-ID below automatic threshold"
            else:
                decision = "DIFFERENT"
                reason = "Re-ID clearly incompatible"
        else:
            decision = "SAME"
            reason = "All mandatory gates passed"
            # Inherit Trajectory ID
            self.trajectory_map[trackB] = self.trajectory_map[trackA]
            
        self.audit_log.append({
            "trackA": trackA,
            "trackB": trackB,
            "gap": gap,
            "geom": geom_score,
            "reid": reid_score,
            "class_match": class_match,
            "decision": decision,
            "reason": reason,
            "trajA": self.trajectory_map[trackA],
            "trajB": self.trajectory_map[trackB]
        })
        return decision
