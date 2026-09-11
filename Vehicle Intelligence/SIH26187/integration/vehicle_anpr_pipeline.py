from __future__ import annotations

import os
import cv2
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from core.config import config, PipelineConfig
from core.schemas import (
    ReportJSON,
    VehicleRecord,
    CameraInfo,
    ProcessingMetrics,
    PipelineMetrics,
    ANPRResult,
    PlateStatus,
    PlateQualityState,
    VehicleClass
)
from vehicle_ai.tracker import VehicleTracker, TrackedVehicle
from anpr.detector import PlateDetector
from anpr.plate_quality import PlateQualityAnalyzer
from anpr.ocr import PlateOCR
from anpr.plate_format import IndianFormatValidator
from anpr.fuzzy_temporal_voter import FuzzyTemporalVoter, TrackANPRState, PlateObservation
from integration.trajectory_association import TrajectoryAssociator

class VehicleANPRPipeline:
    """
    Unified Production Vehicle Intelligence and ANPR Pipeline for SIH26187.
    
    Track-Centric Architecture:
    1. Vehicle Detection (YOLO11n) + ByteTrack Tracking
    2. Vehicle Classification (car, motorcycle, bus, truck)
    3. Intelligent Scheduling Gate (stride=6, min_vehicle_area=10000, verification_stride=60)
    4. Plate Detection (best.pt)
    5. Multi-Dimensional Plate Quality Gate (physical sanity bounds & metrics)
    6. Fine-Tuned FastPlateOCR ONNX Engine
    7. Indian Registration Format Validator
    8. Fail-Closed Fuzzy Temporal Voter
    9. Trajectory Associator (Re-ID & Geometry persistence)
    10. Unified ReportJSON & Frame Streaming Output
    """
    def __init__(
        self,
        camera_id: str = "CAM_01",
        source_type: str = "video",
        custom_config: Optional[PipelineConfig] = None
    ):
        self.cfg = custom_config or config
        self.camera_info = CameraInfo(id=camera_id, source_type=source_type)

        # Core Engines
        self.tracker = VehicleTracker(
            model_path=self.cfg.vehicle_model_path,
            conf_threshold=self.cfg.vehicle_conf_threshold
        )
        self.plate_detector = PlateDetector(
            model_path=self.cfg.plate_model_path,
            conf_threshold=self.cfg.plate_det_conf_threshold
        )
        self.quality_analyzer = PlateQualityAnalyzer(
            min_dim=self.cfg.sanity_min_crop_dim,
            min_area=self.cfg.sanity_min_crop_area,
            min_ar=self.cfg.sanity_min_aspect_ratio,
            max_ar=self.cfg.sanity_max_aspect_ratio
        )
        self.ocr_engine = PlateOCR(
            onnx_model_path=self.cfg.ocr_model_path,
            plate_config_path=self.cfg.ocr_config_path
        )
        self.format_validator = IndianFormatValidator()
        self.temporal_voter = FuzzyTemporalVoter(
            min_observations=self.cfg.min_observations_for_confirmation,
            min_evidence=self.cfg.min_evidence_for_confirmation
        )
        self.trajectory_associator = TrajectoryAssociator()

        # Track State Management
        self.track_anpr_states: Dict[int, TrackANPRState] = {}
        self.track_frames_seen: Dict[int, int] = {}
        self.active_vehicles: Dict[int, VehicleRecord] = {}

        # Pipeline Performance Metrics
        self.metrics = PipelineMetrics()
        self.frames_processed = 0
        self.start_time: Optional[float] = None
        self.last_process_time: Optional[float] = None

    def reset(self):
        """Resets pipeline tracking and metrics state."""
        self.track_anpr_states.clear()
        self.track_frames_seen.clear()
        self.active_vehicles.clear()
        self.metrics = PipelineMetrics()
        self.frames_processed = 0
        self.start_time = None
        self.last_process_time = None

    def should_schedule_plate(self, track_id: int, veh_area: int, current_status: PlateStatus) -> bool:
        """
        Scheduling Gate adhering to frozen A.10 baseline:
        - min_vehicle_area: 10,000 px² (vehicle-crop gate)
        - plate_stride: every 6th frame
        - verification_stride: every 60th frame if already CONFIRMED
        """
        if veh_area < self.cfg.min_vehicle_area:
            return False

        seen = self.track_frames_seen[track_id]

        if current_status == PlateStatus.CONFIRMED:
            # Re-verify infrequently to check stability without wasting compute
            return (seen - 1) % self.cfg.verification_stride == 0

        # Normal active track: check every plate_stride frames
        return (seen - 1) % self.cfg.plate_stride == 0

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        frame_idx: Optional[int] = None
    ) -> List[VehicleRecord]:
        """
        Processes a single video or camera stream frame.
        
        Returns:
            List of VehicleRecord objects active in the current frame.
        """
        if frame is None or frame.size == 0:
            return []

        now = timestamp if timestamp is not None else time.time()
        if self.start_time is None:
            self.start_time = now
        self.frames_processed += 1
        f_idx = frame_idx if frame_idx is not None else self.frames_processed

        frame_h, frame_w = frame.shape[:2]

        # 1. Vehicle Detection + Tracking
        tracked_vehicles: List[TrackedVehicle] = self.tracker.track(frame)
        self.metrics.vehicle_detections += len(tracked_vehicles)

        current_frame_records: List[VehicleRecord] = []

        for veh in tracked_vehicles:
            tid = veh.track_id
            
            # Initialize track state if new
            if tid not in self.track_frames_seen:
                self.track_frames_seen[tid] = 0
                self.track_anpr_states[tid] = TrackANPRState(track_id=tid)
                self.metrics.unique_tracks += 1
                
            self.track_frames_seen[tid] += 1
            anpr_state = self.track_anpr_states[tid]

            # 2. Scheduling Gate Check
            if self.should_schedule_plate(tid, veh.area, anpr_state.current_status):
                anpr_state.add_attempt()
                self.metrics.plate_detector_calls += 1

                # Safe vehicle crop extraction
                vx1, vy1, vx2, vy2 = veh.bbox
                vx1_c = max(0, min(frame_w, vx1))
                vy1_c = max(0, min(frame_h, vy1))
                vx2_c = max(0, min(frame_w, vx2))
                vy2_c = max(0, min(frame_h, vy2))

                veh_crop = frame[vy1_c:vy2_c, vx1_c:vx2_c]

                if veh_crop.size > 0:
                    # 3. Plate Detection
                    plate_detections = self.plate_detector.detect(veh_crop)
                    
                    if plate_detections:
                        self.metrics.plate_detections += len(plate_detections)

                        for (px1, py1, px2, py2), pconf in plate_detections:
                            # Clamp plate coordinates inside vehicle crop
                            vh, vw = veh_crop.shape[:2]
                            px1_c = max(0, min(vw, px1))
                            py1_c = max(0, min(vh, py1))
                            px2_c = max(0, min(vw, px2))
                            py2_c = max(0, min(vh, py2))

                            plate_crop = veh_crop[py1_c:py2_c, px1_c:px2_c]

                            # 4. Multi-Dimensional Plate Quality Gate
                            qstate, qmetrics, qscore = self.quality_analyzer.analyze(plate_crop)

                            if qstate != PlateQualityState.READABLE:
                                anpr_state.add_rejected_crop(qstate, qmetrics)
                                continue

                            # 5. FastPlateOCR Recognition Engine
                            self.metrics.lpr_calls += 1
                            plate_text, ocr_conf, char_probs = self.ocr_engine.recognize(plate_crop)

                            if plate_text:
                                is_valid = self.format_validator.is_valid(plate_text)
                                obs = PlateObservation(
                                    text=plate_text,
                                    ocr_confidence=ocr_conf,
                                    detector_confidence=pconf,
                                    quality_score=qscore,
                                    frame_index=f_idx,
                                    optical_quality=qmetrics,
                                    is_valid_format=is_valid
                                )
                                anpr_state.add_observation(obs)

            # 6. Fuzzy Temporal Evaluation (Fail-Closed State Machine)
            anpr_result = self.temporal_voter.evaluate(anpr_state, f_idx)

            # 7. Trajectory Association
            traj_id = self.trajectory_associator.get_trajectory_id(tid)

            # 8. Record Vehicle Intelligence
            if tid not in self.active_vehicles:
                vrecord = VehicleRecord(
                    track_id=tid,
                    trajectory_id=traj_id,
                    vehicle_class=veh.vehicle_class,
                    vehicle_confidence=round(veh.confidence, 3),
                    bbox=[veh.x1, veh.y1, veh.x2, veh.y2],
                    anpr=anpr_result,
                    first_seen=now,
                    last_seen=now,
                    last_frame=f_idx
                )
            else:
                vrecord = self.active_vehicles[tid]
                vrecord.trajectory_id = traj_id
                vrecord.vehicle_class = veh.vehicle_class
                vrecord.vehicle_confidence = round(veh.confidence, 3)
                vrecord.bbox = [veh.x1, veh.y1, veh.x2, veh.y2]
                vrecord.anpr = anpr_result
                vrecord.last_seen = now
                vrecord.last_frame = f_idx

            self.active_vehicles[tid] = vrecord
            current_frame_records.append(vrecord)

        self.last_process_time = now
        return current_frame_records

    def generate_report(self) -> ReportJSON:
        """
        Generates standardized production ReportJSON schema summary.
        """
        # Calculate summary metrics across all recorded vehicles
        confirmed_count = 0
        probable_count = 0
        unknown_count = 0

        for v in self.active_vehicles.values():
            if v.anpr.status == PlateStatus.CONFIRMED:
                confirmed_count += 1
            elif v.anpr.status == PlateStatus.PROBABLE:
                probable_count += 1
            else:
                unknown_count += 1

        self.metrics.confirmed = confirmed_count
        self.metrics.probable = probable_count
        self.metrics.unknown = unknown_count

        elapsed = (self.last_process_time - self.start_time) if (self.start_time and self.last_process_time) else 0.0
        fps = (self.frames_processed / elapsed) if elapsed > 0 else 0.0

        return ReportJSON(
            schema_version="1.0",
            camera=self.camera_info,
            processing=ProcessingMetrics(
                frames_processed=self.frames_processed,
                effective_fps=round(fps, 2)
            ),
            vehicles=list(self.active_vehicles.values()),
            metrics=self.metrics
        )

    def process_video(
        self,
        video_path: str,
        output_json: Optional[str] = None,
        annotate_video: Optional[str] = None,
        max_frames: Optional[int] = None
    ) -> ReportJSON:
        """
        Runs the full pipeline end-to-end on an input video file.
        """
        p = Path(video_path)
        if not p.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.reset()
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if annotate_video:
            out_p = Path(annotate_video)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))

        frame_idx = 0
        wall_start = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break

                timestamp = wall_start + (frame_idx / fps)
                records = self.process_frame(frame, timestamp=timestamp, frame_idx=frame_idx)

                if writer is not None:
                    annotated = self.annotate_frame(frame, records)
                    writer.write(annotated)

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        report = self.generate_report()

        if output_json:
            out_json_p = Path(output_json)
            out_json_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json_p, "w") as f:
                f.write(report.model_dump_json(indent=2))

        return report

    def annotate_frame(self, frame: np.ndarray, records: List[VehicleRecord]) -> np.ndarray:
        """
        Visual annotation of vehicle bounding boxes, trajectory IDs, and ANPR results.
        """
        canvas = frame.copy()

        STATUS_COLORS = {
            PlateStatus.CONFIRMED: (0, 255, 0),       # Green
            PlateStatus.PROBABLE: (0, 215, 255),      # Gold / Amber
            PlateStatus.OCR_UNCERTAIN: (0, 140, 255), # Orange
            PlateStatus.LOW_QUALITY: (0, 0, 255),     # Red
            PlateStatus.INSUFFICIENT_RESOLUTION: (128, 128, 128), # Gray
            PlateStatus.NO_PLATE_DETECTED: (180, 180, 180),       # Light Gray
            PlateStatus.UNKNOWN: (200, 200, 200)
        }

        for rec in records:
            x1, y1, x2, y2 = rec.bbox
            color = STATUS_COLORS.get(rec.anpr.status, (255, 255, 255))

            # Bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Labels
            traj_tag = f"[{rec.trajectory_id}]" if rec.trajectory_id else f"[T{rec.track_id}]"
            vclass_label = f"{traj_tag} {rec.vehicle_class.value.upper()}"
            
            if rec.anpr.plate:
                anpr_label = f"{rec.anpr.plate} ({rec.anpr.status.value})"
            else:
                anpr_label = f"{rec.anpr.status.value}"

            # Draw vehicle label banner
            cv2.putText(canvas, vclass_label, (x1, max(20, y1 - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, anpr_label, (x1, max(40, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        return canvas
