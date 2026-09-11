import sys
import unittest
import numpy as np
import cv2
import json

from core.schemas import (
    ReportJSON,
    VehicleRecord,
    CameraInfo,
    ProcessingMetrics,
    PipelineMetrics,
    ANPRResult,
    PlateStatus,
    PlateQualityState,
    VehicleClass,
    OpticalQuality
)
from core.config import config
from anpr.plate_format import IndianFormatValidator
from anpr.plate_quality import PlateQualityAnalyzer
from anpr.fuzzy_temporal_voter import FuzzyTemporalVoter, TrackANPRState, PlateObservation
from vehicle_ai.classifier import VehicleClassifier
from integration.vehicle_anpr_pipeline import VehicleANPRPipeline

class TestProductionPipeline(unittest.TestCase):
    
    def test_schema_serialization(self):
        record = VehicleRecord(
            track_id=1,
            trajectory_id="T001",
            vehicle_class=VehicleClass.CAR,
            vehicle_confidence=0.95,
            bbox=[100, 200, 300, 400],
            anpr=ANPRResult(
                plate="TN01AB1234",
                status=PlateStatus.CONFIRMED,
                confidence=0.92,
                format_valid=True,
                optical_quality=OpticalQuality(width=120, height=35, area=4200, aspect_ratio=3.43, sharpness=145.2, contrast=32.1),
                observations=5,
                temporal_support=5,
                first_confirmed_frame=12,
                stable_confirmed_frame=18
            ),
            first_seen=1000.0,
            last_seen=1005.0,
            last_frame=20
        )
        report = ReportJSON(
            camera=CameraInfo(id="CAM_TEST", source_type="video"),
            processing=ProcessingMetrics(frames_processed=20, effective_fps=45.0),
            vehicles=[record],
            metrics=PipelineMetrics(
                vehicle_detections=20,
                unique_tracks=1,
                plate_detector_calls=3,
                plate_detections=3,
                lpr_calls=3,
                confirmed=1
            )
        )
        json_data = json.loads(report.model_dump_json())
        self.assertEqual(json_data["vehicles"][0]["anpr"]["plate"], "TN01AB1234")
        self.assertEqual(json_data["vehicles"][0]["anpr"]["status"], "CONFIRMED")
        self.assertTrue(json_data["vehicles"][0]["anpr"]["format_valid"])
        self.assertEqual(json_data["vehicles"][0]["anpr"]["optical_quality"]["width"], 120)

    def test_plate_format_validator(self):
        validator = IndianFormatValidator()
        self.assertTrue(validator.is_valid("TN01AB1234"))
        self.assertTrue(validator.is_valid("DL3C1234"))
        self.assertTrue(validator.is_valid("MH12DE1433"))
        self.assertTrue(validator.is_valid("KA05M9999"))
        self.assertTrue(validator.is_valid("22BH1234AA")) # Bharat series
        
        # Invalid cases
        self.assertFalse(validator.is_valid("ZZ99ZZ9999")) # Invalid state code
        self.assertFalse(validator.is_valid("HELLO123"))
        self.assertFalse(validator.is_valid("12345"))
        self.assertFalse(validator.is_valid(""))

    def test_plate_quality_analyzer(self):
        analyzer = PlateQualityAnalyzer()
        
        # Microscopic crop
        micro_crop = np.zeros((4, 4, 3), dtype=np.uint8)
        state, metrics, score = analyzer.analyze(micro_crop)
        self.assertEqual(state, PlateQualityState.INSUFFICIENT_RESOLUTION)
        self.assertEqual(score, 0.0)

        # Degenerate aspect ratio (tall vertical crop)
        tall_crop = np.ones((100, 10, 3), dtype=np.uint8) * 128
        state, metrics, score = analyzer.analyze(tall_crop)
        self.assertEqual(state, PlateQualityState.LOW_QUALITY)

        # Standard plate crop with texture
        plate_crop = np.random.randint(0, 256, (30, 100, 3), dtype=np.uint8)
        state, metrics, score = analyzer.analyze(plate_crop)
        self.assertEqual(state, PlateQualityState.READABLE)
        self.assertGreater(metrics.area, 0)
        self.assertGreater(score, 0.0)

    def test_fuzzy_temporal_voter(self):
        voter = FuzzyTemporalVoter(min_observations=3, min_evidence=0.10)
        
        # Case 1: No plates detected
        state1 = TrackANPRState(track_id=1)
        res1 = voter.evaluate(state1, current_frame=10)
        self.assertEqual(res1.status, PlateStatus.NO_PLATE_DETECTED)
        self.assertIsNone(res1.plate)

        # Case 2: Rejected for resolution
        state2 = TrackANPRState(track_id=2)
        state2.add_rejected_crop(PlateQualityState.INSUFFICIENT_RESOLUTION, OpticalQuality(area=50))
        res2 = voter.evaluate(state2, current_frame=10)
        self.assertEqual(res2.status, PlateStatus.INSUFFICIENT_RESOLUTION)

        # Case 3: 1 valid observation -> PROBABLE
        state3 = TrackANPRState(track_id=3)
        obs1 = PlateObservation(
            text="TN01AB1234", ocr_confidence=0.85, detector_confidence=0.90,
            quality_score=0.80, frame_index=1, optical_quality=OpticalQuality(area=4000),
            is_valid_format=True
        )
        state3.add_observation(obs1)
        res3 = voter.evaluate(state3, current_frame=1)
        self.assertEqual(res3.status, PlateStatus.PROBABLE)
        self.assertEqual(res3.plate, "TN01AB1234")

        # Case 4: 3 consistent valid observations -> CONFIRMED
        obs2 = PlateObservation(
            text="TN01AB1234", ocr_confidence=0.88, detector_confidence=0.92,
            quality_score=0.80, frame_index=7, optical_quality=OpticalQuality(area=4200),
            is_valid_format=True
        )
        obs3 = PlateObservation(
            text="TN01AB1234", ocr_confidence=0.90, detector_confidence=0.95,
            quality_score=0.85, frame_index=13, optical_quality=OpticalQuality(area=4500),
            is_valid_format=True
        )
        state3.add_observation(obs2)
        state3.add_observation(obs3)
        res4 = voter.evaluate(state3, current_frame=13)
        self.assertEqual(res4.status, PlateStatus.CONFIRMED)
        self.assertEqual(res4.plate, "TN01AB1234")
        self.assertEqual(res4.first_confirmed_frame, 13)

    def test_vehicle_classifier(self):
        self.assertEqual(VehicleClassifier.classify_by_id(2), VehicleClass.CAR)
        self.assertEqual(VehicleClassifier.classify_by_id(3), VehicleClass.MOTORCYCLE)
        self.assertEqual(VehicleClassifier.classify_by_id(5), VehicleClass.BUS)
        self.assertEqual(VehicleClassifier.classify_by_id(7), VehicleClass.TRUCK)
        self.assertEqual(VehicleClassifier.classify_by_id(999), VehicleClass.UNKNOWN)

if __name__ == "__main__":
    unittest.main()
