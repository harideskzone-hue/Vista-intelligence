from typing import Optional
from core.schemas import VehicleClass

COCO_TO_VEHICLE_CLASS = {
    2: VehicleClass.CAR,
    3: VehicleClass.MOTORCYCLE,
    5: VehicleClass.BUS,
    7: VehicleClass.TRUCK
}

NAME_TO_VEHICLE_CLASS = {
    "car": VehicleClass.CAR,
    "motorcycle": VehicleClass.MOTORCYCLE,
    "motorbike": VehicleClass.MOTORCYCLE,
    "bus": VehicleClass.BUS,
    "truck": VehicleClass.TRUCK
}

class VehicleClassifier:
    """
    Classifies vehicle detections into standardized VehicleClass enums.
    Supported classes:
      - car
      - motorcycle
      - bus
      - truck
    """
    @staticmethod
    def classify_by_id(coco_class_id: int) -> VehicleClass:
        return COCO_TO_VEHICLE_CLASS.get(coco_class_id, VehicleClass.UNKNOWN)

    @staticmethod
    def classify_by_name(class_name: str) -> VehicleClass:
        return NAME_TO_VEHICLE_CLASS.get(class_name.lower().strip(), VehicleClass.UNKNOWN)
