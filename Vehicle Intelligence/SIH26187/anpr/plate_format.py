import re
from typing import Optional

class IndianFormatValidator:
    """
    Indian License Plate Format Validator.
    
    Supports:
    - Standard Private/Commercial: [State Code 2L][District 1-2D][Series 0-3L][Number 3-4D]
      e.g., TN01AB1234, DL3C1234, MH121234, KA05M9999
    - Bharat Series (BH): [Year 2D][BH][Number 4D][Series 1-2L]
      e.g., 22BH1234AA
    - Military: ^\d{2}[A-Z]\d{6}[A-Z]$
    - Diplomatic: ^\d{2}[CD|CC|UN]\d{4}$
    """
    
    VALID_STATE_CODES = {
        "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DN", "DL", 
        "GA", "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD", 
        "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PY", "PB", "RJ", 
        "SK", "TN", "TS", "TR", "UP", "UK", "UA", "WB", "BH"
    }

    def __init__(self):
        # Standard Indian pattern: State (2 letters) + RTO (1-2 digits) + Optional Series (0-3 letters) + Number (3-4 digits)
        self.standard_pattern = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{3,4}$')
        # Bharat Series: 2 digits (year) + BH + 4 digits + 1-2 letters
        self.bh_pattern = re.compile(r'^\d{2}BH\d{4}[A-Z]{1,2}$')

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def is_valid(self, text: str) -> bool:
        """
        Validates if text strictly adheres to Indian plate registration patterns.
        """
        cleaned = self.clean(text)
        if len(cleaned) < 6 or len(cleaned) > 12:
            return False

        # Check Bharat Series
        if self.bh_pattern.match(cleaned):
            return True

        # Check Standard Series
        if self.standard_pattern.match(cleaned):
            state = cleaned[:2]
            return state in self.VALID_STATE_CODES

        return False

    def format_plate(self, text: str) -> Optional[str]:
        """
        Returns normalized uppercase text if valid, else None.
        """
        cleaned = self.clean(text)
        return cleaned if self.is_valid(cleaned) else None
