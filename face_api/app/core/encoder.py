"""
Face Encoder Module
Handles face detection, alignment, and embedding extraction using InsightFace.
"""
import numpy as np  # type: ignore
import cv2  # type: ignore
from insightface.app import FaceAnalysis  # type: ignore


class FaceEncoder:
    """Wraps InsightFace for face detection and embedding extraction."""
    
    def __init__(self):
        """Initialize the InsightFace model (CPU-optimized).
        
        Using buffalo_sc for speed; det_size=640 so faces at normal
        office-camera distance (20-80px at 320 input) are reliably detected.
        Env override: MC_DET_SIZE=320 to trade accuracy for speed.
        """
        import os
        det_px = int(os.environ.get("MC_DET_SIZE", "640"))
        
        # Point to the local models directory in face_engine
        # FaceAnalysis(name='N', root='R') looks for 'R/models/N/'
        from app.config import BASE_DIR  # type: ignore
        # BASE_DIR is .../face_api. One level up is project root.
        # Models are in face_engine/models/
        face_engine_dir = os.path.join(os.path.dirname(BASE_DIR), "face_engine")
        
        self.app = FaceAnalysis(
            name='buffalo_sc',
            root=face_engine_dir,  # <--- Looks for face_engine_dir/models/buffalo_sc
            providers=['CPUExecutionProvider']
        )


        self.app.prepare(ctx_id=-1, det_size=(det_px, det_px))


    
    def detect_and_encode(self, image: np.ndarray) -> tuple[np.ndarray | None, dict | None]:
        """
        Detect face in image and extract embedding.
        
        Args:
            image: BGR image as numpy array (OpenCV format)
            
        Returns:
            Tuple of (embedding, face_info) or (None, None) if no face found.
            embedding: 512-dim normalized face embedding
            face_info: Dict with bbox, landmarks, det_score
        """
        faces = self.app.get(image)
        
        if not faces:
            return None, None
        
        # Get the largest face (by bounding box area)
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        embedding = face.embedding
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        face_info = {
            'bbox': face.bbox.tolist(),
            'det_score': float(face.det_score),
        }
        
        return embedding, face_info
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray | None:
        """
        Extract face embedding from image (convenience method).
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            512-dim normalized embedding or None if no face found
        """
        embedding, _ = self.detect_and_encode(image)
        return embedding
    
    def encode_cropped_face(self, face_image: np.ndarray) -> np.ndarray | None:
        """
        Extract embedding from a pre-cropped face image (no detection needed).
        
        This method is designed for images that are already cropped to contain
        just a face, such as output from another face detector.
        
        Args:
            face_image: BGR image of cropped face as numpy array
            
        Returns:
            512-dim normalized embedding or None if encoding fails
        """
        try:
            # Get the recognition model from InsightFace
            # app.models is a dict like {'recognition': model_obj, ...}
            rec_model = None
            if hasattr(self.app, 'models') and isinstance(self.app.models, dict):
                rec_model = self.app.models.get('recognition')
            
            # Fallback: search through model_zoo if available
            if rec_model is None and hasattr(self.app, 'rec_model'):
                rec_model = self.app.rec_model
            
            if rec_model is None:
                print(f"DEBUG: Recognition model not found! models={type(self.app.models)}")
                return None
            
            # Resize face to 112x112 (standard ArcFace input size)
            face_resized = cv2.resize(face_image, (112, 112))
            
            # get_feat expects a list of BGR images in HWC format (uint8)
            # It does its own preprocessing internally
            embedding = rec_model.get_feat([face_resized]).flatten()
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"DEBUG: Error encoding cropped face: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def compute_average_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
        """
        Compute average of multiple embeddings.
        
        Args:
            embeddings: List of 512-dim embeddings
            
        Returns:
            Normalized average embedding
        """
        if not embeddings:
            raise ValueError("Cannot compute average of empty list")
        
        avg = np.mean(embeddings, axis=0)
        # Re-normalize after averaging
        avg = avg / np.linalg.norm(avg)
        return avg
    
    @staticmethod
    def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First 512-dim embedding
            embedding2: Second 512-dim embedding
            
        Returns:
            Similarity score between -1 and 1 (higher = more similar)
        """
        return float(np.dot(embedding1, embedding2))


# Global encoder instance (lazy initialization)
_encoder: FaceEncoder | None = None


def get_encoder() -> FaceEncoder:
    """Get or create the global FaceEncoder instance."""
    global _encoder
    if _encoder is None:
        _encoder = FaceEncoder()
    return _encoder
