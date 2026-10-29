"""
Face Recognition Service Module
Encapsulates OpenCV YuNet (Face Detection) and OpenCV SFace (Face Recognition/Embeddings).
Provides image validation, face alignment, feature extraction, and cosine matching.
"""

import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Default model paths
DEFAULT_YUNET_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")
DEFAULT_SFACE_PATH = os.path.join("models", "face_recognition_sface_2021dec.onnx")

# Standard OpenCV SFace cosine similarity threshold
DEFAULT_COSINE_THRESHOLD = 0.363
DEFAULT_DETECTION_SCORE_THRESHOLD = 0.65
DEFAULT_NMS_THRESHOLD = 0.30


class FaceRecognitionService:
    """
    High-level service for face detection, alignment, embedding generation,
    and cosine similarity matching using OpenCV's deep learning modules.
    """

    def __init__(
        self,
        yunet_path: str = DEFAULT_YUNET_PATH,
        sface_path: str = DEFAULT_SFACE_PATH,
        cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
        score_threshold: float = DEFAULT_DETECTION_SCORE_THRESHOLD,
    ):
        self.yunet_path = yunet_path
        self.sface_path = sface_path
        self.cosine_threshold = cosine_threshold
        self.score_threshold = score_threshold

        self.detector: Optional[cv2.FaceDetectorYN] = None
        self.recognizer: Optional[cv2.FaceRecognizerSF] = None
        self._load_models()

    def _load_models(self) -> None:
        """Loads YuNet and SFace models if ONNX files exist on disk."""
        if os.path.exists(self.yunet_path) and os.path.getsize(self.yunet_path) > 10_000:
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=self.yunet_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=self.score_threshold,
                    nms_threshold=DEFAULT_NMS_THRESHOLD,
                    top_k=5000,
                )
            except Exception as e:
                print(f"[ERROR] Failed to initialize YuNet detector: {e}")
                self.detector = None

        if os.path.exists(self.sface_path) and os.path.getsize(self.sface_path) > 10_000:
            try:
                self.recognizer = cv2.FaceRecognizerSF.create(
                    model=self.sface_path,
                    config="",
                )
            except Exception as e:
                print(f"[ERROR] Failed to initialize SFace recognizer: {e}")
                self.recognizer = None

    def is_operational(self) -> Tuple[bool, str]:
        """Checks if both models are loaded and ready."""
        if not os.path.exists(self.yunet_path):
            return False, f"YuNet model missing at '{self.yunet_path}'. Run 'python models/download_models.py'."
        if not os.path.exists(self.sface_path):
            return False, f"SFace model missing at '{self.sface_path}'. Run 'python models/download_models.py'."
        if self.detector is None:
            return False, "YuNet face detector failed to initialize."
        if self.recognizer is None:
            return False, "SFace face recognizer failed to initialize."
        return True, "Face recognition service is operational."

    def detect_faces(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects faces in a BGR image using YuNet.
        Dynamically adjusts input resolution.
        Returns array of detected faces or None.
        """
        if self.detector is None or image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        return faces

    def extract_embedding(self, image: np.ndarray, face_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Aligns and crops the face using 5 facial landmarks, then extracts
        a 128-dimensional L2-normalized feature embedding via SFace.
        """
        if self.recognizer is None or image is None or face_data is None:
            return None

        try:
            aligned_face = self.recognizer.alignCrop(image, face_data)
            embedding = self.recognizer.feature(aligned_face)
            return embedding
        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {e}")
            return None

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Computes cosine similarity between two 128-d face embeddings.
        Returns float between ~0.0 (unrelated) and 1.0 (identical match).
        """
        if self.recognizer is not None:
            score = self.recognizer.match(embedding1, embedding2, cv2.FaceRecognizerSF_FR_COSINE)
            return float(score)

        # Fallback manual cosine similarity
        v1 = embedding1.flatten()
        v2 = embedding2.flatten()
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def validate_for_registration(
        self, image: np.ndarray
    ) -> Tuple[bool, str, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Validates an image for student registration:
        - Image must have valid dimensions.
        - Exactly one face must be detected.
        - Face detection score and size must satisfy quality thresholds.
        Returns: (is_valid, message, embedding, annotated_image)
        """
        ready, msg = self.is_operational()
        if not ready:
            return False, msg, None, None

        if image is None or image.size == 0:
            return False, "Invalid image: empty or unreadable.", None, None

        h, w = image.shape[:2]
        if h < 100 or w < 100:
            return False, f"Image resolution too low ({w}x{h}). Minimum required is 100x100.", None, None

        faces = self.detect_faces(image)
        if faces is None or len(faces) == 0:
            return (
                False,
                "No face detected in the image. Please ensure good lighting and face the camera directly.",
                None,
                None,
            )

        if len(faces) > 1:
            return (
                False,
                f"Multiple faces detected ({len(faces)}). Registration requires exactly one face in frame.",
                None,
                None,
            )

        face = faces[0]
        confidence = float(face[-1])
        x, y, box_w, box_h = map(int, face[:4])

        if confidence < self.score_threshold:
            return (
                False,
                f"Face detection confidence is too low ({confidence*100:.1f}%). Please retake in clearer lighting.",
                None,
                None,
            )

        if box_w < 50 or box_h < 50:
            return (
                False,
                f"Detected face is too small ({box_w}x{box_h}px). Please move closer to the camera.",
                None,
                None,
            )

        embedding = self.extract_embedding(image, face)
        if embedding is None:
            return False, "Failed to extract face embedding representation.", None, None

        # Annotate valid face
        annotated = image.copy()
        cv2.rectangle(annotated, (x, y), (x + box_w, y + box_h), (0, 220, 0), 2)

        # Draw 5 landmarks (right eye, left eye, nose, right mouth, left mouth)
        landmarks = [(int(face[i]), int(face[i + 1])) for i in range(4, 14, 2)]
        for pt in landmarks:
            cv2.circle(annotated, pt, 3, (0, 255, 255), -1)

        label = f"Face Validated ({confidence*100:.0f}%)"
        cv2.putText(
            annotated,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

        return True, "Face validated successfully.", embedding, annotated

    def recognize_faces(
        self,
        image: np.ndarray,
        registered_records: List[Dict],
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detects all faces in an image and matches each against registered student embeddings.
        Returns:
            annotated_image: BGR image with bounding boxes, landmarks, and identification labels.
            recognitions: List of dicts with match details.
        """
        if threshold is None:
            threshold = self.cosine_threshold

        ready, _ = self.is_operational()
        if not ready or image is None or image.size == 0:
            return image, []

        faces = self.detect_faces(image)
        if faces is None or len(faces) == 0:
            return image, []

        annotated = image.copy()
        recognitions = []

        for face in faces:
            x, y, w, h = map(int, face[:4])
            x, y = max(0, x), max(0, y)
            confidence = float(face[-1])

            query_embedding = self.extract_embedding(image, face)

            best_match: Optional[Dict] = None
            highest_sim = -1.0

            if query_embedding is not None and len(registered_records) > 0:
                for rec in registered_records:
                    target_emb = rec["embedding"]
                    sim = self.compute_similarity(query_embedding, target_emb)
                    if sim > highest_sim:
                        highest_sim = sim
                        best_match = rec

            # Determine recognition decision
            is_recognized = (
                best_match is not None
                and highest_sim >= threshold
            )

            if is_recognized:
                student_id = best_match["student_id"]
                name = best_match["name"]
                class_name = best_match["class_name"]
                box_color = (0, 200, 0)  # Green
                tag = f"{name} ({student_id}) [{highest_sim*100:.1f}%]"
                status_label = "Recognized"
            else:
                student_id = "UNKNOWN"
                name = "Unknown Face"
                class_name = "N/A"
                box_color = (0, 100, 255)  # Orange/Red
                tag = f"Unknown ({highest_sim*100:.1f}%)" if highest_sim > 0 else "Unknown"
                status_label = "Needs Review"

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)

            # Draw facial landmarks
            landmarks = [(int(face[i]), int(face[i + 1])) for i in range(4, 14, 2)]
            for pt in landmarks:
                cv2.circle(annotated, pt, 2, (255, 255, 0), -1)

            # Draw label banner
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            banner_y1 = max(0, y - th - 8)
            banner_y2 = y
            cv2.rectangle(annotated, (x, banner_y1), (x + tw + 10, banner_y2), box_color, -1)
            cv2.putText(
                annotated,
                tag,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            recognitions.append({
                "student_id": student_id,
                "name": name,
                "class_name": class_name,
                "is_recognized": is_recognized,
                "status_label": status_label,
                "similarity": round(highest_sim, 4),
                "confidence": round(confidence, 3),
                "box": (x, y, w, h),
            })

        return annotated, recognitions
