"""
Unit tests for face_recognition_service.py
Tests model presence, initialization, face detection, feature extraction,
cosine similarity computation, and validation rules.
"""

import os
import unittest
import cv2
import numpy as np

from face_recognition_service import FaceRecognitionService


class TestFaceRecognitionService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = FaceRecognitionService()
        cls.sample_face_path = os.path.join("data", "samples", "sample_student_1.jpg")

    def test_service_operational(self):
        """Verifies YuNet and SFace models are properly loaded."""
        ready, msg = self.service.is_operational()
        self.assertTrue(ready, f"Service not operational: {msg}")

    def test_cosine_similarity_metrics(self):
        """Verifies cosine similarity properties (1.0 for self, low for orthogonal/random)."""
        emb1 = np.random.randn(1, 128).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)

        sim_self = self.service.compute_similarity(emb1, emb1)
        self.assertAlmostEqual(sim_self, 1.0, places=3)

        emb2 = np.random.randn(1, 128).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)
        sim_random = self.service.compute_similarity(emb1, emb2)
        self.assertLess(sim_random, 0.40)

    def test_validate_for_registration_sample(self):
        """Tests validation on a real face portrait."""
        if not os.path.exists(self.sample_face_path):
            self.skipTest("Sample face image not present.")

        img = cv2.imread(self.sample_face_path)
        is_valid, msg, embedding, annotated = self.service.validate_for_registration(img)
        self.assertTrue(is_valid, f"Validation failed: {msg}")
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape, (1, 128))
        self.assertIsNotNone(annotated)

    def test_validate_invalid_inputs(self):
        """Tests rejection of invalid or empty images."""
        # 1. Blank/black image (no faces)
        black_img = np.zeros((200, 200, 3), dtype=np.uint8)
        valid_black, msg_black, _, _ = self.service.validate_for_registration(black_img)
        self.assertFalse(valid_black)
        self.assertIn("no face", msg_black.lower())

        # 2. Too small image
        tiny_img = np.zeros((40, 40, 3), dtype=np.uint8)
        valid_tiny, msg_tiny, _, _ = self.service.validate_for_registration(tiny_img)
        self.assertFalse(valid_tiny)
        self.assertIn("too low", msg_tiny.lower())


if __name__ == "__main__":
    unittest.main()
