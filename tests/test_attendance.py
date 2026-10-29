"""
Unit tests for attendance_service.py
Tests end-to-end registration, frame recognition, duplicate rejection,
DataFrame generation, and CSV export.
"""

import os
import unittest
import cv2
import pandas as pd

from attendance_service import AttendanceService


class TestAttendanceService(unittest.TestCase):
    def setUp(self):
        self.test_db = os.path.join("data", "test_attendance_service_unittest.db")
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        self.service = AttendanceService(db_path=self.test_db)
        self.sample_face_path = os.path.join("data", "samples", "sample_student_1.jpg")

    def tearDown(self):
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except Exception:
                pass

    def test_end_to_end_registration_and_attendance(self):
        """Tests complete registration and attendance flow using a test face."""
        if not os.path.exists(self.sample_face_path):
            self.skipTest("Sample face image not present.")

        img = cv2.imread(self.sample_face_path)

        # 1. Register Student
        ok, msg, ann = self.service.register_student("21CS701", "John Doe", "Sem-VII-CSE", img)
        self.assertTrue(ok, msg)

        # 2. Duplicate registration rejection
        ok_dup, msg_dup, _ = self.service.register_student("21CS701", "John Duplicate", "Sem-VII-CSE", img)
        self.assertFalse(ok_dup)

        # 3. Process frame
        ann_frame, candidates = self.service.process_attendance_frame(img, attendance_date="2026-10-26")
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["is_recognized"])
        self.assertEqual(candidates[0]["student_id"], "21CS701")

        # 4. Confirm attendance
        ok_att, msg_att, is_dup = self.service.confirm_attendance(
            "21CS701", attendance_date="2026-10-26", attendance_time="09:30:00"
        )
        self.assertTrue(ok_att, msg_att)

        # 5. Block duplicate attendance on same date
        ok_att2, msg_att2, is_dup2 = self.service.confirm_attendance(
            "21CS701", attendance_date="2026-10-26", attendance_time="10:00:00"
        )
        self.assertFalse(ok_att2)
        self.assertTrue(is_dup2)

        # 6. Check DataFrame & CSV Export
        df = self.service.get_records_dataframe(date="2026-10-26")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Student ID"], "21CS701")

        csv_str = self.service.export_records_to_csv(date="2026-10-26")
        self.assertIn("John Doe", csv_str)
        self.assertIn("21CS701", csv_str)


if __name__ == "__main__":
    unittest.main()
