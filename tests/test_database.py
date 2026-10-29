"""
Unit tests for database.py
Tests database initialization, CRUD operations, duplicate student handling,
uniqueness constraints on daily attendance, and daily summary calculations.
"""

import os
import unittest
import numpy as np

import database


class TestDatabaseModule(unittest.TestCase):
    def setUp(self):
        self.test_db = os.path.join("data", "test_unittest_db.db")
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        database.init_db(self.test_db)

    def tearDown(self):
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except Exception:
                pass

    def test_database_initialization(self):
        """Verifies tables are created properly."""
        with database.get_db(self.test_db) as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row["name"] for row in cursor.fetchall()}
            self.assertIn("students", tables)
            self.assertIn("face_templates", tables)
            self.assertIn("attendance", tables)

    def test_add_and_get_student(self):
        """Tests adding and retrieving a student record."""
        emb = np.random.randn(1, 128).astype(np.float32)
        ok, msg = database.add_student("21CS001", "Alice Smith", "Sem-VII-CSE", emb, self.test_db)
        self.assertTrue(ok, msg)

        student = database.get_student("21CS001", self.test_db)
        self.assertIsNotNone(student)
        self.assertEqual(student["name"], "Alice Smith")
        self.assertEqual(student["class_name"], "Sem-VII-CSE")

    def test_prevent_duplicate_student_id(self):
        """Ensures duplicate student_id insertion is blocked."""
        emb = np.random.randn(1, 128).astype(np.float32)
        ok1, msg1 = database.add_student("21CS002", "Bob Vance", "Sem-VII-CSE", emb, self.test_db)
        self.assertTrue(ok1)

        ok2, msg2 = database.add_student("21CS002", "Bob Duplicate", "Sem-VII-CSE", emb, self.test_db)
        self.assertFalse(ok2)
        self.assertIn("already registered", msg2.lower())

    def test_delete_student_cascade(self):
        """Tests student deletion and cascade cleanup of template & attendance."""
        emb = np.random.randn(1, 128).astype(np.float32)
        database.add_student("21CS003", "Charlie Brown", "Sem-VII-CSE", emb, self.test_db)
        database.mark_attendance("21CS003", "2026-10-26", "09:00:00", "Present", self.test_db)

        deleted = database.delete_student("21CS003", self.test_db)
        self.assertTrue(deleted)
        self.assertIsNone(database.get_student("21CS003", self.test_db))

        # Check cascade in attendance
        records = database.get_attendance_records(student_id="21CS003", db_path=self.test_db)
        self.assertEqual(len(records), 0)

    def test_attendance_and_duplicate_prevention(self):
        """Tests attendance recording and same-day duplicate prevention constraint."""
        emb = np.random.randn(1, 128).astype(np.float32)
        database.add_student("21CS004", "Diana Prince", "Sem-VII-CSE", emb, self.test_db)

        # 1. First record on 2026-10-26
        ok1, msg1, is_dup1 = database.mark_attendance("21CS004", "2026-10-26", "09:00:00", "Present", self.test_db)
        self.assertTrue(ok1)
        self.assertFalse(is_dup1)

        # 2. Duplicate record on the same date should be rejected
        ok2, msg2, is_dup2 = database.mark_attendance("21CS004", "2026-10-26", "11:30:00", "Present", self.test_db)
        self.assertFalse(ok2)
        self.assertTrue(is_dup2)

        # 3. Attendance on a different date should succeed
        ok3, msg3, is_dup3 = database.mark_attendance("21CS004", "2026-10-27", "09:05:00", "Present", self.test_db)
        self.assertTrue(ok3)
        self.assertFalse(is_dup3)

    def test_daily_summary_calculation(self):
        """Tests total, present, absent, and percentage statistics."""
        emb = np.random.randn(1, 128).astype(np.float32)
        database.add_student("21CS005", "Eve Adams", "Sem-VII-CSE", emb, self.test_db)
        database.add_student("21CS006", "Frank Castle", "Sem-VII-CSE", emb, self.test_db)

        # Mark only Eve as present
        database.mark_attendance("21CS005", "2026-10-26", "09:00:00", "Present", self.test_db)

        summary = database.get_daily_summary("2026-10-26", "Sem-VII-CSE", self.test_db)
        self.assertEqual(summary["total_registered"], 2)
        self.assertEqual(summary["present_count"], 1)
        self.assertEqual(summary["absent_count"], 1)
        self.assertEqual(summary["attendance_percentage"], 50.0)


if __name__ == "__main__":
    unittest.main()
