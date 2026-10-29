"""
Attendance Service Module
Coordinates student registration, face recognition processing,
attendance confirmation, duplicate prevention, analytics, and CSV exports.
"""

from datetime import datetime
import io
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import database
from face_recognition_service import FaceRecognitionService


class AttendanceService:
    """
    Business logic layer for the Smart Attendance Management System.
    Bridges face recognition algorithms with SQLite persistent storage.
    """

    def __init__(
        self,
        db_path: str = database.DEFAULT_DB_PATH,
        face_service: Optional[FaceRecognitionService] = None,
    ):
        self.db_path = db_path
        self.face_service = face_service or FaceRecognitionService()
        database.init_db(self.db_path)

    # ------------------------------------------------------------------------
    # Student Registration
    # ------------------------------------------------------------------------
    def register_student(
        self,
        student_id: str,
        name: str,
        class_name: str,
        image_bgr: np.ndarray,
    ) -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        Validates student input and facial image, extracts the 128-d embedding,
        and saves the student record and embedding into SQLite.
        Returns: (success: bool, message: str, annotated_image: Optional[np.ndarray])
        """
        student_id = student_id.strip()
        name = name.strip()
        class_name = class_name.strip()

        if not student_id or not name or not class_name:
            return False, "Student ID, Name, and Class Name are all required.", None

        # Check if student ID already exists
        existing = database.get_student(student_id, db_path=self.db_path)
        if existing:
            return False, f"Student ID '{student_id}' is already registered to '{existing['name']}'.", None

        # Validate facial image
        is_valid, msg, embedding, annotated_img = self.face_service.validate_for_registration(image_bgr)
        if not is_valid:
            return False, f"Image validation failed: {msg}", annotated_img

        # Save to database
        ok, db_msg = database.add_student(
            student_id=student_id,
            name=name,
            class_name=class_name,
            embedding=embedding,
            db_path=self.db_path,
        )

        return ok, db_msg, annotated_img

    # ------------------------------------------------------------------------
    # Attendance Recognition & Evaluation
    # ------------------------------------------------------------------------
    def process_attendance_frame(
        self,
        image_bgr: np.ndarray,
        attendance_date: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detects faces in the frame, matches against registered students,
        and checks if attendance was already recorded today.
        Returns:
            annotated_image: BGR frame with visual overlays.
            candidates: List of detected candidate dicts with flags for confirmation.
        """
        if not attendance_date:
            attendance_date = datetime.now().strftime("%Y-%m-%d")

        registered_records = database.get_all_face_embeddings(db_path=self.db_path)
        annotated_img, recognitions = self.face_service.recognize_faces(
            image=image_bgr,
            registered_records=registered_records,
            threshold=threshold,
        )

        processed_candidates = []
        for rec in recognitions:
            sid = rec["student_id"]
            if rec["is_recognized"]:
                already_marked = database.is_attendance_marked(sid, attendance_date, db_path=self.db_path)
                status_text = "Already Marked Today" if already_marked else "Ready to Confirm"
            else:
                already_marked = False
                status_text = "Unknown Face"

            cand = {
                **rec,
                "already_marked": already_marked,
                "status_text": status_text,
                "attendance_date": attendance_date,
            }
            processed_candidates.append(cand)

        return annotated_img, processed_candidates

    def confirm_attendance(
        self,
        student_id: str,
        attendance_date: Optional[str] = None,
        attendance_time: Optional[str] = None,
        status: str = "Present",
    ) -> Tuple[bool, str, bool]:
        """
        Records attendance for a verified student.
        Prevents duplicate entries via database uniqueness constraints.
        Returns: (success: bool, message: str, is_duplicate: bool)
        """
        return database.mark_attendance(
            student_id=student_id,
            attendance_date=attendance_date,
            attendance_time=attendance_time,
            status=status,
            db_path=self.db_path,
        )

    # ------------------------------------------------------------------------
    # Reports, Summaries & Exports
    # ------------------------------------------------------------------------
    def get_summary(
        self,
        attendance_date: str,
        class_name: Optional[str] = None,
    ) -> Dict:
        """Computes total registered, present, absent, and attendance percentage."""
        return database.get_daily_summary(
            attendance_date=attendance_date,
            class_name=class_name,
            db_path=self.db_path,
        )

    def get_records_dataframe(
        self,
        date: Optional[str] = None,
        student_id: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Returns a formatted Pandas DataFrame of attendance records."""
        records = database.get_attendance_records(
            date=date,
            student_id=student_id,
            class_name=class_name,
            db_path=self.db_path,
        )

        if not records:
            return pd.DataFrame(
                columns=["ID", "Student ID", "Student Name", "Class", "Date", "Time", "Status"]
            )

        df = pd.DataFrame(records)
        df.rename(
            columns={
                "id": "ID",
                "student_id": "Student ID",
                "name": "Student Name",
                "class_name": "Class",
                "attendance_date": "Date",
                "attendance_time": "Time",
                "status": "Status",
            },
            inplace=True,
        )
        return df

    def export_records_to_csv(
        self,
        date: Optional[str] = None,
        student_id: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> str:
        """Exports attendance records to a CSV formatted string."""
        df = self.get_records_dataframe(date=date, student_id=student_id, class_name=class_name)
        return df.to_csv(index=False)
