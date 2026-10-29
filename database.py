"""
Database Module for Smart Attendance Management System
Manages SQLite connection, schema initialization, student records,
face templates (embeddings), and attendance transactions.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple
import numpy as np

DEFAULT_DB_PATH = os.path.join("data", "attendance.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Returns an SQLite connection with Row factory and foreign keys enabled.
    Creates parent directories if they don't exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def get_db(db_path: str = DEFAULT_DB_PATH) -> Generator[sqlite3.Cursor, None, None]:
    """
    Context manager that yields a cursor and automatically commits & closes the connection.
    Prevents file lock issues on Windows NTFS.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initializes the database schema if tables do not already exist.
    Enforces uniqueness constraints and foreign key relationships.
    """
    with get_db(db_path) as cursor:
        # 1. Students Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                class_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

        # 2. Face Templates (Embeddings) Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS face_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
            );
            """
        )

        # 3. Attendance Table with Uniqueness Constraint on (student_id, attendance_date)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                attendance_time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Present',
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
                CONSTRAINT unique_daily_attendance UNIQUE (student_id, attendance_date)
            );
            """
        )


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Converts a numpy float32 embedding vector to bytes for BLOB storage."""
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Converts BLOB bytes back into a 128-dimensional float32 numpy vector."""
    return np.frombuffer(blob, dtype=np.float32).reshape(1, -1)


def add_student(
    student_id: str,
    name: str,
    class_name: str,
    embedding: np.ndarray,
    db_path: str = DEFAULT_DB_PATH,
) -> Tuple[bool, str]:
    """
    Registers a new student and stores their face embedding.
    Ensures student_id is unique.
    """
    student_id = student_id.strip()
    name = name.strip()
    class_name = class_name.strip()

    if not student_id or not name or not class_name:
        return False, "Student ID, Name, and Class cannot be empty."

    if embedding is None or embedding.size == 0:
        return False, "Invalid face embedding vector."

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    embedding_bytes = serialize_embedding(embedding)

    try:
        with get_db(db_path) as cursor:
            cursor.execute(
                """
                INSERT INTO students (student_id, name, class_name, created_at)
                VALUES (?, ?, ?, ?);
                """,
                (student_id, name, class_name, created_at),
            )
            cursor.execute(
                """
                INSERT INTO face_templates (student_id, embedding, created_at)
                VALUES (?, ?, ?);
                """,
                (student_id, embedding_bytes, created_at),
            )
        return True, f"Student '{name}' ({student_id}) registered successfully."
    except sqlite3.IntegrityError as e:
        if "UNIQUE constraint failed" in str(e) or "PRIMARY KEY" in str(e):
            return False, f"Student ID '{student_id}' is already registered."
        return False, f"Database integrity error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error during registration: {str(e)}"


def get_student(student_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict]:
    """Retrieves a single student's record by student ID."""
    with get_db(db_path) as cursor:
        cursor.execute(
            """
            SELECT student_id, name, class_name, created_at
            FROM students
            WHERE student_id = ?;
            """,
            (student_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_all_students(class_name: Optional[str] = None, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """Retrieves all registered students, optionally filtered by class."""
    with get_db(db_path) as cursor:
        if class_name and class_name.lower() != "all":
            cursor.execute(
                """
                SELECT student_id, name, class_name, created_at
                FROM students
                WHERE class_name = ?
                ORDER BY student_id ASC;
                """,
                (class_name,),
            )
        else:
            cursor.execute(
                """
                SELECT student_id, name, class_name, created_at
                FROM students
                ORDER BY class_name ASC, student_id ASC;
                """
            )
        return [dict(row) for row in cursor.fetchall()]


def delete_student(student_id: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Deletes a student and their face template/attendance via CASCADE."""
    with get_db(db_path) as cursor:
        cursor.execute("DELETE FROM students WHERE student_id = ?;", (student_id,))
        return cursor.rowcount > 0


def get_all_face_embeddings(db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """
    Retrieves all registered face embeddings with corresponding student metadata.
    Returns list of dicts with keys: student_id, name, class_name, embedding (numpy ndarray).
    """
    with get_db(db_path) as cursor:
        cursor.execute(
            """
            SELECT s.student_id, s.name, s.class_name, ft.embedding
            FROM students s
            JOIN face_templates ft ON s.student_id = ft.student_id;
            """
        )
        records = []
        for row in cursor.fetchall():
            records.append({
                "student_id": row["student_id"],
                "name": row["name"],
                "class_name": row["class_name"],
                "embedding": deserialize_embedding(row["embedding"]),
            })
        return records


def is_attendance_marked(
    student_id: str,
    attendance_date: str,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Checks whether attendance has already been recorded for a student on a specific date."""
    with get_db(db_path) as cursor:
        cursor.execute(
            """
            SELECT 1 FROM attendance
            WHERE student_id = ? AND attendance_date = ?;
            """,
            (student_id, attendance_date),
        )
        return cursor.fetchone() is not None


def mark_attendance(
    student_id: str,
    attendance_date: Optional[str] = None,
    attendance_time: Optional[str] = None,
    status: str = "Present",
    db_path: str = DEFAULT_DB_PATH,
) -> Tuple[bool, str, bool]:
    """
    Records attendance for a student on a given date.
    Returns: (success: bool, message: str, is_duplicate: bool)
    """
    if not attendance_date:
        attendance_date = datetime.now().strftime("%Y-%m-%d")
    if not attendance_time:
        attendance_time = datetime.now().strftime("%H:%M:%S")

    # Verify student exists
    student = get_student(student_id, db_path=db_path)
    if not student:
        return False, f"Student ID '{student_id}' does not exist.", False

    # Check for existing record on this date
    if is_attendance_marked(student_id, attendance_date, db_path=db_path):
        return (
            False,
            f"Attendance already recorded today ({attendance_date}) for {student['name']} ({student_id}).",
            True,
        )

    try:
        with get_db(db_path) as cursor:
            cursor.execute(
                """
                INSERT INTO attendance (student_id, attendance_date, attendance_time, status)
                VALUES (?, ?, ?, ?);
                """,
                (student_id, attendance_date, attendance_time, status),
            )
        return True, f"Attendance marked for {student['name']} ({student_id}) at {attendance_time}.", False
    except sqlite3.IntegrityError:
        return (
            False,
            f"Attendance duplicate constraint triggered for {student['name']} on {attendance_date}.",
            True,
        )
    except Exception as e:
        return False, f"Database error while marking attendance: {str(e)}", False


def get_attendance_records(
    date: Optional[str] = None,
    student_id: Optional[str] = None,
    class_name: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict]:
    """
    Retrieves attendance records with optional filtering by date, student_id, or class_name.
    Joins with students table to include student name and class.
    """
    query = """
        SELECT a.id, a.student_id, s.name, s.class_name, a.attendance_date, a.attendance_time, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
        WHERE 1=1
    """
    params = []

    if date:
        query += " AND a.attendance_date = ?"
        params.append(date)

    if student_id:
        query += " AND (a.student_id LIKE ? OR s.name LIKE ?)"
        params.extend([f"%{student_id}%", f"%{student_id}%"])

    if class_name and class_name.lower() != "all":
        query += " AND s.class_name = ?"
        params.append(class_name)

    query += " ORDER BY a.attendance_date DESC, a.attendance_time DESC;"

    with get_db(db_path) as cursor:
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_daily_summary(
    attendance_date: str,
    class_name: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict:
    """
    Calculates attendance summary for a given date and class:
    - Total registered students
    - Present students count
    - Absent students count
    - Attendance percentage
    - List of present students
    - List of absent students
    """
    all_students = get_all_students(class_name=class_name, db_path=db_path)
    total_registered = len(all_students)

    attended_records = get_attendance_records(date=attendance_date, class_name=class_name, db_path=db_path)
    present_student_ids = {r["student_id"] for r in attended_records if r["status"] == "Present"}

    present_count = len(present_student_ids)
    absent_count = max(0, total_registered - present_count)
    attendance_pct = (present_count / total_registered * 100.0) if total_registered > 0 else 0.0

    present_students = [s for s in all_students if s["student_id"] in present_student_ids]
    absent_students = [s for s in all_students if s["student_id"] not in present_student_ids]

    return {
        "date": attendance_date,
        "class_name": class_name or "All Classes",
        "total_registered": total_registered,
        "present_count": present_count,
        "absent_count": absent_count,
        "attendance_percentage": round(attendance_pct, 2),
        "present_students": present_students,
        "absent_students": absent_students,
        "records": attended_records,
    }


def get_unique_classes(db_path: str = DEFAULT_DB_PATH) -> List[str]:
    """Returns a sorted list of all unique class names present in the students table."""
    with get_db(db_path) as cursor:
        cursor.execute("SELECT DISTINCT class_name FROM students ORDER BY class_name ASC;")
        return [row["class_name"] for row in cursor.fetchall()]


def get_distinct_attendance_dates(db_path: str = DEFAULT_DB_PATH) -> List[str]:
    """Returns a sorted list of all unique dates when attendance was recorded."""
    with get_db(db_path) as cursor:
        cursor.execute("SELECT DISTINCT attendance_date FROM attendance ORDER BY attendance_date DESC;")
        return [row["attendance_date"] for row in cursor.fetchall()]


def get_student_attendance_summary(class_name: Optional[str] = None, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """
    Computes overall attendance statistics per student:
    total sessions attended vs total distinct attendance dates recorded in the database.
    """
    all_dates = get_distinct_attendance_dates(db_path=db_path)
    total_dates = max(1, len(all_dates))

    students = get_all_students(class_name=class_name, db_path=db_path)
    stats = []

    with get_db(db_path) as cursor:
        for s in students:
            cursor.execute(
                "SELECT COUNT(*) as attended FROM attendance WHERE student_id = ? AND status = 'Present';",
                (s["student_id"],),
            )
            row = cursor.fetchone()
            attended = row["attended"] if row else 0
            pct = round((attended / total_dates) * 100.0, 1) if len(all_dates) > 0 else 0.0

            stats.append({
                "student_id": s["student_id"],
                "name": s["name"],
                "class_name": s["class_name"],
                "attended_sessions": attended,
                "total_sessions": len(all_dates),
                "attendance_percentage": pct,
            })

    return stats
