"""Student record helpers."""

import logging
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from src.utils.state import StudentRecord

logger = logging.getLogger(__name__)


class StudentTools:
    """
    Student record CRUD operations and data extraction.

    In production, replace the in-memory storage with a proper database.
    """

    def __init__(self):
        self._records: dict[str, StudentRecord] = {}
        self._email_index: dict[str, str] = {}

    def create_record(self, record: StudentRecord) -> StudentRecord:
        """
        Create a new student record.

        Args:
            record: StudentRecord to create

        Returns:
            Created record with generated ID if not provided
        """
        if not record.student_id:
            record.student_id = f"STU_{uuid.uuid4().hex[:8].upper()}"

        timestamp = datetime.now(UTC)
        record.created_at = timestamp
        record.updated_at = timestamp

        self._records[record.student_id] = record
        if record.email:
            self._email_index[record.email] = record.student_id

        logger.info("Created student record: %s", record.student_id)
        return record

    def get_record(self, student_id: str) -> StudentRecord | None:
        """Retrieve a student record by ID."""
        return self._records.get(student_id)

    def find_by_email(self, email: str) -> StudentRecord | None:
        """Find a student record by email address."""
        student_id = self._email_index.get(email)
        if student_id:
            return self._records.get(student_id)
        return None

    def update_record(
        self,
        student_id: str,
        updates: dict[str, Any],
    ) -> StudentRecord | None:
        """
        Update an existing student record.

        Args:
            student_id: Student identifier
            updates: Dictionary of fields to update

        Returns:
            Updated record or None if not found
        """
        record = self._records.get(student_id)
        if not record:
            return None

        old_email = record.email
        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)

        record.updated_at = datetime.now(UTC)
        self._update_email_index(student_id, old_email, updates.get("email"))

        logger.info("Updated student record: %s", student_id)
        return record

    def _update_email_index(
        self,
        student_id: str,
        old_email: str | None,
        new_email: Any,
    ) -> None:
        """Keep the email lookup index consistent after record updates."""
        if new_email == old_email:
            return
        if old_email:
            self._email_index.pop(old_email, None)
        if new_email:
            self._email_index[str(new_email)] = student_id

    def delete_record(self, student_id: str) -> bool:
        """Delete a student record."""
        record = self._records.pop(student_id, None)
        if not record:
            return False
        if record.email:
            self._email_index.pop(record.email, None)
        logger.info("Deleted student record: %s", student_id)
        return True

    def list_records(
        self,
        program: str | None = None,
        limit: int = 100,
    ) -> list[StudentRecord]:
        """List student records with optional filtering."""
        records = list(self._records.values())
        if program:
            records = [record for record in records if record.program == program]
        return records[:limit]

    def extract_from_text(self, text: str) -> dict[str, Any]:
        """
        Extract structured student information from raw text.

        This is a simple rule-based extractor. In production,
        use an LLM for more sophisticated extraction.

        Args:
            text: Raw text from OCR

        Returns:
            Dictionary of extracted fields
        """
        extracted: dict[str, Any] = {}
        for raw_line in text.lower().splitlines():
            line = raw_line.strip()
            if "name:" in line or "full name:" in line:
                extracted["full_name"] = self._extract_text_value(line).title()
            elif "student id:" in line or "id:" in line:
                extracted["student_id"] = self._extract_text_value(line).upper()
            elif "email:" in line or "e-mail:" in line:
                extracted["email"] = self._extract_text_value(line)
            elif "gpa:" in line or "grade point average:" in line:
                with suppress(ValueError):
                    extracted["gpa"] = float(self._extract_text_value(line))
            elif "program:" in line or "degree:" in line:
                extracted["program"] = self._extract_text_value(line).title()
            elif "major:" in line:
                extracted["major"] = self._extract_text_value(line).title()

        return extracted

    def _extract_text_value(self, line: str) -> str:
        """Extract the value segment from a key-value line."""
        return line.split(":", 1)[-1].strip()
