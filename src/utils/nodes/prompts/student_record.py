"""Student record node prompts."""

STUDENT_RECORD_PROMPT = """Extract structured student information from the following text.
If information is missing, leave those fields empty.

Available text:
{text}

Extract and format as JSON with these fields:
- student_id
- full_name
- date_of_birth (YYYY-MM-DD format)
- email
- phone
- program
- major
- gpa (number)
- enrollment_date (YYYY-MM-DD format)
- expected_graduation (YYYY-MM-DD format)

Respond with valid JSON only."""
