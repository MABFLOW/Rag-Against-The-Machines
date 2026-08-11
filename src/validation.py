from pydantic import ValidationError
from .models import AnsweredQuestion, UnansweredQuestion, RagDataset, \
    StudentSearchResults
from .exceptions import ParsingError
from typing import Any


class Validation:
    """Validates raw data against the project's pydantic schemas."""

    def validate_answered_questions(self, data: Any) -> AnsweredQuestion:
        """Validates data against the :class:`AnsweredQuestion` schema.

        Args:
            data: Raw data to validate.

        Returns:
            The validated :class:`AnsweredQuestion` instance.

        Raises:
            ParsingError: If ``data`` does not match the schema.
        """
        try:
            return AnsweredQuestion.model_validate(data)
        except ValidationError as e:
            raise ParsingError("Invalid AnsweredQuestion schema.") from e

    def validate_unanswered_questions(self, data: Any) -> UnansweredQuestion:
        """Validates data against the :class:`UnansweredQuestion` schema.

        Args:
            data: Raw data to validate.

        Returns:
            The validated :class:`UnansweredQuestion` instance.

        Raises:
            ParsingError: If ``data`` does not match the schema.
        """
        try:
            return UnansweredQuestion.model_validate(data)
        except ValidationError as e:
            raise ParsingError("Invalid UnansweredQuestion schema.") from e

    def validate_rag_dataset(self, data: Any) -> RagDataset:
        """Validates data against the :class:`RagDataset` schema.

        Args:
            data: Raw data to validate.

        Returns:
            The validated :class:`RagDataset` instance.

        Raises:
            ParsingError: If ``data`` does not match the schema.
        """
        try:
            return RagDataset.model_validate(data)
        except ValidationError as e:
            raise ParsingError("Invalid RagDataset schema.") from e

    def validate_student_search_results(
        self, data: Any
    ) -> StudentSearchResults:
        """Validates data against the :class:`StudentSearchResults` schema.

        Args:
            data: Raw data to validate.

        Returns:
            The validated :class:`StudentSearchResults` instance.

        Raises:
            ParsingError: If ``data`` does not match the schema.
        """
        try:
            return StudentSearchResults.model_validate(data)
        except ValidationError as e:
            raise ParsingError("Invalid StudentSearchResults schema.") from e
