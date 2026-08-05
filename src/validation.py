from pydantic import ValidationError
from .models import AnsweredQuestion, UnansweredQuestion, RagDataset, \
    StudentSearchResults
from .exceptions import ParsingError


class Validation:

    def validate_answered_questions(self, data):
        try:
            return AnsweredQuestion.model_validate(data)
        except ValidationError as e:
            raise ParsingError("Invalid AnsweredQuestion schema.") from e

    def validate_unanswered_questions(self, data):
        try:
            return UnansweredQuestion.model_validate(data)
        except ValidationError as e:
            raise ParsingError("Invalid UnansweredQuestion schema.") from e

    def validate_rag_dataset(self, data):
        try:
            return RagDataset.model_validate(data)
        except ValidationError as e:
            raise ParsingError("Invalid RagDataset schema.") from e

    def validate_student_search_results(self, data):
        try:
            return StudentSearchResults.model_validate(data)
        except ValidationError as e:
            raise ParsingError("Invalid StudentSearchResults schema.") from e
