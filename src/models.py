from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List
import uuid
from .exceptions import *


class ChunkModel(BaseModel):
    id: int
    file_path: str
    type: str | None
    name: str | None
    part_id: int
    total_parts: int | None
    content: str | None
    first_character: int
    last_character: int



class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int




class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda:
    str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str

class MinimalSearchResults(BaseModel):
    question_id: str
    question_str: str
    retrieved_sources: List[MinimalSource]

class MinimalAnswer(MinimalSearchResults):
    answer: str

class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int

class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int

class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]