from pydantic import BaseModel, Field
from typing import List
import uuid


class ChunkModel(BaseModel):
    """A single indexed chunk of content produced by the chunker.

    Attributes:
        id: Unique identifier of the chunk.
        file_path: Path to the source file the chunk was taken from.
        type: Kind of chunk (e.g. ``"class"``, ``"function"``, ``"text"``).
        name: Name of the enclosing symbol or file stem.
        part_id: Index of this chunk among the parts of its source content.
        total_parts: Total number of parts the source content was split into.
        content: The chunk's text content.
        first_character: Start character offset within the source file.
        last_character: End character offset within the source file.
    """

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
    """A pointer to a slice of content within a source file.

    Attributes:
        file_path: Path to the source file.
        first_character_index: Start character offset within the file.
        last_character_index: End character offset within the file.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A dataset question without an associated answer yet.

    Attributes:
        question_id: Unique identifier of the question, auto-generated if
            not provided.
        question: The question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """An :class:`UnansweredQuestion` augmented with its ground-truth answer.

    Attributes:
        sources: Source locations that support the answer.
        answer: The ground-truth answer text.
    """

    sources: List[MinimalSource]
    answer: str


class MinimalSearchResults(BaseModel):
    """Retrieval results for a single query.

    Attributes:
        question_id: Identifier of the query this result belongs to.
        question: The query text.
        retrieved_sources: Sources retrieved for the query.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """A :class:`MinimalSearchResults` augmented with a generated answer.

    Attributes:
        answer: The generated answer text.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """A collection of search results produced for a dataset.

    Attributes:
        search_results: Per-query retrieval results.
        k: Number of sources retrieved per query.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """A collection of search results with generated answers.

    Attributes:
        search_results: Per-query retrieval results and answers.
        k: Number of sources retrieved per query.
    """

    search_results: List[MinimalAnswer]
    k: int


class RagDataset(BaseModel):
    """A dataset of questions, answered or not, used for evaluation.

    Attributes:
        rag_questions: The dataset's questions.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]
