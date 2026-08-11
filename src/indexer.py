import bm25s
from .models import StudentSearchResults, MinimalSearchResults, MinimalSource
from pathlib import Path
from .parser import Parser
from typing import Any


class Indexer:
    """Builds and queries a BM25 search index over chunked content."""

    def __init__(self, file: Path | str) -> None:
        """Initializes the Indexer.

        Args:
            file: Path to the JSON file containing chunks to index.
        """
        self.parser = Parser()
        self.file = file
        self.chunks: Any = ""
        self.content: list[str] = []
        self.path = "data/processed/bm25_index"

    def _load_info(self) -> None:
        """Loads chunks from ``self.file`` and builds display text for each.

        Populates ``self.chunks`` with the raw chunk data and ``self.content``
        with a formatted text representation of each chunk.
        """
        self.chunks = self.parser.load_data(self.file)
        for chunk in self.chunks:
            text = f"""
            File: {Path(chunk["file_path"]).name}
            Path: {chunk["file_path"]}
            Type: {chunk["type"]}
            Name: {chunk["name"]}

            {chunk["content"]}
            """
            self.content.append(text)

    def index(self) -> None:
        """Builds a BM25 index over the loaded chunks and saves it to disk."""
        self._load_info()
        corpus_tokens = bm25s.tokenize(self.content, show_progress=True)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        self.retriever.save(self.path, corpus=self.chunks)

    def load(self) -> None:
        """Loads a previously saved BM25 index from disk."""
        self.retriever = bm25s.BM25.load(self.path, load_corpus=True)

    def search(
        self,
        query: str | list[str],
        k: int,
        ids: list[str] | None = None,
    ) -> StudentSearchResults:
        """Searches the index for one or more queries.

        Args:
            query: A single query string or a list of query strings.
            k: Number of top results to retrieve per query.
            ids: Optional identifiers to associate with each query. If not
                provided, ids are generated as ``q1``, ``q2``, etc.

        Returns:
            The search results for all queries.
        """
        outputs = []

        queries = [query] if isinstance(query, str) else query
        tokens = bm25s.tokenize(queries)
        results, scores = self.retriever.retrieve(tokens, k=k)

        if ids is None:
            ids = [f"q{i}" for i in range(1, len(queries) + 1)]
        for q, docs, id in zip(queries, results, ids):
            source = [MinimalSource(
                file_path=doc['file_path'],
                first_character_index=doc["first_character"],
                last_character_index=doc['last_character']
            )
                for doc in docs
            ]

            outputs.append(MinimalSearchResults(
                question_id=id,
                question_str=q,
                retrieved_sources=source
            ))

        return StudentSearchResults(search_results=outputs, k=k)
