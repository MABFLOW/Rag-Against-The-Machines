import bm25s
from .models import StudentSearchResults, MinimalSearchResults, MinimalSource
from pathlib import Path
from .parser import Parser


class Indexer:

    def __init__(self, file):
        self.parser = Parser()
        self.file = file
        self.chunks = ""
        self.content = []
        self.path = "data/processed/bm25_index"

    def _load_info(self):
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

    def index(self):
        self._load_info()
        corpus_tokens = bm25s.tokenize(self.content, show_progress=True)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        self.retriever.save(self.path, corpus=self.chunks)  

    def load(self):
        self.retriever = bm25s.BM25.load(self.path, load_corpus=True)

    def search(self, query, k, ids=None):
        outputs = []

        queries = [query] if isinstance(query, str) else query
        tokens = bm25s.tokenize(queries)
        results, scores = self.retriever.retrieve(tokens, k=k)

        i = 0
        if ids is None:
            ids = [f"q{i}" for i in range(1, len(queries) + 1)]
        for q, docs, id in zip(queries, results, ids):
            i += 1

            source = [MinimalSource(
                id=i,
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
