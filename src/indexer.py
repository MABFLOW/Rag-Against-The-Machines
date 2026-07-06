import bm25s
import json


class Indexer:
    
    def __init__(self, file):
        with open(file, 'r') as f:
            self.chunks = json.load(f)
        self.content = [chunk['content'] for chunk in self.chunks]

        

    def index(self):
        corpus_tokens = bm25s.tokenize(self.content)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        self.retriever.save("bm25_index", corpus=self.chunks)

    def load(self):
        self.retriever = bm25s.BM25.load("bm25_index", load_corpus=True)
    
    def search(self):
        pass