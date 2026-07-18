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
    
    def search(self, query, k):
        outputs = []
        
        queries = [query] if isinstance(query, str) else query
        tokens = bm25s.tokenize(queries)
        results, scores = self.retriever.retrieve(tokens,k=k)

            
        for q, docs, doc_scores in zip(queries, results[0], scores[0]):
            query_result = [
                {
                "file_path": doc["file"],
                "first_character_index": doc[
                    "first_character"
                ],
                "last_character_index": doc[
                    "last_character"
                ],
                "content": doc["content"],
                "score": float(score),
            }
            for doc, score in zip(docs, doc_scores)
            ]

        outputs.append({
            "query": q,
            "results": query_result,
        })

        return outputs