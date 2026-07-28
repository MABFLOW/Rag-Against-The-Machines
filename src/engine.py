from .chunker import Chunker
from .indexer import Indexer
from .generator import Generator
import json
from .parser import Parser


class Engine:

    def __init__(self, file="data/processed/chunks.json"):
        self.file = file
        self.generator = Generator()
        self.parser = Parser()
        self.indexer = Indexer(self.file)


    
    def index(self, max_chunk_size=2000):
        chunker = Chunker(max_chunk_size=max_chunk_size)
        chunker.run()

        self.indexer.index()

    
    def search(self, query, k):
        self.indexer.load()

        search_results = self.indexer.search(query, k)
        
        return search_results
    def search_dataset(self, dataset_path, k, save_directory):
        self.indexer = Indexer(self.file)
        self.indexer.load()

        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        entries = dataset["rag_questions"]
        queries = [entry["question"] for entry in entries]

        search_results = self.indexer.search(queries, k)

        # Overwrite the auto-generated question_ids with the dataset's real ones,
        # matching by position (search_results preserves query order).
        for result, entry in zip(search_results.search_results, entries):
            result.question_id = entry["question_id"]
        from pathlib import Path
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "StudentSearchResults.json", 'w') as f:
            json.dump(search_results.model_dump(), f, indent=2)

    
    def answer(self, query, k):
        self.indexer.load()

        search_results = self.indexer.search(query, k)

        # search_results.search_results[0] is this single query's MinimalSearchResults
        sources = search_results.search_results[0].retrieved_sources

        # MinimalSource only has file_path + offsets, no content —
        # so read the actual text back from disk using those offsets
        contexts = []
        for source in sources:
            with open(source.file_path, 'r') as f:
                text = f.read()
            contexts.append(text[source.first_character_index:source.last_character_index])

        answer_text = self.generator.generate(query, contexts)

        return answer_text


    def answer_dataset(self, student_search_results_path, save_directory=None):

        with open(student_search_results_path, 'r') as f:
            student_search_results = json.load(f)


        entries = student_search_results["rag_questions"]
        queries = [entry["question"] for entry in entries]
        print(queries)
        



    def evaluate(self, student_search_results_path, dataset_path):
        pass

