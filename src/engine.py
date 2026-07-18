from .chunker import Chunker
from .indexer import Indexer
from .generator import Generator
import json


class Engine:

    def __init__(self, file="data_chunked.json"):
        self.file = file
        self.generator = Generator()

    
    def index(self, max_chunk_size=2000):
        chunker = Chunker(max_chunk_size=max_chunk_size)
        chunker.run()
        self.indexer = Indexer(self.file)

        self.indexer.index()
        self.indexer.load()

    
    def search(self, query, k):
        res = self.indexer.search(query, k)
        
        contexts = [
            source["content"]
            for source in res[0]['results']
        ]

        print(contexts)
        return contexts
    
    def search_dataset(self, dataset_path, k, save_directory):
        pass

    def answer(self, query, k):
        pass

    def answer_dataset(self, student_search_results_path, save_directory):
        pass

    def evaluate(self, student_search_results_path, dataset_path):
        pass

