from .chunker import Chunker
from .indexer import Indexer



class Engine:

    def __init__(self, file):
        self.chunker = Chunker()
        self.indexer = Indexer(file)

    
    def run(self):
        self.chunker.run()
        self.indexer.index()
        