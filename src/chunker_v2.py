import ast
from .parser import Parser


class AstChunker():

    def __init__(self, file):
        self.parser = Parser()
        self.source = self.parser.read_from_file(file)
        self.tree = ast.parse(self.source)
    
    