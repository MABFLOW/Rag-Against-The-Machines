from .parser import Parser
from enum import Enum
import json 


class Separators(Enum):
    DEF = "def"
    CLASS = "class"
    ASY_DEF = "async def"

class Chuncker:

    def __init__(self, file):
        self.files = [file]
        self.parser = Parser()
        self.file = ""
        self.chunks = []
        self.tokens = 2000

    def run(self):
        for file in self.files:
            self.file = file
            data = self.parser.read_from_file(file)
            self.process(data)
    
    def process(self, data):
        current = None
        i = 0

        for line in data:
            if line.strip().startswith(('class', 'def', 'async def')):
                if current is not None:
                    self.chunks.append(current)
                    i += 1

                definition = self.check(line)
                
                current = {
                    'id': i + 1,
                    'file': self.file,
                    'type': definition,
                    'name': self.extract_name(line),
                    'content': line
                }
            else:
                if current is not None:
                    current['content'] += line
            
            
        if current:
            self.chunks.append(current)
            i+=1
        
        self.save_output()
        
    
    def extract_name(self, line):
        definition = self.check(line)

        # remove leading spaces
        line = line.strip()

        if definition == "class":
            # class User:
            # → take word after "class"
            name = line.replace("class", "").split(":")[0].strip()
            return name

        if definition in ["def", "async def"]:
            # def login(user):
            # async def fetch_data():
            name = line.replace("async def", "").replace("def", "").split("(")[0].strip()
            return name

        return ""
    

    def check(self, line):
        if line.startswith("class"):
            return "class"
        if line.startswith("def"):
            return "def"
        if line.startswith("async def"):
            return "async def"
        return 0
    
    def save_output(self):
        print("data saved")
        with open('res.json', 'w') as f:
            json.dump(self.chunks, f, indent=2)

