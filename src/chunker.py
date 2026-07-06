import ast
from .parser import Parser
import json
from pathlib import Path


class Chunker:

    def __init__(self):
        self.parser = Parser()
        self.max_tokens = 2000
        self.output = "data_chunked.json"
        self.folder_to_chunk = "vllm-0.10.1"
    
    def parsing(self, file):
        source = self.parser.read_from_file(file)
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        return tree, lines

    def splitting_content(self, content):
        chunks = []
        start = 0

        while start < len(content):
            end = min(start + self.max_tokens, len(content))

            # Last chunk
            if end == len(content):
                chunks.append(content[start:end])
                break

            # Prefer blank line
            split = content.rfind("\n\n", start, end)

            # Otherwise normal line break
            if split == -1:
                split = content.rfind("\n", start, end)

            # If no line break exists, hard split
            if split == -1 or split <= start:
                split = end

            chunks.append(content[start:split])
            start = split

        return chunks
        


    def run(self):
        all = []
        i = 1
        folder = Path(self.folder_to_chunk)

        for file in folder.rglob("*.py"):
            if any(skip in file.parts for skip in [".git", ".venv", "__pycache__"]):
                continue

            chunks, i = self.process(file, i)
            all.extend(chunks)

        self.save_output(all)
    
    def process(self, file, i):
        tree, lines = self.parsing(file)
        chunks = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                chunk_type = "class"
            elif isinstance(node, ast.FunctionDef):
                chunk_type = "function"
            elif isinstance(node, ast.AsyncFunctionDef):
                chunk_type = "async_function"
            else:
                continue
            
            start = node.lineno - 1
            end = node.end_lineno
            content = "".join(lines[start:end])

            parts = self.splitting_content(content)
            for part_num, part_content in enumerate(parts):
                chunks.append({
                    "id": i,
                    "file": str(file),
                    "type": chunk_type,
                    "name": node.name,
                    "part_id": part_num,
                    "total_parts": len(parts),
                    "content": part_content
                })

                i += 1
        
        return chunks, i
    
    def save_output(self, data):
        with open(self.output, 'w') as f:
            json.dump(data, f, indent=2)





       
    
    