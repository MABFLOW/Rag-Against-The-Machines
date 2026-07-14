import ast
from .parser import Parser
import json
from pathlib import Path
from tqdm import tqdm

class Chunker:

    def __init__(self, output= "data_chunked.json", folder_to_chunk = "vllm-0.10.1", max_tokens = 2000):
        self.parser = Parser()
        self.output = output
        self.folder_to_chunk = folder_to_chunk
        self.max_tokens = max_tokens
    
    def parse_python_files(self, file):
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
        all_chunks = []
        chunk_id = 1
        folder = Path(self.folder_to_chunk)

        skip_folders = {".git", ".venv", "__pycache__"}

        files = [file for file in folder.rglob("*") if file.is_file() and not any(part in skip_folders for part in file.parts)
        and file.suffix in {".md", ".txt", ".py"}]
        

        for file in tqdm(files, desc="Chuncking files", unit="file"):
            if not file.is_file():
                continue

            if any(part in skip_folders for part in file.parts):
                continue

            if file.suffix == ".md":
                chunks, chunk_id = self.process_readme(file, chunk_id)

            elif file.suffix == ".txt":
                chunks, chunk_id = self.process_txt(file, chunk_id)

            elif file.suffix == ".py":
                chunks, chunk_id = self.process_python(file, chunk_id)

            else:
                continue

            all_chunks.extend(chunks)

        self.save_output(all_chunks)

    def process_python(self, file, i):
        tree, lines = self.parse_python_files(file)
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

            start_char = sum(len(line) for line in lines[: node.lineno - 1]) + node.col_offset
            end_char = (
                sum(len(line) for line in lines[: node.end_lineno - 1])
                + node.end_col_offset
            )

            parts = self.splitting_content(content)
            for part_num, part_content in enumerate(parts):
                chunks.append({
                    "id": i,
                    "file": str(file),
                    "type": chunk_type,
                    "name": node.name,
                    "part_id": part_num,
                    "total_parts": len(parts),
                    "content": part_content,
                    "first_character": start_char,
                    "last_character": end_char
                })

                i += 1
        
        return chunks, i
    
    def process_readme(self, file, chunk_id):
        content = self.parser.read_from_file(file)
        lines = content.splitlines(keepends=True)

        current = []
        sections = []
        chunks = []
        for line in lines:
            if line.startswith("#") and current:
                sections.append("".join(current))
                current = []            
            current.append(line)
        
        if current:
            sections.append("".join(current))
        
        for section in sections:
            parts = self.splitting_content(section)

            heading = "Introduction"

            first_line = section.splitlines()[0] if section.splitlines() else ""

            if first_line.startswith("#"):
                heading = first_line.lstrip("#").strip()

            for part_num, part_content in enumerate(parts, start=1):
                chunks.append({
                    "id": chunk_id,
                    "file": str(file),
                    "type": "markdown_section",
                    "name": heading,
                    "part_id": part_num,
                    "total_parts": len(parts),
                    "content": part_content,
                    "first_character": 0,
                    "last_character": 1,
                })

                chunk_id += 1

        return chunks, chunk_id
        

    def process_txt(self, file, chunk_id):
        content = self.parser.read_from_file(file)
        parts = self.splitting_content(content)

        chunks = []

        for part_number, part_content in enumerate(parts, start=1):
            chunks.append({
                "id": chunk_id,
                "file": str(file),
                "type": "text",
                "name": file.stem,
                "part_id": part_number,
                "total_parts": len(parts),
                "content": part_content,
                "first_character": 0,
                "last_character": 1,

            })

            chunk_id += 1

        return chunks, chunk_id

        

    def save_output(self, data):
        with open(self.output, 'w') as f:
            json.dump(data, f, indent=2)





       
    
    