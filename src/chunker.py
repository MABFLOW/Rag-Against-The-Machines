import ast
from .parser import Parser
import json
from pathlib import Path
from tqdm import tqdm
from .models import ChunkModel


class Chunker:

    def __init__(self, max_chunk_size=2000):
        self.parser = Parser()
        self.max_chunk_size = max_chunk_size
        self.folder_to_chunk = "data/raw/vllm-0.10.1"
        self.output = "data/processed"

    def parse_python_files(self, file):
        source = self.parser.read_from_file(file)
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        return tree, lines

    def splitting_content(self, content):
        chunks = []
        start = 0
        while start < len(content):
            end = min(start + self.max_chunk_size, len(content))

            if end == len(content):
                chunks.append(
                    {
                        "content": content[start:end],
                        "start_char": start,
                        "last_char": end
                    }
                )
                break

            split = content.rfind("\n\n", start, end)

            if split == -1:
                split = content.rfind("\n", start, end)

            if split == -1 or split <= start:
                split = end

            chunks.append(

                    {
                        "content": content[start:split],
                        "start_char": start,
                        "last_char": split
                    }
            )

            next_start = split - 200

            if next_start <= start:
                next_start = split

            start = next_start

        return chunks

    def run(self):
        all_chunks = []
        chunk_id = 1
        folder = Path(self.folder_to_chunk)

        skip_folders = {".git", ".venv", "__pycache__"}

        files = [
            file for file in folder.rglob("*")
            if file.is_file()
            and not any(part in skip_folders for part in file.parts)
            and file.suffix in {".md", ".txt", ".py"}
        ]

        for file in tqdm(files, desc="Chuncking files", unit="file"):
            if not file.is_file():
                continue

            if any(part in skip_folders for part in file.parts):
                continue

            if file.suffix in {".md", ".txt"}:
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

            start_char = sum(
                len(line)
                for line in lines[: node.lineno - 1]) + node.col_offset

            parts = self.splitting_content(content)
            for part_num, part_content in enumerate(parts):
                chunks.append(ChunkModel(
                    id=i,
                    file_path=str(file),
                    type=chunk_type,
                    name=node.name,
                    part_id=part_num,
                    total_parts=len(parts),
                    content=part_content['content'],
                    first_character=start_char + part_content["start_char"],
                    last_character=start_char + part_content["last_char"],
                ))

                i += 1

        return chunks, i

    def process_txt(self, file, chunk_id):
        content = self.parser.read_from_file(file)
        parts = self.splitting_content(content)

        chunks = []

        for part_number, part_content in enumerate(parts, start=1):
            chunks.append(ChunkModel(
                id=chunk_id,
                file_path=str(file),
                type="text",
                name=file.stem,
                part_id=part_number,
                total_parts=len(parts),
                content=part_content['content'],
                first_character=part_content['start_char'],
                last_character=part_content['last_char']
            ))

            chunk_id += 1

        return chunks, chunk_id

    def save_output(self, data):
        path = Path(self.output)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "chunks.json", 'w') as f:
            json.dump([chunk.model_dump() for chunk in data], f, indent=2)
