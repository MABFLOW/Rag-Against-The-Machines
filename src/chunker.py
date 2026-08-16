import ast
from .parser import Parser
import json
from pathlib import Path
from tqdm import tqdm
from .models import ChunkModel
from typing import Any
from .exceptions import FileAccessError


class Chunker:
    """Splits raw source and text files into chunks for indexing.

    Walks a folder of raw files, splits each file's content into
    size-bounded chunks, and writes the resulting chunks to a JSON file.
    """

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initializes the Chunker.

        Args:
            max_chunk_size: Maximum number of characters allowed per chunk.
        """
        self.parser = Parser()
        self.max_chunk_size = max_chunk_size
        self.folder_to_chunk = "data/raw/vllm-0.10.1"
        self.output = "data/processed"

    def parse_python_files(self, file: Path) -> tuple[ast.Module, list[str]]:
        """Parses a Python source file into an AST and its source lines.

        Args:
            file: Path to the Python file to parse.

        Returns:
            A tuple of the parsed AST module and the list of source lines
            (each line retaining its trailing newline character).
        """
        source = self.parser.read_from_file(file)
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        return tree, lines

    def splitting_content(self, content: str) -> list[dict[str, Any]]:
        """Splits content into chunks no larger than ``max_chunk_size``.

        Prefers splitting at paragraph breaks, falling back to line breaks,
        and applies a small overlap between consecutive chunks.

        Args:
            content: The full text content to split.

        Returns:
            A list of dicts, each with ``content``, ``start_char``, and
            ``last_char`` keys describing a chunk and its offsets relative
            to the original content.
        """
        chunks: list[dict[str, Any]] = []
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

    def run(self) -> None:
        """Runs the full chunking pipeline over ``folder_to_chunk``.

        Discovers eligible files, chunks each of them, and writes the
        combined chunks to ``output`` via :meth:`save_output`.
        """
        all_chunks: list[ChunkModel] = []
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
        print("Ingestion complete!", end=" ")
        print(f"Indexed {len(all_chunks)} chunks under {self.output}")

    def process_python(
        self, file: Path, i: int
    ) -> tuple[list[ChunkModel], int]:
        """Chunks a Python file by its top-level classes and functions.

        Args:
            file: Path to the Python file to process.
            i: The starting chunk id to assign to the first produced chunk.

        Returns:
            A tuple of the list of produced :class:`ChunkModel` instances
            and the next available chunk id.
        """
        tree, lines = self.parse_python_files(file)
        chunks: list[ChunkModel] = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                chunk_type = "class"
            elif isinstance(node, ast.FunctionDef):
                chunk_type = "function"
            elif isinstance(node, ast.AsyncFunctionDef):
                chunk_type = "async_function"
            else:
                chunk_type = "code"

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
                    name=getattr(node, "name", file.stem),
                    part_id=part_num,
                    total_parts=len(parts),
                    content=part_content['content'],
                    first_character=start_char + part_content["start_char"],
                    last_character=start_char + part_content["last_char"],
                ))

                i += 1

        return chunks, i

    def process_txt(
        self, file: Path, chunk_id: int
    ) -> tuple[list[ChunkModel], int]:
        """Chunks a text or Markdown file.

        Args:
            file: Path to the ``.txt`` or ``.md`` file to process.
            chunk_id: The starting chunk id to assign to the first produced
                chunk.

        Returns:
            A tuple of the list of produced :class:`ChunkModel` instances
            and the next available chunk id.
        """
        content = self.parser.read_from_file(file)
        parts = self.splitting_content(content)

        chunks: list[ChunkModel] = []

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

    def save_output(self, data: list[ChunkModel]) -> None:
        """Writes chunks to ``chunks.json`` under the output directory.

        Args:
            data: The list of :class:`ChunkModel` instances to serialize.
        """
        path = Path(self.output)
        path.mkdir(parents=True, exist_ok=True)

        try:
            with open(path / "chunks.json", 'w') as f:
                json.dump([chunk.model_dump() for chunk in data], f, indent=2)
        except FileNotFoundError:
            raise FileAccessError(f"{path}/chunks.json File doesn't exist...")