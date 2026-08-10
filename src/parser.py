from .exceptions import CLIError, ParsingError, FileAccessError
from pathlib import Path
import json


class Parser:

    def read_from_file(self, file_path):
        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                content = f.read()
            return content
        except PermissionError:
            raise PermissionError(f"Permission Denied: ({file_path})")
        except FileNotFoundError:
            raise FileNotFoundError(f"File Not Found: ({file_path})")
        except Exception:
            raise Exception(f"Something Went Wrong: {file_path}")

    def validate_number(self, var, name):
        if isinstance(var, bool):
            raise CLIError("--k must be an integer (e.g. --k=5).")
        if not isinstance(var, int):
            raise CLIError(f"{name} should be a valid integer.")
        if var <= 0:
            raise CLIError(f"{name} must be greater than 0.")

    def validate_argument(self, var, name):
        if not isinstance(var, str):
            raise CLIError(f"{name} must be a valid str.")

    def validate_file(self, path, name):
        file = Path(str(path))
        if not file.is_file():
            raise CLIError(f"'{name}' Path must be valid.")
        content = self.read_from_file(file)
        if not content:
            raise ParsingError(f"'{name}' is empty.")

    def validate_dir(self, path, name):
        try:
            dir = Path(str(path))
            dir.mkdir(parents=True, exist_ok=True)
            return dir
        except Exception:
            pass
        if not dir.is_dir():
            raise CLIError(f"'{name}' Path must be valid.")

    def load_data(self, path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data
        except PermissionError:
            raise FileAccessError(
                f"Permission denied while accessing '{path}'.")
        except json.JSONDecodeError:
            raise ParsingError(f"Data in '{path}' must be valid JSON.")

    def dump_to_dir(self, path, data):
        # path = Path(path)
        # path.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data.model_dump(), f, indent=2)
